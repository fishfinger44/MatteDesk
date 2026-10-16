# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/ingest.py - Media ingestion, metadata reading, project creator and proxy generator.

Complies with SPEC_INTERACTIVE_MATTE_TOOL.md (Section 1.0, 3, 4, 12: Stage E1):
- Auto-detects input media: EXR sequences, PNG/TIFF/DPX sequences, MP4/MOV video files.
- Reads resolution, fps, frame range, format, colorspace and bit depth.
- Creates the `<work>/matte_desk/` working structure (via engine.project) and, for new shots only, `shot_manifest_<shot>.json`.
- Zero shot paths in code: all paths and options come from manifest / defaults.json.
- Generates 1920px proxies using exact PlateToDisplay tonemapping for scene-linear material.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    import OpenEXR
    import Imath
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False


def tonemap_plate_to_display(img_linear: np.ndarray, white: float = 1.0) -> np.ndarray:
    """Exact tonemapping matching ComfyUI-PlateMatte PlateToDisplay node.
    
    x = clamp(images, 0, inf) / white
    x = ((x / (x + 0.6)) * 1.6).clamp(0, 1) ** (1 / 1.6)
    
    Args:
        img_linear: float32 image in 0..N range (RGB or RGBA)
        white: linear value mapping to mid-high grey (default 1.0)
    Returns:
        uint8 image in 0..255 sRGB range.
    """
    return (plate_to_display_float(img_linear, white) * 255.0).astype(np.uint8)


def plate_to_display_float(img_linear: np.ndarray, white: float = 1.0) -> np.ndarray:
    """Same PlateToDisplay curve as float32 0..1 – the input of SAM3 / MatAnyone / LTX and of measurements."""
    x = np.maximum(img_linear, 0.0) / float(white)
    x = (x / (x + 0.6)) * 1.6
    return (np.clip(x, 0.0, 1.0) ** (1.0 / 1.6)).astype(np.float32)


class MediaInspector:
    """Inspects file sequences and videos to determine format, resolution, fps, and frame range."""

    FRAME_PATTERNS = [
        re.compile(r"^(.*?)[_.-](\d{3,8})\.([a-zA-Z0-9]+)$"),
        re.compile(r"^(.*?)(\d{3,8})\.([a-zA-Z0-9]+)$")
    ]

    @classmethod
    def scan_directory(cls, dir_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Scan directory and group image sequences by base prefix and extension."""
        p = Path(dir_path)
        if not p.is_dir():
            return []

        files = sorted(os.listdir(str(p)))
        groups: Dict[Tuple[str, str], List[Tuple[int, str]]] = {}

        for f in files:
            full = p / f
            if not full.is_file():
                continue
            ext = full.suffix.lower().lstrip(".")
            if ext not in ("exr", "png", "jpg", "jpeg", "tif", "tiff", "dpx"):
                continue

            name = full.name
            matched = False
            for pat in cls.FRAME_PATTERNS:
                m = pat.match(name)
                if m:
                    prefix, num_str, _ = m.groups()
                    frame_num = int(num_str)
                    delim = name[len(prefix)] if len(name) > len(prefix) and name[len(prefix)] in "_.-" else ""
                    key = (prefix, ext)
                    groups.setdefault(key, []).append((frame_num, name, num_str, delim))
                    matched = True
                    break

        results = []
        for (prefix, ext), frames in groups.items():
            if not frames:
                continue
            frames.sort(key=lambda x: x[0])
            first_frame = frames[0][0]
            last_frame = frames[-1][0]
            count = len(frames)
            num_len = len(frames[0][2])
            delim = frames[0][3]
            hash_pattern = "#" * num_len
            pattern_str = str(p / f"{prefix}{delim}{hash_pattern}.{ext}").replace("\\", "/")

            source_type = "exr_seq" if ext == "exr" else "img_seq"
            results.append({
                "source_type": source_type,
                "extension": ext,
                "first_frame": first_frame,
                "last_frame": last_frame,
                "frame_count": count,
                "pattern": pattern_str,
                "sample_file": str(p / frames[0][1])
            })

        return results

    @classmethod
    def read_metadata(cls, sample_file: str) -> Dict[str, Any]:
        """Read image or video dimensions, channels, bit depth, colorspace hint."""
        p = Path(sample_file)
        ext = p.suffix.lower()

        if ext == ".exr" and HAS_OPENEXR:
            try:
                exr_file = OpenEXR.InputFile(str(p))
                header = exr_file.header()
                dw = header['dataWindow']
                w = dw.max.x - dw.min.x + 1
                h = dw.max.y - dw.min.y + 1
                channels = list(header['channels'].keys())
                bit_depth = 16  # half by default
                if any(c.type == Imath.PixelType(Imath.PixelType.FLOAT) for c in header['channels'].values()):
                    bit_depth = 32

                return {
                    "width": w,
                    "height": h,
                    "channels": channels,
                    "bit_depth": bit_depth,
                    "colorspace": "ACES2065-1",
                    "source_type": "exr_seq"
                }
            except Exception:
                pass

        if ext in (".mp4", ".mov", ".mkv", ".avi"):
            cap = cv2.VideoCapture(sample_file)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                return {
                    "width": w,
                    "height": h,
                    "fps": round(fps, 3),
                    "first_frame": 1,
                    "last_frame": total_frames,
                    "frame_count": total_frames,
                    "bit_depth": 8,
                    "colorspace": "sRGB",
                    "source_type": "video"
                }

        # Standard image fallback
        img = cv2.imread(sample_file, cv2.IMREAD_UNCHANGED)
        if img is not None:
            h, w = img.shape[:2]
            bit_depth = 16 if img.dtype == np.uint16 else 8
            return {
                "width": w,
                "height": h,
                "channels": img.shape[2] if len(img.shape) > 2 else 1,
                "bit_depth": bit_depth,
                "colorspace": "sRGB",
                "source_type": "img_seq"
            }

        # Unknown: no guessed resolution (the wizard asks the user instead)
        return {"width": None, "height": None, "bit_depth": None, "colorspace": "", "source_type": "unknown"}

    @classmethod
    def resolve_sequence_pattern(cls, path_or_pattern: str) -> Dict[str, Any]:
        """Convert a path, folder, or pattern into a resolved sequence pattern and info."""
        raw = path_or_pattern.strip().replace("\\", "/")
        if not raw:
            return {"pattern": "", "exists": False, "frame_count": 0}

        # If it's already a pattern with '#'
        if "#" in raw:
            p = Path(raw)
            parent = p.parent
            if parent.is_dir():
                prefix = p.name.split("#")[0]
                suffix = p.name.split("#")[-1]
                matches = [f for f in os.listdir(str(parent)) if f.startswith(prefix) and f.endswith(suffix)]
                nums = sorted(int(f[len(prefix):len(f) - len(suffix)]) for f in matches
                              if f[len(prefix):len(f) - len(suffix)].isdigit())
                if matches:
                    sample = parent / sorted(matches)[0]
                    meta = cls.read_metadata(str(sample))
                    frame_range = {"first_frame": nums[0], "last_frame": nums[-1]} if nums else {}
                    return {
                        "pattern": raw,
                        "exists": True,
                        "frame_count": len(matches),
                        "sample_file": str(sample),
                        **frame_range,
                        **meta
                    }
                return {"pattern": raw, "exists": True, "frame_count": 0}
            return {"pattern": raw, "exists": False, "frame_count": 0}

        p = Path(raw)
        # If it's a directory
        if p.is_dir():
            seqs = cls.scan_directory(p)
            if seqs:
                best = max(seqs, key=lambda s: s.get("frame_count", 0))
                meta = cls.read_metadata(best["sample_file"])
                return {**best, **meta, "exists": True}
            return {"pattern": raw, "exists": True, "frame_count": 0}

        # If it's a specific file in a sequence
        if p.is_file():
            parent = p.parent
            seqs = cls.scan_directory(parent)
            if seqs:
                ext = p.suffix.lower().lstrip(".")
                for s in seqs:
                    if s.get("extension") == ext:
                        meta = cls.read_metadata(s["sample_file"])
                        return {**s, **meta, "exists": True}
            meta = cls.read_metadata(str(p))
            return {"pattern": raw, "exists": True, "frame_count": 1, **meta}

        return {"pattern": raw, "exists": False, "frame_count": 0}


VIDEO_EXT = (".mp4", ".mov", ".mkv", ".avi", ".mxf")


def _ffmpeg_bin(name: str = "ffmpeg") -> str:
    from engine.config import load_config
    p = load_config().get("ffmpeg_path") or "ffmpeg"
    if name == "ffprobe":
        q = Path(p)
        return str(q.with_name(q.name.replace("ffmpeg", "ffprobe"))) if q.name.startswith("ffmpeg") else "ffprobe"
    return p


def probe_video(path: Union[str, Path]) -> Dict[str, Any]:
    """Resolution, fps, frame count, bit depth and colour hints from the file header (ffprobe)."""
    import subprocess
    out = subprocess.run([_ffmpeg_bin("ffprobe"), "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=width,height,r_frame_rate,nb_read_packets,pix_fmt,bits_per_raw_sample,"
                          "color_space,color_transfer,color_primaries", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True)
    st = json.loads(out.stdout)["streams"][0]
    num, den = (st.get("r_frame_rate") or "24/1").split("/")
    pix = st.get("pix_fmt", "")
    bits = int(st.get("bits_per_raw_sample") or (10 if "10" in pix else 12 if "12" in pix else 16 if "16" in pix else 8))
    return {"width": int(st["width"]), "height": int(st["height"]), "fps": round(float(num) / float(den or 1), 3),
            "frame_count": int(st.get("nb_read_packets") or 0), "bit_depth": bits, "pix_fmt": pix,
            "colorspace": "sRGB" if st.get("color_transfer") in (None, "", "bt709", "iec61966-2-1", "unknown") else st.get("color_transfer"),
            "source_type": "video"}


def decode_video(path: Union[str, Path], out_dir: Union[str, Path], first_frame: int = 1) -> Dict[str, Any]:
    """Video -> image sequence in the shot work folder (SPEC 1.0): PNG 8-bit for 8-bit sources, PNG 16-bit
    otherwise; frame numbers from `first_frame`. Returns probe info + `pattern`."""
    import subprocess
    info = probe_video(path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(path).stem
    pix = "rgb24" if info["bit_depth"] <= 8 else "rgb48be"
    subprocess.run([_ffmpeg_bin(), "-v", "error", "-y", "-i", str(path), "-start_number", str(first_frame),
                    "-pix_fmt", pix, str(out / f"{stem}_%04d.png")], check=True)
    n = len([x for x in os.listdir(out) if x.startswith(stem + "_") and x.endswith(".png")])
    info.update({"pattern": str(out / f"{stem}_####.png").replace("\\", "/"), "first_frame": first_frame,
                 "last_frame": first_frame + n - 1, "frame_count": n, "source_video": str(path)})
    return info


class ProjectCreator:
    """Initializes a MatteDesk project inside <work>/matte_desk/ (manifest is read-only)."""

    def __init__(self, defaults_path: Optional[Path] = None):
        repo_dir = Path(__file__).resolve().parent.parent
        self.defaults_path = defaults_path or (repo_dir / "defaults.json")
        self.defaults = self._load_defaults()

    def _load_defaults(self) -> Dict[str, Any]:
        if self.defaults_path.exists():
            try:
                with open(self.defaults_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "project": {"colorspace": "ACES2065-1", "shutter_deg": 180, "proxy_max_side": 1920},
            "object": {"edge_mode": "auto", "use_backlit": True, "band_scale": 1.0}
        }

    @classmethod
    def inspect_working_dir(cls, dir_path: Union[str, Path]) -> Dict[str, Any]:
        """Inspect a directory to see if shot_manifest_*.json exists, or scan for plates."""
        p = Path(dir_path)
        if not p.is_dir():
            return {
                "exists": False,
                "manifest_found": False,
                "manifest_path": "",
                "shot": p.name,
                "plates": {},
                "detected_plates": {},
                "dir_path": str(p).replace("\\", "/")
            }

        # 1. Search for existing manifest files
        manifest_files = list(p.glob("shot_manifest_*.json"))
        if manifest_files:
            m_file = manifest_files[0]
            try:
                with open(m_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return {
                    "exists": True,
                    "manifest_found": True,
                    "manifest_path": str(m_file.resolve()).replace("\\", "/"),
                    "shot": data.get("shot", p.name),
                    "plates": data.get("plates", {}),
                    "detected_plates": data.get("plates", {}),
                    "first": data.get("first", 1),
                    "last": data.get("last", 100),
                    "plate_w": data.get("plate_w"),
                    "plate_h": data.get("plate_h"),
                    "fps": data.get("fps", 24),
                    "colorspace": data.get("colorspace", "ACES2065-1"),
                    "dir_path": str(p.resolve()).replace("\\", "/")
                }
            except Exception:
                pass

        # 2. If no manifest found, search for plates in directory and immediate subdirectories
        detected_plates: Dict[str, str] = {}
        folders_to_check = [p]
        try:
            folders_to_check.extend([d for d in p.iterdir() if d.is_dir()])
        except Exception:
            pass

        for fld in folders_to_check:
            name_lower = fld.name.lower()
            seqs = MediaInspector.scan_directory(fld)
            if not seqs:
                continue
            best = max(seqs, key=lambda s: s.get("frame_count", 0))
            pat = best["pattern"]
            if any(k in name_lower for k in ("clean", "03", "bg", "empty")):
                detected_plates.setdefault("cleanplate", pat)
            elif any(k in name_lower for k in ("backlit", "x2", "silhouette")):
                detected_plates.setdefault("backlit", pat)
            elif any(k in name_lower for k in ("depth", "zdepth")):
                detected_plates.setdefault("depth", pat)
            elif any(k in name_lower for k in ("beauty", "x1", "main", "plate", "fg")):
                detected_plates.setdefault("beauty", pat)

        return {
            "exists": True,
            "manifest_found": False,
            "manifest_path": "",
            "shot": p.name,
            "plates": detected_plates,
            "detected_plates": detected_plates,
            "dir_path": str(p.resolve()).replace("\\", "/")
        }

    def init_project_from_manifest(self, manifest_path: Union[str, Path]) -> Dict[str, Any]:
        """Open a shot: the manifest is only read; tool state goes to <work>/<work_subdir>/project.json."""
        from engine.project import ShotProject

        proj = ShotProject.open(manifest_path, self.defaults)
        return {
            "project": proj.to_dict(),
            "manifest": proj.effective_manifest(),
            "work_dir": str(proj.work_dir),
            "shot_project": proj,
        }

    def create_manifest(
        self,
        shot_name: str,
        work_dir: str,
        beauty_pattern: str,
        first_frame: int,
        last_frame: int,
        plate_w: int,
        plate_h: int,
        fps: float = 24.0,
        colorspace: str = "ACES2065-1",
        source_type: str = "exr_seq",
        bit_depth: int = 16,
        backlit_pattern: str = "",
        cleanplate_pattern: str = ""
    ) -> Dict[str, Any]:
        """Generate shot_manifest_<shot>.json from template (new-shot wizard only, never overwrites)."""
        manifest = {
            "_help": "Generated by MatteDesk Project Creator",
            "shot": shot_name,
            "project": "PROJECT",
            "fps": fps,
            "first": first_frame,
            "last": last_frame,
            "plate_w": plate_w,
            "plate_h": plate_h,
            "colorspace": colorspace,
            "source_type": source_type,
            "bit_depth": bit_depth,
            "shutter_deg": self.defaults.get("project", {}).get("shutter_deg", 180),
            "plates": {
                "beauty": beauty_pattern,
                "backlit": backlit_pattern,
                "cleanplate": cleanplate_pattern,
                "depth": ""
            },
            "work": work_dir,
            "objects": []
        }

        target_file = Path(work_dir) / f"shot_manifest_{shot_name}.json"
        if target_file.exists():
            # The manifest is user input: the wizard creates it only when none exists.
            raise FileExistsError(f"Manifest already exists (read-only for the tool): {target_file}")
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        return manifest


class ProxyGenerator:
    """Generates 1920-scale JPEG proxies for UI timeline and display."""

    @staticmethod
    def resolve_frame_path(pattern: str, frame_num: int) -> Path:
        """Replace #### or %04d in pattern with zero-padded frame number."""
        if "####" in pattern:
            return Path(pattern.replace("####", f"{frame_num:04d}"))
        elif "#" in pattern:
            hashes = len(re.findall(r"#+", pattern)[0])
            fmt = f"{{:0{hashes}d}}"
            return Path(re.sub(r"#+", fmt.format(frame_num), pattern))
        elif "%0" in pattern:
            return Path(pattern % frame_num)
        return Path(pattern)

    @classmethod
    def load_frame_rgb(cls, frame_path: Union[str, Path], colorspace: str = "ACES2065-1") -> np.ndarray:
        """Load single frame and convert to uint8 display sRGB."""
        p = Path(frame_path)
        ext = p.suffix.lower()

        if ext == ".exr" and HAS_OPENEXR:
            try:   # Z30: one OpenEXR call (half kept) + display LUT – bit-identical to the per-channel FLOAT path below
                from engine.exr_io import display_half_u8_rgb, read_half_rgb
                bgr = read_half_rgb(p)
            except Exception:
                bgr = None
            if bgr is not None:
                is_linear = any(term in colorspace.upper() for term in ("ACES", "LINEAR", "RAW"))
                if is_linear:
                    if bgr.dtype == np.float16:
                        return display_half_u8_rgb(bgr)
                    else:
                        out = tonemap_plate_to_display(bgr)
                else:
                    out = np.clip(bgr.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
                return np.ascontiguousarray(out[..., ::-1])
            try:
                exr_file = OpenEXR.InputFile(str(p))
                header = exr_file.header()
                dw = header['dataWindow']
                w = dw.max.x - dw.min.x + 1
                h = dw.max.y - dw.min.y + 1
                pt = Imath.PixelType(Imath.PixelType.FLOAT)

                r_str = exr_file.channel('R', pt)
                g_str = exr_file.channel('G', pt)
                b_str = exr_file.channel('B', pt)

                r = np.frombuffer(r_str, dtype=np.float32).reshape((h, w))
                g = np.frombuffer(g_str, dtype=np.float32).reshape((h, w))
                b = np.frombuffer(b_str, dtype=np.float32).reshape((h, w))
                rgb_linear = np.stack([r, g, b], axis=-1)

                is_linear = any(term in colorspace.upper() for term in ("ACES", "LINEAR", "RAW"))
                if is_linear:
                    return tonemap_plate_to_display(rgb_linear)
                else:
                    return np.clip(rgb_linear * 255.0, 0, 255).astype(np.uint8)
            except Exception:
                pass

        # Standard image formats via OpenCV
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        raise FileNotFoundError(f"Could not read frame from {frame_path}")

    @classmethod
    def render_proxy_file(
        cls,
        source_frame_path: Union[str, Path],
        dest_proxy_path: Union[str, Path],
        colorspace: str = "ACES2065-1",
        max_side: int = 1920,
        jpeg_quality: int = 92
    ) -> Path:
        """Load source frame, scale down to max_side, and save as JPEG proxy."""
        dest = Path(dest_proxy_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        rgb = cls.load_frame_rgb(source_frame_path, colorspace=colorspace)
        h, w = rgb.shape[:2]
        scale = min(max_side / float(w), max_side / float(h), 1.0)
        target_w = max(1, round(w * scale))
        target_h = max(1, round(h * scale))

        if scale < 1.0:
            resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            resized = rgb

        bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(dest), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        return dest

    @classmethod
    def generate_plate_proxies(
        cls,
        plate_pattern: str,
        first_frame: int,
        last_frame: int,
        output_dir: Union[str, Path],
        colorspace: str = "ACES2065-1",
        max_side: int = 1920,
        progress_cb: Optional[Callable[[int, int], None]] = None
    ) -> List[Path]:
        """Generate proxies for an entire frame sequence."""
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        total = last_frame - first_frame + 1
        # Z30: frames rendered in threads (EXR decode, cv2 resize and JPEG encode release the GIL); results in order
        from concurrent.futures import ThreadPoolExecutor
        from engine.qa import default_workers

        def one(f):
            src = cls.resolve_frame_path(plate_pattern, f)
            dest = out_p / f"p_{f:04d}.jpg"
            if src.exists():
                cls.render_proxy_file(src, dest, colorspace=colorspace, max_side=max_side)
                return dest
            return None

        results = []
        pool = ThreadPoolExecutor(max_workers=default_workers())
        try:
            for idx, dest in enumerate(pool.map(one, range(first_frame, last_frame + 1))):
                if dest is not None:
                    results.append(dest)
                if progress_cb:
                    progress_cb(idx + 1, total)   # may raise (job cancelled) -> pending frames dropped
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

        return results
