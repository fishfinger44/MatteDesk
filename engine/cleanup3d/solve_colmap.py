# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cleanup 3D, package P10 – COLMAP (CUDA) as a subprocess: feature extraction (moving things masked), sequential
matching, mapper, the sparse model read from COLMAP's TEXT export (no pycolmap: the pip build is CPU only and a
separate Python environment is fragile – user decision 2026-11-08).

Every `colmap.exe` command runs inside the GPU lease (`ColmapRunner.run` -> `GPU.use`, engine/gpu_manager.py), a process
per command (nothing stays on the card). Pixel convention: COLMAP puts the centre of the top-left pixel at (0.5, 0.5);
`to_plate` converts its pixels / intrinsics to plate pixels (centre at integer coordinates), undoing the working scale.
"""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


class SolveError(RuntimeError):
    """A camera solve that cannot be finished – the message is Polish, for the user (409 in the API)."""


class SolveCancelled(SolveError):
    def __init__(self, msg: str = "Solve kamery przerwany"):
        super().__init__(msg)


MSG_NO_COLMAP = "Brak programu COLMAP (CUDA) – Doktor → „Zainstaluj COLMAP (CUDA)”"


def find_exe(cfg: Optional[Dict[str, Any]]) -> Path:
    """colmap.exe the Doctor installed (the gate `stage_models.cleanup3d_solve`), else the environment variable
    MD_COLMAP_EXE (tests / a build kept elsewhere). Missing = SolveError with the gate's reason."""
    reason = MSG_NO_COLMAP
    try:
        from engine.stage_models import cleanup3d_solve
        st = cleanup3d_solve(cfg or {})
        if st.get("ok") and st.get("exe"):
            return Path(st["exe"])
        reason = st.get("reason") or reason
    except ImportError:
        pass
    env = os.environ.get("MD_COLMAP_EXE", "")
    if env and Path(env).is_file():
        return Path(env)
    raise SolveError(reason)


class ColmapRunner:
    """Runs `colmap.exe <command> --option value ...`; output goes to `log` (appended)."""

    def __init__(self, exe: Path, log: Optional[Path] = None, timeout_s: float = 6 * 3600.0):
        self.exe = Path(exe)
        self.log = Path(log) if log else None
        self.timeout_s = float(timeout_s)

    def version(self) -> str:
        """First line of `colmap help`, e.g. "COLMAP 4.2.1 (Commit bd1fcf6 on 2026-09-29 with CUDA, ...)"."""
        try:
            r = subprocess.run([str(self.exe), "help"], capture_output=True, text=True, timeout=60)
            for line in (r.stdout or r.stderr or "").splitlines():
                if line.strip():
                    return line.strip()
        except (OSError, subprocess.SubprocessError):
            pass
        return "COLMAP (wersja nieznana)"

    def run(self, command: str, args: Dict[str, Any], cancel_check: Optional[Callable[[], None]] = None) -> None:
        """One COLMAP command under the GPU lease (waits when another work holds the card)."""
        from engine.gpu_manager import GPU
        with GPU.use(("colmap", str(self.exe)), "COLMAP", None, cancel_check):
            self._run(command, args, cancel_check)

    def _run(self, command: str, args: Dict[str, Any], cancel_check: Optional[Callable[[], None]]) -> None:
        cmd = [str(self.exe), command]
        for k, v in args.items():
            cmd += [f"--{k}", str(v)]
        out = open(self.log, "ab") if self.log else subprocess.DEVNULL
        try:
            if self.log:
                out.write(("\n>>> " + " ".join(cmd) + "\n").encode("utf-8"))
                out.flush()
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, creationflags=flags)
            t0 = time.time()
            try:
                while True:
                    try:
                        proc.wait(timeout=0.5)
                        break
                    except subprocess.TimeoutExpired:
                        if cancel_check:
                            cancel_check()
                        if time.time() - t0 > self.timeout_s:
                            raise SolveError(f"COLMAP ({command}): przekroczony czas {self.timeout_s:.0f} s")
            except BaseException:
                proc.kill()
                proc.wait()
                raise
        finally:
            if self.log:
                out.close()
        if proc.returncode != 0:
            raise SolveError(f"COLMAP ({command}) zakończył się błędem {proc.returncode}: {self._tail()}")

    def _tail(self, n: int = 600) -> str:
        try:
            return self.log.read_text(encoding="utf-8", errors="replace").strip()[-n:] if self.log else ""
        except OSError:
            return ""


# ----------------------------------------------------------------------------- the sparse model (COLMAP text export)
@dataclass
class ColmapModel:
    model: str                                       # camera model name (OPENCV, ...)
    width: int
    height: int
    params: np.ndarray                               # the camera model's parameters, COLMAP pixels
    R: Dict[str, np.ndarray] = field(default_factory=dict)        # image name -> 3x3 (cam from world)
    t: Dict[str, np.ndarray] = field(default_factory=dict)        # image name -> (3,)
    obs: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)   # name -> (xy (k, 2), point3D id (k,))
    point_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    errors: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def intrinsics(self) -> np.ndarray:
        """(fx, fy, cx, cy, k1, k2, p1, p2) in COLMAP pixels, from any of the supported camera models."""
        p = [float(v) for v in self.params]
        m = self.model.upper()
        if m == "OPENCV":
            return np.array(p[:8])
        if m == "PINHOLE":
            return np.array(p[:4] + [0, 0, 0, 0])
        if m == "SIMPLE_PINHOLE":
            return np.array([p[0], p[0], p[1], p[2], 0, 0, 0, 0])
        if m == "SIMPLE_RADIAL":
            return np.array([p[0], p[0], p[1], p[2], p[3], 0, 0, 0])
        if m == "RADIAL":
            return np.array([p[0], p[0], p[1], p[2], p[3], p[4], 0, 0])
        raise SolveError(f"COLMAP: nieobsługiwany model kamery {self.model}")


def _quat_to_R(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz) or 1.0
    w, x, y, z = qw / n, qx / n, qy / n, qz / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _data_lines(path: Path) -> List[str]:
    return [ln.rstrip("\r\n") for ln in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if not ln.startswith("#")]


def read_text_model(folder: Path) -> ColmapModel:
    """cameras.txt / images.txt / points3D.txt of `colmap model_converter --output_type TXT` (one camera)."""
    folder = Path(folder)
    cams = [ln.split() for ln in _data_lines(folder / "cameras.txt") if ln.strip()]
    if not cams:
        raise SolveError("COLMAP: pusty model (brak kamery)")
    c = cams[0]
    m = ColmapModel(c[1], int(c[2]), int(c[3]), np.array([float(v) for v in c[4:]]))
    lines = _data_lines(folder / "images.txt")
    while lines and not lines[0].strip():
        lines.pop(0)
    for i in range(0, len(lines) - 1, 2):                     # two lines per image; the second may be empty
        h = lines[i].split()
        if len(h) < 10:
            continue
        name = " ".join(h[9:])
        m.R[name] = _quat_to_R(*[float(v) for v in h[1:5]])
        m.t[name] = np.array([float(v) for v in h[5:8]])
        a = np.array(lines[i + 1].split(), np.float64).reshape(-1, 3) if lines[i + 1].strip() else np.zeros((0, 3))
        keep = a[:, 2] >= 0
        m.obs[name] = (a[keep, :2], a[keep, 2].astype(np.int64))
    ids, xyz, err = [], [], []
    for ln in _data_lines(folder / "points3D.txt"):
        p = ln.split()
        if len(p) < 8:
            continue
        ids.append(int(p[0])); xyz.append([float(v) for v in p[1:4]]); err.append(float(p[7]))
    m.point_ids, m.points, m.errors = np.array(ids, np.int64), np.array(xyz).reshape(-1, 3), np.array(err)
    return m


def to_plate(intr: Sequence[float], scale: float) -> np.ndarray:
    """Intrinsics in COLMAP pixels of the working image (plate x `scale`) -> plate pixels, centre at integers."""
    i = np.array(intr, np.float64)
    i[0], i[1] = i[0] / scale, i[1] / scale
    i[2], i[3] = i[2] / scale - 0.5, i[3] / scale - 0.5
    return i


def xy_to_plate(xy: np.ndarray, scale: float) -> np.ndarray:
    return np.asarray(xy, np.float64) / scale - 0.5


# ----------------------------------------------------------------------------- the COLMAP stages
def image_name(frame: int) -> str:
    return f"f_{int(frame):06d}.jpg"


def frame_of(name: str) -> int:
    return int(Path(name).stem.split("_")[-1])


def default_overlap(n_frames: int) -> int:
    """How many following frames each frame is matched with: all of them for a short shot (57 frames = every pair, as
    the experiment's exhaustive matching), fewer for a long one (about 3600 pairs in total, at least 20)."""
    return max(2, min(n_frames - 1, max(20, 3600 // max(n_frames, 1))))


def run_sfm(runner: Any, ws: Path, n_frames: int, *, masks: bool, max_features: int = 16000,
            peak_threshold: float = 0.002, max_image_size: int = 3840, overlap: Optional[int] = None,
            stage: Optional[Callable[[str], None]] = None, cancel_check: Optional[Callable[[], None]] = None,
            ) -> ColmapModel:
    """feature_extractor -> sequential_matcher -> mapper -> model_converter (TXT) in workspace `ws` (images/ and,
    with `masks`, masks/<image name>.png where 0 = ignore). Returns the model with the most registered images.
    `stage(key)` is called before each step: "features" | "matching" | "mapper"."""
    ws = Path(ws)
    db, sparse = ws / "database.db", ws / "sparse"
    for p in ws.glob("database.db*"):
        p.unlink()
    sparse.mkdir(parents=True, exist_ok=True)
    step = stage or (lambda k: None)
    step("features")
    a = {"database_path": db, "image_path": ws / "images", "ImageReader.camera_model": "OPENCV",
         "ImageReader.single_camera": 1, "SiftExtraction.max_num_features": int(max_features),
         "SiftExtraction.peak_threshold": peak_threshold, "FeatureExtraction.max_image_size": int(max_image_size),
         "FeatureExtraction.use_gpu": 1}
    if masks:
        a["ImageReader.mask_path"] = ws / "masks"
    runner.run("feature_extractor", a, cancel_check)
    step("matching")
    # every pair up to `overlap` frames apart. quadratic_overlap would match ONLY i + 2^k (COLMAP 4.2.1: 279 pairs
    # for 57 frames -> short feature tracks, frames registered on a handful of points)
    runner.run("sequential_matcher", {"database_path": db, "FeatureMatching.use_gpu": 1,
                                      "FeatureMatching.guided_matching": 1,
                                      "SequentialMatching.overlap": default_overlap(n_frames) if overlap is None
                                      else max(2, min(int(overlap), n_frames - 1)),
                                      "SequentialMatching.quadratic_overlap": 0}, cancel_check)
    step("mapper")
    runner.run("mapper", {"database_path": db, "image_path": ws / "images", "output_path": sparse}, cancel_check)
    best: Optional[ColmapModel] = None
    for d in sorted(p for p in sparse.glob("*") if p.is_dir() and not p.name.endswith("_txt")):
        txt = sparse / f"{d.name}_txt"
        txt.mkdir(exist_ok=True)
        runner.run("model_converter", {"input_path": d, "output_path": txt, "output_type": "TXT"}, cancel_check)
        if not (txt / "images.txt").is_file():
            continue
        m = read_text_model(txt)
        if best is None or len(m.R) > len(best.R):
            best = m
    if best is None or not best.R:
        raise SolveError(f"COLMAP nie zarejestrował żadnej z {n_frames} klatek – za mało wspólnych punktów "
                         "(sprawdź maski ruchomych obiektów i ruch kamery)")
    return best
