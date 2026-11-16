# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/plate_write.py – a cleaned frame written in the PLATE'S OWN format (user decision 2026-11-12, after the
export rule of AInVFX Paint a plate, Apache-2.0: „every untouched pixel is the source pixel, in the plate's format").

Cleanup (Inpaint / Cleanup 3D) delivers RGB – a patch or the whole frame – so its full frame (`comp/c_####`) is written
as the plate is: the same file type, bit depth, channels and metadata, and outside the patch (mask == 0) the SOURCE
VALUES THEMSELVES (read again from the plate's file, never through float and back):
  EXR   the source header (pixel types half / float, compression, chromaticities, custom attributes, other layers and
        channels – only R, G, B change where the mask is > 0; engine/exr_io.write_exr_like),
  DPX   a copy of the source file with only the image data re-encoded (header, timecode, packing, endianness, padding
        unchanged; engine/dpx.write_like); RGB / RGBA / ABGR descriptors,
  TIFF / PNG  the source's dtype (8 / 16-bit, float TIFF) and channels (alpha copied) – OpenCV writes no metadata,
  else (JPG and the rest)  PNG of the source's depth – lossless, said in the run's notes.
A layout we do not mirror exactly (EXR data window ≠ the frame, a DPX luma plate…) falls back to the old render format
with a note – never a silent change of the picture.
The patch (`patch/p_####`, premultiplied RGBA) of an EXR plate takes the plate's pixel type, compression and colour
attributes (`patch_exr_header`); of other plates it stays PNG at the plate's depth (Nuke reads it over any Read)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

PathLike = Union[str, Path]
# EXR header fields that describe the layout of the file, not the picture: the patch gets its own
_EXR_STRUCT = {"channels", "dataWindow", "displayWindow", "tiles", "type", "chunkCount", "name", "version", "view",
               "multiView"}


def _suffix(p: PathLike) -> str:
    return Path(str(p)).suffix.lower().lstrip(".")


class PlateFormat:
    """The format of the plate's frames, read once from a sample frame. `comp_ext` = the suffix of the full frames."""

    def __init__(self, sample: PathLike, w: int, h: int):
        self.w, self.h = int(w), int(h)
        self.kind = _suffix(sample)
        self.notes: List[str] = []
        self.info: Dict[str, Any] = {"plate": self.kind}
        self.mode: Optional[str] = None      # "exr" | "dpx" | "cv" (same file type via OpenCV) | "png" | None = old format
        try:
            self._probe(Path(str(sample)))
        except Exception as e:      # an unreadable sample: the old render format, said
            self.notes.append(f"format plate'u nieczytelny ({str(e)[:120]}) – zapis jak dotąd")
            self.mode = None
        self.comp_ext = {"exr": "exr", "dpx": "dpx", "cv": self.kind, "png": "png"}.get(self.mode or "")

    @property
    def native(self) -> bool:
        return self.mode is not None

    def _probe(self, p: Path) -> None:
        if self.kind == "exr":
            from engine import exr_io
            import OpenEXR
            f = OpenEXR.InputFile(str(p))
            try:
                hdr = f.header()
            finally:
                f.close()
            dw = hdr["dataWindow"]
            size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
            names = exr_io.rgb_channel_names({k: None for k in hdr["channels"]})
            types = sorted({str(hdr["channels"][n].type) for n in names})
            self.info.update({"pixel_type": "/".join(t.lower() for t in types), "compression": str(hdr["compression"]),
                              "channels": sorted(hdr["channels"])})
            if size != (self.w, self.h) or exr_io._simple_layout(hdr) is None:
                self.notes.append("EXR: okno danych ≠ kadr – pełna klatka jako EXR float32 (bez nagłówka plate'u)")
                return
            self._exr_hdr = hdr
            self.mode = "exr"
        elif self.kind == "dpx":
            from engine import dpx
            i = dpx.info(p)
            self.info.update({"bits": i["bits"], "packing": i["packing"], "descriptor": i["descriptor"],
                              "transfer": i["transfer_name"], "endian": i["endian"]})
            if i["descriptor"] not in (50, 51, 52) or (i["width"], i["height"]) != (self.w, self.h):
                self.notes.append(f"DPX: deskryptor {i['descriptor']} / rozmiar – pełna klatka jak dotąd (PNG 16-bit)")
                return
            self.mode = "dpx"
        else:
            raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise ValueError("brak klatki")
            self.info.update({"dtype": str(raw.dtype), "channels": 1 if raw.ndim == 2 else int(raw.shape[2])})
            if raw.shape[:2] != (self.h, self.w):
                self.notes.append("rozmiar klatki ≠ plate – zapis jak dotąd")
                return
            ints = raw.dtype in (np.uint8, np.uint16)
            if self.kind in ("tif", "tiff") and (ints or raw.dtype == np.float32):
                self.mode = "cv"
                self.notes.append("TIFF: piksele, głębia i kanały jak w plate'cie; metadane TIFF nie są przenoszone")
            elif self.kind == "png" and ints:
                self.mode = "cv"
            elif ints:
                self.mode = "png"
                self.notes.append(f"plate {self.kind.upper()} – pełna klatka jako PNG "
                                  f"{8 if raw.dtype == np.uint8 else 16}-bit (bezstratnie)")
            else:
                self.notes.append(f"plate {self.kind.upper()} {raw.dtype} – zapis jak dotąd")

    def describe(self) -> Dict[str, Any]:
        return {**self.info, "comp": self.comp_ext, "native": self.native, "notes": list(self.notes)}

    # ------------------------------------------------------------------------------------------------- the full frame
    def write_comp(self, out_path: PathLike, src_path: PathLike, comp: np.ndarray, m: np.ndarray) -> None:
        """`comp` = BGR float in the plate's encoding (as the Cleaner reads it: /255, /65535 or scene-linear);
        `m` = the mask; where m == 0 the written value IS the source value."""
        sel = np.asarray(m) > 0
        if self.kind == "exr":
            from engine import exr_io
            ch, hdr = exr_io.read_exr(src_path)
            r, g, b = exr_io.rgb_channel_names(ch)
            for name, i in ((b, 0), (g, 1), (r, 2)):
                ch[name] = np.where(sel, comp[..., i], ch[name])
            exr_io.write_exr_like(out_path, ch, hdr)
            return
        if self.kind == "dpx":
            from engine import dpx
            codes, h = dpx.read(src_path)
            top = (1 << h["bits"]) - 1
            new = np.rint(np.clip(comp, 0.0, 1.0) * top).astype(codes.dtype)
            idx = (1, 2, 3) if h["descriptor"] == 52 else (2, 1, 0)      # B, G, R positions in the file's order
            codes = codes.copy()
            for i, k in enumerate(idx):
                codes[..., k] = np.where(sel, new[..., i], codes[..., k])
            dpx.write_like(out_path, codes, src_path)
            return
        raw = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.shape[:2] != sel.shape:
            raise RuntimeError(f"Brak / zły rozmiar klatki plate'u: {src_path}")
        def enc_of(x: np.ndarray) -> np.ndarray:
            if raw.dtype == np.uint8:
                return np.rint(np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)
            if raw.dtype == np.uint16:
                return np.rint(np.clip(x, 0.0, 1.0) * 65535.0).astype(np.uint16)
            return x.astype(raw.dtype)
        out = raw.copy()
        if raw.ndim == 2:                                                 # a grey plate stays grey
            g = cv2.cvtColor(comp.astype(np.float32), cv2.COLOR_BGR2GRAY)[..., None]
            gv = enc_of(g)[..., 0]
            out = np.where(sel, gv, raw)
        else:
            out[..., :3] = np.where(sel[..., None], enc_of(comp), raw[..., :3])   # alpha (4th channel) copied
        if not cv2.imwrite(str(out_path), out):
            raise RuntimeError(f"Zapis klatki nie powiódł się: {out_path}")

    def comp_name(self, f: int) -> str:
        return f"c_{int(f):04d}.{self.comp_ext}"

    # ------------------------------------------------------------------------------------------------- the patch (EXR)
    def patch_exr(self) -> bool:
        return self.mode == "exr"

    def write_patch_exr(self, out_path: PathLike, patch_bgra: np.ndarray) -> None:
        """The premultiplied RGBA patch with the plate's pixel type, compression and colour attributes."""
        import Imath
        import OpenEXR
        src = self._exr_hdr
        from engine import exr_io
        r, _, _ = exr_io.rgb_channel_names({k: None for k in src["channels"]})
        pt = src["channels"][r].type
        hdr = OpenEXR.Header(self.w, self.h)
        for k, v in src.items():
            if k not in _EXR_STRUCT:
                hdr[k] = v
        hdr["channels"] = {c: Imath.Channel(pt) for c in ("R", "G", "B", "A")}
        dt = np.float16 if str(pt) == "HALF" else np.float32
        planes = {"B": patch_bgra[..., 0], "G": patch_bgra[..., 1], "R": patch_bgra[..., 2], "A": patch_bgra[..., 3]}
        Path(str(out_path)).parent.mkdir(parents=True, exist_ok=True)
        o = OpenEXR.OutputFile(str(out_path), hdr)
        try:
            o.writePixels({c: np.ascontiguousarray(v).astype(dt).tobytes() for c, v in planes.items()})
        finally:
            o.close()


def plate_format(sample: Optional[PathLike], w: int, h: int) -> Optional[PlateFormat]:
    """The plate's format from its first frame, None without a sample (the old render format)."""
    if not sample:
        return None
    return PlateFormat(sample, w, h)
