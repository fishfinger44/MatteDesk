# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/align_grid.py - the local alignment field (SPEC 1b) stored as a small displacement grid (PLAN Z18 step 1).

The thin-plate field is only ever evaluated on a coarse grid (one node per `step` px, default 16) and upsampled
bicubically, so a full-resolution STMap (2 x float32 x plate, ~57 MB per 4096x2731 frame) carries no more
information than the grid itself (~350 KB uncompressed). The grid is the single source of truth: preview,
on-the-fly correction, `<plate>_align` and the Nuke group all go through `upsample()`.

Node positions (explicit, stored in the EXR header): gw = ceil(plate_w / step) columns, gh = ceil(plate_h / step)
rows, node (i, k) at plate pixel x = (k + 0.5) * plate_w / gw - 0.5, y = (i + 0.5) * plate_h / gh - 0.5 - i.e. the
cell centres of the plate split into gw x gh equal cells. That is exactly the sampling `cv2.resize` (half-pixel
centre convention), torch `F.interpolate(align_corners=False)` and a Nuke Reformat "distort" use when scaling the
gw x gh image up to the plate, so the field is not shifted by half a cell (the old code evaluated at 0, 16, 32... px
and resized, which shifted it by ~7.5 px and stretched 171 x 16 = 2736 rows onto 2731). One guard node (PAD) is
evaluated beyond each edge (k = -1 and k = gw), so the bicubic interpolates right up to the plate edge instead of
extrapolating a constant; the arrays are (gh + 2, gw + 2).

File: `grid_####.exr`, displayWindow gw x gh, dataWindow one node larger on every side (the guard nodes),
channels R = dx, G = dy in plate px (positive = aux feature is right of / below beauty, the same sign as
AlignmentFrameResult.dx/dy and the old STMap), float32, ZIP; header attributes md_grid (version), md_plate_w,
md_plate_h, md_step_x, md_step_y, md_pad.

Nuke: Read grid (raw; the bbox is the padded data window) -> Reformat to the plate format (distort, pbb, filter
Simon = cubic a = -0.75 like OpenCV INTER_CUBIC) -> Expression (dx, dy -> ST) -> STMap. The grid is the residual
over the frame's global model (engine.align.compose_remap): the STMap's source input is the aux AFTER the global
Transform / CornerPin2D (`nuke_nodes(src_stack=...)`).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2
import numpy as np

GRID_VERSION = 1
DEFAULT_STEP = 16
PAD = 1                 # guard nodes beyond each plate edge
NUKE_FILTER = "Simon"   # cubic a = -0.75 = OpenCV INTER_CUBIC (Keys = -0.5, Rifman = -1.0)


def grid_shape(plate_w: int, plate_h: int, step: int = DEFAULT_STEP) -> Tuple[int, int]:
    """(gw, gh): nodes per row / column inside the plate (the EXR display window; arrays carry PAD more per side)."""
    return max(2, math.ceil(plate_w / step)), max(2, math.ceil(plate_h / step))


def node_axes(plate_w: int, plate_h: int, step: int = DEFAULT_STEP) -> Tuple[np.ndarray, np.ndarray]:
    """(xs (gw + 2 PAD,), ys (gh + 2 PAD,)) node positions in plate px (float64), guard nodes included."""
    gw, gh = grid_shape(plate_w, plate_h, step)
    xs = (np.arange(-PAD, gw + PAD, dtype=np.float64) + 0.5) * (plate_w / gw) - 0.5
    ys = (np.arange(-PAD, gh + PAD, dtype=np.float64) + 0.5) * (plate_h / gh) - 0.5
    return xs, ys


def node_points(plate_w: int, plate_h: int, step: int = DEFAULT_STEP) -> Tuple[np.ndarray, Tuple[int, int]]:
    """((rows * cols, 2) node (x, y) in plate px, row-major; (rows, cols)) - what the thin-plate field is
    evaluated at (guard nodes included)."""
    xs, ys = node_axes(plate_w, plate_h, step)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel()]), gy.shape


@dataclass
class AlignGrid:
    dx: np.ndarray        # (gh + 2 PAD, gw + 2 PAD) float32, plate px
    dy: np.ndarray
    plate_w: int
    plate_h: int

    @property
    def shape(self) -> Tuple[int, int]:
        """(gw, gh) nodes inside the plate."""
        return self.dx.shape[1] - 2 * PAD, self.dx.shape[0] - 2 * PAD

    @property
    def step(self) -> Tuple[float, float]:
        gw, gh = self.shape
        return self.plate_w / gw, self.plate_h / gh

    @classmethod
    def zeros(cls, plate_w: int, plate_h: int, step: int = DEFAULT_STEP) -> "AlignGrid":
        gw, gh = grid_shape(plate_w, plate_h, step)
        z = np.zeros((gh + 2 * PAD, gw + 2 * PAD), np.float32)
        return cls(z, z.copy(), plate_w, plate_h)


def upsample(grid: AlignGrid, out_w: Optional[int] = None, out_h: Optional[int] = None,
             scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Dense (out_h, out_w) displacement field (float32) from the grid - the one upsampling everything uses.
    Bicubic (OpenCV INTER_CUBIC), edge replicated; out size defaults to the plate. `scale` multiplies the
    displacement (a proxy at 0.25 of the plate wants the field at the proxy size with displacements x 0.25)."""
    out_w = out_w or grid.plate_w
    out_h = out_h or grid.plate_h
    gw, gh = grid.shape
    # output pixel centre (x + 0.5) / out_w of the plate -> grid pixel coordinate (cv2.resize convention),
    # + PAD for the guard nodes at the start of the arrays
    mx = ((np.arange(out_w, dtype=np.float32) + 0.5) * (gw / out_w) - 0.5 + PAD)[None, :].repeat(out_h, 0)
    my = ((np.arange(out_h, dtype=np.float32) + 0.5) * (gh / out_h) - 0.5 + PAD)[:, None].repeat(out_w, 1)
    out = []
    for g in (grid.dx, grid.dy):
        f = cv2.remap(np.ascontiguousarray(g, np.float32), mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        out.append(f if scale == 1.0 else (f * np.float32(scale)))
    return out[0], out[1]


def remap_maps(grid: AlignGrid, out_w: Optional[int] = None, out_h: Optional[int] = None,
               scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """cv2.remap maps of the grid ALONE: sample at (x - dx, y - dy). A frame's full alignment (global model, then
    the grid) is engine.align.compose_remap - use that for anything the user sees."""
    out_w = out_w or grid.plate_w
    out_h = out_h or grid.plate_h
    dx, dy = upsample(grid, out_w, out_h, scale)
    Y, X = np.indices((out_h, out_w), dtype=np.float32)
    return X - dx, Y - dy


def write_grid(path: Path, grid: AlignGrid) -> Path:
    import Imath
    import OpenEXR
    gw, gh = grid.shape
    sx, sy = grid.step
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = OpenEXR.Header(gw, gh)      # displayWindow = the plate's nodes; dataWindow = + guard nodes
    header["dataWindow"] = Imath.Box2i(Imath.V2i(-PAD, -PAD), Imath.V2i(gw - 1 + PAD, gh - 1 + PAD))
    header["channels"] = {c: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)) for c in ("R", "G")}
    header["compression"] = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)
    header["md_grid"] = GRID_VERSION
    header["md_plate_w"] = int(grid.plate_w)
    header["md_plate_h"] = int(grid.plate_h)
    header["md_step_x"] = float(sx)
    header["md_step_y"] = float(sy)
    header["md_pad"] = int(PAD)
    out = OpenEXR.OutputFile(str(path), header)
    out.writePixels({"R": np.ascontiguousarray(grid.dx, np.float32).tobytes(),
                     "G": np.ascontiguousarray(grid.dy, np.float32).tobytes()})
    out.close()
    return path


def read_grid(path: Path) -> AlignGrid:
    import Imath
    import OpenEXR
    f = OpenEXR.InputFile(str(path))
    try:
        hdr = f.header()
        dw = hdr["dataWindow"]
        rows, cols = dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1
        if "md_plate_w" not in hdr or "md_plate_h" not in hdr:
            raise ValueError(f"{path}: to nie jest siatka dopasowania MatteDesk (brak md_plate_w / md_plate_h)")
        if int(hdr.get("md_pad", 0)) != PAD or dw.min.x != -PAD or dw.min.y != -PAD:
            raise ValueError(f"{path}: siatka w innym układzie (md_pad / dataWindow) – zmierz dopasowanie ponownie")
        ch = {c: np.frombuffer(f.channel(c, Imath.PixelType(Imath.PixelType.FLOAT)), np.float32).reshape(rows, cols).copy()
              for c in ("R", "G")}
    finally:
        f.close()
    return AlignGrid(ch["R"], ch["G"], int(hdr["md_plate_w"]), int(hdr["md_plate_h"]))


def stmap_from_grid(grid: AlignGrid, out_exr_path: Path) -> None:
    """Full-resolution STMap EXR (Nuke convention, float32 R = u, G = v) from the grid - only on request."""
    from engine.align import LocalPatchAligner
    dx, dy = upsample(grid)
    LocalPatchAligner(grid.plate_w, grid.plate_h).generate_stmap(dx, dy, Path(out_exr_path))


def nuke_nodes(grid_pattern: str, first_frame: int, last_frame: int, plate_w: int, plate_h: int,
               grid_w: int, grid_h: int, src_stack: str = "$N_dot1") -> str:
    """Inner nodes of the Align group for a grid: Read (raw) -> Reformat to the plate -> Expression -> STMap.
    `src_stack` is the stack variable of the STMap's source (`$N_dot1` = the group input; `$N_global` = the input
    after the global Transform / CornerPin2D when the grid is the residual over it); the STMap takes it as
    `inputs 2` (src, stmap). The Read's format is the display window (grid_w x grid_h); its bbox carries the guard
    nodes, which the Reformat (pbb) scales along, so the cubic filter sees them at the plate edges."""
    return f""" Read {{
  inputs 0
  file "{grid_pattern}"
  first {first_frame}
  last {last_frame}
  origfirst {first_frame}
  origlast {last_frame}
  format "{grid_w} {grid_h} 0 0 {grid_w} {grid_h} 1 grid_format"
  raw true
  name Read_Grid
  xpos 240
  ypos -80
 }}
 Reformat {{
  type "to format"
  format "{plate_w} {plate_h} 0 0 {plate_w} {plate_h} 1 plate_format"
  resize distort
  center true
  filter {NUKE_FILTER}
  pbb true
  name Reformat_Grid
  xpos 240
  ypos -40
 }}
 Expression {{
  expr0 "(x - r + 0.5) / width"
  expr1 "(y + g + 0.5) / height"
  expr2 0
  channel3 none
  name Grid_to_ST
  xpos 240
  ypos 0
 }}
push {src_stack}
 STMap {{
  inputs 2
  uv rgb
  channels all
  filter Cubic
  name STMap1
  xpos 120
  ypos 60
 }}"""
