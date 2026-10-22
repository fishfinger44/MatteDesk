# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/coords.py - Coordinate and frame index transformations for MatteDesk.

Complies with SPEC_INTERACTIVE_MATTE_TOOL.md (Section 3, 4, 5.1, 12: Stage E1):
- Single source of truth for all spatial transformations:
    plate (full res) <-> proxy (UI display) <-> crop (square section) <-> 1008x1008 (SAM 3.1 input)
- Frame number mapping: global plate frame number (e.g. 1..193 or 1001..1193) <-> Comfy batch index (0..N-1).
- Mirror padding & bounds calculation.
- Round-trip accuracy guaranteed < 0.5 px.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class CropBox:
    """Represents a square or rectangular crop in plate pixel coordinates.
    
    Attributes:
        x: Left edge (int/float)
        y: Top edge (int/float)
        w: Width (must equal h for SAM 3.1 square crops)
        h: Height
        pad_l: Left mirror padding pixels (if crop exceeds plate left bound)
        pad_t: Top mirror padding pixels
        pad_r: Right mirror padding pixels
        pad_b: Bottom mirror padding pixels
    """
    x: float
    y: float
    w: float
    h: float
    pad_l: float = 0.0
    pad_t: float = 0.0
    pad_r: float = 0.0
    pad_b: float = 0.0

    @property
    def side(self) -> float:
        return max(self.total_w, self.total_h)

    @property
    def total_w(self) -> float:
        return self.w + self.pad_l + self.pad_r

    @property
    def total_h(self) -> float:
        return self.h + self.pad_t + self.pad_b


class FrameMapper:
    """Maps between absolute plate frame numbers and 0-based Comfy batch indices."""

    def __init__(self, first_frame: int, last_frame: int):
        if last_frame < first_frame:
            raise ValueError(f"last_frame ({last_frame}) must be >= first_frame ({first_frame})")
        self.first_frame = first_frame
        self.last_frame = last_frame

    @property
    def count(self) -> int:
        return self.last_frame - self.first_frame + 1

    def frame_to_index(self, frame_num: int) -> int:
        idx = frame_num - self.first_frame
        if idx < 0 or idx >= self.count:
            raise IndexError(f"Frame {frame_num} is outside range [{self.first_frame}, {self.last_frame}]")
        return idx

    def index_to_frame(self, index: int) -> int:
        if index < 0 or index >= self.count:
            raise IndexError(f"Index {index} is outside range [0, {self.count - 1}]")
        return self.first_frame + index

    def slice_to_frames(self, start_idx: int, end_idx: int) -> List[int]:
        """Convert a 0-based slice range [start_idx, end_idx] (inclusive) to frame numbers."""
        return [self.index_to_frame(i) for i in range(start_idx, end_idx + 1)]


class CoordTransformer:
    """Performs spatial coordinate conversions between plate, proxy, crop, and SAM 3.1 (1008x1008)."""

    def __init__(self, plate_w: int, plate_h: int, proxy_max_side: int = 1920):
        self.plate_w = float(plate_w)
        self.plate_h = float(plate_h)
        self.proxy_scale = min(proxy_max_side / self.plate_w, proxy_max_side / self.plate_h, 1.0)
        self.proxy_w = round(self.plate_w * self.proxy_scale)
        self.proxy_h = round(self.plate_h * self.proxy_scale)

    # -------------------------------------------------------------------------
    # Plate <-> Proxy (UI Display)
    # -------------------------------------------------------------------------
    def plate_to_proxy(self, x: float, y: float) -> Tuple[float, float]:
        """Convert plate coords (full resolution) to proxy display coords."""
        px = x * (self.proxy_w / self.plate_w)
        py = y * (self.proxy_h / self.plate_h)
        return (px, py)

    def proxy_to_plate(self, px: float, py: float) -> Tuple[float, float]:
        """Convert proxy display coords to full resolution plate coords."""
        x = px * (self.plate_w / self.proxy_w)
        y = py * (self.plate_h / self.proxy_h)
        return (x, y)

    # -------------------------------------------------------------------------
    # Plate <-> Crop
    # -------------------------------------------------------------------------
    def plate_to_crop(self, x: float, y: float, crop: CropBox) -> Tuple[float, float]:
        """Convert plate coords to crop relative coords (including pad offset)."""
        cx = (x - crop.x) + crop.pad_l
        cy = (y - crop.y) + crop.pad_t
        return (cx, cy)

    def crop_to_plate(self, cx: float, cy: float, crop: CropBox) -> Tuple[float, float]:
        """Convert crop relative coords back to plate full resolution coords."""
        x = (cx - crop.pad_l) + crop.x
        y = (cy - crop.pad_t) + crop.y
        return (x, y)

    # -------------------------------------------------------------------------
    # Crop <-> SAM3 1008x1008
    # -------------------------------------------------------------------------
    def crop_to_sam1008(self, cx: float, cy: float, crop: CropBox) -> Tuple[float, float]:
        """Convert crop relative coords to 1008x1008 SAM 3.1 input coords."""
        total_w = crop.total_w
        total_h = crop.total_h
        if total_w <= 0 or total_h <= 0:
            raise ValueError(f"Crop box dimensions must be positive, got {total_w}x{total_h}")
        sx = cx * (1008.0 / total_w)
        sy = cy * (1008.0 / total_h)
        return (sx, sy)

    def sam1008_to_crop(self, sx: float, sy: float, crop: CropBox) -> Tuple[float, float]:
        """Convert SAM 3.1 1008x1008 coords back to crop relative coords."""
        total_w = crop.total_w
        total_h = crop.total_h
        cx = sx * (total_w / 1008.0)
        cy = sy * (total_h / 1008.0)
        return (cx, cy)

    # -------------------------------------------------------------------------
    # Plate <-> SAM3 1008x1008 (Full Pipeline)
    # -------------------------------------------------------------------------
    def plate_to_sam1008(self, x: float, y: float, crop: CropBox) -> Tuple[float, float]:
        """Direct transformation from plate full res to SAM 3.1 1008x1008."""
        cx, cy = self.plate_to_crop(x, y, crop)
        return self.crop_to_sam1008(cx, cy, crop)

    def sam1008_to_plate(self, sx: float, sy: float, crop: CropBox) -> Tuple[float, float]:
        """Direct transformation from SAM 3.1 1008x1008 to plate full res."""
        cx, cy = self.sam1008_to_crop(sx, sy, crop)
        return self.crop_to_plate(cx, cy, crop)

    # -------------------------------------------------------------------------
    # Batch points transformation
    # -------------------------------------------------------------------------
    def points_plate_to_sam1008(self, points: Sequence[Sequence[float]], crop: CropBox) -> List[Dict[str, int]]:
        """Convert a list of [x, y] plate points to KJNodes PointsEditor JSON format [{"x": int, "y": int}, ...]."""
        res = []
        for pt in points:
            sx, sy = self.plate_to_sam1008(pt[0], pt[1], crop)
            res.append({"x": int(round(sx)), "y": int(round(sy))})
        return res

    def points_sam1008_to_plate(self, points: Sequence[Dict[str, float]], crop: CropBox) -> List[Tuple[float, float]]:
        """Convert KJNodes PointsEditor points from SAM 3.1 1008 to plate coords."""
        res = []
        for pt in points:
            px, py = self.sam1008_to_plate(pt["x"], pt["y"], crop)
            res.append((px, py))
        return res

    # -------------------------------------------------------------------------
    # Crop geometry utilities
    # -------------------------------------------------------------------------
    def create_square_crop(
        self,
        center_x: float,
        center_y: float,
        min_side: float,
        pad_ratio: float = 1.10,
        margin_px: float = 64.0,
        snap_multiple: int = 16,
        allow_shift: bool = True
    ) -> CropBox:
        """Create a square crop box around a center point with padding and multiple-of-N snapping.
        
        Args:
            center_x: Center X on plate
            center_y: Center Y on plate
            min_side: Minimum required dimension (e.g. max(bbox_w, bbox_h))
            pad_ratio: Factor to scale dimension (default 1.10)
            margin_px: Minimum margin around object
            snap_multiple: Multiple to snap side length to (16 for SAM 3.1/MatAnyone2, 32 for LTX-2.5)
            allow_shift: If True, shift crop to stay inside plate bounds before falling back to mirror pad.
        """
        # Snap side up to multiple (integer)
        raw_side = max(min_side * pad_ratio, min_side + margin_px)
        rem = raw_side % snap_multiple
        if rem != 0:
            side = int(round(raw_side + (snap_multiple - rem)))
        else:
            side = int(round(raw_side))

        # Origin rounded to exact whole integer pixels of the plate
        x0 = int(round(center_x - side / 2.0))
        y0 = int(round(center_y - side / 2.0))

        pad_l = 0
        pad_t = 0
        pad_r = 0
        pad_b = 0

        plate_w_int = int(round(self.plate_w))
        plate_h_int = int(round(self.plate_h))

        if allow_shift:
            # Shift within bounds if possible
            if side <= plate_w_int:
                if x0 < 0:
                    x0 = 0
                elif x0 + side > plate_w_int:
                    x0 = plate_w_int - side
            if side <= plate_h_int:
                if y0 < 0:
                    y0 = 0
                elif y0 + side > plate_h_int:
                    y0 = plate_h_int - side

        # If still exceeding bounds (or plate smaller than crop), compute mirror padding
        if x0 < 0:
            pad_l = -x0
            x0 = 0
        if y0 < 0:
            pad_t = -y0
            y0 = 0
        if x0 + (side - pad_l) > plate_w_int:
            pad_r = int((x0 + (side - pad_l)) - plate_w_int)
        if y0 + (side - pad_t) > plate_h_int:
            pad_b = int((y0 + (side - pad_t)) - plate_h_int)

        w = int(side - pad_l - pad_r)
        h = int(side - pad_t - pad_b)

        return CropBox(
            x=float(x0),
            y=float(y0),
            w=float(w),
            h=float(h),
            pad_l=float(pad_l),
            pad_t=float(pad_t),
            pad_r=float(pad_r),
            pad_b=float(pad_b)
        )
