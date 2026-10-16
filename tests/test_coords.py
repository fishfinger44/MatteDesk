# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""tests/test_coords.py - Unit tests for coords.py.

Acceptance criterion (SPEC Section 12, E1):
- Round-trip accuracy plate <-> proxy <-> crop <-> 1008 < 0.5 px.
- Frame number mapping round-trip and boundary handling.
- Proper padding and bounds calculation.
"""

import pytest
import math
import cv2
import numpy as np
from engine.coords import CoordTransformer, CropBox, FrameMapper


def test_frame_mapper():
    mapper = FrameMapper(first_frame=1001, last_frame=1193)
    assert mapper.count == 193

    # Check boundaries
    assert mapper.frame_to_index(1001) == 0
    assert mapper.frame_to_index(1193) == 192
    assert mapper.index_to_frame(0) == 1001
    assert mapper.index_to_frame(192) == 1193

    # Check arbitrary frame
    f = 1089
    idx = mapper.frame_to_index(f)
    assert idx == 88
    assert mapper.index_to_frame(idx) == f

    # Out of range
    with pytest.raises(IndexError):
        mapper.frame_to_index(1000)
    with pytest.raises(IndexError):
        mapper.frame_to_index(1194)
    with pytest.raises(IndexError):
        mapper.index_to_frame(-1)
    with pytest.raises(IndexError):
        mapper.index_to_frame(193)


def test_plate_to_proxy_roundtrip():
    # 4K plate (4096 x 2731) downscaled to 1920 proxy
    ct = CoordTransformer(plate_w=4096, plate_h=2731, proxy_max_side=1920)

    test_points = [
        (0.0, 0.0),
        (4096.0, 2731.0),
        (2048.0, 1365.5),
        (1412.0, 508.0),
        (2712.0, 2064.0),
        (123.45, 678.90),
    ]

    for x, y in test_points:
        px, py = ct.plate_to_proxy(x, y)
        rx, ry = ct.proxy_to_plate(px, py)
        err = math.hypot(rx - x, ry - y)
        assert err < 0.5, f"Plate <-> Proxy roundtrip error {err:.4f} px at ({x}, {y})"


def test_plate_to_crop_roundtrip():
    ct = CoordTransformer(plate_w=4096, plate_h=2731)

    # Standard interior crop
    crop = CropBox(x=1000.0, y=500.0, w=1024.0, h=1024.0)

    points = [
        (1000.0, 500.0),
        (1500.0, 800.0),
        (2024.0, 1524.0),
        (1234.5, 678.9),
    ]

    for x, y in points:
        cx, cy = ct.plate_to_crop(x, y, crop)
        rx, ry = ct.crop_to_plate(cx, cy, crop)
        err = math.hypot(rx - x, ry - y)
        assert err < 0.5, f"Plate <-> Crop roundtrip error {err:.4f} px at ({x}, {y})"


def test_crop_to_sam1008_roundtrip():
    ct = CoordTransformer(plate_w=4096, plate_h=2731)
    crop = CropBox(x=500.0, y=300.0, w=1536.0, h=1536.0)

    crop_points = [
        (0.0, 0.0),
        (1536.0, 1536.0),
        (768.0, 768.0),
        (123.4, 567.8),
    ]

    for cx, cy in crop_points:
        sx, sy = ct.crop_to_sam1008(cx, cy, crop)
        rcx, rcy = ct.sam1008_to_crop(sx, sy, crop)
        err = math.hypot(rcx - cx, rcy - cy)
        assert err < 0.5, f"Crop <-> SAM1008 roundtrip error {err:.4f} px at ({cx}, {cy})"


def test_full_pipeline_plate_to_sam1008_roundtrip():
    ct = CoordTransformer(plate_w=4096, plate_h=2731)
    crop = CropBox(x=1400.0, y=500.0, w=1280.0, h=1280.0)

    plate_points = [
        (1400.0, 500.0),
        (2000.0, 1100.0),
        (2680.0, 1780.0),
        (1850.7, 950.3),
    ]

    for x, y in plate_points:
        sx, sy = ct.plate_to_sam1008(x, y, crop)
        rx, ry = ct.sam1008_to_plate(sx, sy, crop)
        err = math.hypot(rx - x, ry - y)
        assert err < 0.5, f"Plate <-> SAM1008 full pipeline roundtrip error {err:.4f} px at ({x}, {y})"


def test_crop_with_padding_roundtrip():
    """Test crop that extends beyond the plate bounds and requires padding."""
    ct = CoordTransformer(plate_w=4096, plate_h=2731)
    # Crop positioned near left/top edge with padding
    crop = CropBox(x=0.0, y=0.0, w=900.0, h=950.0, pad_l=100.0, pad_t=50.0, pad_r=0.0, pad_b=0.0)

    points = [
        (0.0, 0.0),
        (500.0, 400.0),
        (900.0, 950.0),
        (123.0, 456.0),
    ]

    for x, y in points:
        sx, sy = ct.plate_to_sam1008(x, y, crop)
        rx, ry = ct.sam1008_to_plate(sx, sy, crop)
        err = math.hypot(rx - x, ry - y)
        assert err < 0.5, f"Padded crop roundtrip error {err:.4f} px at ({x}, {y})"


def test_create_square_crop_geometry():
    ct = CoordTransformer(plate_w=4096, plate_h=2731)

    # 1. Interior crop: snap multiple 16
    crop = ct.create_square_crop(center_x=2000.0, center_y=1300.0, min_side=800.0, snap_multiple=16)
    assert crop.side % 16 == 0
    assert crop.pad_l == 0.0 and crop.pad_r == 0.0
    assert crop.pad_t == 0.0 and crop.pad_b == 0.0
    assert crop.x >= 0 and crop.x + crop.side <= ct.plate_w
    assert crop.y >= 0 and crop.y + crop.side <= ct.plate_h

    # 2. Edge crop shifted to stay in bounds
    crop_edge = ct.create_square_crop(center_x=100.0, center_y=200.0, min_side=500.0, snap_multiple=16, allow_shift=True)
    assert crop_edge.x >= 0
    assert crop_edge.y >= 0
    assert crop_edge.pad_l == 0.0

    # 3. Huge crop exceeding plate dimension -> mirror padding calculated
    crop_huge = ct.create_square_crop(center_x=2048.0, center_y=1365.5, min_side=3000.0, snap_multiple=16, allow_shift=True)
    assert crop_huge.side > ct.plate_h
    assert crop_huge.pad_t > 0 or crop_huge.pad_b > 0


def test_points_json_formatting():
    ct = CoordTransformer(plate_w=4096, plate_h=2731)
    crop = CropBox(x=1000.0, y=500.0, w=1024.0, h=1024.0)

    pts = [[1200.0, 700.0], [1500.0, 900.0]]
    sam_pts = ct.points_plate_to_sam1008(pts, crop)

    assert len(sam_pts) == 2
    assert "x" in sam_pts[0] and "y" in sam_pts[0]
    assert isinstance(sam_pts[0]["x"], int)
    assert isinstance(sam_pts[0]["y"], int)

    recovered = ct.points_sam1008_to_plate(sam_pts, crop)
    for orig, rec in zip(pts, recovered):
        # Rounding to int might introduce up to ~1-2 px error in full 4K plate, check bound
        err = math.hypot(orig[0] - rec[0], orig[1] - rec[1])
        assert err < 2.0


def test_crop_origin_integer_pixel_alignment():
    """Verify crop origin (x, y) is strictly rounded to whole integer plate pixels."""
    ct = CoordTransformer(plate_w=4096, plate_h=2731)

    test_cases = [
        (2000.33, 1500.67, 800.0, True),
        (123.45, 678.91, 550.0, True),
        (3800.1, 2500.9, 1000.0, False),
        (0.0, 0.0, 400.0, True),
        (4096.0, 2731.0, 600.0, False),
    ]

    for cx, cy, min_side, allow_shift in test_cases:
        crop = ct.create_square_crop(center_x=cx, center_y=cy, min_side=min_side, allow_shift=allow_shift)
        assert crop.x % 1.0 == 0.0, f"crop.x {crop.x} not integer pixel"
        assert crop.y % 1.0 == 0.0, f"crop.y {crop.y} not integer pixel"
        assert crop.w % 1.0 == 0.0, f"crop.w {crop.w} not integer pixel"
        assert crop.h % 1.0 == 0.0, f"crop.h {crop.h} not integer pixel"
        assert crop.pad_l % 1.0 == 0.0, f"crop.pad_l {crop.pad_l} not integer pixel"
        assert crop.pad_t % 1.0 == 0.0, f"crop.pad_t {crop.pad_t} not integer pixel"
        assert crop.pad_r % 1.0 == 0.0, f"crop.pad_r {crop.pad_r} not integer pixel"
        assert crop.pad_b % 1.0 == 0.0, f"crop.pad_b {crop.pad_b} not integer pixel"


def test_crop_image_resize_sam1008_edge_accuracy():
    """Verify consistency with real image scaling: cv2.resize to 1008 and back with test mask.
    
    Requirement: Edge discrepancy mapped back to plate coordinates must be < 0.5 px.
    """
    ct = CoordTransformer(plate_w=4096, plate_h=2731)

    test_crops = [
        # 1. Interior crop, centered
        ct.create_square_crop(center_x=2000.0, center_y=1350.0, min_side=800.0),
        # 2. Padded crop near bottom-right edge
        ct.create_square_crop(center_x=3800.0, center_y=2400.0, min_side=1000.0, allow_shift=False),
        # 3. Arbitrary scale crop
        ct.create_square_crop(center_x=1412.0, center_y=508.0, min_side=950.0, allow_shift=True),
    ]

    for crop in test_crops:
        side = int(crop.side)
        assert side % 16 == 0

        # Construct anti-aliased circular test mask inside crop
        radius = float(side) * 0.25
        center_crop_x = float(side) / 2.0
        center_crop_y = float(side) / 2.0

        Y, X = np.ogrid[:side, :side]
        dist = np.sqrt((X - center_crop_x)**2 + (Y - center_crop_y)**2)
        # Anti-aliased transition at radius
        test_mask = np.clip(0.5 - (dist - radius), 0.0, 1.0).astype(np.float32)

        # 1. Resize to SAM3 target resolution (1008x1008)
        mask_1008 = cv2.resize(test_mask, (1008, 1008), interpolation=cv2.INTER_LINEAR)

        # 2. Resize back to crop size
        restored_mask = cv2.resize(mask_1008, (side, side), interpolation=cv2.INTER_LINEAR)

        # 3. Measure edge position along cardinal axes (e.g. horizontal slice at center_y)
        y_slice = int(round(center_crop_y))
        row = restored_mask[y_slice, int(center_crop_x):]
        cross_idx = np.where(row < 0.5)[0][0]
        # Linear subpixel interpolation for 0.5 crossing
        subpixel_offset = (cross_idx - 1) + (0.5 - row[cross_idx - 1]) / (row[cross_idx] - row[cross_idx - 1])
        measured_crop_x = center_crop_x + subpixel_offset
        expected_crop_x = center_crop_x + radius

        # Convert both to full resolution plate coordinates
        measured_plate_x, _ = ct.crop_to_plate(measured_crop_x, center_crop_y, crop)
        expected_plate_x, _ = ct.crop_to_plate(expected_crop_x, center_crop_y, crop)

        edge_err = abs(measured_plate_x - expected_plate_x)
        assert edge_err < 0.5, f"Edge error {edge_err:.4f} px exceeds 0.5 px threshold for crop side={side}"


def test_crop_pipeline_full_plate_resize_paste_edge_accuracy():
    """Verify full plate pipeline: test mask -> crop -> cv2.resize to 1008 -> cv2.resize back -> paste into plate.
    
    Demonstrates edge error < 0.5 px directly on the reconstructed plate.
    """
    plate_w, plate_h = 4096, 2731
    ct = CoordTransformer(plate_w, plate_h)

    test_scenarios = [
        # (center_x, center_y, radius, min_side, allow_shift)
        (2100.0, 1400.0, 350.0, 900, True),     # interior crop
        (3800.0, 2400.0, 250.0, 1000, False),   # padded bottom-right
        (200.0, 300.0, 150.0, 600, False),      # padded top-left
        (1412.3, 508.7, 280.0, 950, True)       # subpixel plate center
    ]

    for center_plate_x, center_plate_y, radius, min_side, allow_shift in test_scenarios:
        crop = ct.create_square_crop(center_plate_x, center_plate_y, min_side=min_side, allow_shift=allow_shift)
        assert crop.x % 1.0 == 0.0 and crop.y % 1.0 == 0.0
        side = int(crop.side)
        x0, y0 = int(crop.x), int(crop.y)
        w, h = int(crop.w), int(crop.h)
        pad_l, pad_t, pad_r, pad_b = int(crop.pad_l), int(crop.pad_t), int(crop.pad_r), int(crop.pad_b)

        # 1. Full plate test mask
        plate_mask = np.zeros((plate_h, plate_w), dtype=np.float32)
        y_indices, x_indices = np.ogrid[y0:y0+h, x0:x0+w]
        dist = np.sqrt((x_indices - center_plate_x)**2 + (y_indices - center_plate_y)**2)
        plate_mask[y0:y0+h, x0:x0+w] = np.clip(0.5 - (dist - radius), 0.0, 1.0)

        # 2. Extract crop from plate (with mirror pad if needed)
        unpadded_crop = plate_mask[y0:y0+h, x0:x0+w]
        if pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
            crop_img = cv2.copyMakeBorder(unpadded_crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT_101)
        else:
            crop_img = unpadded_crop

        assert crop_img.shape == (side, side)

        # 3. cv2.resize to 1008x1008
        img_1008 = cv2.resize(crop_img, (1008, 1008), interpolation=cv2.INTER_LINEAR)

        # 4. cv2.resize back to crop side
        restored_crop = cv2.resize(img_1008, (side, side), interpolation=cv2.INTER_LINEAR)

        # 5. Remove padding
        if pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
            unpadded_restored = restored_crop[pad_t:pad_t+h, pad_l:pad_l+w]
        else:
            unpadded_restored = restored_crop

        # 6. Paste back into reconstructed full plate
        reconstructed_plate = np.zeros_like(plate_mask)
        reconstructed_plate[y0:y0+h, x0:x0+w] = unpadded_restored

        # 7. Measure edge on reconstructed plate
        y_row = int(round(center_plate_y))
        x_col = int(round(center_plate_x))
        row = reconstructed_plate[y_row, x_col:]
        cross_idx = np.where(row < 0.5)[0][0]
        subpixel = (cross_idx - 1) + (0.5 - row[cross_idx - 1]) / (row[cross_idx] - row[cross_idx - 1])
        measured_edge = x_col + subpixel
        expected_edge = center_plate_x + radius

        edge_error = abs(measured_edge - expected_edge)
        assert edge_error < 0.5, f"Reconstructed plate edge error {edge_error:.4f} px exceeds 0.5 px"
