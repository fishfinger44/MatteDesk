"""Stage 2 speed-up (2026-11-23): the brush correction computed on the object's window instead of the whole frame must
be bit-identical to the full-frame code (kept here as the reference)."""
import cv2
import numpy as np
import pytest

from engine import zones
from engine.zones import LocalWarp, _connected_to_full, _crossfade, connected_to, local_warp_mask, nonzero_bbox


def ref_local_warp_mask(mask, lw):
    """The full-frame local_warp_mask before 2026-11-23 (verbatim)."""
    H, W = mask.shape
    m = mask.astype(np.float32)
    v = lw.vis.astype(bool)
    if v.sum() < 6:
        return cv2.warpAffine(m, lw.fallback, (W, H), flags=cv2.INTER_LINEAR) > 0.5
    g = 8
    gh, gw = (H + g - 1) // g, (W + g - 1) // g
    acc = np.zeros((2, gh, gw), np.float32)
    wsum = np.zeros((gh, gw), np.float32)
    d = (lw.dst[v] - lw.pts[v]).astype(np.float32)
    q = lw.dst[v] / g
    ix = np.clip(np.round(q[:, 0]).astype(int), 0, gw - 1)
    iy = np.clip(np.round(q[:, 1]).astype(int), 0, gh - 1)
    np.add.at(wsum, (iy, ix), 1.0)
    np.add.at(acc[0], (iy, ix), d[:, 0])
    np.add.at(acc[1], (iy, ix), d[:, 1])
    sg = max(1.0, lw.sigma / g)
    k = int(round(sg * 3)) * 2 + 1
    wb = cv2.GaussianBlur(wsum, (k, k), sg)
    ax, ay = cv2.GaussianBlur(acc[0], (k, k), sg), cv2.GaussianBlur(acc[1], (k, k), sg)
    Mi = cv2.invertAffineTransform(lw.fallback.astype(np.float32))
    gy, gx = np.mgrid[0:gh, 0:gw].astype(np.float32) * g
    fx = gx - (Mi[0, 0] * gx + Mi[0, 1] * gy + Mi[0, 2])
    fy = gy - (Mi[1, 0] * gx + Mi[1, 1] * gy + Mi[1, 2])
    peak = 1.0 / (2 * np.pi * sg * sg)
    trust = np.clip(wb / (0.25 * peak), 0.0, 1.0)
    dx = trust * (ax / np.maximum(wb, 1e-9)) + (1 - trust) * fx
    dy = trust * (ay / np.maximum(wb, 1e-9)) + (1 - trust) * fy
    dx = cv2.resize(dx, (W, H), interpolation=cv2.INTER_LINEAR)
    dy = cv2.resize(dy, (W, H), interpolation=cv2.INTER_LINEAR)
    X, Y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    return cv2.remap(m, X - dx, Y - dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.5


def blob(shape, cx, cy, r, rng, n=5):
    m = np.zeros(shape, np.uint8)
    for _ in range(n):
        cv2.circle(m, (int(cx + rng.normal(0, r / 2)), int(cy + rng.normal(0, r / 2))), int(r * rng.uniform(0.3, 1)), 1, -1)
    return m.astype(bool)


SHAPES = [(768, 1024), (757, 1001), (416, 624)]   # divisible by 8 (sub-grid resize) and not (full resize + window)


@pytest.mark.parametrize("shape", SHAPES)
def test_local_warp_mask_window_bit_identical(shape):
    H, W = shape
    rng = np.random.default_rng(7)
    for trial in range(12):
        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
        if trial % 4 == 0:          # touching the frame edge (the rig_02 case: bottom of the plate)
            cy = H - 3
        mask = blob(shape, cx, cy, rng.uniform(5, 40), rng)
        if trial % 3 == 0:
            mask = mask.astype(np.float32) * rng.uniform(0.4, 1.0)   # float mask
        n = int(rng.integers(6, 60))
        pts = np.column_stack([rng.uniform(0, W, n), rng.uniform(0, H, n)]).astype(np.float32)
        dst = pts + rng.normal(0, rng.uniform(1, 30), (n, 2)).astype(np.float32)
        vis = rng.uniform(0, 1, n) > 0.2
        a = rng.uniform(-0.05, 0.05)
        fb = np.array([[np.cos(a), -np.sin(a), rng.normal(0, 15)], [np.sin(a), np.cos(a), rng.normal(0, 15)]], np.float32)
        lw = LocalWarp(pts=pts, dst=dst, vis=vis, fallback=fb, sigma=float(rng.uniform(20, 90)))
        got, ref = local_warp_mask(mask, lw), ref_local_warp_mask(mask, lw)
        assert got.dtype == ref.dtype and got.shape == ref.shape
        assert np.array_equal(got, ref), f"trial {trial}: {int((got != ref).sum())} px differ"


def test_local_warp_mask_empty_and_big():
    H, W = 400, 640
    lw = LocalWarp(pts=np.zeros((8, 2), np.float32) + 50, dst=np.zeros((8, 2), np.float32) + 60,
                   vis=np.ones(8, bool), fallback=np.eye(2, 3, dtype=np.float32))
    assert not local_warp_mask(np.zeros((H, W), bool), lw).any()
    big = np.zeros((H, W), bool)
    big[10:390, 10:630] = True          # window > 60 % of the frame: the full-frame path
    assert np.array_equal(local_warp_mask(big, lw), ref_local_warp_mask(big, lw))


@pytest.mark.parametrize("shape", SHAPES + [(4160 // 4, 6240 // 4)])
def test_connected_to_window_bit_identical(shape):
    H, W = shape
    rng = np.random.default_rng(3)
    for trial in range(10):
        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
        core = blob(shape, cx, cy, rng.uniform(5, 30), rng)
        cand = np.zeros(shape, bool)
        for _ in range(int(rng.integers(1, 8))):    # islands near and far, some touching the core
            cand |= blob(shape, cx + rng.normal(0, 150), cy + rng.normal(0, 150), rng.uniform(3, 25), rng, n=2)
        cand |= blob(shape, cx, cy, 20, rng, n=2)
        for reach in (3, 20, 200):
            got, ref = connected_to(cand, core, reach), _connected_to_full(cand, core, reach)
            assert np.array_equal(got, ref), f"trial {trial} reach {reach}: {int((got != ref).sum())} px differ"


def test_crossfade_window():
    rng = np.random.default_rng(1)
    a = blob((300, 500), 200, 150, 30, rng)
    b = blob((300, 500), 230, 160, 30, rng)
    for t in (0.1, 0.5, 0.73):
        ref = ((1.0 - t) * a.astype(np.float32) + t * b.astype(np.float32)) >= 0.5
        assert np.array_equal(_crossfade(a, b, t), ref)
    assert not _crossfade(np.zeros((5, 5), bool), np.zeros((5, 5), bool), 0.5).any()


def test_core_png_memory_crops(tmp_path):
    from engine import pipeline
    m = np.zeros((300, 400), bool)
    m[280:300, 10:30] = True
    p = tmp_path / "c_0001.png"
    cv2.imwrite(str(p), m.astype(np.uint8) * 255)
    first = pipeline._read_core_png(p)
    first[:] = False                                     # callers may write into what they get
    again = pipeline._read_core_png(p)
    assert np.array_equal(again, m)
    e = next(v for k, v in pipeline._CORE_LRU.items() if k[0] == str(p))
    assert e[2].shape == (20, 20)                        # only the bbox is kept in memory
    empty = tmp_path / "c_0002.png"
    cv2.imwrite(str(empty), np.zeros((30, 40), np.uint8))
    assert not pipeline._read_core_png(empty).any() and pipeline._read_core_png(empty).shape == (30, 40)
    assert nonzero_bbox(m) == (10, 280, 30, 300)


def ref_object_points(z, core, n=48, margin=40, exclude=None):
    """_object_points before 2026-11-23 (verbatim, full frame)."""
    if core is None or not core.any():
        return None
    area = cv2.dilate(zones.zone_mask_at(z, core.shape).astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))).astype(bool)
    inner = cv2.erode(core.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    if exclude is not None and exclude.any():
        inner &= ~cv2.dilate(exclude.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
    ys, xs = np.where(inner & area)
    if len(xs) < 12:
        return None
    idx = np.linspace(0, len(xs) - 1, min(n, len(xs))).astype(int)
    return np.column_stack([xs[idx], ys[idx]]).astype(np.float32)


def test_object_points_and_bbox_transforms_window():
    rng = np.random.default_rng(11)
    shape = (500, 700)
    for trial in range(10):
        cx, cy = rng.uniform(0, 700), rng.uniform(0, 500)
        if trial % 3 == 0:
            cy = 498                                   # object on the frame edge
        core = blob(shape, cx, cy, rng.uniform(8, 60), rng)
        pts = [[float(cx + rng.normal(0, 20)), float(cy + rng.normal(0, 20))] for _ in range(6)]
        z = zones.Zone(zone_id="z_01", mode="add", criterion="all", frame=1, polygon=[],
                       strokes=[{"p": pts, "radius_px": float(rng.uniform(3, 15))}])
        excl = blob(shape, cx + 10, cy, 10, rng, n=1) if trial % 2 else None
        for local in (True, False):
            n, m = (96, 60) if local else (48, 40)
            got = zones._object_points(z, core, n=n, margin=m, exclude=excl)
            ref = ref_object_points(z, core, n=n, margin=m, exclude=excl)
            assert (got is None) == (ref is None)
            if got is not None:
                assert np.array_equal(got, ref)
        cores = {1: core, 2: blob(shape, cx + 5, cy - 3, 30, rng), 3: np.zeros(shape, bool)}
        got = zones._bbox_transforms(z, [1, 2, 3], cores)

        def bb(mm):
            ys, xs = np.where(mm)
            return (xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else None
        ref_b = bb(core)
        for f in (1, 2, 3):
            b = bb(cores[f])
            if b is None or ref_b is None:
                assert np.array_equal(got[f], np.eye(2, 3, dtype=np.float32))
                continue
            s = ((b[2] - b[0]) + (b[3] - b[1])) / max(1e-6, (ref_b[2] - ref_b[0]) + (ref_b[3] - ref_b[1]))
            exp = np.array([[s, 0, (b[0] + b[2]) / 2 - s * (ref_b[0] + ref_b[2]) / 2],
                            [0, s, (b[1] + b[3]) / 2 - s * (ref_b[1] + ref_b[3]) / 2]], np.float32)
            assert np.array_equal(got[f], exp)
