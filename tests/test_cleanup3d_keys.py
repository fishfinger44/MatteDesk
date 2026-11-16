# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cleanup 3D package P2 – key store: versions, deletion, export / import, areas -> mask, carrying marks to keys."""
import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from engine import curves as CV  # noqa: E402
from engine.cleanup3d import keys as KS  # noqa: E402
from engine.cleanup3d import store  # noqa: E402
from engine.cleanup3d import types as T  # noqa: E402

H, W = 48, 64


@pytest.fixture
def work(tmp_path):
    w = tmp_path / "shot" / "matte_desk"
    w.mkdir(parents=True)
    return w


def _img(seed=0, h=H, w=W):
    rng = np.random.default_rng(seed)
    return rng.random((h, w, 3)).astype(np.float32)


def _plate(f):
    return _img(seed=f)


# ----------------------------------------------------------------------------- versions
def test_versions_undo_redo_set_active(work):
    k = KS.add_version(work, "sea", 10, _img(1), "plate", "png")
    assert k.active == "v001" and (T.key_dir(work, "sea", 10) / "v001.png").is_file()
    k = KS.add_version(work, "sea", 10, _img(2), "lama", "exr")
    assert k.active == "v002" and (T.key_dir(work, "sea", 10) / "v002.exr").is_file()
    assert [v["source"] for v in k.versions] == ["plate", "lama"]
    on_disk = store.load_key(work, "sea", 10)
    assert on_disk.active == "v002" and len(on_disk.versions) == 2
    np.testing.assert_allclose(KS.read_key_image(work, on_disk), _img(2), atol=0)       # exr exact

    k = KS.undo(work, "sea", 10)
    assert k.active == "v001"
    assert (T.key_dir(work, "sea", 10) / "v002.exr").is_file()                          # files stay
    np.testing.assert_allclose(KS.read_key_image(work, k), _img(1), atol=0.5 / 255)
    with pytest.raises(ValueError, match="wcześniejszej"):
        KS.undo(work, "sea", 10)
    assert KS.redo(work, "sea", 10).active == "v002"
    with pytest.raises(ValueError, match="późniejszej"):
        KS.redo(work, "sea", 10)
    assert KS.set_active(work, "sea", 10, "v001").active == "v001"
    with pytest.raises(ValueError):
        KS.set_active(work, "sea", 10, "v009")
    # a new version after undo is appended and becomes active
    k = KS.add_version(work, "sea", 10, _img(3), "klein_restore", "png16")
    assert k.active == "v003" and [v["v"] for v in k.versions] == ["v001", "v002", "v003"]
    np.testing.assert_allclose(KS.read_key_image(work, k, "v002"), _img(2), atol=0)


def test_read_key_image_depths(work):
    u8 = (np.arange(H * W * 3) % 256).astype(np.uint8).reshape(H, W, 3)
    u16 = (np.arange(H * W * 3) * 37 % 65536).astype(np.uint16).reshape(H, W, 3)
    k = KS.add_version(work, "sky", 0, u8, "a", "png")
    a = KS.read_key_image(work, k)
    assert a.dtype == np.float32 and np.array_equal(a, u8.astype(np.float32) / 255)
    k = KS.add_version(work, "sky", 0, u16, "b", "png16")
    b = KS.read_key_image(work, k)
    assert b.dtype == np.float32 and np.array_equal(b, u16.astype(np.float32) / 65535)
    hdr = _img(5) * 4.0                                                                  # linear > 1 kept in EXR
    k = KS.add_version(work, "sky", 0, hdr, "c", "exr")
    assert np.array_equal(KS.read_key_image(work, k), hdr)
    with pytest.raises(ValueError, match="Nieznany format"):
        KS.add_version(work, "sky", 0, hdr, "d", "tiff")


def test_delete_key_is_permanent(work):
    KS.add_version(work, "sea", 4, _img(), "plate")
    kd = T.key_dir(work, "sea", 4)
    size = sum(p.stat().st_size for p in kd.iterdir())
    assert KS.delete_key(work, "sea", 4) == size > 0
    assert not kd.exists() and store.load_key(work, "sea", 4) is None
    assert not (T.root_dir(work) / "trash").exists()               # user decision 2026-11-13: no trash
    KS.add_version(work, "sea", 4, _img(), "plate")                # a new key of the same frame starts again
    assert KS.delete_key(work, "sea", 4) > 0
    assert KS.delete_key(work, "sea", 4) is None


def test_frames_every():
    assert KS.frames_every(range(1, 21), 8) == [1, 9, 17, 20]
    assert KS.frames_every(range(1, 18), 8) == [1, 9, 17]
    assert KS.frames_every([5], 3) == [5]
    assert KS.frames_every([], 3) == []
    assert KS.frames_every(range(0, 5), 1) == [0, 1, 2, 3, 4]
    assert KS.frames_every(range(10, 13), 100) == [10, 12]


def test_create_keys_from_plate(work):
    ks = KS.create_keys(work, "sand", [1, 9, 17], _plate)
    assert [k.frame for k in ks] == [1, 9, 17]
    for k in ks:
        assert k.versions[0]["source"] == "plate" and k.active == "v001"
        np.testing.assert_allclose(KS.read_key_image(work, k), _plate(k.frame), atol=0.5 / 65535)
    assert [k.frame for k in store.list_keys(work, "sand")] == [1, 9, 17]
    KS.add_version(work, "sand", 9, _img(99), "lama")
    ks = KS.create_keys(work, "sand", [9], _plate)                                       # existing key untouched
    assert ks[0].active == "v002" and len(ks[0].versions) == 2


# ----------------------------------------------------------------------------- export / import
@pytest.mark.parametrize("fmt,tol", [("png", 1 / 255), ("png16", 1 / 65535), ("exr", 0.0)])
def test_export_import_round_trip(work, tmp_path, fmt, tol):
    src = _img(7)
    KS.add_version(work, "sea", 3, src, "plate", "exr")
    p = KS.export_key(work, "sea", 3, fmt, tmp_path / "out")
    assert p.is_file() and p.suffix == (".exr" if fmt == "exr" else ".png")
    if fmt != "exr":
        raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        assert raw.dtype == (np.uint8 if fmt == "png" else np.uint16)
    k = KS.import_key(work, "sea", 3, p)
    assert k.active == "v002" and k.versions[-1]["source"] == f"import:{p.name}"
    got = KS.read_key_image(work, k)
    assert np.abs(got - src).max() <= tol + 1e-7
    # exported again from the imported version = the same file content
    p2 = KS.export_key(work, "sea", 3, fmt, tmp_path / "out2")
    a, _ = KS.read_image(p)
    b, _ = KS.read_image(p2)
    assert np.array_equal(a, b)


def test_import_drops_alpha(work, tmp_path):
    KS.add_version(work, "sea", 3, _img(1), "plate", "png")
    bgra = np.dstack([(_img(2) * 255).astype(np.uint8), np.full((H, W), 128, np.uint8)])
    cv2.imwrite(str(tmp_path / "painted.png"), bgra)
    k = KS.import_key(work, "sea", 3, tmp_path / "painted.png")
    assert np.array_equal(KS.read_key_image(work, k), bgra[..., :3].astype(np.float32) / 255)


def test_import_refusals(work, tmp_path):
    with pytest.raises(ValueError, match="Brak klucza"):
        KS.import_key(work, "sea", 3, tmp_path / "x.png")
    KS.add_version(work, "sea", 3, _img(1), "plate", "png")
    # wrong size, same aspect
    cv2.imwrite(str(tmp_path / "half.png"), np.zeros((H // 2, W // 2, 3), np.uint8))
    with pytest.raises(ValueError) as e:
        KS.import_key(work, "sea", 3, tmp_path / "half.png")
    assert f"plik {W // 2}×{H // 2}" in str(e.value) and f"klucz {W}×{H}" in str(e.value)
    assert "Inny rozmiar niż klucz" in str(e.value)
    # wrong aspect
    cv2.imwrite(str(tmp_path / "sq.png"), np.zeros((H, H, 3), np.uint8))
    with pytest.raises(ValueError, match="proporcje") as e:
        KS.import_key(work, "sea", 3, tmp_path / "sq.png")
    assert f"plik {H}×{H}" in str(e.value) and f"klucz {W}×{H}" in str(e.value)
    # unreadable
    (tmp_path / "broken.png").write_bytes(b"not an image at all")
    with pytest.raises(ValueError, match="Nie można odczytać pliku broken.png"):
        KS.import_key(work, "sea", 3, tmp_path / "broken.png")
    (tmp_path / "broken.exr").write_bytes(b"\x76\x2f\x31\x01garbage")
    with pytest.raises(ValueError, match="Nie można odczytać pliku broken.exr"):
        KS.import_key(work, "sea", 3, tmp_path / "broken.exr")
    with pytest.raises(ValueError, match="Nie ma pliku"):
        KS.import_key(work, "sea", 3, tmp_path / "missing.png")
    # grey = no 3 colour channels
    cv2.imwrite(str(tmp_path / "grey.png"), np.zeros((H, W), np.uint8))
    with pytest.raises(ValueError, match="3 kanały koloru"):
        KS.import_key(work, "sea", 3, tmp_path / "grey.png")
    from engine import exr_io
    exr_io.write_gray_exr(tmp_path / "y.exr", np.zeros((H, W), np.float32), channels=("Y",))
    with pytest.raises(ValueError, match="3 kanały koloru"):
        KS.import_key(work, "sea", 3, tmp_path / "y.exr")
    assert len(store.load_key(work, "sea", 3).versions) == 1                             # nothing added


# ----------------------------------------------------------------------------- areas -> mask
def _ellipse_curve(cx, cy, rx, ry, n=48):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return CV.from_points(np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], 1), closed=True)


def _edge_ok(a, b, px=1):
    """Masks equal except within `px` of an edge of either."""
    a8, b8 = (a > 0.5).astype(np.uint8), (b > 0.5).astype(np.uint8)
    k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    diff = a8 != b8
    edge = (cv2.dilate(a8, k) != cv2.erode(a8, k)) | (cv2.dilate(b8, k) != cv2.erode(b8, k))
    return not (diff & ~edge).any(), int(diff.sum())


def test_curve_mask_equals_brush_mask(work):
    curve = _ellipse_curve(30.3, 22.7, 17.2, 11.6)
    store.save_areas(work, "sea", 5, [T.Area("c1", "curve", "add", curve)])
    m_curve = KS.areas_mask(work, "sea", 5, (H, W))
    assert m_curve.dtype == np.float32 and 0 < m_curve.mean() < 1
    # the same shape drawn as a brush layer (analytic ellipse, pixel centres)
    yy, xx = np.mgrid[0:H, 0:W]
    brush = (((xx - 30.3) / 17.2) ** 2 + ((yy - 22.7) / 11.6) ** 2 <= 1).astype(np.float32)
    store.save_areas(work, "sea", 6, [])
    KS.save_brush_layer(work, "sea", 6, "b1", brush)
    assert (T.key_dir(work, "sea", 6) / "a_b1.png").is_file()
    assert [a.kind for a in store.load_areas(work, "sea", 6)] == ["brush"]
    m_brush = KS.areas_mask(work, "sea", 6, (H, W))
    ok, nd = _edge_ok(m_curve, m_brush)
    assert ok, nd
    assert np.abs(m_curve - brush).sum() < 0.08 * brush.sum()


def test_add_remove_order(work):
    big = CV.from_points([[5, 5], [55, 5], [55, 40], [5, 40]], closed=True)
    big = CV.spline([CV.point(q["p"], (0, 0), (0, 0), True) for q in big["pts"]], True)     # a rectangle
    hole = np.zeros((H, W), np.float32)
    hole[15:30, 20:40] = 1
    small = CV.spline([CV.point(p, (0, 0), (0, 0), True) for p in [[24, 18], [36, 18], [36, 26], [24, 26]]], True)
    KS.save_brush_layer(work, "sea", 1, "hole", hole, op="remove")
    areas = store.load_areas(work, "sea", 1)
    store.save_areas(work, "sea", 1, [T.Area("big", "curve", "add", big)] + areas)
    m = KS.areas_mask(work, "sea", 1, (H, W))
    assert m[10, 10] == 1 and m[22, 30] == 0 and m[2, 2] == 0
    # add after remove brings the centre back; remove before add does nothing
    store.save_areas(work, "sea", 1, store.load_areas(work, "sea", 1) + [T.Area("small", "curve", "add", small)])
    m = KS.areas_mask(work, "sea", 1, (H, W))
    assert m[22, 30] == 1 and m[16, 21] == 0
    a = store.load_areas(work, "sea", 1)
    store.save_areas(work, "sea", 1, [a[1], a[0]])                                        # remove, then add big
    m = KS.areas_mask(work, "sea", 1, (H, W))
    assert m[22, 30] == 1
    # saving an existing brush id again keeps its place and op
    KS.save_brush_layer(work, "sea", 1, "hole", hole * 0)
    assert [(x.id, x.op) for x in store.load_areas(work, "sea", 1)] == [("hole", "remove"), ("big", "add")]
    assert KS.areas_mask(work, "sea", 1, (H, W)).shape == (H, W)


def test_missing_brush_layer(work):
    store.save_areas(work, "sea", 2, [T.Area("gone", "brush", "add", None, "a_gone.png")])
    with pytest.raises(ValueError, match="Brak warstwy pędzla a_gone.png"):
        KS.areas_mask(work, "sea", 2, (H, W))


# ----------------------------------------------------------------------------- carry to keys
class ShiftGeometry:
    """Fake P1: key pixel = frame pixel + (key.frame - frame) * (dx, dy); NaN left of x = 0 in the key."""
    dx, dy = 2.0, -0.5

    def __init__(self):
        self.calls = []

    def points_frame_to_key(self, pts, frame, key, card):
        self.calls.append((frame, key.frame, card.id))
        d = key.frame - frame
        out = np.asarray(pts, np.float32) + np.array([self.dx * d, self.dy * d], np.float32)
        out[out[:, 0] < -50] = np.nan
        return out

    def frame_to_key(self, *a):
        raise AssertionError("carry_areas must only use points_frame_to_key")

    key_to_frame = card_weight = coverage = frame_to_key


def _card():
    return T.Card("sea", "Morze", "plane", {"n": [0, -1, 0], "d": 1.0})


def _keys(*frames):
    return [T.Key("sea", f, [{"v": "v001", "ext": "png"}], "v001") for f in frames]


def test_keys_for_frame():
    ks = _keys(1, 9, 17)
    assert [k.frame for k in KS.keys_for_frame(9, ks)] == [9]
    assert [k.frame for k in KS.keys_for_frame(12, ks)] == [9, 17]
    assert [k.frame for k in KS.keys_for_frame(0, ks)] == [1]
    assert [k.frame for k in KS.keys_for_frame(20, ks)] == [17]
    assert KS.keys_for_frame(5, []) == []


def test_carry_curve_own_key_and_two_nearest():
    g, card = ShiftGeometry(), _card()
    curve = _ellipse_curve(30, 22, 10, 8, n=12)
    areas = [T.Area("c1", "curve", "remove", curve, label="kubek")]
    own = KS.carry_areas(g, areas, 9, _keys(1, 9, 17), card)
    assert list(own) == [9]
    a = own[9][0]
    assert (a.kind, a.op, a.label) == ("curve", "remove", "kubek")
    for q0, q1 in zip(curve["pts"], a.curve["pts"]):
        np.testing.assert_allclose(q1["p"], q0["p"], atol=1e-4)
        np.testing.assert_allclose(q1["in"], q0["in"], atol=1e-4)

    out = KS.carry_areas(g, areas, 12, _keys(1, 9, 17), card)
    assert sorted(out) == [9, 17]
    for kf, d in ((9, -3), (17, 5)):
        c = out[kf][0].curve
        assert c["closed"] and len(c["pts"]) == 12
        for q0, q1 in zip(curve["pts"], c["pts"]):
            np.testing.assert_allclose(q1["p"], np.array(q0["p"]) + [2.0 * d, -0.5 * d], atol=1e-4)
            np.testing.assert_allclose(q1["out"], q0["out"], atol=1e-4)                 # translation keeps handles
    assert {c[0] for c in g.calls} == {9, 12} and {c[2] for c in g.calls} == {"sea"}


def test_carry_brush_becomes_curve_with_same_mask(tmp_path):
    g, card = ShiftGeometry(), _card()
    yy, xx = np.mgrid[0:H, 0:W]
    m = ((((xx - 28) / 12.0) ** 2 + ((yy - 24) / 9.0) ** 2) <= 1).astype(np.float32)
    m[5:12, 45:58] = 1                                                                   # a second stroke
    areas = [T.Area("b1", "brush", "add", None, "a_b1.png")]
    out = KS.carry_areas(g, areas, 13, _keys(1, 9, 17), card, brush={"b1": m})
    assert sorted(out) == [9, 17]
    for kf in (9, 17):
        res = out[kf]
        assert [a.kind for a in res] == ["curve", "curve"] and {a.id for a in res} == {"b1_1", "b1_2"}
        d = kf - 13
        shifted = cv2.warpAffine(m, np.float32([[1, 0, 2.0 * d], [0, 1, -0.5 * d]]), (W, H),
                                 flags=cv2.INTER_NEAREST)
        got = KS.render_areas(res, (H, W))
        ok, nd = _edge_ok(got, shifted)
        assert ok, (kf, nd)
    # the same from a folder of a_<id>.png files
    cv2.imwrite(str(tmp_path / "a_b1.png"), (m * 255).astype(np.uint8))
    out2 = KS.carry_areas(g, areas, 13, _keys(1, 9, 17), card, brush=tmp_path)
    assert [a.curve for a in out2[9]] == [a.curve for a in out[9]]
    with pytest.raises(ValueError, match="Brak warstwy pędzla"):
        KS.carry_areas(g, areas, 13, _keys(1, 9, 17), card)


def test_carry_drops_points_that_miss_the_card():
    g, card = ShiftGeometry(), _card()
    tri = CV.spline([CV.point(p) for p in [[0, 0], [5, 0], [2, 10]]], True)
    out = KS.carry_areas(g, [T.Area("t", "curve", "add", tri)], 30, _keys(1), card)
    assert out == {1: []}                                                                # all moved past x < -50


# ----------------------------------------------------------------------------- the plate + a fill only in the hole
def _smooth(seed, h=96, w=128):
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.random((h, w, 3)).astype(np.float32), (0, 0), 3) * 0.6 + 0.2


def test_fill_region_grow_and_soft_edge():
    m = np.zeros((96, 128), np.uint8)
    m[40:56, 50:70] = 255
    a = KS.fill_region(m, grow_px=6, soft_px=8)
    assert a[m > 0].min() == 1.0 and a[48, 44] == 1.0              # the mask and the grown ring: fully the fill
    assert 0.0 < a[48, 40] < 1.0 and a[48, 30] == 0.0              # soft edge outside the grown mask, then nothing
    assert KS.fill_region(np.zeros((8, 8), np.uint8)).max() == 0.0
    hard = KS.fill_region(m, grow_px=6, soft_px=0)
    assert set(np.unique(hard)) == {0.0, 1.0} and hard[48, 44] == 1.0 and hard[48, 42] == 0.0


def test_patch_on_plate_keeps_the_plate_outside_the_region():
    """The user's case 2026-11-10: a video model's key repaints (and re-exposes) the whole frame; only the removed
    object's place may come from it."""
    plate = _smooth(1)
    yy, xx = np.mgrid[0:96, 0:128].astype(np.float32)
    stripes = (0.08 * np.sin(xx / 3.0) * np.cos(yy / 4.0))[..., None]
    fill = np.clip((plate + stripes) * 0.8, 0, 1)                  # another texture everywhere + darker (gain 0.8)
    m = np.zeros((96, 128), np.uint8)
    m[36:60, 48:80] = 255
    out = KS.patch_on_plate(plate, fill, m, grow_px=4, soft_px=6)
    a = KS.fill_region(m, 4, 6)
    assert np.array_equal(out[a == 0], plate[a == 0])              # bit for bit
    inside = m > 0
    assert np.abs(out[inside] - plate[inside]).mean() > 0.01       # the hole is the fill, not the plate
    # ... tone-matched: the fill's exposure shift is gone (mean of the hole close to a plate-toned fill)
    raw = KS.patch_on_plate(plate, fill, m, grow_px=4, soft_px=6, tone=False)
    ring = (KS.fill_region(m, 14, 0) > 0) & (a == 0)
    assert abs(out[inside].mean() - plate[ring].mean()) < abs(raw[inside].mean() - plate[ring].mean())
    # an empty mask = the plate itself; sizes are checked
    assert np.array_equal(KS.patch_on_plate(plate, fill, np.zeros((96, 128), np.uint8)), plate)
    with pytest.raises(ValueError):
        KS.patch_on_plate(plate, fill[:50], m)
    # 8-bit inputs are read as 0..1
    o8 = KS.patch_on_plate((plate * 255).astype(np.uint8), (fill * 255).astype(np.uint8), m, 4, 6)
    assert o8.dtype == np.float32 and abs(float(o8.mean()) - float(out.mean())) < 0.01


def test_rebase_on_plate_adds_one_version_and_skips_plate_keys(work):
    plate = _smooth(3)
    m = np.zeros((96, 128), np.uint8)
    m[30:50, 40:60] = 255
    KS.add_version(work, "sand", 5, np.clip(plate * 0.7, 0, 1), "experiment void", "png")
    k = KS.rebase_on_plate(work, "sand", 5, plate, m, 4, 6)
    assert k.active == "v002" and k.versions[-1]["source"] == "plate+fill:v001" and k.versions[-1]["ext"] == "png"
    img = KS.read_key_image(work, k)
    out = KS.fill_region(m, 4 + KS.EFFECT_EDGE_PX + 6, 0) == 0         # grown mask + the effects' margin + soft edge
    assert np.abs(img[out] - plate[out]).max() <= 1.0 / 255 + 1e-6     # the plate at 8-bit precision
    assert KS.rebase_on_plate(work, "sand", 5, plate, m) is None        # already a patch on the plate
    assert len(store.load_key(work, "sand", 5).versions) == 2
    KS.undo(work, "sand", 5)                                            # the model's whole frame is one undo away
    assert store.load_key(work, "sand", 5).active == "v001"
    KS.create_keys(work, "sea", [5], lambda f: plate, "png")
    assert KS.rebase_on_plate(work, "sea", 5, plate, m) is None         # a plate key stays


def test_effects_mask_takes_the_shadow_next_to_the_object_only():
    """sh060 2026-11-10: the woman's shadow and the sand she kicks up lie outside her matte; the model removed them."""
    fill = _smooth(4, 240, 320)
    plate = fill * 0.9                                             # the model's frame is globally brighter
    m = np.zeros((240, 320), np.uint8)
    m[60:140, 100:140] = 255
    plate[m > 0] = 0.1                                             # the object
    plate[140:200, 110:170] *= 0.5                                 # its shadow, touching it
    plate[20:50, 250:300] *= 0.5                                   # another dark thing far away: not an effect
    e = KS.effects_mask(plate, fill, m, grow_px=6)
    assert e[170, 140] == 255 and e[100, 120] == 255               # shadow + object
    assert e[35, 275] == 0 and e[220, 30] == 0                     # the far thing and plain ground stay the plate
    out = KS.patch_on_plate(plate, fill, m, grow_px=6, soft_px=4, effects=True)
    assert abs(float(out[170, 140].mean()) - float(plate[220, 140].mean())) < 0.1    # shadow gone, plate-toned
    assert np.array_equal(out[35, 275], plate[35, 275])
    off = KS.patch_on_plate(plate, fill, m, grow_px=6, soft_px=4, effects=False)
    assert np.array_equal(off[170, 140], plate[170, 140])          # without effects the shadow is the plate's


def test_rebase_again_rebuilds_from_the_model_frame(work):
    plate = _smooth(5)
    m = np.zeros((96, 128), np.uint8)
    m[30:50, 40:60] = 255
    KS.add_version(work, "sand", 7, np.clip(plate * 0.7, 0, 1), "experiment void", "png")
    KS.rebase_on_plate(work, "sand", 7, plate, m, 4, 6)
    assert KS.rebase_on_plate(work, "sand", 7, plate, m, 4, 6) is None
    k = KS.rebase_on_plate(work, "sand", 7, plate, m, 10, 6, again=True)
    assert k.active == "v003" and k.versions[-1]["source"] == "plate+fill:v001"


# ----------------------------------------------------------------------------- key steps (user 2026-11-10)
def test_delete_version_is_permanent_and_its_number_never_reused(work):
    KS.add_version(work, "sea", 3, _img(1), "plate", "png")
    KS.add_version(work, "sea", 3, _img(2), "regen:lama", "png", base="v001")
    k = KS.add_version(work, "sea", 3, _img(3), "regen:qwen21_edit", "png", prompt="sand", base="v001",
                       extra={"area": "patch", "strength": None})
    assert k.versions[1]["base"] == "v001" and k.versions[2]["area"] == "patch" and "strength" not in k.versions[2]
    kd = T.key_dir(work, "sea", 3)
    size = (kd / "v002.png").stat().st_size
    assert KS.delete_version(work, "sea", 3, "v002") == size
    assert not (kd / "v002.png").exists() and not (T.root_dir(work) / "trash").exists()
    k = store.load_key(work, "sea", 3)
    assert [v["v"] for v in k.versions] == ["v001", "v003"] and k.active == "v003"
    np.testing.assert_allclose(KS.read_key_image(work, k, "v001"), _img(1), atol=0.5 / 255)
    # the name of a deleted version is never given to a new one (another version's `base` may name it)
    assert KS.add_version(work, "sea", 3, _img(4), "plate", "png").active == "v004"
    KS.set_active(work, "sea", 3, "v003")
    KS.delete_version(work, "sea", 3, "v004")
    assert KS.add_version(work, "sea", 3, _img(5), "plate", "png").active == "v005"
    assert (kd / KS.LAST_VERSION).read_text(encoding="utf-8") == "5"


def test_version_numbers_of_versions_deleted_into_the_old_trash_stay_taken(work):
    """Versions deleted before 2026-11-13 sit in <work>/inpaint/trash/ as keyver_*: read only, their numbers stay taken."""
    KS.add_version(work, "sea", 3, _img(1), "plate", "png")
    kd = T.key_dir(work, "sea", 3)
    (kd / KS.LAST_VERSION).unlink()                                # a key from before .last_version existed
    old = T.root_dir(work) / "trash"
    old.mkdir(parents=True)
    (old / f"keyver_sea_{kd.name}_v004_20261112_101010.json").write_text("{}", encoding="utf-8")
    assert KS.add_version(work, "sea", 3, _img(2), "plate", "png").active == "v005"
    assert (old / f"keyver_sea_{kd.name}_v004_20261112_101010.json").is_file()   # the old trash is never touched


def test_delete_version_refusals(work):
    KS.add_version(work, "sea", 3, _img(1), "plate", "png")
    KS.add_version(work, "sea", 3, _img(2), "regen:lama", "png")
    with pytest.raises(ValueError, match="aktywna"):
        KS.delete_version(work, "sea", 3, "v002")
    with pytest.raises(ValueError, match="nie ma wersji"):
        KS.delete_version(work, "sea", 3, "v009")
    with pytest.raises(ValueError, match="Brak klucza"):
        KS.delete_version(work, "sea", 4, "v001")
    KS.delete_version(work, "sea", 3, "v001")
    with pytest.raises(ValueError, match="aktywna"):              # the last one is also the active one
        KS.delete_version(work, "sea", 3, "v002")
    # a key whose active version is broken off (never expected) still keeps its last version
    k = store.load_key(work, "sea", 3)
    k.active = None
    store.save_key(work, k)
    with pytest.raises(ValueError, match="ostatnia wersja"):
        KS.delete_version(work, "sea", 3, "v002")
    assert [v["v"] for v in store.load_key(work, "sea", 3).versions] == ["v002"]


def test_rebase_again_skips_when_its_model_frame_was_deleted(work):
    plate = _smooth(5)
    m = np.zeros((96, 128), np.uint8)
    m[30:50, 40:60] = 255
    KS.add_version(work, "sand", 7, np.clip(plate * 0.7, 0, 1), "experiment void", "png")
    KS.rebase_on_plate(work, "sand", 7, plate, m, 4, 6)
    KS.delete_version(work, "sand", 7, "v001")
    assert KS.rebase_on_plate(work, "sand", 7, plate, m, 10, 6, again=True) is None


def test_patch_on_plate_with_a_ready_weight():
    plate = _smooth(6)
    fill = np.clip(plate * 0.6, 0, 1)
    m = np.zeros((96, 128), np.float32)
    m[30:50, 40:60] = 1
    wgt = m.copy()
    wgt[25:30, 40:60] = 0.5                                        # a soft edge above the hole only
    out = KS.patch_on_plate(plate, fill, m, 0.0, 0.0, tone=False, weight=wgt)
    assert np.array_equal(out[wgt == 0], plate[wgt == 0])
    np.testing.assert_allclose(out[m > 0], fill[m > 0], atol=1e-6)
    np.testing.assert_allclose(out[27, 50], 0.5 * plate[27, 50] + 0.5 * fill[27, 50], atol=1e-6)
