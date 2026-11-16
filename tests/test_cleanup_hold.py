# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Removed objects lost for a few frames keep the nearest mask (engine/cleanup_hold.py, 2026-11-12): a short hole in
the removal alpha (SAM 3.1 lost the object) is filled with the nearest present frame's alpha; a long hole and a real
exit through the frame border are left alone; every frame that is not lost comes back bit for bit."""
from pathlib import Path

import cv2
import numpy as np

import api.cleanup as AC
from engine import cleanup_hold as CH
from engine.objects import ObjectStore

W, H = 160, 120
FRAMES = list(range(1, 21))


def blob(f, x0=40, y0=40, w=30, h=30, drift=1):
    """A soft-edged square drifting right, different in every frame (so the source of a held frame is known)."""
    a = np.zeros((H, W), np.float32)
    x = x0 + drift * f
    a[y0:y0 + h, max(0, x):max(0, x + w)] = 1.0
    a[y0:y0 + h, max(0, x - 1):max(0, x)] = 0.4          # a soft column: not only 0 / 1
    return a


class Loader:
    """f -> alpha; `lost` frames are empty; counts its reads."""

    def __init__(self, lost=(), fn=blob, tiny=()):
        self.lost, self.fn, self.tiny, self.reads = set(lost), fn, set(tiny), 0

    def __call__(self, f):
        self.reads += 1
        if f in self.lost:
            return np.zeros((H, W), np.float32)
        if f in self.tiny:                                # a few stray pixels instead of the object
            a = np.zeros((H, W), np.float32)
            a[5:10, 5:10] = 1.0
            return a
        return self.fn(f)


def hold(ld, frames=FRAMES, **kw):
    return CH.Hold(ld, kw.pop("name", "kobieta"), frames, **kw)


def test_short_hole_is_held_from_the_nearest_frame_others_bit_identical():
    ld = Loader(lost={9, 10, 11})
    h = hold(ld)
    assert h.plan()["held"] == {9: 8, 10: 8, 11: 12}                    # the tie (F10) takes the earlier frame
    for f in FRAMES:
        got = h(f)
        want = blob({9: 8, 10: 8, 11: 12}.get(f, f))
        assert got.dtype == want.dtype and np.array_equal(got, want), f
    assert np.array_equal(h(30), blob(30))                              # outside the work range: untouched


def test_nothing_lost_changes_nothing_and_costs_one_scan():
    ld = Loader()
    h = hold(ld)
    out = [h(f) for f in FRAMES]
    assert all(np.array_equal(a, blob(f)) for a, f in zip(out, FRAMES)) and h.plan()["held"] == {}
    assert ld.reads == 2 * len(FRAMES)                                  # one scan of the range + the reads themselves
    assert CH.held_report([h]) == {"objects": {}, "warnings": []}


def test_stray_pixels_count_as_lost():
    h = hold(Loader(tiny={10, 11}))
    assert h.plan()["held"] == {10: 9, 11: 12} and np.array_equal(h(10), blob(9))


def test_hole_longer_than_the_limit_is_not_held():
    lost = set(range(6, 6 + CH.HOLD_MAX_GAP + 1))                       # one frame too long
    h = hold(Loader(lost=lost))
    assert h.plan()["held"] == {} and not h(8).any()
    lost.discard(max(lost))                                             # exactly the limit: held
    assert set(hold(Loader(lost=lost)).plan()["held"]) == lost


def test_exit_through_the_frame_border_is_not_held():
    def leaving(f):                                                     # slides out through the right border, gone from F15
        return blob(f, x0=60, drift=7)
    assert not leaving(15).any() and leaving(14).any()
    h = hold(Loader(fn=leaving))
    assert h.plan()["held"] == {} and not h(16).any()

    def passing(f):                                                     # out through the right border and back in
        return blob(f if f <= 10 else 21 - f, x0=110, drift=6)
    gone = [f for f in FRAMES if not passing(f).any()]
    assert gone and len(gone) <= CH.HOLD_MAX_GAP
    assert hold(Loader(fn=passing)).plan()["held"] == {}

    def entering(f):                                                    # not in the frame yet in F1–F4
        return blob(f, x0=-60, drift=7)
    assert not entering(4).any() and entering(5).any()
    assert hold(Loader(fn=entering)).plan()["held"] == {}


def test_object_at_the_border_that_does_not_shrink_is_held():
    def edge(f):                                                        # stands cut by the left border all the time
        return blob(f, x0=-10, drift=0)
    h = hold(Loader(lost={9, 10}, fn=edge))
    assert h.plan()["held"] == {9: 8, 10: 11}


def test_hole_at_an_end_of_the_range_is_held_from_the_only_side():
    h = hold(Loader(lost={1, 2, 3}))
    assert h.plan()["held"] == {1: 4, 2: 4, 3: 4}
    h = hold(Loader(lost={18, 19, 20}))
    assert h.plan()["held"] == {18: 17, 19: 17, 20: 17}
    assert hold(Loader(lost=set(range(1, 11)))).plan()["held"] == {}    # too long: it was never there
    assert hold(Loader(lost=set(FRAMES))).plan()["held"] == {}          # never present: nothing to hold from


def test_single_detection_is_no_source_and_life_is_respected():
    present_once = set(FRAMES) - {6}                                    # one stray detection in the whole range
    assert hold(Loader(lost=present_once)).plan()["held"] == {}
    life = lambda f: f <= 18                                            # noqa: E731 – the object's life ends at F18
    assert hold(Loader(lost={17, 18, 19, 20}), alive=life).plan()["held"] == {17: 16, 18: 16}


def test_two_objects_are_judged_independently_and_reported():
    a, b = Loader(lost={9, 10, 11}), Loader(lost={5})
    ha, hb, hc = CH.wrap([a, b, Loader()], ["kobieta", "pies", "ptak"], FRAMES)
    assert ha.plan()["held"] == {9: 8, 10: 8, 11: 12} and hb.plan()["held"] == {5: 4} and hc.plan()["held"] == {}
    assert np.array_equal(ha(3), blob(3)) and np.array_equal(hb(10), blob(10))
    rep = CH.held_report([ha, hb, hc, lambda f: None])
    assert rep["objects"] == {"kobieta": [{"frames": [9, 10], "from": 8}, {"frames": [11], "from": 12}],
                              "pies": [{"frames": [5], "from": 4}]}
    assert rep["warnings"] == [
        "Obiekt usuwany „kobieta” nie ma maski w F9–F11 (zgubiony) – podstawiona maska z najbliższej klatki "
        "(F9–F10 ← F8, F11 ← F12); popraw maskę w tych klatkach, jeśli łata nie trafia w obiekt",
        "Obiekt usuwany „pies” nie ma maski w F5 (zgubiony) – podstawiona maska z najbliższej klatki "
        "(F5 ← F4); popraw maskę w tych klatkach, jeśli łata nie trafia w obiekt"]
    assert CH.held_report([ha, hb], frames=[1, 2, 3, 4, 5])["objects"] == {"pies": [{"frames": [5], "from": 4}]}


def test_scan_is_cached_by_key_and_redone_when_a_mask_changes():
    ld = Loader(lost={9})
    h1 = hold(ld, key=("t", "cache"))
    assert h1.plan()["held"] == {9: 8} and ld.reads == len(FRAMES)
    h2 = hold(ld, key=("t", "cache"))                                  # the next request: new loaders, the same masks
    assert np.array_equal(h2(9), blob(8)) and ld.reads == len(FRAMES) + 2   # F9 itself + its source, no new scan
    ld.lost = set()                                                     # the user repaired F9
    assert np.array_equal(h2(9), blob(9)) and h2.plan()["held"] == {}   # seen at F9 (area differs): scanned again
    assert hold(ld, frames=lambda: FRAMES[:5], key=("t", "cache")).plan()["held"] == {}   # another range: its own scan


def test_a_failing_scan_holds_nothing():
    def bad(f):
        if f == 7:
            raise RuntimeError("broken file")
        return blob(f)
    h = hold(bad)
    assert h.plan() == {"area": {}, "held": {}} and np.array_equal(h(3), blob(3))


def _core(store, name, lost=(), depth=0):
    store.create(name, [name], depth_order=depth)
    d = store.dir(name) / "core"
    d.mkdir(exist_ok=True)
    for f in FRAMES:
        m = np.zeros((H, W), np.uint8) if f in lost else (blob(f) > 0.5).astype(np.uint8) * 255
        cv2.imwrite(str(d / f"c_{f:04d}.png"), m)


def test_api_loaders_hold_and_the_run_reports(tmp_path):
    store = ObjectStore(tmp_path / "shot" / "matte_desk")
    _core(store, "kobieta", lost={9, 10, 11})
    _core(store, "pies", depth=1)
    STATE = {"work_dir": str(store.root), "manifest": {"plates": {"beauty": str(tmp_path / "b_####.png")}}}
    _, ex = AC.make(STATE=STATE, objects=lambda: store, plate_size=lambda: (W, H), work_frames=lambda: FRAMES,
                    proxy_scale=lambda: 1.0, comfy_client=lambda: None, long_job=lambda t: (lambda f: f),
                    refresh_manifest=lambda: STATE["manifest"])
    s = AC.cu.settings({"remove": ["kobieta"]})
    rm, keep = ex["alpha_loaders"](s, W, H)
    hard = lambda f: (blob(f) > 0.5).astype(np.float32)                # noqa: E731 – what the core loader gives
    assert len(rm) == 1 and [k.name for k in keep] == ["pies"]
    assert not rm[0](10).any() and rm[0].cached() is None               # a call never scans: the preview does not wait
    ex["held"](rm, FRAMES)                                              # a run scans up front
    for f in FRAMES:
        assert np.array_equal(rm[0](f), hard({9: 8, 10: 8, 11: 12}.get(f, f))), f
    assert np.array_equal(keep[0](10), hard(10))                        # the kept object's loader is not wrapped
    mask10 = AC.cu.mask_fn({**s, "holdout": False}, rm, keep)(10)       # what the preview / the video models get
    assert mask10[hard(8) > 0].min() == 1.0

    class Cl:                                                           # the Cleaner's two public fields
        def __init__(self):
            self.report, self.notes = {}, []
    cl = Cl()
    rep = ex["held"](rm, FRAMES, cl)
    assert rep["objects"] == {"kobieta": [{"frames": [9, 10], "from": 8}, {"frames": [11], "from": 12}]}
    assert cl.report["held"] == rep and cl.notes == rep["warnings"] and "F9–F11" in cl.notes[0]
    cl2 = Cl()
    assert ex["held"](rm, [1, 2, 3], cl2)["objects"] == {} and cl2.report == {} and cl2.notes == []
    rm2, _ = ex["alpha_loaders"](AC.cu.settings({"remove": ["pies"]}), W, H)
    assert ex["held"](rm2, FRAMES) == {"objects": {}, "warnings": []}
    assert Path(store.root).is_dir()
