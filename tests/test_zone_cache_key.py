"""Audit 2026-11-02 #5: the key of a matte zone's frame covers the CONTENT of what forces the trimap: the holdout of the
objects in front and the forced background. It used to be `bgm.sum()` alone, without the holdout, so a new version of
the front object (same or other area) gave „Zastosuj” the old alpha from the cache."""
import sys
from pathlib import Path

import numpy as np
import pytest

from engine import vitmatte as vm
from engine.zonecache import ZoneCache, mask_token
from engine.zones import Zone, _matte_zone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_zone_vitmatte import MockRunner, _frames_scene  # noqa: E402


def _two_masks(region, core):
    """Two masks of the same pixel count at different places of the zone (a `sum` cannot tell them apart)."""
    ys, xs = np.where(region & ~core)
    a, b = np.zeros(core.shape, bool), np.zeros(core.shape, bool)
    a[ys[:40], xs[:40]] = True
    b[ys[-40:], xs[-40:]] = True
    assert a.sum() == b.sum() and (a != b).any()
    return a, b


def _ma_run(seg, sf, core_, keys_, start, scale, tag):
    return {f: np.full(core_[f].shape, 0.7, np.float32) for f in sf}


def _run(tmp_path, matting, monkeypatch, runner, hold=None, bg=None):
    monkeypatch.setattr(vm, "get_runner", lambda model_id=None: runner)
    img, core, _hair, region, frames = _frames_scene(2)
    ys, xs = np.where(region)
    poly = [[int(xs.min()), int(ys.min())], [int(xs.max()), int(ys.min())], [int(xs.max()), int(ys.max())], [int(xs.min()), int(ys.max())]]
    z = Zone("z_01", "add", "matte", 1, polygon=poly, matting=matting, params={"temporal": "none"})
    empty = np.zeros(core.shape, bool)
    _, _, st = _matte_zone(z, frames, {f: core for f in frames}, lambda f: img, {f: region for f in frames},
                           {f: np.eye(2, 3, dtype=np.float32) for f in frames}, {f: empty for f in frames},
                           (lambda f: hold) if hold is not None else None, ma_run=_ma_run,
                           bg_masks={f: (bg if bg is not None else empty) for f in frames},
                           cache=ZoneCache(tmp_path, "z_01"))
    return st


def _masks():
    _img, core, _hair, region, _frames = _frames_scene(2)
    return _two_masks(region, core)


@pytest.mark.parametrize("matting", ["knn", "vitmatte", "matanyone"])
def test_same_inputs_come_from_the_cache(tmp_path, monkeypatch, matting):
    a, _b = _masks()
    r = MockRunner()
    assert _run(tmp_path, matting, monkeypatch, r, hold=a)["computed"] == 2
    st = _run(tmp_path, matting, monkeypatch, r, hold=a)
    assert st["computed"] == 0 and st["cached"] == 2


@pytest.mark.parametrize("matting", ["knn", "vitmatte", "matanyone"])
def test_new_front_object_invalidates_the_frame(tmp_path, monkeypatch, matting):
    a, b = _masks()
    r = MockRunner()
    _run(tmp_path, matting, monkeypatch, r, hold=a)
    st = _run(tmp_path, matting, monkeypatch, r, hold=b)     # same area, another place: a new version of the front
    assert st["computed"] == 2 and st["cached"] == 0
    st = _run(tmp_path, matting, monkeypatch, r)             # the front object is gone
    assert st["computed"] == 2 and st["cached"] == 0


@pytest.mark.parametrize("matting", ["knn", "vitmatte", "matanyone"])
def test_forced_background_counts_by_content(tmp_path, monkeypatch, matting):
    a, b = _masks()
    r = MockRunner()
    _run(tmp_path, matting, monkeypatch, r, bg=a)
    st = _run(tmp_path, matting, monkeypatch, r, bg=b)       # the same number of pixels: `bgm.sum()` called this equal
    assert st["computed"] == 2 and st["cached"] == 0


def test_holdout_off_in_the_zone_is_not_in_the_key(tmp_path, monkeypatch):
    """A zone with holdout off does not read it: another front object changes nothing, no recomputing."""
    a, b = _masks()
    monkeypatch.setattr(vm, "get_runner", lambda model_id=None: MockRunner())
    img, core, _hair, region, frames = _frames_scene(2)
    ys, xs = np.where(region)
    poly = [[int(xs.min()), int(ys.min())], [int(xs.max()), int(ys.min())], [int(xs.max()), int(ys.max())], [int(xs.min()), int(ys.max())]]
    z = Zone("z_01", "add", "matte", 1, polygon=poly, matting="knn", holdout=False)
    empty = np.zeros(core.shape, bool)
    cache = ZoneCache(tmp_path, "z_01")

    def go(hold):
        return _matte_zone(z, frames, {f: core for f in frames}, lambda f: img, {f: region for f in frames},
                           {f: np.eye(2, 3, dtype=np.float32) for f in frames}, {f: empty for f in frames},
                           lambda f: hold, cache=cache)[2]
    go(a)
    assert go(b)["computed"] == 0


def test_mask_token_is_content_and_shape():
    a = np.zeros((8, 8), bool)
    b = a.copy()
    a[1, 1] = b[2, 2] = True
    assert mask_token(a) != mask_token(b)
    assert mask_token(a) == mask_token(a.copy())
    assert mask_token(None) is None
    assert mask_token(np.zeros((8, 8), bool)) != mask_token(np.zeros((4, 16), bool))
