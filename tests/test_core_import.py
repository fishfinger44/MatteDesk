"""External mask as the stage-1 core (engine/core_import.py, user request 2026-11-20) – synthetic."""
import json

import cv2
import numpy as np
import pytest

from engine.core_import import core_source, frame_map, import_core, read_mask, remove_import
from engine.objects import ObjectStore
from engine.stages import object_stages

W, H = 160, 96      # not the SH010 size on purpose


def _seq(d, first, n, w=W, h=H, ext="png", rgba_const_alpha=False):
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        m = np.zeros((h, w), np.uint8)
        x = 20 + i * 3
        cv2.circle(m, (x * w // W, h // 2), max(4, 18 * w // W), 255, -1)
        if rgba_const_alpha:
            img = np.dstack([m, m, m, np.full_like(m, 255)])
        else:
            img = m
        cv2.imwrite(str(d / f"roto.{first + i:04d}.{ext}"), img)
    return d


def test_import_writes_core_crops_keys_and_stage_done(tmp_path):
    store = ObjectStore(tmp_path / "work")
    store.create("obj", ["x"])
    # an old SAM 3.1 key is replaced
    store.save_key("obj", 3, np.ones((H, W), bool), author="user", kind="points")
    src = _seq(tmp_path / "masks", 1, 10)
    info = import_core(store, "obj", str(src), list(range(1, 11)), W, H)
    od = store.dir("obj")
    assert info["visible"] == 10 and info["offset"] == 0
    for f in range(1, 11):
        c = cv2.imread(str(od / "core" / f"c_{f:04d}.png"), cv2.IMREAD_GRAYSCALE)
        assert c.shape == (H, W) and c.max() == 255
        assert (od / "core_soft" / f"p_{f:04d}.png").exists()
    assert (od / "crops.json").exists()
    keys = store.keys("obj")
    assert 3 not in keys and set(keys) == set(info["keys"]) and keys[info["keys"][0]].get("source") == "import"
    assert core_source(od)["kind"] == "import"
    st = {s["id"]: s["state"] for s in object_stages(store, "obj", tmp_path / "work", (1, 10), mplate=False)["stages"]}
    assert st["detect"] == "done" and st["crops"] == "done" and st["core"] == "done"


def test_offset_numbering_and_proxy_size_and_constant_alpha(tmp_path):
    store = ObjectStore(tmp_path / "work")
    store.create("obj", ["x"])
    src = _seq(tmp_path / "m", 1001, 6, w=W // 2, h=H // 2, rgba_const_alpha=True)   # half size, mask in RGB
    info = import_core(store, "obj", str(src / "roto.1003.png"), list(range(1, 7)), W, H)
    assert info["offset"] == 1000
    c = cv2.imread(str(store.dir("obj") / "core" / "c_0001.png"), cv2.IMREAD_GRAYSCALE)
    assert c.shape == (H, W) and c.max() == 255


def test_refusals(tmp_path):
    with pytest.raises(ValueError):
        frame_map([1, 2, 3], [1, 2, 3, 4, 5], 1)
    store = ObjectStore(tmp_path / "work")
    store.create("obj", ["x"])
    src = _seq(tmp_path / "m", 1, 4, w=100, h=100)   # other aspect
    with pytest.raises(ValueError, match="proporcje"):
        import_core(store, "obj", str(src), [1, 2, 3, 4], W, H)
    assert not (store.dir("obj") / "core" / "core.json").exists()   # nothing written on a refusal


def test_life_range_and_remove(tmp_path):
    store = ObjectStore(tmp_path / "work")
    store.create("obj", ["x"])
    store.update("obj", life=[[3, 6]])
    src = _seq(tmp_path / "m", 3, 4)          # only the visible frames exist
    import_core(store, "obj", str(src), list(range(1, 9)), W, H)
    od = store.dir("obj")
    assert cv2.imread(str(od / "core" / "c_0001.png"), cv2.IMREAD_GRAYSCALE).max() == 0
    assert json.loads((od / "core" / "core.json").read_text(encoding="utf-8"))["life"] == [[3, 6]]
    assert remove_import(store, "obj") > 0
    assert core_source(od) is None and not (od / "core").exists() and store.keys("obj")


def test_read_mask_16bit(tmp_path):
    p = tmp_path / "a.0001.png"
    cv2.imwrite(str(p), np.full((8, 8), 65535, np.uint16))
    assert abs(read_mask(str(p)).max() - 1.0) < 1e-6
