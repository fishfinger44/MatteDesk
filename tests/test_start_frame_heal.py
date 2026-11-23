"""A deleted start key (points cleared / empty SAM 3.1 result) must not block „Update changes” (2026-11-23, rig F25 -> F28)."""
import numpy as np

from engine.objects import ObjectStore


def test_stale_start_frame_heals_to_nearest_key(tmp_path):
    store = ObjectStore(tmp_path)
    store.create("rig")
    m = np.zeros((8, 8), bool)
    m[2:5, 2:5] = True
    store.save_key("rig", 28, m, author="user", kind="points")
    store.save_key("rig", 40, m, author="user", kind="points")
    store.update("rig", tool={"start_frame": 25})          # key 25 was deleted
    assert store.start_frame("rig") == 28
    assert store.get("rig")["tool"]["start_frame"] == 28    # saved
    store.update("rig", tool={"start_frame": 40})
    assert store.start_frame("rig") == 40                   # a valid start stays


def test_start_frame_none_without_keys(tmp_path):
    store = ObjectStore(tmp_path)
    store.create("rig")
    store.update("rig", tool={"start_frame": 25})
    assert store.start_frame("rig") is None
