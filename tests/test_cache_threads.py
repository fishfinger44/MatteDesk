# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Audit 2026-11-02 #20: sync endpoints run in a thread pool next to the job, so the small in-memory caches and the
read-modify-write of object.json are used in parallel. A cache is only a cache: same values as without it, no
`KeyError` -> 500 from another thread's eviction, no slider write lost to a concurrent update()."""

import threading
import time

import numpy as np
import pytest

import app as app_module
from engine import groups
from engine.lru import LockedLRU
from engine.objects import ObjectStore
from engine.pipeline import ObjectPipeline


def _together(*fns):
    """Run the functions at the same moment; the first exception of any of them is raised here."""
    errors = []
    gate = threading.Barrier(len(fns))

    def run(fn):
        try:
            gate.wait()
            fn()
        except Exception as e:  # the assertion below names it
            errors.append(e)
    ts = [threading.Thread(target=run, args=(fn,)) for fn in fns]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors, repr(errors[0])


@pytest.fixture
def slow_writes(monkeypatch):
    """Every object.json write takes a while, so an unlocked read-modify-write of a second thread always overlaps it."""
    write = ObjectStore._write

    def slow(p, data):
        time.sleep(0.05)
        write(p, data)
    monkeypatch.setattr(ObjectStore, "_write", staticmethod(slow))


def test_parallel_updates_keep_every_field(tmp_path, slow_writes):
    ObjectStore(tmp_path).create("o")
    # two store instances, as in the app (each request builds its own): the lock belongs to the object, not the instance
    _together(lambda: ObjectStore(tmp_path).update("o", slider_a=1, tool={"status": "liczony"}),
              lambda: ObjectStore(tmp_path).update("o", slider_b=2, tool={"final_valid": True}),
              lambda: ObjectStore(tmp_path).update("o", slider_c=3))
    o = ObjectStore(tmp_path).get("o")
    assert (o["slider_a"], o["slider_b"], o["slider_c"]) == (1, 2, 3)
    assert o["tool"]["status"] == "liczony" and o["tool"]["final_valid"] is True


def test_object_lock_is_per_object_and_reentrant(tmp_path):
    s = ObjectStore(tmp_path)
    s.create("a")
    s.create("b")
    with s.lock("a"):
        s.update("a", x=1)       # the same thread may nest (update inside a locked edit)
        done = []
        t = threading.Thread(target=lambda: done.append(s.update("b", y=2)))
        t.start()
        t.join(5)
        assert done, "a lock on one object must not block another"


def test_object_life_edit_does_not_lose_a_parallel_update(tmp_path, slow_writes, monkeypatch):
    """PUT life is the one endpoint with its own get + write (a key can be removed, update() only sets)."""
    store = ObjectStore(tmp_path)
    store.create("o")
    monkeypatch.setattr(app_module, "_objects", lambda: ObjectStore(tmp_path))
    monkeypatch.setattr(app_module, "_shot_range", lambda: (1, 100))
    monkeypatch.setattr(app_module, "get_object_life", lambda name: {})
    req = app_module.LifeRequest(life=[[10, 20]])
    _together(lambda: app_module.set_object_life("o", req), lambda: ObjectStore(tmp_path).update("o", slider=7))
    o = store.get("o")
    assert o["life"] == [[10, 20]] and o["slider"] == 7


def test_locked_lru_is_a_plain_lru():
    c = LockedLRU(2)
    c.put("a", 1)
    c.put("b", 2)
    assert c.get("a") == 1        # most recent now
    c.put("c", 3)                 # the oldest goes
    assert c.get("b") is None and c.get("a") == 1 and c.get("c") == 3
    assert len(c) == 2 and "a" in c and "b" not in c
    assert c.get("nope", 5) == 5
    c.clear()
    assert len(c) == 0


def test_locked_lru_parallel_use_never_raises():
    c = LockedLRU(4)
    stop = time.time() + 0.5

    def worker(seed):
        def run():
            i = seed
            while time.time() < stop:
                i += 1
                c.put(i % 11, i)
                c.get((i + 3) % 11)
                assert len(c) <= 4
                if i % 97 == 0:
                    c.clear()
        return run
    _together(*[worker(s) for s in range(8)])


def test_the_caches_of_the_audit_use_it(tmp_path, monkeypatch):
    assert isinstance(app_module._LINES_CACHE, LockedLRU)
    assert isinstance(app_module._FT_BASE_CACHE, LockedLRU)
    assert isinstance(groups._CACHE, LockedLRU)


def test_display_cache_in_parallel_threads(tmp_path, monkeypatch):
    """_full_display_u8 from the tile endpoints: an eviction in another thread is not an error, the limit holds, and
    the image is the one of that frame."""
    frame = tmp_path / "f.png"
    frame.write_bytes(b"x")
    monkeypatch.setitem(app_module.STATE, "manifest", {"plates": {"p": "ignored"}, "colorspace": "sRGB"})
    monkeypatch.setitem(app_module.STATE, "display_cache", None)
    app_module.STATE.pop("display_cache")
    monkeypatch.setattr(app_module.ProxyGenerator, "resolve_frame_path", staticmethod(lambda pattern, f: frame))
    seen = {}

    def load(src, colorspace=None):
        n = threading.current_thread().ident
        seen[n] = seen.get(n, 0) + 1
        return np.full((2, 2, 3), 7, np.uint8)
    monkeypatch.setattr(app_module.ProxyGenerator, "load_frame_rgb", staticmethod(load))

    def worker(seed):
        def run():
            for i in range(400):
                img = app_module._full_display_u8("p", (seed + i) % 9)
                assert img.shape == (2, 2, 3) and int(img[0, 0, 0]) == 7
        return run
    _together(*[worker(s) for s in range(8)])
    assert isinstance(app_module.STATE["display_cache"], LockedLRU)
    assert len(app_module.STATE["display_cache"]) <= app_module.DISPLAY_CACHE_FRAMES


def test_stabilize_gray_cache_survives_another_thread(tmp_path, monkeypatch):
    """pipeline._stab_gray: the frame read in this call is returned even when another member's preview evicts it."""
    class Loader:
        def read(self, f):
            return np.full((2, 2, 3), f / 100.0, np.float32)

    grays = []
    import engine.corestab
    monkeypatch.setattr(engine.corestab, "stabilize_core", lambda core, gray, **kw: (grays.append(gray), ({}, {}))[1])
    s = ObjectStore(tmp_path)
    s.create("o")
    p = ObjectPipeline.__new__(ObjectPipeline)
    p.store, p.loader = s, Loader()
    p.load_core_soft = lambda name, frames, core: {}
    p.stabilized_core("o", [1], {}, {}, only=[1])
    gray = grays[0]
    assert isinstance(p._stab_gray, LockedLRU)

    def worker(seed):
        def run():
            for i in range(300):
                f = (seed * 7 + i) % 40
                g = gray(f)
                assert int(g[0, 0]) == int(f / 100.0 * 255)
        return run
    _together(*[worker(s) for s in range(6)])
    assert len(p._stab_gray) <= 17
