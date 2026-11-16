# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""C1 anchors „tylko referencja” (user decision 2026-11-13): the main Qwen-Image-2.1 anchor goes to the video model only
as the whole-clip reference image – no frame of the clip is pinned (MiniMax-H3 MiniMaxH3ReferenceToVideo without
MiniMaxH3AddGuide, Wan 2.2 Fun VACE reference_image with the plain control video); LTX-2.5 has no reference input.
Also: the patch levelling (`tone`, one gain per channel for the shot) is on by default."""
from pathlib import Path

import numpy as np

from engine import cleanup as cu
from engine import h3_anchors as HA
from engine import h3_fill as H
from engine import vace_fill as V

FILES = {"unet": "u", "clip": "c", "vae": "v", "turbo": "t"}


def test_settings_default_and_active_anchors():
    s = cu.settings({})
    assert s["anchor_ref_only"] is False and s["tone"] is True
    assert "anchor_ref_only" in cu.PER_TARGET
    lst = HA.defaults(31, [5, 50])
    assert len(cu.active_anchors({"anchor": "frame", "anchors": lst})) == 3
    only = cu.active_anchors({"anchor": "frame", "anchors": lst, "anchor_ref_only": True})
    assert [a["kind"] for a in only] == ["main"] and only[0]["frame"] == 31
    assert cu.active_anchors({"anchor": "none", "anchors": lst, "anchor_ref_only": True}) == []
    assert cu.REF_ONLY_MODELS == ("h3", "vace")


def test_h3_graph_reference_without_pinned_frames(tmp_path):
    g = H.graph(tmp_path, tmp_path, "p", "x", 768, 1344, 56, FILES, guides=[(30, tmp_path)], ref_only=True)
    assert g["20"]["class_type"] == "MiniMaxH3ReferenceToVideo" and g["20"]["inputs"]["ref_images.ref_image_0"] == ["60", 0]
    assert not any(n.get("class_type") == "MiniMaxH3AddGuide" for n in g.values())
    assert g["23"]["inputs"]["conditioning"] == ["20", 0]
    pinned = H.graph(tmp_path, tmp_path, "p", "x", 768, 1344, 56, FILES, guides=[(30, tmp_path)])
    assert any(n.get("class_type") == "MiniMaxH3AddGuide" for n in pinned.values())     # the old mode unchanged


def test_vace_control_video_untouched_and_reference_given(tmp_path, monkeypatch):
    seen = {}
    real = V.control_frames

    def spy(frames, plate, grown, anchors, cw, ch):
        seen["anchors"] = dict(anchors)
        return real(frames, plate, grown, anchors, cw, ch)
    monkeypatch.setattr(V, "control_frames", spy)

    def fake_graph(cd, md, rd, prefix, prompt, w, h, length, f, seed, fast):
        seen["ref"] = rd
        seen["ref_file"] = rd is not None and (Path(rd) / "r.png").exists()   # the job folder goes after the run
        raise RuntimeError("stop after the graph")
    monkeypatch.setattr(V, "graph", fake_graph)
    monkeypatch.setattr(V, "status", lambda *a, **k: {"ok": True, "files": {"high": "h", "low": "l"}}, raising=False)
    frames = list(range(1, 6))
    plate = lambda f: np.full((64, 96, 3), 0.5, np.float32)            # noqa: E731
    mask = lambda f: np.pad(np.ones((16, 16), np.float32), ((24, 24), (40, 40)))   # noqa: E731
    anchor = {3: np.full((64, 96, 3), 255, np.uint8)}
    try:
        V.run(None, tmp_path, {}, frames, plate, mask, "p", anchors=anchor, short_edge=64, ref_only=True)
    except RuntimeError as e:
        assert "stop after the graph" in str(e) or "status" in str(e)
    assert seen["anchors"] == {}                                           # no control frame replaced
    assert seen["ref"] is not None and seen["ref_file"]


def test_run_refuses_ltx_and_passes_ref_only():
    import inspect
    import api.cleanup as AC
    src = inspect.getsource(AC)
    run = src[src.index('ref_only = bool(s.get("anchor_ref_only"))'):src.index("cl = cu.Cleaner(", src.index('ref_only = bool('))]
    assert "cu.REF_ONLY_MODELS" in run and "LTX-2.5 nie ma wejścia referencji" in run
    assert run.count("ref_only=ref_only") == 2                             # MiniMax-H3 and Wan 2.2 Fun VACE


def test_off_for_good_and_out_of_the_ui():
    """User 2026-11-15 (tests v025–v028): „tylko referencja” left the UI; a True saved before is read as False."""
    from pathlib import Path
    assert cu.settings({"anchor_ref_only": True})["anchor_ref_only"] is False
    js = (Path(__file__).resolve().parents[1] / "static" / "cleanup_anchors.js").read_text(encoding="utf-8")
    assert "cuAnchorRef" not in js
