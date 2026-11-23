"""Band export to Nuke (Roto) / After Effects (masks) – engine/band_export.py + api/band_export.py (2026-11-23)."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from engine import band_export as BE
from engine.bandsys import preview_lines

W, H = 320, 240


def ring_lines(cx: float, r_out: float = 60, r_in: float = 30, feather: bool = True):
    """Preview lines of a band ring (disc r_out minus disc r_in) at x = cx, with feather-end outlines."""
    yy, xx = np.mgrid[:H, :W]
    d = np.hypot(xx - cx, yy - 120)
    band = (d <= r_out) & (d > r_in)
    inner = (d <= r_in + 6) if feather else None
    outer = (d <= r_out + 10) if feather else None
    return preview_lines(band, inner, outer)


def moving(frames=range(1001, 1006)):
    return {f: ring_lines(120 + 5 * (f - 1001)) for f in frames}


def test_split_holes_by_orientation():
    sp = BE.split(ring_lines(150))
    assert len(sp["outer"]) == 1 and len(sp["inner"]) == 1
    assert len(sp["inset_end"]) == 1 and len(sp["outer_end"]) == 1
    assert BE._area(sp["outer"][0]) < 0 < BE._area(sp["inner"][0])


def test_shapes_tracked_with_fixed_point_count():
    shp = BE.shapes(moving())
    assert BE.summary(shp) == {"outer": 1, "inner": 1, "inset_end": 1, "outer_end": 1}
    for s in shp:
        assert (s["first"], s["last"]) == (1001, 1005)
        assert s["pts"].shape[0] == 5 and s["pts"].shape[2] == 2
        assert BE.MIN_PTS <= s["pts"].shape[1] <= BE.MAX_PTS
        # the outline moves 5 px a frame: the points follow (no sliding around the outline)
        step = np.hypot(*(s["pts"][1:] - s["pts"][:-1]).transpose(2, 0, 1))
        assert np.median(step) < 7.0
    outer = next(s for s in shp if s["role"] == "outer")
    c = outer["pts"][2].mean(axis=0)
    assert abs(c[0] - 130) < 1.5 and abs(c[1] - 120) < 1.5


def test_vanishing_outline_ends_its_shape():
    lines = moving(range(1001, 1004))
    lines[1004] = ring_lines(135, feather=False)          # feather ends gone on the last frame
    shp = BE.shapes(lines)
    ends = [s for s in shp if s["role"] in ("inset_end", "outer_end")]
    assert all(s["last"] == 1003 for s in ends)
    assert next(s for s in shp if s["role"] == "outer")["last"] == 1004


def test_nk_text_structure():
    shp = BE.shapes(moving())
    t = BE.nk_text(shp, W, H, "PAS_matanyone_obj", "Pas MatAnyone2 · obj")
    assert t.startswith("set cut_paste_input [stack 0]\nversion 15.0 v1\n")
    assert "Roto {" in t and " inputs 0" in t and f'format "{W} {H} 0 0 {W} {H} 1' in t
    assert "{layer pas {f 0} {t 0} {a vis 1}" in t
    assert "{layer koniec_wtopienia_wciecia {f 0} {t 0} {a vis 0}" in t
    assert "{layer koniec_wtopienia_zewn {f 0} {t 0} {a vis 0}" in t
    pas = t[t.index("{layer pas"):]
    assert pas.index("pas_wewn_1") < pas.index("pas_zewn_1")              # holes first (= on top)
    assert re.search(r"\{a bm 12 ltn 1001 ltm 1005 ltt 3 str 1\}", t)     # hole: blending mode out + lifetime
    assert "{a ltn 1001 ltm 1005 ltt 3 str 1}" in t
    assert t.count("{cc {f 8192}") == 2 * len(shp)                         # every curvegroup has its feather block
    assert "{{1001 " in t and t.count("{") == t.count("}")


def test_nk_single_frame_is_static():
    shp = BE.shapes({1001: ring_lines(150)})
    t = BE.nk_text(shp, W, H, "PAS", "x")
    assert "{{1001 " not in t and t.count("{") == t.count("}")


def test_jsx_text():
    shp = BE.shapes(moving())
    j = BE.jsx_text(shp, W, H, 1001, "PAS_ltx_obj", "Pas LTX-2.5 · obj")
    assert "var W = 320, H = 240, FIRST = 1001" in j
    assert '"m":"ADD"' in j and '"m":"SUBTRACT"' in j and '"m":"NONE"' in j
    assert j.index('"m":"ADD"') < j.index('"m":"SUBTRACT"')               # Add before Subtract
    assert "setValuesAtTimes" in j and "KeyframeInterpolationType.HOLD" in j


def test_endpoint(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.band_export import make

    class Objs:
        def dir(self, name):
            return tmp_path / "objects" / name

    def long_job(label, quiet=False):
        return lambda fn: fn

    seen = []

    def lines_fn(n, s, f):
        seen.append((n, s, f))
        return {"frame": f, "approx": False, **ring_lines(120 + 5 * (f - 1001))}

    router, _ = make(STATE={"work_dir": str(tmp_path)}, long_job=long_job, objects=lambda: Objs(), lines_fn=lines_fn,
                     band_frames=lambda n, f: list(range(1001, 1004)), work_frames=lambda: list(range(1001, 1006)),
                     plate_size=lambda: (W, H))
    app = FastAPI()
    app.include_router(router)
    c = TestClient(app)
    r = c.post("/api/object/obj/stage_band/zone:z1/export?fmt=jsx")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["frames"] == [1001, 1003] and [x[2] for x in seen] == [1001, 1002, 1003]
    assert d["node"] == "PAS_strefa_z1_obj" and d["path"].endswith(".jsx") and "FIRST = 1001" in d["text"]
    assert Path(d["nk"]).exists() and Path(d["jsx"]).exists()
    assert c.post("/api/object/obj/stage_band/core/export").status_code == 400
    assert c.post("/api/object/obj/stage_band/ltx/export?fmt=png").status_code == 400


NUKE = os.environ.get("MD_NUKE_EXE") or r"C:\Program Files\Nuke15.0v1\Nuke15.0.exe"


@pytest.mark.real
@pytest.mark.skipif(not Path(NUKE).exists(), reason="Nuke 15 not installed")
def test_nk_renders_ring_in_nuke(tmp_path):
    """The pasted Roto renders the band ring: alpha 1 in the ring, 0 in the hole and outside (feather ends hidden)."""
    shp = BE.shapes(moving())
    (tmp_path / "pas.nk").write_text(BE.nk_text(shp, W, H, "PAS", "test"), "utf-8")
    png = (tmp_path / "out.png").as_posix()
    (tmp_path / "go.py").write_text(
        "import nuke\n"
        f"nuke.addFormat('{W} {H} 1 mdtest'); nuke.root()['format'].setValue('mdtest')\n"
        f"nuke.nodePaste('{(tmp_path / 'pas.nk').as_posix()}')\n"
        f"w = nuke.nodes.Write(file='{png}', channels='rgba', file_type='png'); w.setInput(0, nuke.toNode('PAS'))\n"
        "nuke.execute(w, 1003, 1003)\n", "utf-8")
    subprocess.run([NUKE, "-t", str(tmp_path / "go.py")], check=True, capture_output=True, timeout=300)
    img = cv2.imread(png, cv2.IMREAD_UNCHANGED)
    a = (img[..., -1] if img.ndim == 3 else img) / (65535.0 if img.dtype == np.uint16 else 255.0)
    assert a.shape == (H, W)
    cx = 130                                    # frame 1003; Nuke writes rows top-down like our y
    assert a[120, cx + 45] > 0.9 and a[120, cx - 45] > 0.9      # ring
    assert a[120, cx] < 0.1                                    # hole
    assert a[120, cx + 68] < 0.1 and a[10, 10] < 0.1           # outer feather end hidden, outside
