# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""„👁 Podgląd przez profil” (user request 2026-11-20): the viewer can show the plate through a colour profile curve
(engine/log_curves lin → log) – GET /api/frame/{f}?view=<curve> and …/tile?view=. View only: the proxy the models'
stages read and the default view stay the PlateToDisplay image."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import log_curves as lc  # noqa: E402
from engine.exr_io import read_half_rgb  # noqa: E402
from engine.ingest import ProxyGenerator  # noqa: E402
from engine.profile_view import view_rgb  # noqa: E402


def _png(r):
    assert r.status_code == 200, r.text
    return cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_UNCHANGED)


def test_view_rgb_is_the_curve_of_the_linear_plate(tmp_path):
    from engine.exr_io import write_exr_like
    import Imath
    import OpenEXR
    rng = np.random.default_rng(1)
    lin = (rng.random((12, 16, 3)) * 4.0).astype(np.float16)
    hdr = OpenEXR.Header(16, 12)
    hdr["channels"] = {c: Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)) for c in "RGB"}
    p = tmp_path / "f.exr"
    write_exr_like(p, {"R": lin[..., 0], "G": lin[..., 1], "B": lin[..., 2]}, hdr)
    for cid, c in lc.LOG_CURVES.items():
        want = (np.clip(c.lin_to_log(lin.astype(np.float32)), 0, 1) * 255 + 0.5).astype(np.uint8)
        assert np.array_equal(view_rgb(p, "ACES2065-1", cid), want), cid
    # 8-bit display source: decoded with the sRGB EOTF first – the „video” curve gives the image back
    png = tmp_path / "f.png"
    img = rng.integers(0, 256, (12, 16, 3), dtype=np.uint8)
    cv2.imwrite(str(png), img)
    assert np.abs(view_rgb(png, "sRGB", "video").astype(int) - img[..., ::-1].astype(int)).max() <= 1


def test_frame_and_tile_view_endpoints(tmp_path, monkeypatch):
    from test_step7_faithful_preview import _api_project
    app_module, api, shot, pipe, frames, start = _api_project(tmp_path, monkeypatch, n=3, w=2560, h=1440)
    src = ProxyGenerator.resolve_frame_path(shot.plate_pattern, start)
    normal = api.get(f"/api/frame/{start}").content
    work = Path(app_module.STATE["work_dir"])
    proxy_before = (work / "proxy" / "beauty" / f"p_{start:04d}.jpg").read_bytes()
    full = view_rgb(src, "ACES2065-1", "acescct")[..., ::-1]
    t = _png(api.get(f"/api/frame/{start}/tile?x=1000&y=300&w=640&h=400&fmt=png&view=acescct"))
    assert np.array_equal(t, full[300:700, 1000:1640])
    pv = api.get(f"/api/frame/{start}?view=acescct")
    assert pv.status_code == 200 and pv.content != normal
    vp = work / "proxy" / "beauty_view_acescct" / f"p_{start:04d}.jpg"
    assert vp.exists() and cv2.imread(str(vp)).shape[1] == 1920
    # the models' proxy and the default view are untouched
    assert (work / "proxy" / "beauty" / f"p_{start:04d}.jpg").read_bytes() == proxy_before
    assert api.get(f"/api/frame/{start}").content == normal
    assert np.array_equal(_png(api.get(f"/api/frame/{start}/tile?x=0&y=0&w=64&h=64&fmt=png")),
                          ProxyGenerator.load_frame_rgb(src, colorspace="ACES2065-1")[:64, :64, ::-1])
    assert api.get(f"/api/frame/{start}?view=nope").status_code == 400
    assert read_half_rgb(src) is not None
