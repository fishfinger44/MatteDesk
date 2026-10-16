# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage 3 local correction (user request 2026-10-12): a click changes ONLY the hole / leak under it.

- engine/correct.py: seeds inside the mask, component under the click (with snapping), every pixel outside the
  changed region identical to the current mask.
- POST /api/object/{o}/correct/{f}: + in a hole of the core -> the hole is filled, nothing else changes;
  - on a leak -> only the leak is removed; result saved at once as a key `correction` (undo works);
  overlay views delta_add / delta_remove; brush keys build on the existing key of the frame.
"""

import cv2
import numpy as np

from engine.correct import component_at, local_correction, seed_points
from test_ui_stages_api import _open


def test_seed_points_inside_and_spread():
    m = np.zeros((200, 300), bool)
    m[40:160, 50:250] = True
    pts = seed_points(m, n=8)
    assert 3 <= len(pts) <= 8
    assert all(m[int(y), int(x)] for x, y in pts)
    far = seed_points(m, n=8, exclude=(150, 100), exclude_r=60)
    assert all(np.hypot(x - 150, y - 100) >= 60 for x, y in far)


def test_local_correction_changes_only_the_clicked_region():
    base = np.zeros((100, 100), bool)
    base[20:80, 20:80] = True
    base[45:55, 45:55] = False            # hole
    sam = np.zeros_like(base)
    sam[18:82, 18:82] = True              # SAM3 also grows the outline a bit (must NOT be taken)
    new, comp = local_correction(base, sam, (50, 50), "add")
    assert comp.sum() == 100 and new[45:55, 45:55].all()
    outside = ~comp
    assert np.array_equal(new[outside], base[outside])
    # snapping: click 5 px beside the hole still finds it; far away -> nothing
    assert component_at(sam & ~base, (50, 58)).sum() > 0
    assert local_correction(base, sam, (5, 5), "add")[1].sum() == 0

    leak = base.copy()
    leak[30:40, 80:95] = True             # leak attached to the right edge
    new, comp = local_correction(leak, base | (sam & False), (90, 35), "remove")
    assert comp.sum() == 150 and not new[30:40, 80:95].any()
    assert np.array_equal(new[~comp], leak[~comp])


def _core_dir(tmp_path):
    return tmp_path / "work" / "matte_desk" / "objects" / "puppet"


def test_api_correct_fills_hole_and_removes_leak(tmp_path, monkeypatch):
    api, shot, mf = _open(tmp_path, monkeypatch)
    f = shot.first + 2
    puppet = shot.masks[f] > 0.5
    ys, xs = np.nonzero(puppet)
    cx, cy = int(np.median(xs)), int(np.median(ys))
    # core of the frame = puppet with a hole
    base = puppet.copy()
    cv2.circle(base.view(np.uint8), (cx, cy), 6, 0, -1)
    hole = puppet & ~base
    assert hole.sum() > 50
    od = _core_dir(tmp_path)
    (od / "core").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(od / "core" / f"c_{f:04d}.png"), base.astype(np.uint8) * 255)
    before = mf.read_bytes()

    r = api.post(f"/api/object/puppet/correct/{f}", json={"mode": "add", "point": [cx, cy]})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["changed_px"] > 0 and d["key"]["kind"] == "correction"
    key = cv2.imread(str(od / "keys" / f"k_{f:04d}_mask.png"), cv2.IMREAD_GRAYSCALE) > 127
    assert key[hole].mean() > 0.95                                    # the hole is filled
    changed = key != base
    assert (changed & ~hole).sum() == 0, "pixels outside the hole changed"
    assert api.get(f"/api/object/puppet/frame/{f}?view=delta_add").status_code == 200

    # leak: a blob of background glued to the puppet in the key -> "-" click removes only the blob
    leak = key.copy()
    x1 = int(xs.max())
    leak[cy - 4:cy + 4, x1 - 2:min(shot.w, x1 + 18)] = True
    blob = leak & ~key
    cv2.imwrite(str(od / "keys" / f"k_{f:04d}_mask.png"), leak.astype(np.uint8) * 255)
    r = api.post(f"/api/object/puppet/correct/{f}", json={"mode": "remove", "point": [min(shot.w - 1, x1 + 10), cy]})
    assert r.status_code == 200, r.text
    assert r.json()["changed_px"] > 0
    key2 = cv2.imread(str(od / "keys" / f"k_{f:04d}_mask.png"), cv2.IMREAD_GRAYSCALE) > 127
    assert not (key2 & blob).any()
    assert ((key2 != leak) & ~blob).sum() == 0, "pixels outside the leak changed"

    # undo -> the mask as it was before the last correction (the leak was written into the key file above)
    assert api.post("/api/object/puppet/undo").json()["success"]
    assert np.array_equal(cv2.imread(str(od / "keys" / f"k_{f:04d}_mask.png"), cv2.IMREAD_GRAYSCALE) > 127, leak)

    # brush key builds on the frame key (not on the old core with the hole)
    r = api.post(f"/api/object/puppet/keys/{f}", json={"source": "core", "strokes": [{"mode": "add", "radius_px": 2, "p": [[2, 2]]}]})
    assert r.status_code == 200
    key3 = cv2.imread(str(od / "keys" / f"k_{f:04d}_mask.png"), cv2.IMREAD_GRAYSCALE) > 127
    assert key3[hole].mean() > 0.95

    # outside the frame -> 400; manifest untouched
    assert api.post(f"/api/object/puppet/correct/{f}", json={"mode": "add", "point": [-5, 10]}).status_code == 400
    assert mf.read_bytes() == before
