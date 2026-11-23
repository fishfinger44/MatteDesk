"""engine/track_export: the band's tracker (band points + mask centre) as Nuke Tracker4 / After Effects nulls."""
import re

import numpy as np

from engine import track_export as TE


def _square(cx, cy, r):
    # OpenCV-like outer contour (image coords): signed area < 0
    return [[cx - r, cy - r], [cx - r, cy + r], [cx + r, cy + r], [cx + r, cy - r]]


def test_centroid_of_square():
    c = TE.centroid([np.array(_square(100, 50, 10), float)])
    assert abs(c[0] - 100) < 1e-6 and abs(c[1] - 50) < 1e-6


def test_tracks_points_and_centre():
    pts = {1: [{"id": "p1", "x": 10, "y": 20}, {"id": "p2", "x": 5, "y": 5, "visible": False}],
           2: [{"id": "p1", "x": 12, "y": 21}, {"id": "p2", "x": 6, "y": 6}]}
    lines = {1: {"band_contours": [_square(100, 50, 10)]}, 2: {"band_contours": [_square(104, 52, 10)]}}
    trk = TE.tracks(pts, lines)
    names = [t["name"] for t in trk]
    assert names == ["pkt_1", "pkt_2", TE.CENTRE]
    assert trk[0]["keys"] == {1: (10.0, 20.0), 2: (12.0, 21.0)}
    assert set(trk[1]["keys"]) == {2}                       # hidden in frame 1 -> left out
    assert abs(trk[2]["keys"][2][0] - 104) < 1e-6
    assert TE.summary(trk) == {"tracks": 3, "points": 2, "centre": 1}


def test_tracks_without_points_still_have_centre():
    trk = TE.tracks({}, {5: {"band_contours": [_square(10, 10, 4)]}})
    assert [t["name"] for t in trk] == [TE.CENTRE]


def test_nk_tracker_format():
    trk = [{"name": "pkt_1", "id": "p1", "keys": {3: (10.0, 20.0), 4: (11.0, 22.0)}}]
    txt = TE.nk_text(trk, 200, 100, "TRK_x", "Tracker {x}")
    assert "Tracker4 {" in txt and "{ 1 31 1 }" in txt and "name TRK_x" in txt
    assert txt.count("{ 1 1 0 ") == 10 and len(re.findall(r"\{ \d \d \d+ \w+ \w+ 1 \}", txt)) == 31
    # plate px -> Nuke: x + 0.5, H - y - 0.5
    assert "{curve x3 10.500 x4 11.500}" in txt and "{curve x3 79.500 x4 77.500}" in txt
    assert "{x}" not in txt.split("label", 1)[1]            # braces in the label would break the script


def test_jsx_nulls():
    trk = [{"name": TE.CENTRE, "id": None, "keys": {1: (0.0, 0.0), 2: (1.0, 1.0)}}]
    js = TE.jsx_text(trk, 200, 100, 1, "TRK_x", "label")
    assert "addNull" in js and '"srodek_maski"' in js and "setValuesAtTimes" in js
