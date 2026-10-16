# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Bug 2026-10-15: after opening another project the preview kept showing the previous one. Frames / tiles / mask
overlays still loading for the old project could land in the new project's caches (the "loading" set was not cleared,
the tile cache was never cleared, object names repeat between shots). Each project open bumps a generation: late
responses of the old one are dropped and the URLs carry it."""
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "static"


def test_project_generation_guards_frames_tiles_overlays():
    app = (STATIC / "app.js").read_text(encoding="utf-8")
    apply_ = app[app.index("  applyProject(data) {"):app.index("  setActivePlate(name) {")]
    for s in ("this.projectGen = (this.projectGen || 0) + 1;", "this.loadingFrames.clear();", "this.tiles.cache.clear();"):
        assert s in apply_, s
    load = app[app.index("  loadFrame(frameNum, callback, plateName, alignedFlag) {"):]
    assert "&g=${gen}" in load and "if (gen !== (this.projectGen || 0)) return;" in load
    assert app.count("&g=${gen}") >= 2                                       # frames + tiles
    obj = (STATIC / "objects.js").read_text(encoding="utf-8")
    assert "const gen = S.projectGen || 0;" in obj
    assert "S.projectGen = (S.projectGen || 0) + 1; clearOverlays();" in obj
