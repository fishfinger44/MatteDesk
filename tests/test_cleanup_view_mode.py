# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""User report 2026-11-10: the Alpha view of „✂ Wycinanie” stayed on the canvas after switching to Cleanup. In Cleanup
mode objects.js setView always shows the plate (without forgetting the user's cutting view); leaving Cleanup restores
it. Checked in the browser (Alpha -> Cleanup = plate -> Cut-out = Alpha again)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_cleanup_mode_forces_the_plate_view_and_restores_the_cut_view():
    obj = (ROOT / "static" / "objects.js").read_text(encoding="utf-8")
    body = obj[obj.index("function setView(v, render = true, remember = true) {"):]
    body = body[:body.index("S.view = viewAllowed(v)")]
    assert "document.body.classList.contains('mode-clean')" in body and "v = 'none'; remember = false;" in body
    cu = (ROOT / "static" / "cleanup.js").read_text(encoding="utf-8")
    assert "if (clean) md.setView('none', true, false); else if (md.restoreUserView) md.restoreUserView();" in cu
    # a plate that disappears (the last Cleanup version deleted) is switched away from WITH md-plate, so the object tools
    # unlock (user 2026-11-10: „przełącz plate na beauty” while beauty was already shown)
    app = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    sel = app[app.index("  updatePlateSelector(plates) {"):app.index("  bgRefLabel() {")]
    assert sel.count("new CustomEvent('md-plate'") == 2
    # the body class is set before the view is reset (setView reads it)
    assert cu.index("document.body.classList.toggle('mode-clean', clean)") < cu.index("md.setView('none', true, false)")
