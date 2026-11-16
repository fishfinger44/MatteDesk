"""Collapse ▾ of floating palettes = ONE script (user decision 2026-11-11): static/palette_fold.js, attached by
makeFloating to every palette; no module keeps its own palette fold button."""
import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "static"


def _js():
    return {p.name: p.read_text(encoding="utf-8") for p in STATIC.glob("*.js")}


def test_shared_script_loaded_before_the_palettes():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert html.index("/static/palette_fold.js") < html.index("/static/app.js") < html.index("/static/objects.js")
    fold = (STATIC / "palette_fold.js").read_text(encoding="utf-8")
    assert "window.MD_FOLD = { attach, isFolded }" in fold and "md_fold_${k}" in fold


def test_make_floating_attaches_it_to_every_palette():
    obj = (STATIC / "objects.js").read_text(encoding="utf-8")
    a = obj.index("function makeFloating(")
    body = obj[a:obj.index("\n  }\n", a)]
    assert "window.MD_FOLD.attach(box, head)" in body
    # every floating palette goes through makeFloating (the one entry point)
    for name, src in _js().items():
        if name == "palette_fold.js":
            continue
        assert "md-fold" not in src or name == "objects.js", name


def test_objects_palette_uses_the_shared_fold_after_rerender():
    obj = (STATIC / "objects.js").read_text(encoding="utf-8")
    assert 'id="olFold"' not in obj and "olFold" not in obj
    assert "window.MD_FOLD.attach(box, '.ol-head', 'otherLayers'" in obj


def test_no_own_palette_fold_elsewhere():
    for name, src in _js().items():
        if name == "palette_fold.js":
            continue
        assert "function foldable(" not in src, name
        assert not re.search(r"md_cu_fold_|pp-fold", src), name
    css = (STATIC / "index.css").read_text(encoding="utf-8")
    assert "pp-fold" not in css and ".pts-palette.folded" not in css
    assert ".md-folded > :not(.md-fold-head) { display: none !important; }" in css
