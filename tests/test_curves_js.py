"""static/curves.js and engine/curves.py = one format, the same curve (user decision 2026-11-09: one curve module)."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from engine import curves as C

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


def _js(code):
    src = (ROOT / "static" / "curves.js").as_posix()
    out = subprocess.run([NODE, "-e", f"const M = require({json.dumps(src)}); {code}"], capture_output=True, text=True,
                         timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_js_from_points_equals_python():
    pts = [[0, 0], [120, 40], [260, 5], [400, 70]]
    js = _js(f"console.log(JSON.stringify(M.fromPoints({json.dumps(pts)})))")
    py = C.from_points(pts)
    assert js["type"] == py["type"] and js["closed"] == py["closed"] and len(js["pts"]) == len(py["pts"])
    for a, b in zip(js["pts"], py["pts"]):
        assert a["cusp"] == b["cusp"]
        for k in ("p", "in", "out"):
            assert np.allclose(a[k], b[k], atol=1e-9)


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_js_split_matches_python_and_keeps_shape():
    sp = C.fit([[x, 40 * np.sin(x / 150)] for x in range(0, 900, 5)], tol=0.5)
    js = _js(f"const s = {json.dumps(sp)}; M.split(s, 1.4); console.log(JSON.stringify(s))")
    py = C.split(json.loads(json.dumps(sp)), 1.4)
    for a, b in zip(js["pts"], py["pts"]):
        for k in ("p", "in", "out"):
            assert np.allclose(a[k], b[k], atol=1e-9)
    xy_js = np.array(_js(f"console.log(JSON.stringify(M.sample({json.dumps(js)}, 1).xy))"))
    xy0, _, _ = C.sample(sp, 0.5)
    assert max(float(np.min(np.linalg.norm(xy0 - p, axis=1))) for p in xy_js[::5]) < 0.5


def test_static_js_has_no_ui_strings():
    """The editor is host-agnostic: no UI text inside (hosts translate their own labels - i18n rule)."""
    s = (ROOT / "static" / "curves.js").read_text(encoding="utf-8")
    code = "\n".join(line for line in s.splitlines() if not line.strip().startswith(("/*", "*", "//")))
    assert "textContent" not in code and "innerHTML" not in code and "alert(" not in code
