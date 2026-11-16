# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/deletion.py – the one permanent delete of the program (user decision 2026-11-13: no project trash)."""
import os
import stat
from pathlib import Path

import pytest

from engine import deletion as DEL


def _tree(root: Path) -> Path:
    d = root / "obj"
    (d / "runs" / "v001").mkdir(parents=True)
    (d / "object.json").write_bytes(b"x" * 10)
    (d / "runs" / "v001" / "a.png").write_bytes(b"y" * 1000)
    return d


def test_size_of(tmp_path):
    d = _tree(tmp_path)
    assert DEL.size_of(d) == 1010 and DEL.size_of(d / "object.json") == 10
    assert DEL.size_of(tmp_path / "missing") == 0 and DEL.size_of(None) == 0


def test_remove_returns_bytes_freed_and_ignores_missing(tmp_path):
    d = _tree(tmp_path)
    f = tmp_path / "loose.txt"
    f.write_bytes(b"z" * 5)
    os.chmod(f, stat.S_IREAD)                                   # read-only (Windows) is deleted too
    assert DEL.remove(d, f, tmp_path / "missing", None) == 1015
    assert not d.exists() and not f.exists()
    assert DEL.remove(d) == 0
    assert list(tmp_path.iterdir()) == []                       # nothing moved anywhere (no trash folder)


def test_remove_refusals(tmp_path):
    d = _tree(tmp_path)
    with pytest.raises(ValueError):
        DEL.remove("")
    with pytest.raises(ValueError):
        DEL.remove(Path(Path.cwd().anchor))                    # a drive / filesystem root
    with pytest.raises(ValueError):
        DEL.remove(d, inside=d / "runs")                       # outside the allowed folder
    with pytest.raises(ValueError):
        DEL.remove(d / "runs", inside=d / "runs")              # the allowed folder itself
    other = tmp_path / "other.txt"
    other.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError):                            # one bad path = nothing is deleted
        DEL.remove(d / "object.json", other, inside=d)
    assert (d / "object.json").exists() and other.exists()
    assert DEL.remove(d / "runs" / "v001", inside=d) == 1000 and (d / "object.json").exists()


def test_no_project_trash_left_in_code_and_ui():
    """User decision 2026-11-13: nothing moves to a project trash any more. No code writes into a `trash` folder (the
    Doctor's weight bin engine/model_trash.py deletes for real and is left as is; keys.py only READS the old trash for
    version numbers) and no UI text promises a trash; the deleting dialogs use UI.confirmDelete."""
    import re
    root = Path(__file__).resolve().parents[1]
    py = [root / "app.py", *sorted((root / "api").glob("*.py")), *sorted((root / "engine").rglob("*.py"))]
    bad = [p.name for p in py if p.name != "model_trash.py"
           and re.search(r"""/\s*["']trash["']""", p.read_text(encoding="utf-8"))
           and p.name != "keys.py"]
    assert not bad, bad
    keys = (root / "engine" / "cleanup3d" / "keys.py").read_text(encoding="utf-8")
    assert keys.count('"trash"') == 1 and "mkdir" not in keys.split('"trash"')[1].split("\n")[0]
    for js in sorted((root / "static").glob("*.js")):
        t = js.read_text(encoding="utf-8")
        assert not re.search(r"kosz(a|u)? projektu|(?<!nie )trafi\w* do kosza|przenie\w* do kosza|w koszu|można (ją|go|je) (stamtąd )?przywrócić", t), js.name
    for name in ("cleanup.js", "cleanup3d.js", "cleanup_anchors.js", "objects.js"):
        assert "UI.confirmDelete(" in (root / "static" / name).read_text(encoding="utf-8"), name
    ui = (root / "static" / "ui.js").read_text(encoding="utf-8")
    assert "Usunięte z dysku na stałe – nie da się cofnąć." in ui and "enter: false" in ui.split("function confirmDelete")[1]


def test_human():
    assert DEL.human(0) == "0 B" and DEL.human(512) == "512 B"
    assert DEL.human(3 * 1024 + 410) == "3,4 kB"
    assert DEL.human(12 * 2**20) == "12,0 MB"
    assert DEL.human(int(1.2 * 2**30)) == "1,2 GB"
    assert DEL.human(None) == "0 B"
