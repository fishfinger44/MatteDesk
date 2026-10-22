# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Full model names in user-visible text (user rule 2026-10-18): MatAnyone2, SAM 3.1, CoTracker3, LTX-2.5 – never the
short forms. tools/full_model_names.py knows which parts of each file are text; every target must already be fixed."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import full_model_names as F   # noqa: E402


def test_every_target_uses_full_names():
    bad = []
    for p, fn in F.targets():
        if not p.exists():
            continue
        with open(p, encoding="utf-8", newline="") as fh:
            src = fh.read()
        if fn(src) != src:
            bad.append(str(p.relative_to(ROOT)))
    assert not bad, f"short model names in UI text – run `python tools/full_model_names.py`: {bad}"


def test_rules_keep_identifiers():
    assert F.fix_text("SAM3 i MatAnyone, CoTracker, LTX Alpha-Gen") == "SAM 3.1 i MatAnyone2, CoTracker3, LTX-2.5 Alpha-Gen"
    for keep in ("SAM3_Detect", "sam3.1_multiplex_fp16", "MatAnyone2-Keyframes", "CoTracker3", "LTX-2.5", "LTX_MODELS", "SAM 3.1"):
        assert F.fix_text(keep) == keep
    js = "const S = { key: 'SAM3', label: 'SAM3' }; x = ST.SAM3; paneTitle('LTX', `LTX ${ST.LTX} z SAM3`); // SAM3"
    assert F.fix_js(js) == "const S = { key: 'SAM3', label: 'SAM 3.1' }; x = ST.SAM3; paneTitle('LTX', `LTX-2.5 ${ST.LTX} z SAM 3.1`); // SAM3"
    py = 'X = {"core": "SAM3"}  # SAM3\nraise ValueError(f"{\'MatAnyone2\' if a else n} / MatAnyone")\n'
    assert F.fix_py(py) == 'X = {"core": "SAM 3.1"}  # SAM3\nraise ValueError(f"{\'MatAnyone2\' if a else n} / MatAnyone2")\n'
    assert F.fix_md("SAM3 w `ST.SAM3` i\n```\nSAM3\n```\n") == "SAM 3.1 w `ST.SAM3` i\n```\nSAM3\n```\n"
