# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Doctor's GPU memory warning before an install (user 2026-11-10, engine/vram_fit.py): a verdict per heavy model on
THIS card, attached to the Doctor's rows, shown in the row and in the install question (static/app.js)."""
from pathlib import Path

import pytest

import doctor
from engine import vram_fit as vf

ROOT = Path(__file__).resolve().parent.parent


def test_table_is_sane_and_keys_are_doctor_components():
    for key, need in vf.NEEDS.items():
        assert key in doctor.LINKS, key                       # rated rows are found by the Doctor's LINKS key
        assert need["basis"] in ("measured", "estimate") and need["why"]
        mins = [t[0] for t in need["tiers"]]
        assert mins == sorted(mins, reverse=True) and mins[-1] == 0.0, key   # largest first, a floor for any card
        assert all(t[1] in vf.LEVELS for t in need["tiers"])


@pytest.mark.parametrize("card,expect", [
    (12.0, {"void": "spill", "ltx": "no", "seedvr2": "ok", "flux3_klein": "spill", "qwen21": "ok",
            "cotracker3": "ok", "videomama": "spill"}),
    (24.0, {"void": "ok", "ltx": "spill", "seedvr2": "ok", "flux3_klein": "ok", "qwen21": "ok",
            "cotracker3": "ok", "videomama": "ok"}),
])
def test_verdicts_on_a_12_and_a_24_gb_card(card, expect):
    got = {k: vf.assess(k, card)["level"] for k in expect}
    assert got == expect


def test_card_reported_a_bit_under_its_size_counts_as_that_size():
    # Comfy /system_stats reports a 12 GB card as ~11.8 GB: Qwen-Image-2.1 must not drop to "spill"
    assert vf.assess("qwen21", 11.8)["level"] == "ok"


def test_unknown_card_or_light_model_gives_no_verdict():
    assert vf.assess("void", None) is None
    assert vf.assess("sam3", 12.0) is None


def test_videomama_crop_follows_the_measured_line():
    r = vf.assess("videomama", 12.0)
    assert r["basis"] == "measured" and "704 px" in r["text"]


def test_doctor_rows_carry_the_verdict(monkeypatch):
    monkeypatch.setattr(vf, "card_gb", lambda comfy=None: 12.0)
    rep = doctor.EnvironmentDoctor().run_all_checks()
    by = {i.name: i for i in rep.items}
    assert by[doctor.VOID_ITEM].vram["level"] == "spill"                # sequential_cpu_offload below 19 GB
    assert by["Model: LTX-2.5 Alpha-Gen"].vram["level"] == "no"
    assert by[doctor.SEEDVR2_ITEM].vram["level"] == "ok"                 # the 3B fp8 on 12 GB (2026-11-11)
    assert by[doctor.QWEN21_ITEM].vram["level"] == "ok"
    assert by[doctor.KLEIN_LORA_ITEM].vram is None            # one warning per model: on the model row, not the LoRA


def test_every_model_row_has_a_module(monkeypatch):
    """Doctor's module filter (user 2026-11-10): every model / node row is Alpha, Cleanup or both – not "system"."""
    monkeypatch.setattr(vf, "card_gb", lambda comfy=None: 12.0)
    rep = doctor.EnvironmentDoctor().run_all_checks()
    assert set(doctor.MODULE_OF.values()) <= {"alpha", "cleanup", "both"}
    for i in rep.items:
        assert i.module in ("system", "alpha", "cleanup", "both")
        if i.name.startswith("Model:") or i.vram is not None:
            assert i.module != "system", i.name
    by = {i.name: i.module for i in rep.items}
    assert by[doctor.QWEN21_ITEM] == "cleanup" and by["Model: SAM 3.1"] == "alpha" and by["Model: CoTracker3"] == "both"


def test_ui_filter_by_module_and_vram():
    js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    for s in ('id="doctorModSel"', 'id="doctorVramSel"', "this.wireDoctorFilter()", 'class="doctor-row" data-module='):
        assert s in js
    body = js[js.index("doctorRowShown(module, vram, mod, vf) {"):js.index("wireDoctorFilter() {")]
    assert "module === 'both' && (mod === 'alpha' || mod === 'cleanup')" in body


def test_ui_shows_the_warning_and_keeps_no_fit_out_of_install_all():
    js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    assert "${vramLine}" in js and 'data-what="${act[0]}"${vramAsk} data-ask=' in js
    assert "html: vramWarn + this.esc(b.dataset.ask" in js     # the install question shows the warning first
    assert "!(vr && vr.level === 'no')" in js
