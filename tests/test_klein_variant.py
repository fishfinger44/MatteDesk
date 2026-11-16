# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""FLUX 3 Image Klein variant by the card (user decision 2026-11-10): int8-convrot under 16 GB of VRAM (model 5.8 GB +
encoder 4.9 GB fit a 12 GB card), bf16 above; the variant complete on disk is used (a project moved between machines
keeps working); the install downloads the variant of this card; the graph loads the files of the variant used."""
import pytest

from engine import stage_models as G


@pytest.fixture
def auto(monkeypatch):
    monkeypatch.delenv("MD_KLEIN_VARIANT", raising=False)


def _put(md, table):
    for folder, fn, _r, _s, size in table.values():
        p = md / folder / fn
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * 8)
    return md


def test_preferred_by_card(auto):
    assert G.klein_preferred(12.0) == "int8" and G.klein_preferred(11.8) == "int8"
    assert G.klein_preferred(24.0) == "bf16" and G.klein_preferred(16.0) == "bf16"
    assert G.klein_preferred(0) == "bf16"                         # unknown card: full precision


def test_variant_on_disk_wins(auto, tmp_path, monkeypatch):
    small = {k: (a, b, c, d, 4) for k, (a, b, c, d, _n) in G.KLEIN_FILES.items()}
    small8 = {k: (a, b, c, d, 4) for k, (a, b, c, d, _n) in G.KLEIN_FILES_INT8.items()}
    monkeypatch.setattr(G, "KLEIN_FILES", small)
    monkeypatch.setattr(G, "KLEIN_FILES_INT8", small8)
    md = tmp_path / "models"
    assert G.klein_variant(md, 12.0) == "int8" and G.klein_variant(md, 24.0) == "bf16"   # nothing: what to download
    _put(md, small)                                               # bf16 on disk (installed on a 24 GB machine)
    assert G.klein_variant(md, 12.0) == "bf16" and all(m["present"] for m in G.klein_status(md, 12.0))
    _put(md, small8)                                              # both: the card's own
    assert G.klein_variant(md, 12.0) == "int8" and G.klein_variant(md, 24.0) == "bf16"
    names = {m["file"] for m in G.klein_status(md, 12.0)}
    assert "flux3-image-klein-int8-convrot.safetensors" in names and "qwen3vl_4b_int8_convrot.safetensors" in names


def test_graph_and_install_follow_the_variant(auto, tmp_path, monkeypatch):
    from engine.cleanup3d.regen import klein_graph
    g = klein_graph(tmp_path, "p", 64, 64, files=G.KLEIN_FILES_INT8)
    assert g["1"]["inputs"]["unet_name"] == "flux3-image-klein-int8-convrot.safetensors"
    assert g["2"]["inputs"]["clip_name"] == "qwen3vl_4b_int8_convrot.safetensors"
    from engine import installer
    got = []
    monkeypatch.setattr("engine.vram_fit.card_gb", lambda comfy=None: 12.0)
    installer.install_flux3_klein({"models_dir": str(tmp_path / "m")}, download=lambda url, dst: got.append(url) or
                                  (dst.parent.mkdir(parents=True, exist_ok=True), dst.write_bytes(b"x")))
    assert any(u.endswith("flux3-image-klein-int8-convrot.safetensors") for u in got)
    assert not any("bf16" in u.rsplit("/", 1)[-1] for u in got)
