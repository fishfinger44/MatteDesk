# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage „LTX Alpha-Gen” (user decision 2026-10-16: its own stage 6 before the export, out of the experiments' edge
modes). Synthetic: graph recipe (OFX addAlphaGen), batches 8k+1 + crossfade, padding (not stretching), the guide with
the other objects painted out, the region limit, the stage / export wiring through the API with a mock Comfy whose
"Alpha-Gen" returns a matte computed from the guide frames it was given. The real model run = LOCAL_CHECKS."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from engine import ltx_alpha as L   # noqa: E402


# ----------------------------------------------------------------------------- pure helpers
def test_settings_validation():
    s = L.ltx_settings({"ltx": {"guide": "nope", "max_side": 5000, "max_frames": 300, "combine": "x", "overlap": 99}})
    assert s["guide"] == "plain" and s["combine"] == "band"                     # one band (2026-10-24): unknown -> band
    assert s["max_side"] == 1920 and s["max_frames"] == 145 and (s["max_frames"] - 1) % 8 == 0
    assert s["overlap"] <= s["max_frames"] // 2
    assert L.ltx_settings({"ltx": {"max_frames": 30}})["max_frames"] == 25        # down to 8k+1
    assert L.settings_sig(L.ltx_settings()) != L.settings_sig(L.ltx_settings({"ltx": {"guide": "isolate"}}))


@pytest.mark.parametrize("n,mx,ov", [(1, 49, 8), (9, 49, 8), (49, 49, 8), (50, 49, 8), (100, 49, 8), (394, 49, 8),
                                     (193, 25, 8), (60, 9, 4)])
def test_batches_cover_overlap_and_8k1(n, mx, ov):
    bl = L.batches(n, mx, ov)
    assert bl[0][0] == 0 and bl[-1][1] == n - 1
    for a, b in bl:
        assert b - a + 1 <= mx
        assert (L.batch_length(b - a + 1) - 1) % 8 == 0 and L.batch_length(b - a + 1) >= b - a + 1
    for (a0, b0), (a1, b1) in zip(bl, bl[1:]):
        assert b0 - a1 + 1 == ov                                        # consecutive batches overlap by `ov`
    lens = [b - a + 1 for a, b in bl]
    assert max(lens) - min(lens) <= 2                                   # even split, no short tail batch


def test_work_size_scales_down_only_and_pads_to_32():
    assert L.work_size(800, 800, 1024) == ((800, 800), (800, 800))
    assert L.work_size(1280, 720, 1280) == ((1280, 720), (1280, 736))   # padded, not stretched (model card)
    (sw, sh), (pw, ph) = L.work_size(4096, 2731, 1920)
    assert sw <= 1920 and sh <= 1088 and pw % 32 == 0 and ph % 32 == 0 and pw - sw < 32 and ph - sh < 32
    assert L.work_size(300, 200, 1024) == ((300, 200), (320, 224))     # never scaled up


def test_graph_recipe():
    g = L.alphagen_graph("C:/x/g_####.exr", 41, 800, 736, "matte_desk_ltx/a", seed=7)
    by = {v["class_type"]: v["inputs"] for v in g.values()}
    assert by["PlateLoadEXR"]["last_frame"] == 40
    assert by["EmptyLTXVLatentVideo"] == {"width": 800, "height": 736, "length": 41, "batch_size": 1}
    assert by["LoraLoaderModelOnly"]["lora_name"].startswith("ltx-2.5-22b-ic-lora-alpha-gen") and by["LoraLoaderModelOnly"]["strength_model"] == 1.0
    assert by["CLIPLoader"]["type"] == "ltxv" and by["CFGGuider"]["cfg"] == 1.0
    texts = sorted(v["inputs"]["text"] for v in g.values() if v["class_type"] == "CLIPTextEncode")
    assert texts == ["", " "]                                            # empty prompt, " " negative (tokenizer)
    assert by["KSamplerSelect"]["sampler_name"] == "euler_ancestral" and by["ManualSigmas"]["sigmas"] == L.SIGMAS
    assert by["RandomNoise"]["noise_seed"] == 7 and by["LTXVAddGuide"]["strength"] == 1.0 and by["LTXVAddGuide"]["frame_idx"] == 0
    assert by["SaveImageAdvanced"]["format.bit_depth"] == "16-bit"
    for nid, n in g.items():                                             # every link points at an existing node
        for v in n["inputs"].values():
            if isinstance(v, list):
                assert v[0] in g, (nid, v)
    assert set(L.LTX_NODES) <= {n["class_type"] for n in g.values()} | {"PlateLoadEXR"}


def test_decode_matte_16bit_and_rgb():
    m = (np.linspace(0, 1, 64 * 32).reshape(32, 64) * 65535).astype(np.uint16)
    ok, buf = cv2.imencode(".png", np.dstack([m, m, m]))
    out = L.decode_matte(buf.tobytes())
    assert out.shape == (32, 64) and abs(out.max() - 1.0) < 1e-4 and np.unique(out).size > 1000   # 16-bit kept


def test_isolate_guide_paints_others_not_target():
    rng = np.random.default_rng(0)
    img = np.full((240, 320, 3), 0.3, np.float32) + rng.normal(0, 0.01, (240, 320, 3)).astype(np.float32)
    sel = np.zeros((240, 320), np.float32)
    sel[60:180, 40:120] = 1
    img[sel > 0] = (0.9, 0.2, 0.2)                                       # target: red
    oth = np.zeros_like(sel)
    oth[80:160, 200:280] = 1
    img[oth > 0] = (0.1, 0.9, 0.1)                                       # another object: green
    plain = L.isolate_guide(img, sel, None)
    assert np.allclose(plain, img)                                       # nothing to paint -> unchanged
    g = L.isolate_guide(img, sel, oth)
    assert np.allclose(g[sel > 0], img[sel > 0], atol=1e-6)              # target untouched
    inner = g[100:140, 220:260]
    assert abs(inner[..., 1].mean() - 0.3) < 0.05                        # the other object filled with the background
    assert inner[..., 0].mean() < 0.4                                    # ... never with the target's red
    st = L.isolate_guide(img, sel, oth, steer=True)
    assert np.allclose(st[sel > 0][:10], img[sel > 0][:10], atol=0.05)
    far = st[5:30, 290:315]
    assert far.std() < img[5:30, 290:315].std()                         # outside defocused


def test_finish_band_and_raw_and_legacy_modes():
    """2026-10-24: the result modes are „pas” (the one band of every stage) and „surowy” (no band, the model decides);
    the former „region” / „+ wnętrze” (max_core) read as „pas”."""
    core = np.zeros((100, 100), np.float32)
    core[30:70, 30:70] = 1
    a = np.zeros_like(core)
    a[30:70, 30:70] = 0.6                                                # a "hole" inside (generative matte too soft)
    a[0:10, 0:10] = 1.0                                                  # another object far away
    a[30:70, 71:80] = 0.5                                                # soft edge just outside (hair)
    for legacy in ("region", "max_core"):
        assert L.ltx_settings({"ltx": {"combine": legacy}})["combine"] == "band"
    s = L.ltx_settings({"ltx": {"region_px": 12, "core_erode_px": 4, "region_soft_px": 0, "combine": "band"}})
    r = L.finish(a, core, s)
    assert r[5, 5] == 0 and abs(r[50, 75] - 0.5) < 1e-6 and r[50, 50] == 1.0         # far away 0, band = LTX, core = 1
    assert np.array_equal(L.finish(a, core, {**s, "combine": "raw"}), a)


# ----------------------------------------------------------------------------- AlphaGen with a mock Comfy
def _alphagen_handler(out_root: Path, record: list):
    """Mock of the whole Alpha-Gen graph: matte = red channel of each guide frame (16-bit PNG, SaveImageAdvanced)."""
    from engine.depth import frame_path

    def h(graph):
        ld = next(v["inputs"] for v in graph.values() if v["class_type"] == "PlateLoadEXR")
        lat = next(v["inputs"] for v in graph.values() if v["class_type"] == "EmptyLTXVLatentVideo")
        sv_id, sv = next((k, v["inputs"]) for k, v in graph.items() if v["class_type"] == "SaveImageAdvanced")
        n = ld["last_frame"] + 1
        assert n == lat["length"] and (n - 1) % 8 == 0
        record.append({"n": n, "w": lat["width"], "h": lat["height"]})
        sub, stem = sv["filename_prefix"].rsplit("/", 1)
        (out_root / sub).mkdir(parents=True, exist_ok=True)
        images = []
        for i in range(n):
            g = cv2.imread(str(frame_path(ld["path_pattern"], i)), cv2.IMREAD_UNCHANGED)
            assert g.shape[1] == lat["width"] and g.shape[0] == lat["height"]
            m = np.clip(g[..., 2], 0, 1)
            name = f"{stem}_{i + 1:05d}_.png"
            cv2.imwrite(str(out_root / sub / name), (m * 65535).astype(np.uint16))
            images.append({"filename": name, "subfolder": sub, "type": "output"})
        return {sv_id: {"images": images}}
    return h


def test_alphagen_run_batches_crossfade_padding(tmp_path):
    from mock_comfy import MockComfyClient
    rec = []
    cl = MockComfyClient(handlers={"SaveImageAdvanced": _alphagen_handler(tmp_path / "out", rec)})
    cl.output_root = tmp_path / "out"
    s = L.ltx_settings({"ltx": {"max_frames": 17, "overlap": 4, "max_side": 1920}})
    H, W = 90, 150                                                        # not multiples of 32 -> padded
    guides = []
    for i in range(40):
        g = np.zeros((H, W, 3), np.float32)
        g[..., 2] = 0.25 + 0.5 * i / 39                                   # red ramps over time (matte follows it)
        g[20:60, 30:90, 2] = 1.0
        guides.append(g)
    gen = L.AlphaGen(cl, tmp_path / "io", s)
    out = gen.run("t", guides, (W, H))
    assert len(out) == 40 and out[0].shape == (H, W)
    assert all(r["w"] == 160 and r["h"] == 96 for r in rec)             # padded to %32, not stretched
    assert len(rec) == len(L.batches(40, 17, 4)) and all((r["n"] - 1) % 8 == 0 for r in rec)
    for i in range(40):                                                   # padding cropped, crossfade continuous
        assert abs(out[i][5, 5] - (0.25 + 0.5 * i / 39)) < 2e-3
        assert abs(out[i][40, 60] - 1.0) < 2e-3
    assert [b["frames"] for b in gen.log] == [list(b) for b in L.batches(40, 17, 4)]


def _links_ok(g):
    for nid, n in g.items():
        for v in n["inputs"].values():
            if isinstance(v, list):
                assert v[0] in g, (nid, v)


def test_graph_empty_prompt_file_load_and_save():
    """2026-11-06: the empty prompt encoded once - load = no CLIPLoader / CLIPTextEncode (no Gemma), the conditioning
    node feeds LTXVConditioning; save = the Gemma graph unchanged plus the save node on both encodes."""
    base = L.alphagen_graph("C:/x/g_####.exr", 41, 800, 736, "p")
    ld = L.alphagen_graph("C:/x/g_####.exr", 41, 800, 736, "p", cond_load="C:/m/ltx_conditioning/te.safetensors")
    kinds = {v["class_type"] for v in ld.values()}
    assert not kinds & {"CLIPLoader", "CLIPTextEncode", "ConditioningPairSave"}
    nid = next(k for k, v in ld.items() if v["class_type"] == "ConditioningPairLoad")
    cond = next(v["inputs"] for v in ld.values() if v["class_type"] == "LTXVConditioning")
    assert cond["positive"] == [nid, 0] and cond["negative"] == [nid, 1] and cond["frame_rate"] == 24.0
    _links_ok(ld)
    assert {k: v for k, v in ld.items() if k not in ("32", "35", "36", "37", nid)} == \
           {k: v for k, v in base.items() if k not in ("32", "35", "36", "37")}   # the rest of the recipe untouched
    sv = L.alphagen_graph("C:/x/g_####.exr", 41, 800, 736, "p", cond_save="C:/m/c.safetensors")
    save = next(v["inputs"] for v in sv.values() if v["class_type"] == "ConditioningPairSave")
    by = {v["class_type"]: k for k, v in sv.items() if v["class_type"] == "CLIPTextEncode" and v["inputs"]["text"] == ""}
    assert save["positive"] == [by["CLIPTextEncode"], 0] and save["path"] == "C:/m/c.safetensors"
    assert {k: v for k, v in sv.items() if v["class_type"] != "ConditioningPairSave"} == base
    _links_ok(sv)
    assert L.cond_file(None) is None
    assert L.cond_file(Path("C:/m")).name == Path(L.LTX_MODELS["text_encoder"][1]).stem + ".empty_prompt.safetensors"
    assert L.cond_file(Path("C:/m"), "other.safetensors") != L.cond_file(Path("C:/m"))   # one file per encoder


def test_alphagen_encodes_once_then_loads(tmp_path):
    from mock_comfy import MockComfyClient
    rec, saved = [], []
    cond = tmp_path / "models" / L.COND_DIR / "te.empty_prompt.safetensors"

    def save(graph):
        p = next(v["inputs"]["path"] for v in graph.values() if v["class_type"] == "ConditioningPairSave")
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_bytes(b"x")
        saved.append(p)
    cl = MockComfyClient(handlers={"SaveImageAdvanced": _alphagen_handler(tmp_path / "out", rec), "ConditioningPairSave": save})
    cl.output_root = tmp_path / "out"
    s = L.ltx_settings({"ltx": {"max_frames": 17, "overlap": 4}})
    g = [np.full((64, 64, 3), 0.5, np.float32)] * 40
    gen = L.AlphaGen(cl, tmp_path / "io", s, cond_path=cond)
    gen.run("t", g, (64, 64))
    assert [b["prompt"] for b in gen.log] == ["gemma+save"] + ["file"] * (len(gen.log) - 1)
    assert saved == [cond.resolve().as_posix()]
    kinds = [{v["class_type"] for v in gr.values()} for gr in cl.sent]
    assert "CLIPLoader" in kinds[0] and all("CLIPLoader" not in k and "ConditioningPairLoad" in k for k in kinds[1:])
    gen0 = L.AlphaGen(cl, tmp_path / "io", s)                                    # no file given = Gemma every batch
    gen0.run("u", g[:9], (64, 64))
    assert [b["prompt"] for b in gen0.log] == ["gemma"]


def test_conditioning_pair_node_roundtrip(tmp_path):
    """The PlateMatte node file: tensors back bit-identical (dtype too), plain / None entries kept, no pickle."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pm_cond", ROOT / "handoff" / "comfy" / "custom_nodes" / "ComfyUI-PlateMatte" / "conditioning.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    g = torch.Generator().manual_seed(0)
    pos = [[torch.randn(1, 7, 64, generator=g), {"pooled_output": None, "unprocessed_ltxav_embeds": True,
                                                  "attention_mask": torch.ones(1, 7, dtype=torch.int64)}]]
    neg = [[torch.randn(1, 3, 64, generator=g).to(torch.bfloat16), {"pooled_output": torch.randn(1, 8, generator=g)}]]
    p = str(tmp_path / "sub" / "c.safetensors")
    mod.ConditioningPairSave().save(pos, neg, p)
    lp, ln = mod.ConditioningPairLoad().load(p)
    for a, b in ((pos, lp), (neg, ln)):
        assert len(a) == len(b)
        for (ta, ea), (tb, eb) in zip(a, b):
            assert ta.dtype == tb.dtype and torch.equal(ta, tb)
            assert set(ea) == set(eb)
            for k, v in ea.items():
                assert (torch.equal(v, eb[k]) and v.dtype == eb[k].dtype) if isinstance(v, torch.Tensor) else v == eb[k]
    with pytest.raises(ValueError, match="cannot be saved"):
        mod.save_pair(p, [[torch.zeros(1), {"x": object()}]], neg)


def test_alphagen_busy_and_error(tmp_path):
    from mock_comfy import MockComfyClient
    g = [np.zeros((64, 64, 3), np.float32)] * 9
    with pytest.raises(RuntimeError, match="zajęte"):
        L.AlphaGen(MockComfyClient(busy=True), tmp_path, L.ltx_settings()).run("t", g, (64, 64))

    def boom(graph):
        raise RuntimeError("CUDA out of memory")
    with pytest.raises(RuntimeError, match="out of memory"):
        L.AlphaGen(MockComfyClient(handlers={"SaveImageAdvanced": boom}), tmp_path, L.ltx_settings()).run("t", g, (64, 64))


def test_models_status_and_installer_urls(tmp_path, monkeypatch):
    from engine import installer
    st = L.models_status(tmp_path)
    assert len(st) == 5 and not any(m["present"] for m in st)
    got = []

    def fake(url, dst):
        got.append(url)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "wb") as f:
            f.truncate(1)                                                  # sparse: size check needs ~gb
    res = installer.install_ltx({"models_dir": str(tmp_path)}, download=fake)
    assert res["success"] and len(got) == 5
    assert "https://huggingface.co/Lightricks/LTX-2.5/resolve/main/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors" in got
    assert "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen/resolve/main/ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors" in got
    assert got[0].endswith("alpha-gen-0.9.safetensors")                   # small files first
    monkeypatch.setattr(installer, "hf_token", lambda: None)
    with pytest.raises(RuntimeError, match="Hugging Face"):
        installer.install_ltx({"models_dir": str(tmp_path / "x")})
    assert L.missing_nodes({n: {} for n in L.LTX_NODES}) == [] and "LTXVAddGuide" in L.missing_nodes({})


def test_gemma_not_needed_once_the_empty_prompt_is_saved(tmp_path):
    """2026-11-06 (CLIP2_A: file = Gemma bit for bit): with the conditioning file the stage, the gate and the installer
    no longer ask for the 15.4 GB text encoder."""
    from engine import installer, stage_models
    md = tmp_path / "models"
    assert {m["key"] for m in L.missing_models(md)} == set(L.LTX_MODELS)
    assert any("gemma" in f for f in stage_models.CHECKS["ltx"]({"models_dir": str(md)}))
    cf = L.cond_file(md)
    cf.parent.mkdir(parents=True)
    cf.write_bytes(b"x")
    st = {m["key"]: m for m in L.models_status(md)}
    assert st["text_encoder"]["needed"] is False and all(m["needed"] for k, m in st.items() if k != "text_encoder")
    assert "text_encoder" not in {m["key"] for m in L.missing_models(md)}
    assert not any("gemma" in f for f in stage_models.CHECKS["ltx"]({"models_dir": str(md)}))
    got = []

    def fake(url, dst):
        got.append(url)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "wb") as f:
            f.truncate(1)
    res = installer.install_ltx({"models_dir": str(md)}, download=fake)
    assert len(got) == 4 and not any("gemma" in u for u in got)
    assert any("niepotrzebne" in line for line in res["log"])


# ----------------------------------------------------------------------------- the stage through the API
@pytest.fixture
def proj(tmp_path, monkeypatch):
    from engine.pipeline import ObjectPipeline
    from test_zone_matte import _api_project
    monkeypatch.setattr(ObjectPipeline, "_zone_tracker", lambda self: None)
    return _api_project(tmp_path, monkeypatch)


def test_stage_ltx_api_export_and_edge_modes(proj, tmp_path, monkeypatch):
    app_module, api, shot, pipe, frames, start = proj
    rec = []
    pipe.client.handlers["SaveImageAdvanced"] = _alphagen_handler(tmp_path / "ltx_out", rec)
    pipe.client.output_root = tmp_path / "ltx_out"
    d = api.get("/api/object/obj/stages").json()
    assert [s["n"] for s in d["steps"]] == [1, 2, 3, 4, 5, 6, 7, 8, 9]    # 8 = Final Touch (2026-10-22)
    assert [s["id"] for s in d["steps"]][-3:] == ["ltx", "final_touch", "export"]
    st = {s["id"]: s["state"] for s in d["stages"]}
    assert st["ltx"] == "todo" and "opcjonalny" in next(s for s in d["stages"] if s["id"] == "ltx")["why"]
    # LTX is no longer an edge mode of the experiments
    assert "ltx" not in [m["mode"] for m in api.get("/api/object/obj/edge_modes").json()["modes"]]
    assert api.put("/api/object/obj/edge_mode", json={"mode": "ltx"}).status_code == 400
    # models missing -> 409 with the sizes
    monkeypatch.setattr(app_module, "_ltx_status", lambda: {"missing_models": ["a.safetensors"], "missing_gb": 21.5,
                                                            "missing_nodes": [], "models": [], "ready": False})
    r = api.post("/api/object/obj/ltx")
    assert r.status_code == 409 and "21.5 GB" in r.json()["detail"]
    monkeypatch.setattr(app_module, "_ltx_status", lambda: {"missing_models": [], "missing_gb": 0, "missing_nodes": [],
                                                            "models": [], "ready": True})
    assert api.put("/api/object/obj/ltx", json={"guide": "plain", "max_frames": 9, "overlap": 2}).status_code == 200
    info = api.get("/api/object/obj/ltx").json()
    assert info["settings"]["guide"] == "plain" and info["settings"]["max_frames"] == 9 and info["run"] is None
    r = api.post("/api/object/obj/ltx")
    assert r.status_code == 200, r.text
    run = r.json()["run"]
    assert rec and all((x["n"] - 1) % 8 == 0 for x in rec)
    rj = json.loads((pipe.store.dir("obj") / "runs" / run / "run.json").read_text(encoding="utf-8"))
    assert rj["stage"] == "ltx" and rj["ltx"]["guide"] == "plain" and rj["ltx_batches"]
    st = {s["id"]: s["state"] for s in api.get("/api/object/obj/stages").json()["stages"]}
    assert st["ltx"] == "done"
    # the matte is limited to the object: nothing far from the mask 1+2
    from engine.export import run_alpha_loader
    a = run_alpha_loader(pipe.store.dir("obj"), run, shot.w, shot.h)(start)
    core = pipe.load_core("obj", [start])[start]
    far = L.sdf_px(core.astype(np.float32)) < -40
    assert a[core].mean() > 0.5 and a[far].max() == 0
    # export from the LTX stage
    r = api.post("/api/export", json={"objects": ["obj"], "source": "ltx"})
    assert r.status_code == 200, r.text
    # a settings change makes the result stale; the export source then refuses it
    assert api.put("/api/object/obj/ltx", json={"region_px": 30}).status_code == 200
    stg = next(s for s in api.get("/api/object/obj/stages").json()["stages"] if s["id"] == "ltx")
    assert stg["state"] == "stale" and "ustawienia LTX" in stg["why"]
    assert api.post("/api/export", json={"objects": ["obj"], "source": "ltx"}).status_code == 409
    # LTX palette (2026-10-16): live band + the raw matte re-limited with unsaved values; „suwaki na wynik” = a new
    # version from the raw mattes without a model run, current again
    nrec = len(rec)
    rb = api.get(f"/api/object/obj/ltx_preview/{start}?what=band&region_px=10")
    rb2 = api.get(f"/api/object/obj/ltx_preview/{start}?what=band&region_px=60")
    assert rb.status_code == 200 and rb2.status_code == 200 and rb.content != rb2.content
    assert api.get(f"/api/object/obj/ltx_preview/{start}?what=alpha&region_px=5").status_code == 200
    r = api.post("/api/object/obj/ltx/refinish")
    assert r.status_code == 200, r.text
    run2 = r.json()["run"]
    assert len(rec) == nrec and run2 != run                              # no model call
    rj2 = json.loads((pipe.store.dir("obj") / "runs" / run2 / "run.json").read_text(encoding="utf-8"))
    assert rj2["refinished_from"] == run and rj2["ltx"]["region_px"] == 30
    stg = next(s for s in api.get("/api/object/obj/stages").json()["stages"] if s["id"] == "ltx")
    assert stg["state"] == "done"
    a2 = run_alpha_loader(pipe.store.dir("obj"), run2, shot.w, shot.h)(start)
    assert a2[L.sdf_px(core.astype(np.float32)) < -40].max() == 0 and a2[core].mean() > 0.5
    # mask source: a stage result that does not exist -> a clear refusal
    assert api.put("/api/object/obj/ltx", json={"mask_source": "vitmatte"}).status_code == 200
    r = api.get(f"/api/object/obj/ltx_preview/{start}?what=band")
    assert r.status_code == 404 and "ViTMatte" in r.json()["detail"]
    assert L.ltx_settings({"ltx": {"mask_source": "zzz"}})["mask_source"] == "core"
