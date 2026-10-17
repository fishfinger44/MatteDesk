# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage „VideoMaMa” (user decision 2026-10-16: between stage 3 MatAnyone and ViTMatte, after the A/B test
local_results/videomama_m2). Synthetic: settings, the composition over the base (reach / protect), batches + crossfade,
the stage / ViTMatte base / export wiring through the API with a mock Comfy whose "VideoMaMa" returns the guide mask
grown with a soft halo. The real model run = LOCAL_CHECKS."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from engine import videomama as V   # noqa: E402


# ----------------------------------------------------------------------------- pure helpers
def test_settings_validation_and_protect_auto():
    s = V.vmm_settings()
    assert s["base"] == "core" and s["protect"] == 0.0 and s["reach_px"] == 40.0 and s["batch"] == 24 and s["overlap"] == 8
    s = V.vmm_settings({"videomama": {"base": "matanyone"}})
    assert s["protect"] == 3.0 and s["protect_px"] is None                     # auto by base
    s = V.vmm_settings({"videomama": {"base": "x", "protect_px": 5, "max_side": 1283, "batch": 2, "overlap": 99,
                                      "offload": "?", "reach_px": -4}})
    assert s["base"] == "core" and s["protect"] == 5.0 and s["max_side"] == 1280 and s["batch"] == 4
    assert s["overlap"] == 2 and s["offload"] == "auto" and s["reach_px"] == 0.0
    a, b = V.vmm_settings(), V.vmm_settings({"videomama": {"reach_px": 20}})
    assert V.settings_sig(a) != V.settings_sig(b) and V.model_sig(a) == V.model_sig(b)   # slider = no model run
    assert V.model_sig(a) != V.model_sig(V.vmm_settings({"videomama": {"overlap": 4}}))


def _disc(h=120, w=120, r=30):
    m = np.zeros((h, w), np.float32)
    cv2.circle(m, (w // 2, h // 2), r, 1.0, -1)
    return m


def test_compose_reach_and_protect():
    base = _disc()
    d = V.sdf_px(base)
    vmm = np.full(base.shape, 0.4, np.float32)                 # a uniform "strand" everywhere
    a = V.compose(base, vmm, protect=0.0, reach=10.0)
    assert np.allclose(a[d > 0.5], 1.0)                        # inside the base untouched
    assert np.allclose(a[(d < -1) & (d >= -10)], 0.4)           # within reach: max(base, vmm)
    assert np.allclose(a[d < -10.5], 0.0)                      # beyond reach: base (strays cut off)
    a3 = V.compose(base, vmm, protect=3.0, reach=10.0)
    ring = (d < -0.5) & (d > -2.5)
    assert (a3[ring] < 0.4).all() and (a3[ring] >= 0).all()    # protect band: ramp from the base
    assert np.allclose(a3[(d < -3.5) & (d >= -10)], 0.4)
    band = V.band_u8(base, V.vmm_settings({"videomama": {"protect_px": 3, "reach_px": 10}}))
    assert (band[(d < -4) & (d > -9)] == 128).all() and (band[np.abs(d) < 2] == 255).all() and (band[d < -12] == 0).all()


def test_compose_unified_inset_and_feather():
    """User decision 2026-10-17: the same sliders as MatAnyone / ViTMatte – VideoMaMa replaces the base from reach px
    outside to inset px inside its edge, blended into the deeper base over feather px; inset None = the former result."""
    base = _disc()
    d = V.sdf_px(base)
    vmm = np.full(base.shape, 0.4, np.float32)
    a = V.compose(base, vmm, protect=0.0, reach=10.0, inset=6.0, feather=0.0)
    assert np.allclose(a[(d > 0.5) & (d < 5.5)], 0.4)          # inside the base, within the inset: VideoMaMa replaces it
    assert np.allclose(a[d > 6.5], 1.0) and np.allclose(a[d < -10.5], 0.0)
    af = V.compose(base, vmm, protect=0.0, reach=10.0, inset=6.0, feather=4.0)
    near = (d > 4.5) & (d < 5.5)                                # ~1 px from the deep base: mostly the base
    assert (af[near] > 0.7).all() and np.allclose(af[(d < -1) & (d >= -10)], 0.4)
    # inset 0 on the mask 1+2 base = the former composition exactly
    assert np.allclose(V.compose(base, vmm, 0.0, 10.0, inset=0.0), V.compose(base, vmm, 0.0, 10.0))
    s_old = V.vmm_settings({"videomama": {"reach_px": 10}})
    s_new = V.vmm_settings({"videomama": {"reach_px": 10, "inset_px": 0}})
    assert s_old["inset_px"] is None                            # never touched: the former signature (results stay current)
    assert V.settings_sig(s_old) != V.settings_sig(s_new)       # the unified sliders are part of the result
    band = V.band_u8(base, V.vmm_settings({"videomama": {"reach_px": 10, "inset_px": 6}}))
    assert (band[(d > 1) & (d < 5)] == 128).all() and (band[d > 7] == 255).all() and (band[d < -12] == 0).all()


def test_offload_auto_by_vram():
    s = V.vmm_settings()
    assert V.offload_on(s, {"devices": [{"vram_total": 12e9}]}) and not V.offload_on(s, {"devices": [{"vram_total": 25e9}]})
    assert not V.offload_on(s, None)
    assert V.offload_on(V.vmm_settings({"videomama": {"offload": "on"}}), {"devices": [{"vram_total": 25e9}]})


def test_graph_recipe():
    g = V.vmm_graph("C:/io/g_####.exr", "C:/io/m_####.exr", 24, 1280, "p/x", "C:/models/videomama", seed=7, offload=True)
    ld = g["4"]["inputs"]
    assert ld["base_model_path"].endswith("/videomama/stable-video-diffusion-img2vid-xt")
    assert ld["unet_checkpoint_path"].endswith("/videomama/VideoMaMa") and ld["enable_model_cpu_offload"] is True
    smp = g["5"]["inputs"]
    assert smp["max_resolution"] == 1280 and smp["seed"] == 7 and smp["masks"] == ["3", 0]
    assert g["1"]["inputs"]["last_frame"] == 23 and g["3"]["class_type"] == "ImageToMask"
    assert g["7"]["inputs"]["format.bit_depth"] == "16-bit"
    assert V.missing_nodes({n: {} for n in V.VMM_NODES}) == [] and "VideoMaMaSampler" in V.missing_nodes({})
    assert V.work_side(800, 800, 1280) == 800 and V.work_side(2000, 1000, 1280) == 1280 and V.work_side(803, 10, 2048) == 800


def test_models_status(tmp_path):
    st = V.models_status(tmp_path)
    assert len(st) == len(V.VMM_FILES) and not any(m["present"] for m in st)
    for sub, _repo, fn, gb in V.VMM_FILES:
        p = tmp_path / "videomama" / sub / fn
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    st = {m["file"]: m["present"] for m in V.models_status(tmp_path)}
    assert st["unet/config.json"] and not st["unet/diffusion_pytorch_model.safetensors"]   # too small = missing


# ----------------------------------------------------------------------------- mock Comfy
def _vmm_handler(out_root: Path, record: list, batch_value: bool = False):
    """Mock of the VideoMaMa graph: matte = guide mask + a 0.5 halo 4 px around it (16-bit PNG); with `batch_value`
    every pixel = the batch number (crossfade check)."""
    from engine.depth import frame_path

    def h(graph):
        img = graph["1"]["inputs"]
        msk = graph["2"]["inputs"]
        sv = graph["7"]["inputs"]
        assert graph["5"]["class_type"] == "VideoMaMaSampler" and graph["4"]["class_type"] == "VideoMaMaPipelineLoader"
        n = img["last_frame"] + 1
        assert msk["last_frame"] + 1 == n
        record.append({"n": n, "res": graph["5"]["inputs"]["max_resolution"]})
        sub, stem = sv["filename_prefix"].rsplit("/", 1)
        (out_root / sub).mkdir(parents=True, exist_ok=True)
        images = []
        for i in range(n):
            m = cv2.imread(str(frame_path(msk["path_pattern"], i)), cv2.IMREAD_UNCHANGED)[..., 2]
            g = cv2.imread(str(frame_path(img["path_pattern"], i)), cv2.IMREAD_UNCHANGED)
            assert g.shape[:2] == m.shape[:2] and g.max() <= 1.0
            if batch_value:
                a = np.full(m.shape, (len(record) - 1) / 4.0, np.float32)
            else:
                halo = cv2.dilate((m > 0.5).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(np.float32)
                a = np.maximum(m, 0.5 * halo)
            name = f"{stem}_{i + 1:05d}_.png"
            cv2.imwrite(str(out_root / sub / name), (np.clip(a, 0, 1) * 65535).astype(np.uint16))
            images.append({"filename": name, "subfolder": sub, "type": "output"})
        return {"7": {"images": images}}
    return h


def test_run_batches_crossfade(tmp_path):
    from mock_comfy import MockComfyClient
    rec = []
    cl = MockComfyClient(handlers={"SaveImageAdvanced": _vmm_handler(tmp_path / "out", rec, batch_value=True)})
    cl.output_root = tmp_path / "out"
    s = V.vmm_settings({"videomama": {"batch": 8, "overlap": 4}})
    gen = V.VideoMaMa(cl, tmp_path / "io", s, tmp_path / "models")
    imgs = [np.full((40, 40, 3), 0.5, np.float32) for _ in range(20)]
    masks = [_disc(40, 40, 10) for _ in range(20)]
    out = gen.run("t", imgs, masks)
    assert len(out) == 20 and out[0].shape == (40, 40)
    assert [r["n"] for r in rec] == [8, 8, 8, 8] and len(gen.log) == 4
    v = [float(o.mean()) for o in out]
    assert v[0] == pytest.approx(0.0, abs=1e-4) and v[-1] == pytest.approx(0.75, abs=1e-4)   # 16-bit PNG
    assert all(b >= a - 1e-6 for a, b in zip(v, v[1:]))           # smooth: no jump back, crossfaded ramps
    assert any(0.0 < x < 0.25 for x in v)                          # a blended frame between batch 0 and 1
    assert not list((tmp_path / "io").glob("vmm_*"))               # Comfy inputs removed after each batch


# ----------------------------------------------------------------------------- the stage through the API
@pytest.fixture
def proj(tmp_path, monkeypatch):
    from engine.pipeline import ObjectPipeline
    from test_zone_matte import _api_project
    monkeypatch.setattr(ObjectPipeline, "_zone_tracker", lambda self: None)
    return _api_project(tmp_path, monkeypatch)


def test_stage_videomama_api_base_export(proj, tmp_path, monkeypatch):
    app_module, api, shot, pipe, frames, start = proj
    rec = []
    pipe.client.handlers["SaveImageAdvanced"] = _vmm_handler(tmp_path / "vmm_out", rec)
    pipe.client.output_root = tmp_path / "vmm_out"
    d = api.get("/api/object/obj/stages").json()
    assert [s["id"] for s in d["steps"]][2:5] == ["matanyone", "videomama", "vitmatte"]
    stg = next(s for s in d["stages"] if s["id"] == "videomama")
    assert stg["state"] == "todo" and "opcjonalny" in stg["why"]
    # weights missing -> 409 with the size
    monkeypatch.setattr(app_module, "_vmm_status", lambda: {"missing_models": ["VideoMaMa/unet/x"], "missing_gb": 6.1,
                                                            "missing_nodes": [], "models": [], "ready": False})
    r = api.post("/api/object/obj/videomama")
    assert r.status_code == 409 and "6.1 GB" in r.json()["detail"]
    monkeypatch.setattr(app_module, "_vmm_status", lambda: {"missing_models": [], "missing_gb": 0, "missing_nodes": [],
                                                            "models": [], "ready": True})
    # base MatAnyone without a stage-3 result: the stage stays open (its base selector is inside, bug 2026-10-17),
    # says why and refuses to compute
    assert api.put("/api/object/obj/videomama", json={"base": "matanyone"}).status_code == 200
    stg = next(s for s in api.get("/api/object/obj/stages").json()["stages"] if s["id"] == "videomama")
    assert stg["state"] == "todo" and "MatAnyone" in stg["why"]
    r = api.post("/api/object/obj/videomama")
    assert r.status_code == 409 and "MatAnyone" in r.json()["detail"]
    assert api.put("/api/object/obj/videomama", json={"base": "core", "batch": 4, "overlap": 2}).status_code == 200
    info = api.get("/api/object/obj/videomama").json()
    assert info["settings"]["base"] == "core" and info["settings"]["batch"] == 4 and info["run"] is None
    r = api.post("/api/object/obj/videomama")
    assert r.status_code == 200, r.text
    run = r.json()["run"]
    assert rec and all(x["n"] <= 4 for x in rec)
    od = pipe.store.dir("obj")
    rj = json.loads((od / "runs" / run / "run.json").read_text(encoding="utf-8"))
    assert rj["stage"] == "videomama" and rj["base"] == "core" and rj["vmm_batches"] and (od / "runs" / run / "vmm_raw").exists()
    st = {s["id"]: s["state"] for s in api.get("/api/object/obj/stages").json()["stages"]}
    assert st["videomama"] == "done"
    # the result = mask 1+2 + the halo within reach; nothing beyond reach
    from engine.export import run_alpha_loader
    a = run_alpha_loader(od, run, shot.w, shot.h)(start)
    core = pipe.load_core("obj", [start])[start]
    d_ = V.sdf_px(core.astype(np.float32))
    assert a[core].min() > 0.99 and a[(d_ < -1.5) & (d_ > -3.5)].mean() == pytest.approx(0.5, abs=0.05)
    assert a[d_ < -41].max() == 0
    # ViTMatte can take it as its base; export source „VideoMaMa”
    assert api.put("/api/object/obj/settings", json={"vitmatte_base": "videomama"}).status_code == 200
    stv = next(s for s in api.get("/api/object/obj/stages").json()["stages"] if s["id"] == "vitmatte")
    assert stv["state"] != "locked"
    r = api.post("/api/export", json={"objects": ["obj"], "source": "videomama"})
    assert r.status_code == 200, r.text
    # palette: live band / recomposed result; a slider change = stale, „suwaki na wynik” = new version, no model run
    rb = api.get(f"/api/object/obj/videomama_preview/{start}?what=band&reach_px=5")
    rb2 = api.get(f"/api/object/obj/videomama_preview/{start}?what=band&reach_px=60")
    assert rb.status_code == 200 and rb2.status_code == 200 and rb.content != rb2.content
    assert api.get(f"/api/object/obj/videomama_preview/{start}?what=alpha&reach_px=2").status_code == 200
    nrec = len(rec)
    assert api.put("/api/object/obj/videomama", json={"reach_px": 2}).status_code == 200
    stg = next(s for s in api.get("/api/object/obj/stages").json()["stages"] if s["id"] == "videomama")
    assert stg["state"] == "stale" and "ustawienia VideoMaMa" in stg["why"]
    assert api.post("/api/export", json={"objects": ["obj"], "source": "videomama"}).status_code == 409
    r = api.post("/api/object/obj/videomama/refinish")
    assert r.status_code == 200, r.text
    assert len(rec) == nrec                                          # no model run
    a2 = run_alpha_loader(od, r.json()["run"], shot.w, shot.h)(start)
    assert a2[d_ < -2.5].max() == 0                                 # reach 2 px: the halo cut off
    st = {s["id"]: s["state"] for s in api.get("/api/object/obj/stages").json()["stages"]}
    assert st["videomama"] == "done"
    # a model setting change cannot be refinished
    assert api.put("/api/object/obj/videomama", json={"overlap": 0}).status_code == 200
    assert api.post("/api/object/obj/videomama/refinish").status_code == 409


def test_installer_videomama(tmp_path, monkeypatch):
    from engine import installer
    import engine.tools_install as ti
    # tiny stand-ins for the GB-sized files (the presence check compares sizes)
    monkeypatch.setattr(V, "VMM_FILES", [(s, r, f, 0.0) for s, r, f, _g in V.VMM_FILES])
    monkeypatch.setattr(ti, "find_tool", lambda name, cfg=None: "git")
    cfg = {"models_dir": str(tmp_path / "models"), "comfy_dir": str(tmp_path / "comfy")}
    (tmp_path / "comfy" / "custom_nodes").mkdir(parents=True)
    calls = []

    def run(cmd, **kw):   # git clone -> make the folder; pip -> ok
        calls.append(cmd)
        if "clone" in cmd:
            Path(cmd[-1]).mkdir(parents=True)

        class R:
            returncode, stdout, stderr = 0, "", ""
        return R()
    got = []

    def dl(url, dst):
        got.append(url)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"0123456789")
    res = installer.install_videomama(cfg, download=dl, run=run)
    assert res["success"] and res["restart_comfy"] and len(got) == len(V.VMM_FILES)
    assert all(m["present"] for m in V.models_status(Path(cfg["models_dir"])))
    assert any("ComfyUI-VideoMaMa" in str(c[-1]) for c in calls if "clone" in c)
