# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""MatAnyone2RunKeyframes 1.1.0 (2026-10-13): conditional VRAM release, max_mem_frames 0 = auto, refine graph sends
the new inputs only to a node that has them (/object_info); object option ma_scale_mode (auto | 1.0 | 0.75 | 0.5)."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "handoff" / "comfy" / "custom_nodes" / "ComfyUI-MatAnyone2-Keyframes"
GiB = 1024 ** 3


def _load_vram():
    spec = importlib.util.spec_from_file_location("ma_kf_vram", PKG / "vram.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vram = _load_vram()


class FakeCuda:
    """torch.cuda stand-in: reserved memory follows a script, empty_cache drops it to `allocated`."""

    def __init__(self, total=12 * GiB, free=10 * GiB, reserved=(), allocated=1 * GiB):
        self.total, self.free, self.allocated = total, free, allocated
        self.script = list(reserved)
        self.reserved = self.script[0] if self.script else allocated
        self.emptied = 0

    def mem_get_info(self):
        return self.free, self.total

    def memory_reserved(self):
        return self.reserved

    def memory_allocated(self):
        return self.allocated

    def empty_cache(self):
        self.emptied += 1
        self.reserved = self.allocated


def _torch(cuda):
    return types.SimpleNamespace(cuda=cuda)


# ------------------------------------------------------------------------------------------- conditional release
def test_release_only_above_fraction_of_card():
    cuda = FakeCuda(total=12 * GiB)
    rel = vram.VramReleaser(_torch(cuda), 0.85)
    assert rel.limit == pytest.approx(0.85 * 12 * GiB)
    for r in (3 * GiB, 6 * GiB, 10 * GiB, 10.1 * GiB):       # all <= 10.2 GiB
        cuda.reserved = r
        assert rel.step() is False
    assert cuda.emptied == 0
    cuda.reserved = 10.5 * GiB                               # > 85 % of 12 GiB
    assert rel.step() is True and cuda.emptied == 1 and cuda.reserved == cuda.allocated
    assert rel.peak_reserved == 10.5 * GiB and rel.releases == 1 and rel.steps == 5


def test_release_frac_zero_is_old_behaviour_every_frame():
    cuda = FakeCuda()
    rel = vram.VramReleaser(_torch(cuda), 0.0)
    for _ in range(7):
        cuda.reserved = 2 * GiB
        assert rel.step() is True
    assert cuda.emptied == 7


def test_release_disabled_without_cuda_touches_nothing():
    class Boom:
        def __getattr__(self, k):
            raise AssertionError("torch.cuda must not be touched on CPU")
    rel = vram.VramReleaser(_torch(Boom()), 0.85, enabled=False)
    assert rel.step() is False and rel.steps == 0


# ------------------------------------------------------------------------------------------- auto max_mem_frames
def test_auto_max_mem_frames_formula():
    # 12 GB card at 1440 px (even completely free) -> 5 = the official value used so far
    assert vram.auto_max_mem_frames(12 * GiB, 1440, 1440) == 5
    assert vram.auto_max_mem_frames(10 * GiB, 1440, 1440) == 5
    assert vram.auto_max_mem_frames(12 * GiB, 1080, 1920) == 5          # 1080p: same token count
    # bigger card -> more memory frames, clamp 15
    assert vram.auto_max_mem_frames(16 * GiB, 1440, 1440) == 9
    assert vram.auto_max_mem_frames(24 * GiB, 1440, 1440) == 15
    assert vram.auto_max_mem_frames(24 * GiB, 1440, 1440) > vram.auto_max_mem_frames(12 * GiB, 1440, 1440)
    # clamps
    assert vram.auto_max_mem_frames(12 * GiB, 512, 512) == 15
    assert vram.auto_max_mem_frames(2 * GiB, 1440, 1440) == 5
    assert vram.auto_max_mem_frames(0, 4096, 4096) == 5
    # per-frame estimate: HW^2 * 4 B * 3 with HW = ceil(h/16) * ceil(w/16)
    assert vram.bytes_per_memory_frame(1440, 1440) == 8100 ** 2 * 12
    assert vram.bytes_per_memory_frame(1441, 1440) == (91 * 90) ** 2 * 12


def test_resolve_max_mem_frames_explicit_auto_and_cpu():
    cuda = FakeCuda(total=24 * GiB, free=20 * GiB, allocated=1 * GiB)
    cuda.reserved = 3 * GiB                                  # 2 GiB cached but free -> counted as available
    t = _torch(cuda)
    assert vram.available_vram(t) == 22 * GiB
    assert vram.resolve_max_mem_frames(5, t, 1440, 1440, True) == 5      # explicit values unchanged
    assert vram.resolve_max_mem_frames(12, t, 1440, 1440, True) == 12
    assert vram.resolve_max_mem_frames(1, t, 1440, 1440, True) == 2
    assert vram.resolve_max_mem_frames(0, t, 1440, 1440, True) == vram.auto_max_mem_frames(22 * GiB, 1440, 1440) == 15
    assert vram.resolve_max_mem_frames(0, t, 1440, 1440, False) == 5     # CPU: official value


# ------------------------------------------------------------------------------------------- node wiring (fake Comfy)
@pytest.fixture
def node_pkg(monkeypatch, tmp_path):
    """The real package __init__ with ComfyUI / matanyone2 replaced by small fakes (CPU)."""
    calls = {"cfg": [], "release": []}

    class PB:
        def __init__(self, n):
            pass

        def update(self, k):
            pass

    class Core:
        def __init__(self, model, cfg, device):
            calls["cfg"].append(cfg.max_mem_frames)
            self.m = None

        def step(self, img, mask=None, objects=None, first_frame_pred=False, force_permanent=False):
            if mask is not None:
                self.m = mask
            return img.mean(0) * 0.5 + self.m * 0.5

        def output_prob_to_mask(self, prob):
            return prob

    class Cfg(dict):
        __getattr__ = dict.get

        def __setattr__(self, k, v):
            self[k] = v

    class open_dict:
        def __init__(self, c):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    comfy = types.ModuleType("comfy")
    comfy.utils = types.SimpleNamespace(ProgressBar=PB)
    comfy.model_management = types.SimpleNamespace()
    mods = {
        "comfy": comfy, "comfy.utils": comfy.utils, "comfy.model_management": comfy.model_management,
        "folder_paths": types.SimpleNamespace(models_dir=str(tmp_path / "models")),
        "comfy_extras": types.ModuleType("comfy_extras"),
        "comfy_extras.nodes_images": types.SimpleNamespace(_encode_image=None),
        "omegaconf": types.SimpleNamespace(open_dict=open_dict),
        "matanyone2": types.ModuleType("matanyone2"), "matanyone2.inference": types.ModuleType("i"),
        "matanyone2.inference.inference_core": types.SimpleNamespace(InferenceCore=Core),
        "matanyone2.utils": types.ModuleType("u"),
        "matanyone2.utils.inference_utils": types.SimpleNamespace(gen_dilate=lambda m, a, b: m, gen_erosion=lambda m, a, b: m),
    }
    for k, v in mods.items():
        monkeypatch.setitem(sys.modules, k, v)
    spec = importlib.util.spec_from_file_location("ma_kf_pkg", PKG / "__init__.py", submodule_search_locations=[str(PKG)])
    pkg = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "ma_kf_pkg", pkg)
    spec.loader.exec_module(pkg)

    real = pkg.VramReleaser

    def spy(t, frac, enabled=True):
        calls["release"].append((frac, enabled))
        return real(t, frac, enabled)
    monkeypatch.setattr(pkg, "VramReleaser", spy)
    return pkg, calls, Cfg


def _run(pkg, Cfg, **kw):
    n = 6
    imgs = torch.rand(n, 32, 32, 3, generator=torch.Generator().manual_seed(0))
    masks = torch.zeros(2, 32, 32)
    masks[:, 8:24, 8:24] = 1
    pipe = {"model": None, "cfg": Cfg(max_mem_frames=5), "device": torch.device("cpu")}
    args = dict(model_pipe=pipe, images=imgs, keyframe_masks=masks, keyframes="0, 4", start_frame=4, warmup_frames=2,
                erode_kernel=0, dilate_kernel=0, max_size=-1)
    args.update(kw)
    return pkg.MatAnyone2RunKeyframes().run(**args)[0]


def test_node_inputs_version_and_old_graph_compat(node_pkg):
    pkg, calls, Cfg = node_pkg
    it = pkg.MatAnyone2RunKeyframes.INPUT_TYPES()
    assert it["required"]["max_mem_frames"][1]["min"] == 0 and it["required"]["max_mem_frames"][1]["default"] == 5
    assert it["optional"]["vram_release_frac"][1]["default"] == 0.85
    assert pkg.__version__ == "1.2.0"
    old = _run(pkg, Cfg, max_mem_frames=5)                          # graph of an old MatteDesk: no new input
    assert calls["cfg"][-1] == 5 and calls["release"][-1] == (0.85, False)   # CPU: release disabled
    every = _run(pkg, Cfg, max_mem_frames=5, vram_release_frac=0.0)
    assert torch.equal(old, every)                                  # numerics do not depend on the release policy
    _run(pkg, Cfg, max_mem_frames=0, vram_release_frac=0.5)          # auto on CPU -> official 5
    assert calls["cfg"][-1] == 5 and calls["release"][-1] == (0.5, False)
    _run(pkg, Cfg, max_mem_frames=9)
    assert calls["cfg"][-1] == 9


def test_installer_reads_package_version(tmp_path):
    from engine import installer
    assert installer.package_version(PKG) == "1.2.0"
    old = tmp_path / "custom_nodes" / "ComfyUI-MatAnyone2-Keyframes"
    old.mkdir(parents=True)
    (old / "__init__.py").write_text("# old copy without a version\n")
    v = installer.node_versions(str(tmp_path))
    assert v["ComfyUI-MatAnyone2-Keyframes"] == {"installed": None, "repo": "1.2.0"}
    assert installer.node_status(str(tmp_path))["ComfyUI-MatAnyone2-Keyframes"] == "different"   # Doctor offers it


def test_installer_digest_ignores_line_endings(tmp_path):
    """Doctor: a copy that differs only by CRLF / LF line endings is "same" (not "inna wersja niż w repo")."""
    import shutil
    from engine import installer
    dst = tmp_path / "custom_nodes" / "ComfyUI-MatAnyone2-Keyframes"
    shutil.copytree(PKG, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for f in dst.rglob("*.py"):
        lf = f.read_bytes().replace(b"\r\n", b"\n")
        f.write_bytes(lf.replace(b"\n", b"\r\n"))          # every text file with Windows line endings
    assert installer._tree_digest(dst) == installer._tree_digest(PKG)
    assert installer.node_status(str(tmp_path))["ComfyUI-MatAnyone2-Keyframes"] == "same"
    (dst / "__init__.py").write_bytes((dst / "__init__.py").read_bytes() + b"# changed\r\n")
    assert installer.node_status(str(tmp_path))["ComfyUI-MatAnyone2-Keyframes"] == "different"   # real change still seen


def test_doctor_message_shows_versions(tmp_path):
    from doctor import EnvironmentDoctor
    old = tmp_path / "custom_nodes" / "ComfyUI-MatAnyone2-Keyframes"
    old.mkdir(parents=True)
    (old / "__init__.py").write_text("# old\n")
    doc = EnvironmentDoctor(simulation={"simulate_comfy_offline": True})
    doc.config.update({"comfy_dir": str(tmp_path)})
    it = doc.check_node_files()
    assert it.action == "install_nodes" and "→ repo 1.2.0" in it.message
    assert it.details["versions"]["ComfyUI-MatAnyone2-Keyframes"]["repo"] == "1.2.0"


# ------------------------------------------------------------------------------------------- refine graph
NEW_INFO = {"MatAnyone2RunKeyframes": {"input": {"required": {"max_mem_frames": ["INT", {"min": 0}]},
                                                 "optional": {"vram_release_frac": ["FLOAT", {}]}}}}
OLD_INFO = {"MatAnyone2RunKeyframes": {"input": {"required": {"max_mem_frames": ["INT", {"min": 2}]}}}}


class InfoClient:
    def __init__(self, info):
        self.info = info
        self.asked = 0

    def get_object_info(self, node_class=None):
        self.asked += 1
        if isinstance(self.info, Exception):
            raise self.info
        return self.info   # real Comfy: {class: {...}}


def test_refine_sends_new_inputs_only_when_node_has_them():
    from engine import refine
    D = refine.ma_node_defaults()                     # defaults.json "matanyone" (user decision 2026-10-13: 0 / 5)
    assert refine.ma_extra_inputs(InfoClient(NEW_INFO)) == {"vram_release_frac": D["vram_release_frac"], "max_mem_frames": D["max_mem_frames"]}
    assert refine.ma_extra_inputs(InfoClient(NEW_INFO["MatAnyone2RunKeyframes"])) != {}      # inner dict form
    assert refine.ma_extra_inputs(InfoClient(NEW_INFO), {"max_mem_frames": 7}) == {"vram_release_frac": D["vram_release_frac"],
                                                                                   "max_mem_frames": 7}
    for info in (OLD_INFO, None, {}, RuntimeError("offline")):
        assert refine.ma_extra_inputs(InfoClient(info)) == {}
    g_old = refine.matanyone_graph("f_####.exr", 5, "k_####.png", [0], 0, "c_####.png", "a_####.exr")
    assert "vram_release_frac" not in g_old["6"]["inputs"] and g_old["6"]["inputs"]["max_mem_frames"] == 5
    g_new = refine.matanyone_graph("f_####.exr", 5, "k_####.png", [0], 0, "c_####.png", "a_####.exr",
                                   ma_extra=refine.ma_extra_inputs(InfoClient(NEW_INFO)))
    assert g_new["6"]["inputs"]["vram_release_frac"] == D["vram_release_frac"] and g_new["6"]["inputs"]["max_mem_frames"] == D["max_mem_frames"]
    g8 = refine.matanyone_graph_png8("fr", "ke", [0], 0, "co", "p", ma_extra={"vram_release_frac": 0.5, "max_mem_frames": 0})
    assert g8["6"]["inputs"]["vram_release_frac"] == 0.5
    # defaults.json "matanyone"
    d = json.loads((REPO / "defaults.json").read_text(encoding="utf-8"))
    assert set(d["matanyone"]) == {"vram_release_frac", "max_mem_frames", "keyframe_anchor", "keyframe_edge_px"} and d["matanyone"] == refine.ma_node_defaults()
    # Finalizer asks /object_info once
    c = InfoClient(NEW_INFO)
    fin = refine.Finalizer(c, Path("."), 64, 64, lambda f, s, sc: None)
    fin.ma_extra(); fin.ma_extra()
    assert c.asked == 1


INFO_111 = {"MatAnyone2RunKeyframes": {"input": {"required": {"max_mem_frames": ["INT", {"min": 0}]},
                                                 "optional": {"vram_release_frac": ["FLOAT", {}], "alpha_only": ["BOOLEAN", {}]}}}}


def test_refine_sends_alpha_only_to_node_1_1_1_and_frees_comfy_cache():
    from engine import refine
    assert refine.ma_extra_inputs(InfoClient(INFO_111))["alpha_only"] is True
    assert "alpha_only" not in refine.ma_extra_inputs(InfoClient(NEW_INFO))            # 1.1.0: unknown input
    g8 = refine.matanyone_graph_png8("fr", "ke", [0], 0, "co", "p", ma_extra=refine.ma_extra_inputs(InfoClient(INFO_111)))
    assert g8["6"]["inputs"]["alpha_only"] is True

    class C(InfoClient):
        def __init__(self, info, fail=False):
            super().__init__(info)
            self.freed, self.fail = [], fail

        def free_vram(self, unload_models=True, free_memory=True):
            if self.fail:
                raise RuntimeError("offline")
            self.freed.append((unload_models, free_memory))
            return True
    c = C(INFO_111)
    refine.Finalizer(c, Path("."), 64, 64, lambda f, s, sc: None)._release_comfy_cache()
    assert c.freed == [(False, True)]                                                     # models stay, cache goes
    refine.Finalizer(C(INFO_111, fail=True), Path("."), 64, 64, lambda f, s, sc: None)._release_comfy_cache()  # no raise
    refine.Finalizer(InfoClient(INFO_111), Path("."), 64, 64, lambda f, s, sc: None)._release_comfy_cache()    # no free_vram


def test_node_alpha_only_returns_same_alpha_and_dummies(node_pkg):
    pkg, calls, Cfg = node_pkg
    n = 6
    imgs = torch.rand(n, 32, 32, 3, generator=torch.Generator().manual_seed(0))
    masks = torch.zeros(2, 32, 32)
    masks[:, 8:24, 8:24] = 1
    pipe = {"model": None, "cfg": Cfg(max_mem_frames=5), "device": torch.device("cpu")}
    args = dict(model_pipe=pipe, images=imgs, keyframe_masks=masks, keyframes="0, 4", start_frame=4, warmup_frames=2,
                erode_kernel=0, dilate_kernel=0, max_size=-1, max_mem_frames=5)
    full = pkg.MatAnyone2RunKeyframes().run(**args)
    lean = pkg.MatAnyone2RunKeyframes().run(**args, alpha_only=True)
    assert torch.equal(full[0], lean[0])
    assert tuple(full[1].shape) == (n, 32, 32, 3) and tuple(full[2].shape) == (n, 32, 32, 3)
    assert tuple(lean[1].shape) == (1, 1, 1, 3) and tuple(lean[2].shape) == (1, 1, 1, 3)
    it = pkg.MatAnyone2RunKeyframes.INPUT_TYPES()
    assert it["optional"]["alpha_only"][1]["default"] is False and pkg.__version__ == "1.2.0"


# ------------------------------------------------------------------------------------------- ma_scale_mode (pipeline)
def _e3(tmp_path, vram_gb, object_info=None, n=12):
    import test_e3_acceptance as e3
    from mock_comfy import MockComfyClient
    orig = MockComfyClient.__init__

    def init(self, *a, **kw):
        orig(self, *a, **{**kw, "object_info": object_info})
    MockComfyClient.__init__ = init
    try:
        return e3._setup(tmp_path, n=n, vram_gb=vram_gb)
    finally:
        MockComfyClient.__init__ = orig


def _ma_graphs(client):
    return [g for g in client.sent if any(v["class_type"] == "MatAnyone2RunKeyframes" for v in g.values())]


def test_fixed_scale_skips_probe_and_logs_vram(tmp_path):
    from engine import refine as refine_mod
    shot, pipe, client, frames = _e3(tmp_path, vram_gb=11.5, object_info=NEW_INFO)
    pipe.run_core("puppet", frames)
    pipe.store.update("puppet", ma_scale_mode="0.75")
    n0 = len(_ma_graphs(client))
    full, rep, run_dir = pipe.finalize("puppet", frames, edge_mode="matanyone")
    assert rep.ma_scale == 0.75 and rep.ma_scale_mode == "0.75"
    graphs = _ma_graphs(client)[n0:]
    assert len(graphs) == len(pipe.segments("puppet"))               # no 9-frame probe batches
    D = refine_mod.ma_node_defaults()
    assert all(v["inputs"]["vram_release_frac"] == D["vram_release_frac"] and v["inputs"]["max_mem_frames"] == D["max_mem_frames"]
               for g in graphs for v in g.values() if v["class_type"] == "MatAnyone2RunKeyframes")
    assert rep.vram_log and all("wybrana, bez próby" in s and "VRAM 11.5 GB" in s for s in rep.vram_log)
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run["ma_scale_mode"] == "0.75" and run["params"]["ma_scale_mode"] == "0.75"
    # NEW_INFO = a 1.1.x node (no keyframe_anchor input): only the inputs that node has are sent and recorded
    assert run["ma_node"] == {k: D[k] for k in ("vram_release_frac", "max_mem_frames")}
    from engine.contract import validate
    assert not validate(run)


def test_auto_probes_from_full_scale_each_time(tmp_path):
    # old node (no object_info entry) -> old graph inputs; auto: probe 1.0 -> 0.75 -> 0.5 like before
    shot, pipe, client, frames = _e3(tmp_path, vram_gb=6.0)
    pipe.run_core("puppet", frames)
    pipe.store.update("puppet", ma_scale=0.5)                         # stored by an earlier fixed 0.5 run
    full, rep, run_dir = pipe.finalize("puppet", frames, edge_mode="matanyone")
    assert rep.ma_scale_mode == "auto" and rep.ma_scale == 1.0 and len(rep.vram_log) == 1 and rep.ma_node == {}
    assert all("vram_release_frac" not in v["inputs"] for g in _ma_graphs(client) for v in g.values()
               if v["class_type"] == "MatAnyone2RunKeyframes")


def test_api_settings_ma_scale_mode_marks_final_stale_and_hash(tmp_path):
    from fastapi.testclient import TestClient
    import app as app_module
    from engine.export import params_hash
    from synth import make_shot
    shot = make_shot(tmp_path, n=3)
    work = tmp_path / "w"
    work.mkdir()
    mf = work / "shot_manifest_S.json"
    mf.write_text(json.dumps({"shot": "S", "first": shot.first, "last": shot.last, "plate_w": shot.w, "plate_h": shot.h,
                              "work": str(work), "plates": {"beauty": shot.plate_pattern}}), encoding="utf-8")
    api = TestClient(app_module.app)
    api.post("/api/project/open", json={"manifest_path": str(mf)})
    o = api.post("/api/object", json={"name": "o", "prompts": ["x"]}).json()
    store = app_module._objects()
    assert store.get("o")["ma_scale_mode"] == "auto"
    store.update("o", tool={"final_valid": True})
    before = store.get("o")
    r = api.put("/api/object/o/settings", json={"ma_scale_mode": "0.5"})
    assert r.status_code == 200
    o = r.json()["object"]
    assert o["ma_scale_mode"] == "0.5" and o["tool"]["final_valid"] is False
    assert api.put("/api/object/o/settings", json={"ma_scale_mode": "0.6"}).status_code == 400
    assert api.put("/api/object/o/settings", json={"ma_scale_mode": "1"}).json()["object"]["ma_scale_mode"] == "1.0"
    run = {"params": {}, "segments_def": []}
    assert params_hash(before, run) != params_hash({**before, "ma_scale_mode": "0.5"}, run)
    assert params_hash(before, run) == params_hash({k: v for k, v in before.items() if k != "ma_scale_mode"}, run)


def test_md_cli_finalize_ma_scale():
    sys.path.insert(0, str(REPO / "tools"))
    import md
    calls, lines = [], []

    class H:
        def request(self, method, path, body=None, timeout=None):
            calls.append((method, path, body))
            if path == "/api/jobs":
                return 200, {"jobs": []}
            return 200, {"run": "v003"} if path.endswith("/finalize") else {"object": {}}

    assert md.main(["finalize", "figure", "--ma-scale", "0.75", "--wait"], http=H(), out=lines.append) == 0
    i = calls.index(("PUT", "/api/object/figure/settings", {"ma_scale_mode": "0.75"}))
    assert any(c[:2] == ("POST", "/api/object/figure/finalize") for c in calls[i:])
    calls.clear()
    assert md.main(["finalize", "figure"], http=H(), out=lines.append) == 0
    assert not any(c[0] == "PUT" for c in calls)
    with pytest.raises(SystemExit):
        md.main(["finalize", "figure", "--ma-scale", "0.6"], http=H(), out=lines.append)


# ------------------------------------------------------------------------------------------- benchmark helpers (no GPU)
def test_bench_script_helpers(tmp_path):
    import cv2
    if not (REPO / "tools" / "bench_matanyone.py").exists():
        pytest.skip("benchmark scripts are not part of the public copy")
    sys.path.insert(0, str(REPO / "tools"))
    import bench_matanyone as bench
    for n in ("f_0001.png", "f_0000.png", "c_0000.png", "k_0000.png"):
        cv2.imwrite(str(tmp_path / n), np.full((8, 8, 3), 128, np.uint8))
    assert [p.name for p in bench.list_frames(tmp_path)] == ["f_0000.png", "f_0001.png"]
    im = bench.read_image(tmp_path / "f_0000.png")
    assert im.shape == (8, 8, 3) and abs(float(im.max()) - 128 / 255) < 1e-6
    assert bench.read_mask(tmp_path / "k_0000.png").max() == 1.0
    assert bench.main(["--frames", str(tmp_path), "--mask", "x.png", "--keyframes", "0,8"]) == 1   # 1 mask, 2 keys: stops before the GPU


# ------------------------------------------------------------------------------------------- CoreCombine core_mode (1.1.2)
def test_core_combine_soft_mode(node_pkg, monkeypatch):
    """core_mode=soft: a frame whose core mask has intermediate values (SAM3 probabilities) is used as the core directly;
    a binary frame keeps the old threshold + erode + feather path; the default (no core_mode) = the old node bit for bit."""
    pkg, _, _ = node_pkg
    node = pkg.MatAnyone2CoreCombine
    # the submodule may be cached from an earlier test with an older fake comfy: patch the module it actually uses
    monkeypatch.setattr(sys.modules[node.__module__].mm, "get_torch_device", lambda: torch.device("cpu"), raising=False)
    it = node.INPUT_TYPES()
    assert it["optional"]["core_mode"][0] == ["binary", "soft"] and it["optional"]["core_mode"][1]["default"] == "binary"
    n, H, W = 3, 48, 48
    alpha = torch.zeros(n, H, W)
    alpha[:, 12:36, 12:36] = 0.6                        # MatAnyone with a weak interior: the core fill shows
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    p = (1.0 / (1.0 + torch.exp(-(10.0 - torch.maximum((yy - 23.5).abs(), (xx - 23.5).abs())) / 2.0)))   # soft square
    binary = (p > 0.5).float()
    cores = torch.stack([p, binary, p])
    kw = dict(core_erode_px=3, core_feather_px=2, gate_px=0, gate_low=0.55, gate_high=0.8, temporal_sigma=0.0,
              edge_band_px=6, motion_limit=0.35)
    old = node().run(alpha, cores, **kw)[0]
    assert torch.equal(old, node().run(alpha, cores, **kw, core_mode="binary")[0])
    soft = node().run(alpha, cores, **kw, core_mode="soft")[0]
    assert torch.equal(soft[1], old[1])                 # binary frame: old path
    ref = torch.maximum(alpha[0], p).half().float()
    assert torch.allclose(soft[0], ref, atol=1e-6) and torch.allclose(soft[2], ref, atol=1e-6)   # core = p as is
    assert not torch.equal(soft[0], old[0])             # the soft start differs from the eroded hard core


def test_node_soft_keyframe_anchor(node_pkg):
    """1.2.0 (2026-10-16): a key frame other than the start one used to output the hard key (CLIP2_A: F20 / F80 with
    13.5k / 16.5k soft px vs ~19k / ~21k around them). soft (default): the memory prediction is kept in a thin band around
    the key's edge, the key elsewhere, and the memory is anchored on that; hard = the old behaviour."""
    pkg, calls, Cfg = node_pkg
    it = pkg.MatAnyone2RunKeyframes.INPUT_TYPES()
    assert it["optional"]["keyframe_anchor"][0] == ["soft", "hard"] and it["optional"]["keyframe_anchor"][1]["default"] == "soft"
    assert it["optional"]["keyframe_edge_px"][1]["default"] == 6
    soft = _run(pkg, Cfg, max_mem_frames=5, keyframe_edge_px=3)      # keys 0 and 4, start 4 -> frame 0 is a re-anchor
    hard = _run(pkg, Cfg, max_mem_frames=5, keyframe_anchor="hard")
    k0 = soft[0]
    assert float(k0[16, 16]) == 1.0 and float(k0[2, 2]) == 0.0        # inside / outside the band: the key itself (0 / 1)
    edge = k0[6:11, 16]                                              # across the key's top edge (row 8): the prediction
    assert ((edge > 0) & (edge < 1)).any() or not torch.equal(edge, (torch.arange(6, 11) >= 8).float())
    assert not torch.equal(soft[0], hard[0]) and torch.equal(soft[4], hard[4])   # the start frame is unchanged
    assert torch.equal(_run(pkg, Cfg, max_mem_frames=5, keyframe_edge_px=0)[0], (torch.zeros(32, 32).index_fill_(0, torch.arange(8, 24), 1) * torch.zeros(32, 32).index_fill_(1, torch.arange(8, 24), 1)))


def test_refine_sends_keyframe_anchor_to_node_1_2_0():
    from engine import refine
    info = {"MatAnyone2RunKeyframes": {"input": {"required": {"max_mem_frames": ["INT", {"min": 0}]},
                                                 "optional": {"vram_release_frac": ["FLOAT", {}], "alpha_only": ["BOOLEAN", {}],
                                                              "keyframe_anchor": [["soft", "hard"], {}], "keyframe_edge_px": ["INT", {}]}}}}
    ex = refine.ma_extra_inputs(InfoClient(info))
    assert ex["keyframe_anchor"] == "soft" and ex["keyframe_edge_px"] == 6
    assert "keyframe_anchor" not in refine.ma_extra_inputs(InfoClient(INFO_111))   # older node: not sent
