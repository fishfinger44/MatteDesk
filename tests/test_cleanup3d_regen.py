# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cleanup 3D package P5 – regeneration of a key's marked area (engine/cleanup3d/regen.py) with a fake ComfyUI client;
one `real` test per engine on a sh060 frame (ComfyUI + GPU, skipped when ComfyUI does not answer)."""
import ast
import math
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from engine import stage_models as SM  # noqa: E402
from engine.cleanup3d import keys as KS  # noqa: E402
from engine.cleanup3d import regen as RG  # noqa: E402
from engine.cleanup3d import store  # noqa: E402
from engine.cleanup3d import types as T  # noqa: E402
from tests.mock_comfy import MockComfyClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
H, W = 160, 224
CARD, FRAME = "sea", 12
AREA = (60, 50, 150, 110)            # x0, y0, x1, y1 of the marked rectangle
OLD_FILL = np.array([1.0, 0.0, 1.0], np.float32)     # BGR magenta: the faulty fill the model must never see
SHIFT = np.array([0.06, -0.04, 0.05], np.float32)    # the fake FLUX 3 Image Klein answers its reference + this colour cast


@pytest.fixture
def work(tmp_path):
    w = tmp_path / "shot" / "matte_desk"
    w.mkdir(parents=True)
    return w


def _background(h=H, w=W):
    """Smooth grey-blue gradient with green >= 0.4 everywhere (so magenta is recognisable)."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.dstack([0.55 + 0.2 * x / w, 0.45 + 0.2 * y / h, 0.35 + 0.1 * (x + y) / (w + h)])
    return (np.round(img * 255) / 255).astype(np.float32)


def _rect(box=AREA, h=H, w=W):
    m = np.zeros((h, w), np.float32)
    m[box[1]:box[3], box[0]:box[2]] = 1.0
    return m


def _make_key(work, ext="png", box=AREA, img=None):
    """A key whose marked rectangle holds the old (magenta checker) fill + the brush area over it."""
    img = _background() if img is None else img
    m = _rect(box) > 0
    yy, xx = np.mgrid[0:H, 0:W]
    img = img.copy()
    img[m] = OLD_FILL * (0.6 + 0.4 * (((xx + yy) // 4) % 2))[m][:, None]
    KS.add_version(work, CARD, FRAME, img, "plate", ext)
    KS.save_brush_layer(work, CARD, FRAME, "a1", _rect(box))
    return KS.read_key_image(work, store.load_key(work, CARD, FRAME))


class Fake:
    """MockComfyClient + handlers standing in for the two engines. Keeps what the models were given."""

    def __init__(self, root):
        self.root = Path(root)
        self.tiles = []          # FLUX 3 Image Klein: (graph, reference tile float BGR)
        self.lama = []           # LaMa: (image float BGR, mask bool)
        self.qwen = []           # Qwen-Image-2.1: (graph, crop float BGR as the model got it)
        self.n = 0
        self.client = MockComfyClient({"KSampler": self._klein, "INPAINT_InpaintWithModel": self._lama},
                                      object_info={"VHS_LoadImagesPath": {"VHS_LoadImagesPath": {}}})
        self.client.output_root = str(self.root)

    def _save(self, node, prefix, imgs):
        sub, stem = prefix.rsplit("/", 1)
        (self.root / sub).mkdir(parents=True, exist_ok=True)
        out = []
        for im in imgs:
            self.n += 1
            name = f"{stem}_{self.n:05d}_.png"
            cv2.imwrite(str(self.root / sub / name), np.clip(np.rint(im * 255), 0, 255).astype(np.uint8))
            out.append({"filename": name, "subfolder": sub, "type": "output"})
        return {node: {"images": out}}

    @staticmethod
    def _load_dir(node):
        assert node["class_type"] == "VHS_LoadImagesPath"
        files = sorted(Path(node["inputs"]["directory"]).glob("*.png"))
        return [cv2.imread(str(f), cv2.IMREAD_COLOR).astype(np.float32) / 255.0 for f in files]

    def _klein(self, g):
        if any(n["class_type"] == "TextEncodeQwenImage21" for n in g.values()):
            return self._qwen(g)
        (tile,) = self._load_dir(g["4"])
        self.tiles.append((g, tile))
        return self._save("14", g["14"]["inputs"]["filename_prefix"], [np.clip(tile + SHIFT, 0, 1)])

    def _qwen(self, g):
        """Qwen-Image-2.1 stand-in: an edit model that removes what the instruction names (the magenta object it SEES)
        and paints the whole crop with a colour cast."""
        (crop,) = self._load_dir(g["4"])
        self.qwen.append((g, crop))
        o = crop.copy()
        obj = _is_magenta(crop)
        if obj.any() and (~obj).any():
            o[obj] = crop[~obj].mean(axis=0)
        return self._save("14", g["14"]["inputs"]["filename_prefix"], [np.clip(o + SHIFT, 0, 1)])

    def _lama(self, g):
        imgs, masks = self._load_dir(g["1"]), self._load_dir(g["2"])
        outs = []
        for im, m in zip(imgs, masks):
            m = m[..., 0] > 0.5
            self.lama.append((im, m))
            o = im.copy()
            o[m] = cv2.GaussianBlur(im, (0, 0), 2.0)[m]          # a fill made of what the model was given
            outs.append(o)
        return self._save("6", g["6"]["inputs"]["filename_prefix"], outs)


@pytest.fixture
def fake(tmp_path):
    return Fake(tmp_path / "comfy_out")


def _run(work, fake, engine, tmp_path, **kw):
    return RG.regenerate(work, CARD, FRAME, engine, client=fake.client, io_dir=tmp_path / "io", cfg={}, **kw)


def _u8(work, version):
    return cv2.imread(str(T.key_dir(work, CARD, FRAME) / f"{version}.png"), cv2.IMREAD_UNCHANGED)


def _is_magenta(img):
    """Pixels that still look like the old fill (green far below the background's 0.4)."""
    return img[..., 1] < 0.3


# ----------------------------------------------------------------------------- the graph
def test_klein_graph_nodes_values_and_file_names():
    g = RG.klein_graph("C:/io/t_000", "matte_desk/x/t000", 1920, 1088)
    by = {}
    for n in g.values():
        by.setdefault(n["class_type"], []).append(n["inputs"])
    assert by["UNETLoader"][0]["unet_name"] == SM.KLEIN_FILES["unet"][1] == "flux3-image-klein-bf16.safetensors"
    lora = by["LoraLoaderModelOnly"][0]
    assert lora["lora_name"] == SM.KLEIN_LORA["lora"][1] == "flux-3-image-klein-restore.safetensors"
    assert lora["strength_model"] == 1.0 and g[lora["model"][0]]["class_type"] == "UNETLoader"
    ms = by["ModelSamplingFlux"][0]
    assert ms["max_shift"] == ms["base_shift"] == pytest.approx(math.log(16))
    assert (ms["width"], ms["height"]) == (1920, 1088) and g[ms["model"][0]]["class_type"] == "LoraLoaderModelOnly"
    clip = by["CLIPLoader"][0]
    assert clip["clip_name"] == SM.KLEIN_FILES["text_encoder"][1] == "qwen3vl_4b_bf16.safetensors"
    assert clip["type"] == "flux3"
    assert by["VAELoader"][0]["vae_name"] == SM.KLEIN_FILES["vae"][1] == "flux2-vae.safetensors"
    texts = sorted(i["text"] for i in by["CLIPTextEncode"])
    assert texts == ["", RG.KLEIN_PROMPT] and RG.KLEIN_PROMPT.startswith("Restore the photo: remove blur, noise")
    ref = by["ReferenceMultiLatent"][0]
    assert ref["resolution"] == "keep resolution"
    assert g[ref["images.image0"][0]]["class_type"] == "VHS_LoadImagesPath"
    assert g[ref["positive"][0]]["inputs"]["text"] == RG.KLEIN_PROMPT and g[ref["negative"][0]]["inputs"]["text"] == ""
    ks = by["KSampler"][0]
    assert (ks["steps"], ks["cfg"], ks["sampler_name"], ks["scheduler"], ks["denoise"]) == (8, 1.0, "euler", "simple", 1.0)
    ref_id = next(k for k, n in g.items() if n["class_type"] == "ReferenceMultiLatent")
    assert ks["positive"] == [ref_id, 0] and ks["negative"] == [ref_id, 1] and ks["latent_image"] == [ref_id, 2]
    assert g[ks["model"][0]]["class_type"] == "ModelSamplingFlux"
    dec = by["VAEDecode"][0]
    assert g[dec["samples"][0]]["class_type"] == "KSampler" and g[dec["vae"][0]]["class_type"] == "VAELoader"
    assert g[by["SaveImage"][0]["images"][0]]["class_type"] == "VAEDecode"
    assert RG.klein_graph("d", "p", 64, 64, prompt="own text")["8"]["inputs"]["text"] == "own text"


# ----------------------------------------------------------------------------- tiles
def test_tile_size_about_2mp_multiples_of_16():
    assert RG.tile_size(1700, 900, 4096, 2731) == (1712, 912)
    tw, th = RG.tile_size(4000, 2600, 4096, 2731)
    assert tw % 16 == 0 and th % 16 == 0 and tw * th <= RG.TILE_PX and tw * th > 0.9 * RG.TILE_PX
    assert max(tw, th) <= RG.TILE_MAX_SIDE
    assert RG.tile_size(50, 50, 100, 90) == (64, 64)              # never larger than the image
    with pytest.raises(ValueError, match="za mały"):
        RG.tile_size(8, 8, 12, 300)


def test_tiles_cover_the_region_and_overlap(monkeypatch):
    monkeypatch.setattr(RG, "TILE_PX", 96 * 96)
    monkeypatch.setattr(RG, "TILE_CONTEXT_PX", 16)
    region = RG.paste_region(_rect((30, 20, 200, 140)))
    tiles = RG.plan_tiles(region)
    assert len(tiles) >= 4
    cover = np.zeros((H, W), np.int32)
    for x0, y0, x1, y1 in tiles:
        assert (x1 - x0) % 16 == 0 and (y1 - y0) % 16 == 0 and (x1 - x0) * (y1 - y0) <= 96 * 96
        assert 0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H
        cover[y0:y1, x0:x1] += 1
    assert (cover[region > 0] >= 1).all()                          # the whole region is inside tiles
    assert (cover >= 2).any()
    for a in tiles:                                                # every tile shares a strip with a neighbour
        best = 0
        for b in tiles:
            if a == b:
                continue
            ox, oy = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
            if ox > 0 and oy > 0:
                best = max(best, ox if oy == a[3] - a[1] else oy if ox == a[2] - a[0] else min(ox, oy))
        assert best >= RG.tile_overlap(min(a[2] - a[0], a[3] - a[1]))
    wt = RG.tile_weight(96, 80)
    assert wt.shape == (80, 96) and wt.max() == 1.0 and 0 < wt.min() < 0.1


def test_two_far_parts_are_two_groups():
    m = _rect((10, 10, 30, 30), 600, 900) + _rect((700, 400, 760, 470), 600, 900)
    lab, boxes = RG.groups(m)
    assert len(boxes) == 2 and set(np.unique(lab)) == {0, 1, 2}
    assert sorted(boxes) == [(10, 10, 30, 30), (700, 400, 760, 470)]
    assert ((lab > 0) == (m > 0)).all()


# ----------------------------------------------------------------------------- FLUX 3 Image Klein + LoRA Restore
def test_klein_regenerates_only_the_area(work, fake, tmp_path):
    old = _make_key(work)
    seen = []
    r = _run(work, fake, "klein_restore", tmp_path, progress=lambda d, t: seen.append((d, t)))
    assert r["engine"] == "klein_restore" and r["version"] == "v002" and r["tiles"] == 1 and r["passes"] == 1
    assert r["area_px"] == int(_rect().sum()) and r["seconds"] >= 0 and seen == [(1, 1)]
    key = store.load_key(work, CARD, FRAME)
    assert key.active == "v002" and key.versions[-1]["source"] == "regen:klein_restore"
    assert key.versions[-1]["ext"] == "png" and key.versions[-1]["bits"] == 8
    assert len(fake.client.sent) == 1 and len(fake.tiles) == 1

    # the model's input: nothing of the old fill in the area (nor in its growth), tile = what the graph declares
    g, tile = fake.tiles[0]
    assert tile.shape[0] % 16 == 0 and tile.shape[1] % 16 == 0
    assert (g["6"]["inputs"]["width"], g["6"]["inputs"]["height"]) == (tile.shape[1], tile.shape[0])
    assert not _is_magenta(tile).any()
    assert tile[..., 1].min() > 0.4

    # the new version: bit-identical outside the grown area, the old fill gone inside
    a, b = _u8(work, "v001"), _u8(work, "v002")
    grown = cv2.dilate(_rect().astype(np.uint8), np.ones((2 * RG.PASTE_GROW_PX + 1,) * 2, np.uint8)) > 0
    assert np.array_equal(a[~grown], b[~grown])
    new = b.astype(np.float32) / 255
    assert not _is_magenta(new).any()
    # colour matched to the surroundings: the fake's colour cast (0.04–0.06) is gone
    bg = _background()
    inner = _rect() > 0
    assert np.abs((new - bg)[inner].mean(axis=0)).max() < 0.02
    assert np.abs(old - a.astype(np.float32) / 255).max() < 1e-6
    assert not list((tmp_path / "io").glob("*"))                    # the tiles are removed afterwards


def test_klein_soften_seams_queues_two_passes(work, fake, tmp_path, monkeypatch):
    monkeypatch.setattr(RG, "TILE_PX", 96 * 96)
    monkeypatch.setattr(RG, "TILE_CONTEXT_PX", 16)
    _make_key(work)
    seen = []
    r = _run(work, fake, "klein_restore", tmp_path, soften_seams=True, prompt="own prompt",
             progress=lambda d, t: seen.append((d, t)))
    n = r["tiles"]
    assert n >= 2 and r["passes"] == 2 and len(fake.client.sent) == 2 * n
    assert seen == [(i + 1, 2 * n) for i in range(2 * n)]
    assert all(g["8"]["inputs"]["text"] == "own prompt" for g, _ in fake.tiles)
    # the description stays with the version and comes back in the state (the UI's prompt field of this key)
    assert store.load_key(work, CARD, FRAME).versions[-1]["prompt"] == "own prompt"
    first, second = fake.tiles[:n], fake.tiles[n:]
    for (_, t1), (_, t2) in zip(first, second):
        assert t1.shape == t2.shape
        assert not _is_magenta(t1).any() and not _is_magenta(t2).any()
    # pass 2 is fed the result of pass 1 (the fake adds a colour cast; the per-tile match removes most of it, the
    # references still differ) and the first reference is blurred
    assert any(np.abs(t1 - t2).max() > 1e-3 for (_, t1), (_, t2) in zip(first, second))
    a, b = _u8(work, "v001"), _u8(work, "v002")
    grown = cv2.dilate(_rect().astype(np.uint8), np.ones((2 * RG.PASTE_GROW_PX + 1,) * 2, np.uint8)) > 0
    assert np.array_equal(a[~grown], b[~grown]) and not np.array_equal(a[grown], b[grown])


def test_soften_reference_blurs_only_around_the_region():
    rng = np.random.default_rng(3)
    ref = rng.random((H, W, 3)).astype(np.float32)
    region = RG.paste_region(_rect())
    s = RG.soften_reference(ref, region)
    inner = region > 0
    assert np.abs(np.diff(s, axis=1))[inner[:, 1:]].mean() < 0.3 * np.abs(np.diff(ref, axis=1))[inner[:, 1:]].mean()
    far = cv2.dilate(region, np.ones((121, 121), np.uint8)) == 0
    assert far.any() and np.abs(s - ref)[far].max() < 1e-4


# ----------------------------------------------------------------------------- LaMa
def test_lama_crop_mask_and_paste(work, fake, tmp_path):
    _make_key(work)
    r = _run(work, fake, "lama", tmp_path)
    assert r == {**r, "engine": "lama", "version": "v002", "tiles": 1, "passes": 1}
    assert store.load_key(work, CARD, FRAME).versions[-1]["source"] == "regen:lama"
    (im, m), = fake.lama
    assert im.shape[:2] == (H, W)                                  # area + 256 px margin, clipped to the key
    grown = RG.paste_region(_rect()) > 0
    assert np.array_equal(m, grown)                                # the mask = the area grown by the paste edge
    assert not _is_magenta(im).any()                               # no old fill even under the mask
    a, b = _u8(work, "v001"), _u8(work, "v002")
    assert np.array_equal(a[~grown], b[~grown])
    assert not _is_magenta(b.astype(np.float32) / 255).any()
    # outside its mask LaMa gets the key's own pixels (a smoothed rim made a grey blob on sh060)
    assert np.array_equal(np.rint(im * 255).astype(np.uint8)[~grown], a[~grown])


def test_lama_masks_the_whole_area_but_pastes_inside_the_extent(work, fake, tmp_path):
    _make_key(work)
    ext = np.zeros((H, W), np.float32)
    ext[:, :100] = 1.0
    _run(work, fake, "lama", tmp_path, extent=ext)
    (im, m), = fake.lama
    assert np.array_equal(m, RG.paste_region(_rect()) > 0)         # the old fill beyond the extent is masked too
    a, b = _u8(work, "v001"), _u8(work, "v002")
    assert np.array_equal(a[:, 100:], b[:, 100:]) and not np.array_equal(a[:, :100], b[:, :100])


def test_lama_big_crop_is_scaled_down_and_back(work, fake, tmp_path, monkeypatch):
    monkeypatch.setattr(RG, "LAMA_MAX_SIDE", 112)
    _make_key(work)
    _run(work, fake, "lama", tmp_path)
    (im, m), = fake.lama
    assert im.shape[:2] == (80, 112) and m.any()
    assert _u8(work, "v002").shape[:2] == (H, W)


def test_lama_one_crop_per_far_group(work, fake, tmp_path, monkeypatch):
    monkeypatch.setattr(RG, "GROUP_PX", 8)
    monkeypatch.setattr(RG, "LAMA_MARGIN_PX", 20)
    _make_key(work, box=(20, 20, 50, 50))
    KS.save_brush_layer(work, CARD, FRAME, "a2", _rect((160, 100, 200, 140)))
    seen = []
    r = _run(work, fake, "lama", tmp_path, progress=lambda d, t: seen.append((d, t)))
    assert r["tiles"] == 2 and len(fake.lama) == 2 and seen == [(1, 2), (2, 2)]
    # each crop = its part grown by the paste edge (4 px) + 20 px of context, clipped to the key
    assert sorted(im.shape[:2] for im, _ in fake.lama) == [(74, 74), (84, 88)]


# ----------------------------------------------------------------------------- Qwen-Image-2.1 (edit)
def test_qwen_graph_nodes_values_and_file_names():
    g = RG.qwen_graph("C:/io/c_001", "matte_desk/x/c001", "do it")
    by = {}
    for n in g.values():
        by.setdefault(n["class_type"], []).append(n["inputs"])
    assert by["UNETLoader"][0]["unet_name"] == SM.QWEN21_FILES["unet"][1] == "qwen_image_2.1_int8_convrot.safetensors"
    clip = by["CLIPLoader"][0]
    assert clip["clip_name"] == SM.QWEN21_FILES["text_encoder"][1] == "qwen3vl_8b_int8_convrot.safetensors"
    assert clip["type"] == "qwen_image"
    assert by["VAELoader"][0]["vae_name"] == SM.QWEN21_FILES["vae"][1] == "qwen_image_2.1_vae_bf16.safetensors"
    enc = by["TextEncodeQwenImage21"][0]
    assert (enc["prompt"], enc["negative_prompt"], enc["resolution"]) == ("do it", "", 0)
    assert g[enc["images.image_1"][0]]["class_type"] == "VHS_LoadImagesPath"
    assert g[enc["vae"][0]]["class_type"] == "VAELoader" and g[enc["clip"][0]]["class_type"] == "CLIPLoader"
    venc = by["VAEEncode"][0]
    assert venc["pixels"] == enc["images.image_1"]                 # the sampler starts from the crop itself
    ks = by["KSampler"][0]
    assert (ks["steps"], ks["cfg"], ks["sampler_name"], ks["scheduler"], ks["denoise"], ks["seed"]) == \
        (40, 1.0, "euler", "simple", 1.0, 42)
    enc_id = next(k for k, n in g.items() if n["class_type"] == "TextEncodeQwenImage21")
    assert ks["positive"] == [enc_id, 0] and ks["negative"] == [enc_id, 1]
    assert g[ks["latent_image"][0]]["class_type"] == "VAEEncode" and g[ks["model"][0]]["class_type"] == "UNETLoader"
    assert g[by["VAEDecode"][0]["samples"][0]]["class_type"] == "KSampler"
    assert g[by["SaveImage"][0]["images"][0]]["class_type"] == "VAEDecode" and RG.QWEN_SAVE_NODE in g


def test_qwen_prompt_default_and_description():
    assert RG.qwen_prompt(None) == RG.qwen_prompt("  ") == RG.QWEN_PROMPT
    assert RG.QWEN_PROMPT.startswith("Remove the marked object and everything it causes (shadow, footprints, kicked-up")
    assert RG.QWEN_PROMPT.endswith("Change nothing else.")
    assert RG.qwen_prompt("wet sand with ripples.") == RG.QWEN_PROMPT + " In its place there should be: wet sand with ripples."


def test_qwen_crop_and_model_size():
    assert RG.qwen_crop((2130, 1030, 2490, 1985), 4096, 2731) == (2130 - 256, 1030 - 256, 2490 + 256, 1985 + 256)
    assert RG.qwen_crop((0, 0, 40, 30), 100, 80) == (0, 0, 100, 80)                  # clamped to the key
    assert RG.qwen_crop((1000, 1000, 3000, 1400), 4096, 2731) == (500, 500, 3500, 1900)   # 25 % of the longer side
    for cw, ch in ((872, 1467), (224, 160), (3000, 1400), (4096, 2731), (40, 30)):
        mw, mh = RG.qwen_model_size(cw, ch)
        assert mw % 32 == 0 and mh % 32 == 0
        assert abs(mw * mh / RG.QWEN_PX - 1.0) < 0.08, (cw, ch, mw, mh)              # about 1.4 MP, up or down
        assert abs(mw / mh - cw / ch) / (cw / ch) < 0.08                             # aspect kept
    assert RG.qwen_model_size(1024, 1344) == (1024, 1344)


def test_qwen_edits_the_crop_and_pastes_only_the_area(work, fake, tmp_path, monkeypatch):
    leases = []
    import engine.gpu_manager as GM
    real_lease = GM.comfy_lease

    def lease(client, label, *a, **k):
        leases.append(label)
        return real_lease(client, label, *a, **k)
    monkeypatch.setattr(GM, "comfy_lease", lease)
    old = _make_key(work)
    seen = []
    r = _run(work, fake, "qwen21_edit", tmp_path, prompt="wet sand", progress=lambda d, t: seen.append((d, t)))
    assert r["engine"] == "qwen21_edit" and r["version"] == "v002" and r["tiles"] == 1 and r["passes"] == 1
    assert seen == [(1, 1)] and leases == ["Qwen-Image-2.1 (dogenerowanie klucza)"]
    key = store.load_key(work, CARD, FRAME)
    assert key.versions[-1]["source"] == "regen:qwen21_edit" and key.versions[-1]["prompt"] == "wet sand"
    assert len(fake.client.sent) == 1 and len(fake.qwen) == 1

    # the model's input: the whole key (crop = area + 256 px, clamped) scaled up to ~1.4 MP, multiples of 32, with the
    # object STILL VISIBLE (an edit model removes what the instruction names), the instruction + the description
    g, crop = fake.qwen[0]
    assert crop.shape[:2] == (992, 1376) and crop.shape[0] % 32 == 0 and crop.shape[1] % 32 == 0
    assert _is_magenta(crop).any()
    assert g["8"]["inputs"]["prompt"] == RG.QWEN_PROMPT + " In its place there should be: wet sand."

    a, b = _u8(work, "v001"), _u8(work, "v002")
    grown = RG.paste_region(_rect()) > 0
    assert np.array_equal(a[~grown], b[~grown])                    # everything outside the area bit for bit
    new = b.astype(np.float32) / 255
    assert not _is_magenta(new).any()
    inner = _rect() > 0
    assert np.abs((new - _background())[inner].mean(axis=0)).max() < 0.03   # the colour cast is matched away on the ring
    assert np.abs(old - a.astype(np.float32) / 255).max() < 1e-6
    assert not list((tmp_path / "io").glob("*"))                    # the crops are removed afterwards


def test_qwen_default_prompt_and_blur_hook(work, fake, tmp_path, monkeypatch):
    monkeypatch.setattr(RG, "QWEN_PX", 64 * 96)
    _make_key(work)
    got = {}

    def blur(gen, real, hole, known):
        got.update(gen=gen, hole=hole, known=known)
        return gen
    _run(work, fake, "qwen21_edit", tmp_path, blur=blur)
    g, crop = fake.qwen[0]
    assert g["8"]["inputs"]["prompt"] == RG.QWEN_PROMPT
    assert crop.shape[0] % 32 == 0 and crop.shape[1] % 32 == 0
    assert store.load_key(work, CARD, FRAME).versions[-1].get("prompt") is None
    assert got["known"].all() and got["hole"][_rect() > 0].all()  # the crop covers the whole small key
    assert not _is_magenta(got["gen"]).any()


def test_qwen_one_crop_per_far_group(work, fake, tmp_path, monkeypatch):
    monkeypatch.setattr(RG, "GROUP_PX", 8)
    monkeypatch.setattr(RG, "QWEN_MARGIN_PX", 12)
    monkeypatch.setattr(RG, "QWEN_PX", 64 * 64)
    _make_key(work, box=(20, 20, 50, 50))
    KS.save_brush_layer(work, CARD, FRAME, "a2", _rect((160, 100, 200, 140)))
    seen = []
    r = _run(work, fake, "qwen21_edit", tmp_path, progress=lambda d, t: seen.append((d, t)))
    assert r["tiles"] == 2 and len(fake.qwen) == 2 and seen == [(1, 2), (2, 2)]
    a, b = _u8(work, "v001"), _u8(work, "v002")
    grown = RG.paste_region(_rect((20, 20, 50, 50)) + _rect((160, 100, 200, 140))) > 0
    assert np.array_equal(a[~grown], b[~grown])


# ----------------------------------------------------------------------------- extent, precision
def test_extent_cuts_the_area_and_its_growth(work, fake, tmp_path):
    _make_key(work)
    ext = np.zeros((H, W), np.float32)
    ext[:, :100] = 1.0                                             # the card ends at x = 100, inside the area
    r = _run(work, fake, "klein_restore", tmp_path, extent=ext)
    assert r["area_px"] == (100 - AREA[0]) * (AREA[3] - AREA[1])
    a, b = _u8(work, "v001"), _u8(work, "v002")
    assert np.array_equal(a[:, 100:], b[:, 100:])                  # the neighbouring card keeps its pixels (old fill too)
    assert _is_magenta(b.astype(np.float32) / 255)[:, 100:].any()
    assert not _is_magenta(b.astype(np.float32) / 255)[:, :96].any()
    _, tile = fake.tiles[0]
    assert not _is_magenta(tile).any()                             # ... but the model never saw that old fill
    with pytest.raises(ValueError, match="poza zasięgiem karty"):
        _run(work, fake, "klein_restore", tmp_path, extent=np.zeros((H, W), np.float32))
    with pytest.raises(ValueError, match="inny rozmiar"):
        _run(work, fake, "klein_restore", tmp_path, extent=np.ones((H // 2, W), np.float32))
    assert store.load_key(work, CARD, FRAME).active == "v002"      # refusals add nothing


@pytest.mark.parametrize("engine", T.ENGINES)
def test_linear_exr_key_goes_through_acescct(work, fake, tmp_path, engine):
    old = _make_key(work, ext="exr", img=_background() * 3.0)       # scene-linear, values above 1
    assert old.max() > 1.5
    _run(work, fake, engine, tmp_path)
    key = store.load_key(work, CARD, FRAME)
    assert key.versions[-1]["ext"] == "exr" and key.versions[-1]["bits"] == 32
    new = KS.read_key_image(work, key)
    grown = RG.paste_region(_rect()) > 0
    assert np.array_equal(old[~grown], new[~grown])
    inner = _rect() > 0
    assert np.isfinite(new).all() and np.abs(new - old)[inner].max() > 0.1
    # the fill has the level of the linear surroundings (ACEScct there and back), not of a clipped 0..1 image
    assert abs(float(new[inner].mean()) / float((_background() * 3.0)[inner].mean()) - 1.0) < 0.1
    model_in = {"klein_restore": lambda: fake.tiles[0][1], "qwen21_edit": lambda: fake.qwen[0][1],
                "lama": lambda: fake.lama[0][0]}[engine]()
    assert 0.3 < float(model_in.mean()) < 0.75 and model_in.max() < 1.0   # log-encoded, not clipped linear


def test_png16_key_keeps_its_precision(work, fake, tmp_path):
    _make_key(work, ext="png16")
    _run(work, fake, "lama", tmp_path)
    v = store.load_key(work, CARD, FRAME).versions[-1]
    assert (v["ext"], v["bits"]) == ("png", 16)
    a, b = _u8(work, "v001"), _u8(work, "v002")
    assert a.dtype == b.dtype == np.uint16
    grown = RG.paste_region(_rect()) > 0
    assert np.array_equal(a[~grown], b[~grown])


def test_regenerate_starts_from_the_active_version(work, fake, tmp_path):
    _make_key(work)
    _run(work, fake, "lama", tmp_path)
    KS.undo(work, CARD, FRAME)                                      # back on v001
    r = _run(work, fake, "klein_restore", tmp_path)
    assert r["version"] == "v003"
    a, c = _u8(work, "v001"), _u8(work, "v003")
    grown = RG.paste_region(_rect()) > 0
    assert np.array_equal(a[~grown], c[~grown])


# ----------------------------------------------------------------------------- refusals
def test_refused_without_area_or_key(work, fake, tmp_path):
    with pytest.raises(ValueError, match="Brak klucza"):
        _run(work, fake, "lama", tmp_path)
    KS.add_version(work, CARD, FRAME, _background(), "plate", "png")
    with pytest.raises(ValueError, match="Brak zaznaczonego obszaru"):
        _run(work, fake, "lama", tmp_path)
    KS.save_brush_layer(work, CARD, FRAME, "a1", np.zeros((H, W), np.float32))
    with pytest.raises(ValueError, match="Brak zaznaczonego obszaru"):
        _run(work, fake, "klein_restore", tmp_path)
    with pytest.raises(ValueError, match="Nieznany silnik"):
        _run(work, fake, "void", tmp_path)
    assert not fake.client.sent and store.load_key(work, CARD, FRAME).active == "v001"


def test_refused_when_the_model_is_missing(work, fake, tmp_path, monkeypatch):
    monkeypatch.setenv("MD_MODEL_GATE", "1")
    _make_key(work)
    md = tmp_path / "models"
    md.mkdir()
    cfg = {"models_dir": str(md)}
    for engine, label in (("klein_restore", "FLUX 3 Image Klein"), ("qwen21_edit", "Qwen-Image-2.1"), ("lama", "LaMa")):
        reason = SM.cleanup3d_engines(cfg)[engine]["reason"]
        assert label in reason and "Doktor" in reason
        with pytest.raises(ValueError) as e:
            RG.regenerate(work, CARD, FRAME, engine, client=fake.client, io_dir=tmp_path / "io", cfg=cfg)
        assert str(e.value) == reason
    assert not fake.client.sent and store.load_key(work, CARD, FRAME).active == "v001"
    (md / "inpaint").mkdir()
    (md / "inpaint" / "big-lama.pt").write_bytes(b"x")
    assert RG.regenerate(work, CARD, FRAME, "lama", client=fake.client, io_dir=tmp_path / "io", cfg=cfg)["version"] == "v002"


def test_failed_prompt_adds_no_version(work, tmp_path):
    def boom(g):
        raise RuntimeError("CUDA out of memory")
    client = MockComfyClient({"KSampler": boom})
    _make_key(work)
    with pytest.raises(RuntimeError, match="FLUX 3 Image Klein \\+ LoRA Restore nie powiódł się: CUDA out of memory"):
        RG.regenerate(work, CARD, FRAME, "klein_restore", client=client, io_dir=tmp_path / "io", cfg={})
    assert store.load_key(work, CARD, FRAME).active == "v001"
    assert not list((tmp_path / "io").glob("*"))


def test_every_prompt_of_regen_runs_under_the_comfy_lease():
    """As tests/test_gpu_manager.py::test_every_comfy_prompt_runs_under_the_lease (which only walks engine/*.py)."""
    tree = ast.parse((ROOT / "engine" / "cleanup3d" / "regen.py").read_text(encoding="utf-8"))
    leased, calls = set(), []
    for w in ast.walk(tree):
        if isinstance(w, ast.With) and any(isinstance(it.context_expr, ast.Call)
                                           and getattr(it.context_expr.func, "id", None) == "comfy_lease"
                                           for it in w.items):
            leased |= {id(n) for n in ast.walk(w)}
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) in ("queue_prompt", "wait_for_prompt"):
            calls.append(n)
    assert len(calls) == 2 and all(id(n) in leased for n in calls)   # one shared runner (_comfy_image) for both models


# ----------------------------------------------------------------------------- key steps (user 2026-11-10)
def test_klein_strength_below_one_starts_from_the_tile_latent():
    g = RG.klein_graph("C:/io/t_000", "p", 64, 64, strength=0.9)
    ks = g["12"]["inputs"]
    enc = g[ks["latent_image"][0]]
    assert enc["class_type"] == "VAEEncode" and ks["denoise"] == pytest.approx(0.9)
    assert g[enc["inputs"]["pixels"][0]]["class_type"] == "VHS_LoadImagesPath"
    assert g[enc["inputs"]["vae"][0]]["class_type"] == "VAELoader"
    assert ks["positive"] == ["15", 0] and ks["negative"] == ["15", 1]         # the reference still conditions
    full = RG.klein_graph("C:/io/t_000", "p", 64, 64, strength=1.0)
    assert full["12"]["inputs"]["latent_image"] == ["15", 2] and full["12"]["inputs"]["denoise"] == 1.0
    assert not any(n["class_type"] == "VAEEncode" for n in full.values())
    assert RG.check_strength(0.3) == 0.3 and RG.check_strength(1) == 1.0
    for bad in (0.29, 1.01, float("nan")):
        with pytest.raises(ValueError, match="Siła FLUX 3 Image Klein"):
            RG.check_strength(bad)


def test_klein_strength_reaches_every_tile(work, fake, tmp_path):
    _make_key(work)
    r = _run(work, fake, "klein_restore", tmp_path, strength=0.5)
    assert r["strength"] == 0.5 and r["area"] == "marked" and r["base"] == "v001"
    assert all(g["12"]["inputs"]["denoise"] == 0.5 and g["12"]["inputs"]["latent_image"] == ["10", 0]
               for g, _t in fake.tiles)
    e = store.load_key(work, CARD, FRAME).versions[-1]
    assert e["strength"] == 0.5 and e["base"] == "v001" and "area" not in e
    with pytest.raises(ValueError, match="Siła"):
        _run(work, fake, "klein_restore", tmp_path, strength=0.1)
    _run(work, fake, "klein_restore", tmp_path)                    # marked default = 1.0 (the graph as before)
    assert fake.tiles[-1][0]["12"]["inputs"]["denoise"] == 1.0
    r = _run(work, fake, "lama", tmp_path, strength=0.1)           # other engines ignore the strength
    assert "strength" not in r and "strength" not in store.load_key(work, CARD, FRAME).versions[-1]


def test_regenerate_from_a_chosen_base_version(work, fake, tmp_path):
    _make_key(work)
    _run(work, fake, "lama", tmp_path)                              # v002, active
    r = _run(work, fake, "qwen21_edit", tmp_path, base="v001")
    assert r["version"] == "v003" and r["base"] == "v001"
    key = store.load_key(work, CARD, FRAME)
    assert key.active == "v003" and key.versions[-1]["base"] == "v001" and key.versions[1]["base"] == "v001"
    # the step started from v001: the crop the model got holds the magenta object of v001 (v002 had it removed)
    assert _is_magenta(fake.qwen[-1][1]).any()
    a, c = _u8(work, "v001"), _u8(work, "v003")
    grown = RG.paste_region(_rect()) > 0
    assert np.array_equal(a[~grown], c[~grown])
    r = _run(work, fake, "lama", tmp_path)                          # default base = the active one
    assert r["base"] == "v003"
    with pytest.raises(ValueError, match="nie ma wersji"):
        _run(work, fake, "lama", tmp_path, base="v042")


PATCH = (40, 30, 120, 90)            # x0, y0, x1, y1 of the fill a model put into the key (the plate around it)


def _patch_key(work, ext="png"):
    """A key: v001 = the plate frame, v002 = the plate + a fill in PATCH (magenta checker = what the step works on)."""
    plate = _background()
    KS.add_version(work, CARD, FRAME, plate, "plate", ext)
    img = plate.copy()
    m = _rect(PATCH) > 0
    yy, xx = np.mgrid[0:H, 0:W]
    img[m] = OLD_FILL * (0.6 + 0.4 * (((xx + yy) // 4) % 2))[m][:, None]
    KS.add_version(work, CARD, FRAME, img, "plate+fill:v001", ext)
    return plate


def test_patch_weight_is_the_difference_with_the_band(work):
    plate = _patch_key(work)
    key = store.load_key(work, CARD, FRAME)
    base = KS.read_key_image(work, key)
    hole = RG.patch_hole(base, plate)
    assert np.array_equal(hole, _rect(PATCH) > 0)
    noisy = base.copy()
    noisy[5, 5] += 2.0 / 255                                        # under the threshold: not the patch
    noisy[150, 200] += 0.3                                          # one speck: dropped
    assert np.array_equal(RG.patch_hole(noisy, plate), hole)
    holed = base.copy()
    holed[55:60, 75:80] = plate[55:60, 75:80]                       # a hole inside the fill still belongs to the patch
    assert np.array_equal(RG.patch_hole(holed, plate), hole)
    wgt = RG.patch_weight(base, plate, 8.0)
    from engine.cleanup3d import band as C3B
    assert np.array_equal(wgt, C3B.patch_weight(hole, hole, 8.0)[0])  # the patch band rule, nothing else
    far = cv2.dilate(hole.astype(np.uint8), np.ones((19, 19), np.uint8)) == 0
    assert (wgt[hole] == 1).all() and ((wgt > 0) & (wgt < 1) & ~hole).any() and not wgt[far].any()
    assert np.array_equal(RG.patch_weight(base, plate, 0.0) > 0, hole)
    assert not RG.patch_weight(plate, plate, 8.0).any()


@pytest.mark.parametrize("engine", ["klein_restore", "lama", "qwen21_edit"])
def test_patch_mode_changes_only_the_patch_and_keeps_the_plate_outside(work, fake, tmp_path, engine):
    plate = _patch_key(work)
    base = KS.read_key_image(work, store.load_key(work, CARD, FRAME))
    wgt = RG.patch_weight(base, plate, 6.0)
    r = _run(work, fake, engine, tmp_path, area_mode="patch", patch=wgt, plate=plate)
    assert r["area"] == "patch" and r["base"] == "v002" and r["version"] == "v003" and r["area_px"] == int((wgt > 0).sum())
    e = store.load_key(work, CARD, FRAME).versions[-1]
    assert e["source"] == f"regen:{engine}" and e["base"] == "v002" and e["area"] == "patch"
    new = _u8(work, "v003")
    pl8 = np.round(plate * 255).astype(np.uint8)
    out = wgt == 0
    assert np.array_equal(new[out], pl8[out]) and np.array_equal(new[out], _u8(work, "v002")[out])
    assert not np.array_equal(new[wgt > 0], _u8(work, "v002")[wgt > 0])


def test_patch_mode_klein_redraws_the_fill_from_its_own_latent(work, fake, tmp_path):
    plate = _patch_key(work)
    base = KS.read_key_image(work, store.load_key(work, CARD, FRAME))
    wgt = RG.patch_weight(base, plate, 6.0)
    seen = []
    r = _run(work, fake, "klein_restore", tmp_path, area_mode="patch", patch=wgt, plate=plate, soften_seams=True,
             blur=lambda gen, real, hole, known: (seen.append((hole.copy(), known.copy())), gen)[1])
    assert r["strength"] == 0.9 and r["passes"] == 1                # cała łata: 0.9 by default, no softening pass
    g, tile = fake.tiles[0]
    assert g["12"]["inputs"]["denoise"] == pytest.approx(0.9) and g[g["12"]["inputs"]["latent_image"][0]]["class_type"] == "VAEEncode"
    assert _is_magenta(tile).any()                                  # a REDRAW: the restore sees the existing fill
    assert len(seen) == 1 and seen[0][0].any() and seen[0][1].any()  # the blur hook of the merge ran
    assert store.load_key(work, CARD, FRAME).versions[-1]["strength"] == 0.9
    # the fake's colour cast is taken out by the tone match on the ring: over the band the result is the plate's colour
    new = _u8(work, "v003").astype(np.float32) / 255
    band = (wgt > 0) & (wgt < 1)
    assert band.any() and np.abs((new - plate)[band].mean(axis=0)).max() < 0.01


def test_patch_mode_refusals_and_extent(work, fake, tmp_path):
    plate = _patch_key(work)
    base = KS.read_key_image(work, store.load_key(work, CARD, FRAME))
    wgt = RG.patch_weight(base, plate, 6.0)
    with pytest.raises(ValueError, match="potrzebuje"):
        _run(work, fake, "lama", tmp_path, area_mode="patch")
    with pytest.raises(ValueError, match="nie ma łaty"):
        _run(work, fake, "lama", tmp_path, area_mode="patch", base="v001", plate=plate,
             patch=RG.patch_weight(plate, plate, 6.0))
    with pytest.raises(ValueError, match="Nieznany obszar"):
        _run(work, fake, "lama", tmp_path, area_mode="all")
    ext = np.zeros((H, W), np.float32)
    ext[:, :80] = 1                                                 # the card covers only the left part
    _run(work, fake, "lama", tmp_path, area_mode="patch", patch=wgt, plate=plate, extent=ext)
    a, b = _u8(work, "v002"), _u8(work, "v003")
    assert np.array_equal(a[:, 80:], b[:, 80:]) and not np.array_equal(a[:, :80], b[:, :80])


# ----------------------------------------------------------------------------- real (ComfyUI + GPU on the workstation)
BEATCH = Path(r"D:/MD_REAL_PROJECT/Documents\shots\shots\matte_desk\source\sh060\sh060_0029.png")


def _real_client():
    if not BEATCH.is_file():
        pytest.skip(f"no source image {BEATCH}")
    from comfy_client import ComfyClient
    from engine.config import load_config
    cfg = load_config()
    for port in dict.fromkeys([int(cfg.get("comfy_port") or 8188), 8188]):
        c = ComfyClient(port=port)
        if c.is_alive():
            return c, cfg
    pytest.skip("ComfyUI does not answer")


def _real_run(engine, tmp_path, monkeypatch, **kw):
    client, cfg = _real_client()
    if cfg.get("models_dir"):
        monkeypatch.setenv("MD_MODEL_GATE", "1")                    # the real gate when this machine's config is known
        st = SM.cleanup3d_engines(cfg)[engine]
        if not st["ok"]:
            pytest.skip(st["reason"])
    work = tmp_path / "shot" / "matte_desk"
    work.mkdir(parents=True)
    src = cv2.imread(str(BEATCH), cv2.IMREAD_UNCHANGED)
    h, w = src.shape[:2]
    KS.add_version(work, "sand", 29, src[..., :3], "plate", "png" if src.dtype == np.uint8 else "png16")
    m = np.zeros((h, w), np.float32)                                # an ellipse of about 9 % x 12 % of the frame
    cv2.ellipse(m, (int(w * 0.5), int(h * 0.62)), (int(w * 0.09), int(h * 0.06)), 0, 0, 360, 1.0, -1)
    KS.save_brush_layer(work, "sand", 29, "a1", m)
    key = store.load_key(work, "sand", 29)
    old = KS.read_key_image(work, key)
    t0 = time.time()
    r = RG.regenerate(work, "sand", 29, engine, client=client, io_dir=tmp_path / "io", cfg=cfg, **kw)
    key = store.load_key(work, "sand", 29)
    new = KS.read_key_image(work, key)
    region = RG.paste_region(m) > 0
    inner = m > 0.5
    diff = float(np.abs(new - old)[inner].mean())
    print(f"\nREAL regen {engine} {kw or ''}: {w}x{h}, area {r['area_px']} px, tiles {r['tiles']}, passes {r['passes']}, "
          f"{r['seconds']} s (wall {time.time() - t0:.1f} s), mean |new - old| in the area {diff:.4f}")
    keep = os.environ.get("MD_REGEN_KEEP")
    if keep:
        Path(keep).mkdir(parents=True, exist_ok=True)
        tag = engine + ("_soft" if kw.get("soften_seams") else "")
        shutil.copy2(KS.version_path(work, key), Path(keep) / f"regen_{tag}.png")
    assert key.active == "v002" and key.versions[-1]["source"] == f"regen:{engine}"
    assert np.array_equal(old[~region], new[~region])
    assert diff > 1e-3 and np.isfinite(new).all()
    # colour matched: the mean colour of the fill stays near the ring of real pixels around it
    ring = (cv2.dilate(region.astype(np.uint8), np.ones((49, 49), np.uint8)) > 0) & ~region
    assert np.abs(new[inner].mean(axis=0) - old[ring].mean(axis=0)).max() < 0.08
    return r


@pytest.mark.real
def test_real_lama_on_sh060_key(tmp_path, monkeypatch):
    _real_run("lama", tmp_path, monkeypatch)


@pytest.mark.real
def test_real_klein_restore_on_sh060_key(tmp_path, monkeypatch):
    _real_run("klein_restore", tmp_path, monkeypatch, soften_seams=bool(os.environ.get("MD_REGEN_SOFT")))


@pytest.mark.real
def test_real_qwen21_edit_on_sh060_key(tmp_path, monkeypatch):
    """Qwen-Image-2.1 edit (user 2026-11-10); MD_REGEN_KEEP=<folder> keeps the result for a look."""
    _real_run("qwen21_edit", tmp_path, monkeypatch, prompt=os.environ.get("MD_REGEN_PROMPT") or None)
