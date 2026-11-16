# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cleanup 3D – the key store (docs/PLAN_CLEANUP_3D.md, package P2).

Key images and their versions, export / import for painting outside (Photoshop), the regeneration areas of a key
(curves + brush layers -> one mask) and carrying areas marked on ANY plate frame to the keys that frame is made of
(only through the `Geometry` Protocol of engine/cleanup3d/types.py – the geometry itself is package P1).

On disk (types.key_dir): key.json, v001.png / v002.exr ... (versions; files are never removed by undo), areas.json,
a_<id>.png (brush layers, 8-bit), .last_version (the highest version number ever given). Deleting a key / a version
removes it from disk for good (engine/deletion.py, user decision 2026-11-13).
Images in memory: float32 BGR (cv2 order); 8-bit -> /255, 16-bit -> /65535, EXR as stored.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")   # before cv2 (as api/__init__.py); EXR goes through exr_io anyway

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from engine import curves as CV  # noqa: E402
from engine import deletion as DEL  # noqa: E402
from engine.cleanup3d import store  # noqa: E402
from engine.cleanup3d import types as T  # noqa: E402

PathLike = Union[str, Path]
LAST_VERSION = ".last_version"   # highest version number ever given in the key folder
# key format -> (file extension, bits)
_FMT = {"png": ("png", 8), "png16": ("png", 16), "exr": ("exr", 32)}
CARRY_PAD_PX = 0.0          # no pad: the curve itself is carried (marks_to_keys.py padded a bbox by 24 px)
BRUSH_EDGE_PX = 16.0        # brush outline -> polygon: an edge longer than this is subdivided before mapping


# ----------------------------------------------------------------------------- image io
def _to_float(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    return img.astype(np.float32)


def _bgr3(img: np.ndarray) -> np.ndarray:
    """Drop an alpha channel; anything without 3 colour channels is refused."""
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    if img.ndim != 3 or img.shape[2] != 3:
        ch = 1 if img.ndim == 2 else img.shape[2]
        raise ValueError(f"Plik musi mieć 3 kanały koloru (RGB), ma {ch}")
    return img


def _write_exr(path: Path, bgr: np.ndarray) -> None:
    import Imath
    import OpenEXR
    from engine import exr_io
    h, w = bgr.shape[:2]
    hdr = OpenEXR.Header(w, h)
    hdr["channels"] = {c: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)) for c in ("R", "G", "B")}
    hdr["compression"] = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)
    exr_io.write_exr_like(path, {"B": bgr[..., 0], "G": bgr[..., 1], "R": bgr[..., 2]}, hdr)


def write_image(path: PathLike, img: np.ndarray, fmt: str) -> Path:
    """Write BGR (float 0..1, uint8 or uint16) as `fmt` in types.KEY_FORMATS. Floats are clipped to 0..1 for PNG."""
    if fmt not in _FMT:
        raise ValueError(f"Nieznany format klucza: {fmt!r} (dozwolone: {', '.join(T.KEY_FORMATS)})")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = _bgr3(np.asarray(img))
    if fmt == "exr":
        _write_exr(path, _to_float(bgr))
        return path
    hi, dt = (255.0, np.uint8) if fmt == "png" else (65535.0, np.uint16)
    if bgr.dtype == dt:
        out = bgr
    else:
        out = np.round(np.clip(_to_float(bgr), 0.0, 1.0) * hi).astype(dt)
    if not cv2.imwrite(str(path), np.ascontiguousarray(out)):
        raise ValueError(f"Nie udało się zapisać {path.name}")
    return path


def read_image(path: PathLike) -> Tuple[np.ndarray, str]:
    """Any image file -> (float32 BGR, the key format it maps to). Alpha dropped. ValueError (Polish) when unreadable."""
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"Nie ma pliku {p.name}")
    if p.suffix.lower() == ".exr":
        from engine import exr_io
        try:
            ch, _ = exr_io.read_exr(p)
        except Exception as e:  # noqa: BLE001 – OpenEXR raises plain OSError / generic errors
            raise ValueError(f"Nie można odczytać pliku {p.name} ({e})")
        try:
            r, g, b = exr_io.rgb_channel_names(ch)
        except KeyError:
            raise ValueError(f"Plik musi mieć 3 kanały koloru (RGB), ma kanały: {', '.join(sorted(ch)) or 'brak'}")
        return np.dstack([ch[b], ch[g], ch[r]]).astype(np.float32), "exr"
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Nie można odczytać pliku {p.name} (uszkodzony albo nieobsługiwany format)")
    img = _bgr3(img)
    fmt = "png" if img.dtype == np.uint8 else "png16" if img.dtype == np.uint16 else "exr"
    return _to_float(img), fmt


# ----------------------------------------------------------------------------- versions
def _require_key(work: PathLike, card: str, frame: int) -> T.Key:
    k = store.load_key(Path(work), card, frame)
    if k is None or not k.versions:
        raise ValueError(f"Brak klucza {card} / klatka {int(frame)}")
    return k


def _version(key: T.Key, v: Optional[str]) -> Dict:
    v = v or key.active
    for e in key.versions:
        if e.get("v") == v:
            return e
    raise ValueError(f"Klucz {key.card} / klatka {key.frame}: nie ma wersji {v!r}")


def version_path(work: PathLike, key: T.Key, version: Optional[str] = None) -> Path:
    e = _version(key, version)
    return T.key_dir(Path(work), key.card, key.frame) / f"{e['v']}.{e['ext']}"


def add_version(work: PathLike, card: str, frame: int, image: np.ndarray, source: str, ext: str = "png",
                prompt: Optional[str] = None, base: Optional[str] = None,
                extra: Optional[Mapping[str, Any]] = None) -> T.Key:
    """Write a NEW version of the key (created when missing) and make it active. `ext` in types.KEY_FORMATS
    (png = 8-bit, png16, exr float32); `prompt`: the description a generation used (kept in the version, the UI puts it
    back in the prompt field); `base`: the version a key step started from (user 2026-11-10: every step starts from a
    chosen base version, so steps can be re-run from the same base and compared); `extra`: more fields of the version
    entry (e.g. the regeneration's area mode and strength). Returns the updated Key."""
    work = Path(work)
    if ext not in _FMT:
        raise ValueError(f"Nieznany format klucza: {ext!r} (dozwolone: {', '.join(T.KEY_FORMATS)})")
    kd = T.key_dir(work, card, frame)
    key = store.load_key(work, card, frame) or T.Key(T.check_name(card), int(frame))
    n = 1 + max([int(str(e["v"])[1:]) for e in key.versions] + [last_version_number(work, card, frame)])
    # a stray file (a crash between image and json) is never overwritten; the number of a deleted version is never
    # reused (another version's `base` / `source` may still name it, browsers cache images by it)
    while any(kd.glob(f"{T.version_name(n)}.*")):
        n += 1
    v = T.version_name(n)
    file_ext, bits = _FMT[ext]
    write_image(kd / f"{v}.{file_ext}", image, ext)
    _remember_version(kd, n)
    entry = {"v": v, "ext": file_ext, "bits": bits, "source": str(source), "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    if prompt:
        entry["prompt"] = str(prompt)
    if base:
        entry["base"] = str(base)
    for k, val in (extra or {}).items():
        if k not in entry and val is not None:
            entry[str(k)] = val
    key.versions.append(entry)
    key.active = v
    store.save_key(work, key)
    return key


def _step(work: PathLike, card: str, frame: int, d: int) -> T.Key:
    key = _require_key(work, card, frame)
    names = [e["v"] for e in key.versions]
    i = names.index(key.active) if key.active in names else len(names) - 1
    j = i + d
    if not 0 <= j < len(names):
        raise ValueError("Nie ma wcześniejszej wersji klucza" if d < 0 else "Nie ma późniejszej wersji klucza")
    key.active = names[j]
    store.save_key(Path(work), key)
    return key


def undo(work: PathLike, card: str, frame: int) -> T.Key:
    """Active version one back (files stay, so `redo` brings it back)."""
    return _step(work, card, frame, -1)


def redo(work: PathLike, card: str, frame: int) -> T.Key:
    return _step(work, card, frame, +1)


def set_active(work: PathLike, card: str, frame: int, version: str) -> T.Key:
    key = _require_key(work, card, frame)
    key.active = _version(key, version)["v"]
    store.save_key(Path(work), key)
    return key


def read_key_image(work: PathLike, key: T.Key, version: Optional[str] = None) -> np.ndarray:
    """Float32 BGR of the active (or the given) version."""
    img, _ = read_image(version_path(work, key, version))
    return img


def last_version_number(work: PathLike, card: str, frame: int) -> int:
    """Highest version number ever given in this key: its version files, `.last_version` (written by add_version /
    delete_version) and - read only, for versions deleted before 2026-11-13 - their `keyver_*` files in the old
    `<work>/inpaint/trash/`. 0 = no key yet."""
    work = Path(work)
    kd = T.key_dir(work, card, frame)
    n = max([int(p.stem[1:]) for p in kd.glob("v[0-9][0-9][0-9]*.*") if p.stem[1:].isdigit()] + [0])
    try:
        n = max(n, int((kd / LAST_VERSION).read_text(encoding="utf-8").strip() or 0))
    except (OSError, ValueError):
        pass
    old = T.root_dir(work) / "trash"   # legacy, read only: never created or emptied here
    if old.is_dir():
        pat = re.compile(rf"keyver_{re.escape(card)}_{re.escape(kd.name)}_v(\d+)_")
        for f in old.glob(f"keyver_{card}_{kd.name}_v*_*.json"):
            m = pat.match(f.name)
            if m:
                n = max(n, int(m.group(1)))
    return n


def _remember_version(kd: Path, n: int) -> None:
    try:
        cur = int((kd / LAST_VERSION).read_text(encoding="utf-8").strip() or 0)
    except (OSError, ValueError):
        cur = 0
    if n > cur:
        kd.mkdir(parents=True, exist_ok=True)
        (kd / LAST_VERSION).write_text(str(n), encoding="utf-8")


def delete_key(work: PathLike, card: str, frame: int) -> Optional[int]:
    """Delete the key folder (all its versions, areas, warps) from disk for good (user decision 2026-11-13).
    Returns the bytes freed (None = no key)."""
    work = Path(work)
    kd = T.key_dir(work, card, frame)
    if not kd.is_dir():
        return None
    return DEL.remove(kd, inside=T.root_dir(work))


def delete_version(work: PathLike, card: str, frame: int, version: str) -> int:
    """Delete ONE version of a key from disk for good (user 2026-11-10: key versions must not pile up; user decision
    2026-11-13: no trash). Its number stays taken (`.last_version`), so a later version never gets its name. Refused
    (ValueError, Polish): no such version, the ACTIVE version (switch to another one first), the last remaining version
    (delete the key instead). Returns the bytes freed."""
    work = Path(work)
    key = _require_key(work, card, frame)
    e = _version(key, version)
    if e["v"] == key.active:
        raise ValueError(f"Wersja {e['v']} jest aktywna – najpierw ustaw aktywną inną wersję klucza")
    if len(key.versions) <= 1:
        raise ValueError("To ostatnia wersja klucza – usuń cały klucz zamiast niej")
    kd = T.key_dir(work, card, frame)
    _remember_version(kd, last_version_number(work, card, frame))   # before the delete: the number stays taken
    key.versions = [x for x in key.versions if x.get("v") != e["v"]]
    store.save_key(work, key)
    return DEL.remove(kd / f"{e['v']}.{e['ext']}", inside=kd)


def frames_every(work_frames: Iterable[int], n: int) -> List[int]:
    """Key mode "every N frames": every n-th frame of the range from its first, always with the first and the last."""
    fr = sorted({int(f) for f in work_frames})
    if not fr:
        return []
    n = max(1, int(n))
    out = [f for f in fr if (f - fr[0]) % n == 0]
    if out[-1] != fr[-1]:
        out.append(fr[-1])
    return out


def create_keys(work: PathLike, card: str, frames: Iterable[int],
                plate_loader: Callable[[int], np.ndarray], ext: str = "png16") -> List[T.Key]:
    """First version of each key = the plate frame itself (`source: "plate"`). A key that already has versions is left
    as it is (returned unchanged). `plate_loader(frame)` -> BGR (float 0..1 / uint8 / uint16). Default 16-bit PNG
    (a float plate keeps its precision without the size of a float EXR; pass "exr" for linear plates > 1)."""
    out = []
    for f in sorted({int(x) for x in frames}):
        k = store.load_key(Path(work), card, f)
        if k is not None and k.versions:
            out.append(k)
            continue
        out.append(add_version(work, card, f, plate_loader(f), "plate", ext))
    return out


# ----------------------------------------------------------------------------- the plate + a fill only in the hole
REBASE_GROW_PX = 40.0       # how far past the removed objects' mattes the fill is taken (shadow edge, kicked sand, halo)
REBASE_SOFT_PX = 24.0       # soft ramp outward from that region (smoothstep), over plate the tone was matched on
REBASE_RING_PX = 48.0       # ring outside the region where plate − fill is measured
REBASE_SOURCE = "plate+fill"
EFFECT_THR = 0.10           # plate vs fill (0..1, after one global gain per channel): above = the object's effect
EFFECT_EDGE_PX = 12.0       # margin around the found effects


def fill_region(mask: np.ndarray, grow_px: float = REBASE_GROW_PX, soft_px: float = REBASE_SOFT_PX) -> np.ndarray:
    """0..1 weight of the fill: the removed objects' mask (any range, > 2 % counts) grown by `grow_px`, with a soft edge
    of `soft_px` that lies OUTSIDE the grown mask (the mask itself stays fully the fill)."""
    m = np.asarray(mask)
    m = (m.astype(np.float32) / (255.0 if m.dtype == np.uint8 else 1.0) > 0.02).astype(np.uint8)
    if not m.any():
        return np.zeros(m.shape, np.float32)
    g, s = max(0.0, float(grow_px)), max(0.0, float(soft_px))
    d = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)             # px outside the mask
    if s <= 0:
        return (d <= g).astype(np.float32)
    t = np.clip((g + s - d) / s, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _diffuse(val: np.ndarray, wgt: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth multi-scale normalised convolution of `val` known where `wgt` > 0 (sum over the scales: a hard switch
    between scales draws contour-shaped steps inside the hole)."""
    num = np.zeros_like(val)
    den = np.zeros(val.shape[:2] + (1,), np.float32)
    for sc in (sigma, sigma * 2, sigma * 4, sigma * 8, sigma * 16, sigma * 32, sigma * 64):
        num += cv2.GaussianBlur(val * wgt[..., None], (0, 0), sc)
        den += cv2.GaussianBlur(wgt, (0, 0), sc)[..., None]
    return num / np.maximum(den, 1e-6)


def match_tone(fill: np.ndarray, plate: np.ndarray, hole: np.ndarray, ring_px: float = REBASE_RING_PX) -> np.ndarray:
    """`fill` + the difference plate − fill measured on a ring of `ring_px` just outside `hole` (bool / 0..1 > 0.5) and
    diffused into the hole at 1/8 size: at the hole's boundary the fill has the plate's brightness and colour, so the
    soft edge over the ring shows no step (a video model re-exposes the whole frame). The method of the sh060
    experiment (removers_test/void_comp.py). Float32 BGR."""
    out = np.asarray(fill, np.float32)
    pl = np.asarray(plate, np.float32)
    m = (np.asarray(hole) > 0.5)
    r = max(1, int(round(ring_px)))
    ring = (cv2.dilate(m.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0) & ~m
    if not m.any() or not ring.any():
        return out.copy()
    h, w = m.shape
    ds = 8 if min(h, w) >= 256 else 1
    sm = (max(1, w // ds), max(1, h // ds))
    d = _diffuse(cv2.resize(pl - out, sm, interpolation=cv2.INTER_AREA),
                 cv2.resize(ring.astype(np.float32), sm, interpolation=cv2.INTER_AREA), max(1.0, r / 2.0 / ds))
    return out + (cv2.resize(d, (w, h), interpolation=cv2.INTER_CUBIC) if ds > 1 else d)


def effects_mask(plate: np.ndarray, fill: np.ndarray, mask: np.ndarray, grow_px: float = REBASE_GROW_PX,
                 thr: float = EFFECT_THR) -> np.ndarray:
    """uint8 0 / 255: the removed objects (grown by `grow_px`) + their EFFECTS the model removed with them – shadow,
    kicked-up sand, reflection: where the fill differs from the plate by more than `thr` AND the difference touches
    the objects. Measured at ~1/4 size after ONE gain per channel (medians outside the objects): a local match would
    hide a shadow, and a model's repaint of the texture stays below the threshold. Float32 BGR 0..1 inputs."""
    base = fill_region(mask, grow_px, 0.0) > 0
    if not base.any() or base.all():
        return base.astype(np.uint8) * 255
    h, w = base.shape
    q = max(1, int(round(max(h, w) / 960.0)))
    out = ~base
    gain = np.array([float(np.median(plate[..., c][out])) / max(float(np.median(fill[..., c][out])), 1e-4)
                     for c in range(3)], np.float32)
    small = (max(1, w // q), max(1, h // q))
    d = np.abs(cv2.resize(fill * gain, small, interpolation=cv2.INTER_AREA)
               - cv2.resize(plate, small, interpolation=cv2.INTER_AREA)).max(axis=2)
    d = cv2.GaussianBlur(d, (0, 0), 3)
    bs = cv2.resize(base.astype(np.uint8), small, interpolation=cv2.INTER_NEAREST)
    _n, lab = cv2.connectedComponents(((d > float(thr)) | (bs > 0)).astype(np.uint8))
    keep = np.unique(lab[bs > 0])
    eff = np.isin(lab, keep[keep > 0]).astype(np.uint8)
    return ((cv2.resize(eff, (w, h), interpolation=cv2.INTER_NEAREST) > 0) | base).astype(np.uint8) * 255


def patch_on_plate(plate: np.ndarray, fill: np.ndarray, mask: np.ndarray, grow_px: float = REBASE_GROW_PX,
                   soft_px: float = REBASE_SOFT_PX, tone: bool = True, effects: bool = False,
                   blur: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = None,
                   weight: Optional[np.ndarray] = None) -> np.ndarray:
    """The plate everywhere, `fill` (a frame with the objects already removed) only in the removed objects' region
    (`fill_region`; with `effects` also their shadow / kicked sand – `effects_mask`), tone-matched to the plate around
    it. `blur(fill, plate, hole) -> fill`: the motion blur of the generated patch matched to the plate before the tone
    (render.match_patch_blur through the API). `weight`: a ready 0..1 patch weight (e.g. the patch band of
    engine/cleanup3d/band.py) used instead of the soft ramp of `soft_px` (the hole is still `mask` grown by `grow_px`).
    Float32 BGR; outside the region the plate bit for bit."""
    pl = _bgr3(_to_float(np.asarray(plate)))
    fl = _bgr3(_to_float(np.asarray(fill)))
    if fl.shape[:2] != pl.shape[:2]:
        raise ValueError(f"Wypełnienie {fl.shape[1]}×{fl.shape[0]} ma inny rozmiar niż plate {pl.shape[1]}×{pl.shape[0]}")
    if np.asarray(mask).shape[:2] != pl.shape[:2]:
        raise ValueError("Maska usuwanych obiektów ma inny rozmiar niż plate")
    if effects:
        hole = fill_region(effects_mask(pl, fl, mask, grow_px), EFFECT_EDGE_PX, 0.0) > 0
    else:
        hole = fill_region(mask, grow_px, 0.0) > 0
    if weight is not None:
        a = np.clip(np.asarray(weight, np.float32), 0.0, 1.0)
        if a.shape != pl.shape[:2]:
            raise ValueError("Waga łaty ma inny rozmiar niż plate")
    else:
        a = fill_region(hole, 0.0, soft_px)
    if blur is not None:
        fl = np.asarray(blur(fl, pl, hole), np.float32)
    if not (a > 0).any():
        return pl.copy()
    if tone:
        fl = match_tone(fl, pl, hole)
    out = pl.copy()
    sel = a > 0
    out[sel] = pl[sel] * (1.0 - a[sel])[:, None] + fl[sel] * a[sel][:, None]
    return out


def rebase_on_plate(work: PathLike, card: str, frame: int, plate: np.ndarray, mask: np.ndarray,
                    grow_px: float = REBASE_GROW_PX, soft_px: float = REBASE_SOFT_PX, effects: bool = True,
                    again: bool = False,
                    blur: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = None) -> Optional[T.Key]:
    """A new active version of the key: the plate frame + the present active version only where the removed objects
    (and, with `effects`, their shadow / kicked sand) are (`source: "plate+fill:<version>"`, same precision). None
    (nothing written) when the active version already is the plate, or the key has its own canvas (not the plate's
    pixels), or it already is such a patch – unless `again`: then it is built anew from the version it was made of."""
    key = _require_key(work, card, frame)
    e = _version(key, None)
    src = str(e.get("source") or "")
    if key.canvas is not None or src == "plate":
        return None
    if src.startswith(REBASE_SOURCE):
        if not again:
            return None
        made_of = src.split(":", 1)[1] if ":" in src else None
        if made_of and made_of not in [x.get("v") for x in key.versions]:
            return None                            # the version it was made of was deleted: nothing to rebuild
        e = _version(key, made_of)
    img, fmt = read_image(version_path(work, key, e["v"]))
    out = patch_on_plate(plate, img, mask, grow_px, soft_px, effects=effects, blur=blur)
    return add_version(work, card, frame, out, f"{REBASE_SOURCE}:{e['v']}", fmt)


# ----------------------------------------------------------------------------- export / import
def export_key(work: PathLike, card: str, frame: int, fmt: str, out_dir: PathLike,
               version: Optional[str] = None) -> Path:
    """Write the active (or given) version for painting outside -> the written path."""
    if fmt not in _FMT:
        raise ValueError(f"Nieznany format klucza: {fmt!r} (dozwolone: {', '.join(T.KEY_FORMATS)})")
    key = _require_key(work, card, frame)
    v = _version(key, version)["v"]
    img = read_key_image(work, key, v)
    suffix = "_16bit" if fmt == "png16" else ""
    path = Path(out_dir) / f"{card}_k{int(frame):04d}_{v}{suffix}.{_FMT[fmt][0]}"
    return write_image(path, img, fmt)


def _size_msg(fw: int, fh: int, kw: int, kh: int) -> str:
    same_aspect = abs(fw * kh - fh * kw) <= max(fw, kw)        # within a pixel of the same proportions
    what = "Inny rozmiar" if same_aspect else "Inny rozmiar i proporcje"
    hint = " – przeskaluj do rozmiaru klucza" if same_aspect else ""
    return f"{what} niż klucz: plik {fw}×{fh}, klucz {kw}×{kh}{hint}"


def import_key(work: PathLike, card: str, frame: int, path: PathLike) -> T.Key:
    """Painted file -> a NEW version of the key (`source: "import:<file name>"`, stored in the file's precision).
    Refused (ValueError, Polish): unreadable file, other size than the key, not 3 colour channels (alpha is dropped)."""
    key = _require_key(work, card, frame)
    p = Path(path)
    img, fmt = read_image(p)
    ref = version_path(work, key)
    if ref.suffix.lower() == ".exr":
        kh, kw = read_key_image(work, key).shape[:2]
    else:
        hdr = cv2.imread(str(ref), cv2.IMREAD_UNCHANGED)
        kh, kw = hdr.shape[:2]
    if img.shape[:2] != (kh, kw):
        raise ValueError(_size_msg(img.shape[1], img.shape[0], kw, kh))
    if fmt == "png":
        img = np.round(img * 255.0).astype(np.uint8)          # exact 8-bit values, no float round trip
    elif fmt == "png16":
        img = np.round(img * 65535.0).astype(np.uint16)
    return add_version(work, card, frame, img, f"import:{p.name}", fmt)


# ----------------------------------------------------------------------------- areas -> mask
def brush_name(area_id: str) -> str:
    return f"a_{T.check_name(area_id)}.png"


def save_brush_layer(work: PathLike, card: str, frame: int, area_id: str, mask: np.ndarray, op: str = "add",
                     label: str = "") -> T.Area:
    """Write brush layer `a_<id>.png` (mask 0..1 float or 0..255 uint8, > 0.5 / 127 = painted) next to the key and
    register it in areas.json (appended at the end when new; an existing area keeps its place and op)."""
    work = Path(work)
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    m8 = m.astype(np.uint8) if m.dtype == np.uint8 else (np.clip(m.astype(np.float32), 0, 1) * 255 + 0.5).astype(np.uint8)
    name = brush_name(area_id)
    kd = T.key_dir(work, card, frame)
    kd.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(kd / name), m8):
        raise ValueError(f"Nie udało się zapisać {name}")
    areas = store.load_areas(work, card, frame)
    for a in areas:
        if a.id == area_id:
            a.kind, a.png, a.curve = "brush", name, None
            store.save_areas(work, card, frame, areas)
            return a
    a = T.Area(T.check_name(area_id), "brush", op, None, name, label)
    areas.append(a)
    store.save_areas(work, card, frame, areas)
    return a


def curve_polygon(curve: Dict, step: float = 0.5) -> np.ndarray:
    """(n, 2) dense points of a closed curve (engine/curves.py sampling)."""
    sp = CV.normalize(curve)
    sp["closed"] = True
    xy, _, _ = CV.sample(sp, step)
    return xy


def fill_polygon(shape: Tuple[int, int], xy: np.ndarray) -> np.ndarray:
    """Binary uint8 mask of a polygon in pixel-centre coordinates (subpixel vertices, cv2 shift 4)."""
    m = np.zeros(shape[:2], np.uint8)
    if len(xy) >= 3 and np.isfinite(xy).all():
        pts = np.round(np.asarray(xy, np.float64) * 16.0).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(m, [pts], 1, lineType=cv2.LINE_8, shift=4)
    return m


def area_layer(area: T.Area, shape: Tuple[int, int], layer_dir: Optional[Path] = None,
               brush: Optional[Mapping[str, np.ndarray]] = None) -> np.ndarray:
    """One area -> float32 (h, w) 0..1. Brush layers: from `brush[id]` when given, else `layer_dir / area.png`."""
    h, w = shape[:2]
    if area.kind == "curve":
        return fill_polygon((h, w), curve_polygon(area.curve)).astype(np.float32)
    if brush is not None and area.id in brush:
        m = np.asarray(brush[area.id])
    else:
        p = (layer_dir or Path(".")) / str(area.png)
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise ValueError(f"Brak warstwy pędzla {area.png}")
    if m.ndim == 3:
        m = m[..., 0]
    m = _to_float(m) if m.dtype in (np.uint8, np.uint16) else m.astype(np.float32)
    if m.shape != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(m, 0.0, 1.0)


def render_areas(areas: Sequence[T.Area], shape: Tuple[int, int], layer_dir: Optional[Path] = None,
                 brush: Optional[Mapping[str, np.ndarray]] = None) -> np.ndarray:
    """Layers in order: add = max, remove = multiply by (1 - layer). Float32 0..1."""
    out = np.zeros(shape[:2], np.float32)
    for a in areas:
        m = area_layer(a, shape, layer_dir, brush)
        out = np.maximum(out, m) if a.op == "add" else out * (1.0 - m)
    return out


def areas_mask(work: PathLike, card: str, frame: int, shape: Tuple[int, int]) -> np.ndarray:
    """The key's regeneration mask: layers of areas.json in order (curves + brush layers a_<id>.png)."""
    work = Path(work)
    return render_areas(store.load_areas(work, card, frame), shape, T.key_dir(work, card, frame))


# ----------------------------------------------------------------------------- marks on any frame -> keys
def keys_for_frame(frame: int, keys_of_card: Sequence[T.Key]) -> List[T.Key]:
    """The frame's own key if it is one, else the two nearest keys of the card (one before, one after; one at the
    ends of the range)."""
    frame = int(frame)
    own = [k for k in keys_of_card if k.frame == frame]
    if own:
        return own[:1]
    before = [k for k in keys_of_card if k.frame < frame]
    after = [k for k in keys_of_card if k.frame > frame]
    out = []
    if before:
        out.append(max(before, key=lambda k: k.frame))
    if after:
        out.append(min(after, key=lambda k: k.frame))
    return out


def _map_curve(curve: Dict, fn: Callable[[np.ndarray], np.ndarray]) -> Optional[Dict]:
    """Map the points and the handle ends of a curve; handles become relative again. Points that miss the card are
    dropped, a handle that misses becomes 0. None when < 3 points are left."""
    sp = CV.normalize(curve)
    P = sp["pts"]
    if not P:
        return None
    p = np.array([q["p"] for q in P], np.float64)
    a = p + np.array([q["in"] for q in P], np.float64)
    b = p + np.array([q["out"] for q in P], np.float64)
    m = np.asarray(fn(np.vstack([p, a, b]).astype(np.float32)), np.float64).reshape(-1, 2)
    n = len(P)
    mp, ma, mb = m[:n], m[n:2 * n], m[2 * n:]
    out = []
    for i, q in enumerate(P):
        if not np.isfinite(mp[i]).all():
            continue
        tin = ma[i] - mp[i] if np.isfinite(ma[i]).all() else np.zeros(2)
        tout = mb[i] - mp[i] if np.isfinite(mb[i]).all() else np.zeros(2)
        out.append(CV.point(mp[i], tin, tout, q.get("cusp", False)))
    if len(out) < 3:
        return None
    return CV.spline(out, True)


def brush_outlines(mask: np.ndarray, edge_px: float = BRUSH_EDGE_PX) -> List[np.ndarray]:
    """Outer outlines of a brush layer as (n, 2) polygons in pixel-centre coordinates, long edges subdivided so a
    non-linear mapping bends them. Holes inside a stroke are filled (a carried brush area is a closed curve)."""
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    m = ((m.astype(np.float32) / (255.0 if m.dtype == np.uint8 else 1.0)) > 0.5).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cs:
        if len(c) < 3:
            continue
        c = cv2.approxPolyDP(c, 0.5, True).reshape(-1, 2).astype(np.float64)
        if len(c) < 3:
            continue
        # contour runs through the centres of the edge pixels; the painted area reaches half a pixel further
        pts = []
        for i in range(len(c)):
            p0, p1 = c[i], c[(i + 1) % len(c)]
            k = max(1, int(np.ceil(np.linalg.norm(p1 - p0) / max(edge_px, 1.0))))
            for j in range(k):
                pts.append(p0 + (p1 - p0) * j / k)
        out.append(_grow_half(np.array(pts)))
    return out


def _grow_half(pts: np.ndarray) -> np.ndarray:
    """Push every outline vertex 0.5 px outward (along the vertex normal), so filling the polygon gives back the
    painted pixels instead of a mask one pixel thinner."""
    prev, nxt = np.roll(pts, 1, 0), np.roll(pts, -1, 0)
    t = nxt - prev
    nrm = np.stack([t[:, 1], -t[:, 0]], 1)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.where(ln > 0, nrm / np.maximum(ln, 1e-9), 0.0)
    area2 = np.sum(pts[:, 0] * nxt[:, 1] - nxt[:, 0] * pts[:, 1])
    if area2 < 0:
        nrm = -nrm
    # y down: positive signed area = clockwise on screen, where (ty, -tx) points outward
    return pts + 0.5 * nrm


def _polygon_curve(pts: np.ndarray) -> Dict:
    """Closed polygon as an engine/curves.py curve (zero handles = straight edges, cusps)."""
    return CV.spline([CV.point(p, (0, 0), (0, 0), True) for p in pts], True)


def carry_areas(geometry: T.Geometry, areas: Sequence[T.Area], frame: int, keys_of_card: Sequence[T.Key],
                card: T.Card, brush: Optional[Union[Mapping[str, np.ndarray], PathLike]] = None
                ) -> Dict[int, List[T.Area]]:
    """Areas drawn in plate frame `frame` -> {key frame: areas in that key's pixels}, only through
    `geometry.points_frame_to_key`. Keys: `keys_for_frame`. Curves: points and handle ends mapped. Brush layers:
    outline polygon mapped and stored as a closed curve (kind "curve", id `<id>` or `<id>_<n>` for several strokes).
    `brush`: the masks of the brush areas (id -> mask) or the folder holding their `a_<id>.png` files."""
    out: Dict[int, List[T.Area]] = {}
    outlines: Dict[str, List[np.ndarray]] = {}
    for a in areas:
        if a.kind != "brush":
            continue
        if isinstance(brush, Mapping) and a.id in brush:
            m = np.asarray(brush[a.id])
        else:
            p = Path(brush) / str(a.png) if brush is not None and not isinstance(brush, Mapping) else None
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p is not None else None
            if m is None:
                raise ValueError(f"Brak warstwy pędzla {a.png}")
        outlines[a.id] = brush_outlines(m)
    for k in keys_for_frame(frame, keys_of_card):
        fn = (lambda pts, _k=k: geometry.points_frame_to_key(np.asarray(pts, np.float32), int(frame), _k, card))
        res: List[T.Area] = []
        for a in areas:
            if a.kind == "curve":
                c = _map_curve(a.curve, fn)
                if c is not None:
                    res.append(T.Area(a.id, "curve", a.op, c, None, a.label))
                continue
            polys = outlines.get(a.id) or []
            for i, poly in enumerate(polys):
                c = _map_curve(_polygon_curve(poly), fn)
                if c is None:
                    continue
                aid = a.id if len(polys) == 1 else T.check_name(f"{a.id}_{i + 1}"[:64])
                res.append(T.Area(aid, "curve", a.op, c, None, a.label))
        out[k.frame] = res
    return out
