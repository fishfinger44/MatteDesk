# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cleanup 3D – the geometry (docs/PLAN_CLEANUP_3D.md, package P1): a solved Camera3D + the cards -> sampling maps
between plate frames and keys, the partition of a frame between the cards, a key's coverage. Implements the `Geometry`
Protocol of engine/cleanup3d/types.py.

Ported from the experiment removers_test/ (cards.py `planes` / `card_map`, master_lib.py `project` / `rays`, vkey3d.py
`KCOV`); the per-pixel maths lives ONCE in engine/cleanup3d/geom_gpu.py, written for a backend (numpy here, torch float64
in the GPU worker), so the CPU and the GPU path compute the same formulas.

- A card = plane n.X + d = 0 (normal toward the cameras) with the user's sliders `adjust` (dh along the normal, pitch /
  roll in degrees about the card's own axes through the point under the middle camera - as cards.py), or the sky at
  infinity (rotation only), or (P18) a DEPTH card: the surface of its reference frame's depth map (depth["frame"],
  snapshot `depth["file"]`, scale / offset / sliders -> metric depth); rays are cast onto it by geom_gpu.depth_cast
  (z-buffer guess + Newton, MISS where the reference does not see the surface) and the card spec carries the approximate
  plane (n, d) for the previews' grid. Its "hit" (card_weights) = where the cast succeeds.
- A key without `canvas` = the plate frame `key.frame` as shot (camera K + distortion); with `canvas` = its own pinhole
  view {"w", "h", "K", "frame"} (pose of that plate frame, no distortion).
- Card EXTENT: areas (closed Bezier curves of engine/curves.py, brush layers passed in as arrays) drawn in the plate
  frame `extent_frame`; they lie on the card, so the camera carries them to every frame. Rasterised once on a canvas
  with a margin around the frame (curves may reach outside the picture), sampled through the card. No `add` layer =
  the whole card (minus `remove` layers).
- card_weight: cards in order (0 = front); a pixel belongs to the first card whose plane its ray hits in front of the
  camera, inside that card's extent; the hard ownership is softened by a box of `seam_px` (linear ramp across the
  border to a neighbour) and renormalised: weights sum to 1 wherever a card is hit, 0 elsewhere.

device "cpu" (NumPy, default) or "gpu" (the worker engine/cleanup3d/geom_gpu.py in ComfyUI's Python, every call under
GPU.use; arrays through .npy files). Maps are float32, (h, w), cv2.remap convention; MISS (< 0) = no data.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from engine.cleanup3d import geom_gpu as K_
from engine.cleanup3d.types import Area, Camera3D, Card, Key
from engine.worker_proto import WorkerDied, WorkerProcess, WorkerTimeout

MISS = K_.MISS
WORKER = Path(K_.__file__).resolve()
EXTENT_MARGIN = 0.5          # canvas of an extent = the frame + this fraction of its size on every side


class GeomGpuError(RuntimeError):
    """The geometry GPU worker failed (did not start, crashed twice, returned an error)."""


# ----------------------------------------------------------------------------------------------- plane of a card
def _rot(axis: np.ndarray, deg: float) -> np.ndarray:
    a = np.radians(deg)
    axis = axis / np.linalg.norm(axis)
    Kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * Kx + (1 - np.cos(a)) * Kx @ Kx


def card_plane(camera: Camera3D, card: Card) -> Optional[Tuple[np.ndarray, float]]:
    """(n, d) of a plane card with its `adjust` applied (removers_test/cards.py `planes`), None for the sky.
    The base normal is turned toward the cameras; pivot = the base plane's point under the middle camera; pitch
    turns about the card's across axis, roll about its forward axis (mean camera forward projected on the card)."""
    if card.kind == "sky":
        return None
    if card.kind == "depth" and not card.plane:
        raise ValueError(f"karta głębi {card.id}: brak przybliżonej płaszczyzny – „Bieżąca klatka” w palecie karty (krok „Karty”)")
    n0 = np.asarray(card.plane["n"], np.float64)
    nn = np.linalg.norm(n0)
    if nn <= 0:
        raise ValueError(f"karta {card.id}: zerowa normalna")
    n0, d0 = n0 / nn, float(card.plane["d"]) / nn
    frames = camera.frames
    if not frames:
        raise ValueError("camera3d: brak klatek")
    centres = np.array([camera.centre(f) for f in frames])
    if np.mean(centres @ n0 + d0) < 0:
        n0, d0 = -n0, -d0
    if card.kind == "depth":                              # the approximate plane of a depth surface: no sliders
        return n0, d0
    adj = {"dh": 0.0, "pitch": 0.0, "roll": 0.0, **(card.adjust or {})}
    if not any(float(adj[k]) for k in ("dh", "pitch", "roll")):
        return n0, d0
    c = centres[len(frames) // 2]
    pivot = c - (n0 @ c + d0) * n0
    fwd = np.mean([camera.R[f].T @ np.array([0, 0, 1.0]) for f in frames], 0)
    ax_u = fwd - (fwd @ n0) * n0
    if np.linalg.norm(ax_u) < 1e-9:                       # cameras looking straight along the normal
        ax_u = np.cross(n0, [1.0, 0, 0]) if abs(n0[0]) < 0.9 else np.cross(n0, [0, 1.0, 0])
    ax_u /= np.linalg.norm(ax_u)
    ax_v = np.cross(n0, ax_u)
    Rm = _rot(ax_v, float(adj["pitch"])) @ _rot(ax_u, float(adj["roll"]))
    n = Rm @ n0
    p = pivot + float(adj["dh"]) * n0
    return n, float(-(n @ p))


# ----------------------------------------------------------------------------------------------- extent canvas
def extent_canvas(card: Card, width: int, height: int, layers: Optional[Mapping[str, np.ndarray]] = None,
                  margin: float = EXTENT_MARGIN) -> Optional[Tuple[np.ndarray, int, int, int]]:
    """(canvas uint8 (H, W), ox, oy, base) of a card's extent in the plate pixels of `extent_frame` (pixel (x, y) of
    the frame = canvas (x + ox, y + oy)); base = the value outside the canvas (1 when no layer adds, i.e. the extent
    starts as the whole card). None = no extent (the whole card). `layers`: area id -> brush mask (uint8 / float 0..1,
    the frame's size or resized to it)."""
    if not card.extent:
        return None
    import cv2
    from engine import curves
    ox, oy = int(round(width * margin)), int(round(height * margin))
    W, H = width + 2 * ox, height + 2 * oy
    base = 0 if any(a.op == "add" for a in card.extent) else 1
    canvas = np.full((H, W), float(base), np.float32)
    for a in card.extent:
        if a.kind == "curve":
            sp = curves.normalize(a.curve)
            if not sp.get("closed"):
                raise ValueError(f"zasięg karty {card.id}: krzywa {a.id} nie jest zamknięta")
            xy, _u, _p = curves.sample(sp, step=1.0)
            pts = np.round((xy + [ox, oy]) * 16).astype(np.int32)[None]
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, pts, 255, lineType=cv2.LINE_8, shift=4)
            lay = m.astype(np.float32) / 255.0
        else:
            arr = None if layers is None else layers.get(a.id)
            if arr is None:
                raise ValueError(f"zasięg karty {card.id}: brak warstwy pędzla {a.id}")
            arr = np.asarray(arr)
            if arr.ndim == 3:
                arr = arr[..., -1]
            scale = 65535.0 if arr.dtype == np.uint16 else 255.0 if arr.dtype == np.uint8 else 1.0
            arr = arr.astype(np.float32) / scale
            if arr.shape != (height, width):
                arr = cv2.resize(arr, (width, height), interpolation=cv2.INTER_LINEAR)
            lay = np.zeros((H, W), np.float32)
            lay[oy:oy + height, ox:ox + width] = np.clip(arr, 0, 1)
        canvas = np.maximum(canvas, lay) if a.op == "add" else canvas * (1.0 - lay)
    return np.round(np.clip(canvas, 0, 1) * 255).astype(np.uint8), ox, oy, base


# ----------------------------------------------------------------------------------------------- GPU worker client
def worker_python() -> Optional[Path]:
    from engine.config import load_config
    from engine.installer import comfy_python
    return comfy_python(load_config())


class GeomGpuWorker(WorkerProcess):
    """One worker process (engine/cleanup3d/geom_gpu.py) in ComfyUI's Python; `call(req)` -> answer dict."""

    worker = WORKER
    label = "worker geometrii 3D"
    doctor_item = "Wyrównanie na GPU"          # same requirement (ComfyUI's Python with torch + CUDA)
    tmp_prefix = "md_geom3d_gpu_"
    error_cls = GeomGpuError

    def __init__(self, python: Optional[Union[str, Path]] = None, env: Optional[Dict[str, str]] = None,
                 timeout_s: float = 600.0):
        super().__init__(python if python is not None else worker_python(), env, timeout_s)
        self.device = "?"
        self.tmp = self.tmp_dir()

    def call(self, req: Dict[str, Any], cancel_check: Optional[Callable[[], None]] = None) -> Dict[str, Any]:
        from engine.gpu_manager import GPU
        with GPU.use(self, "geometria 3D (GPU)", self.kill, cancel_check):
            try:
                resp = self.request(req, cancel_check, restarts=1)
            except WorkerDied as e:
                raise GeomGpuError(f"worker geometrii 3D zakończył się dwa razy: {e}") from None
            except WorkerTimeout as e:
                raise GeomGpuError(str(e)) from None
            if not resp.get("ok"):
                raise GeomGpuError(f"worker geometrii 3D: {resp.get('error')}")
            self.device = resp.get("device") or self.device
            return resp


def check_worker(python: Optional[Union[str, Path]] = None, env: Optional[Dict[str, str]] = None) -> Tuple[bool, str]:
    """(ok, "cuda <gpu>" / "cpu" / "fake" / the reason): can ComfyUI's Python run the geometry worker? Cached."""
    from engine.worker_proto import check_worker as wp_check
    ok, out = wp_check(python if python is not None else worker_python(), WORKER, env=env)
    return ok, out[3:].strip() if ok else out


# ----------------------------------------------------------------------------------------------- the geometry
def _jsonable_view(v: Dict[str, Any]) -> Dict[str, Any]:
    return {"K": np.asarray(v["K"], float).tolist(), "dist": None if v.get("dist") is None else
            np.asarray(v["dist"], float).tolist(), "R": np.asarray(v["R"], float).tolist(),
            "t": np.asarray(v["t"], float).tolist(), "w": int(v["w"]), "h": int(v["h"])}


class CardGeometry:
    """Geometry of a shot: one Camera3D + its cards (types.Geometry). `layers`: card id -> {area id -> brush mask}
    for the brush layers of the cards' extents (the caller loads the PNGs). device "cpu" | "gpu"; `gpu` = an own
    GeomGpuWorker (tests), else one is opened on the first GPU call (close() ends it)."""

    def __init__(self, camera: Camera3D, cards: Sequence[Card],
                 layers: Optional[Mapping[str, Mapping[str, np.ndarray]]] = None, device: str = "cpu",
                 gpu: Optional[GeomGpuWorker] = None, depths: Optional[Mapping[str, np.ndarray]] = None):
        """`depths`: card id -> (h, w) relative depth of a depth card's reference frame, in place of its snapshot file
        (tests, previews of an unsaved card)."""
        if device not in ("cpu", "gpu"):
            raise ValueError(f"device: cpu | gpu, nie {device!r}")
        self.camera = camera
        self.cards = sorted(cards, key=lambda c: c.order)
        self.layers = dict(layers or {})
        self.device = device
        self._gpu = gpu
        self._own_gpu = gpu is None
        self._rays: Dict[Any, Tuple[np.ndarray, np.ndarray]] = {}
        self._canvas: Dict[str, Any] = {}
        self._weights: Dict[int, Dict[str, np.ndarray]] = {}
        self._n = 0
        self._depths = dict(depths or {})
        self._depth_tmp: Optional[str] = None
        self._depth_specs: Dict[str, Tuple[str, Dict[str, Any]]] = {}

    # ---------------------------------------------------------------- views / cards
    def frame_view(self, frame: int) -> Dict[str, Any]:
        cam = self.camera
        if frame not in cam.R:
            raise KeyError(f"camera3d: brak pozy klatki {frame}")
        return {"K": np.asarray(cam.K, np.float64), "dist": np.asarray(cam.dist, np.float64), "R": cam.R[frame],
                "t": cam.t[frame], "w": int(cam.width), "h": int(cam.height)}

    def key_view(self, key: Key) -> Dict[str, Any]:
        cv = key.canvas
        if not cv:
            return self.frame_view(int(key.frame))
        v = self.frame_view(int(cv.get("frame", key.frame)))
        Kc = np.asarray(cv["K"], np.float64)
        if Kc.shape != (3, 3):
            raise ValueError("key.canvas: K musi być 3x3")
        return {**v, "K": Kc, "dist": None, "w": int(cv["w"]), "h": int(cv["h"])}

    def card_spec(self, card: Card) -> Dict[str, Any]:
        if card.kind == "depth":
            return self._depth_spec(card)
        pl = card_plane(self.camera, card)
        if pl is None:
            return {"kind": "sky"}
        return {"kind": "plane", "n": pl[0].tolist(), "d": float(pl[1])}

    # ---------------------------------------------------------------- depth cards (P18)
    def _depth_path(self, card: Card) -> Path:
        if card.id in self._depths:
            if self._depth_tmp is None:
                import tempfile
                self._depth_tmp = tempfile.mkdtemp(prefix="md_depthcard_")
            p = Path(self._depth_tmp) / f"{card.id}.npy"
            if not p.is_file():
                np.save(p, np.asarray(self._depths[card.id], np.float32))
            return p
        p = card.depth_file()
        if p is None or not p.is_file():
            raise ValueError(f"Karta głębi „{card.name}”: brak mapy głębi – „Bieżąca klatka” w palecie karty (krok „Karty”)")
        return p

    def _depth_spec(self, card: Card) -> Dict[str, Any]:
        """The kernel's spec of a depth card (JSON: it goes to the GPU worker too): its snapshot file, the reference
        frame's view, the conversion to metric depth and the approximate plane (computed when the card has none)."""
        import json
        dp = card.depth or {}
        if dp.get("frame") is None:
            raise ValueError(f"Karta głębi „{card.name}”: brak klatki odniesienia")
        sig = json.dumps(card.to_json(), sort_keys=True, default=str)
        got = self._depth_specs.get(card.id)
        if got is not None and got[0] == sig:
            return got[1]
        p = self._depth_path(card)
        st = p.stat()
        spec: Dict[str, Any] = {"kind": "depth", "file": str(p), "stamp": f"{dp.get('stamp') or ''}:{st.st_mtime_ns}:{st.st_size}",
                                "ref": _jsonable_view(self.frame_view(int(dp["frame"]))),
                                "conv": str(dp.get("conv") or "disparity"), "scale": float(dp.get("scale", 1.0)),
                                "offset": float(dp.get("offset", 0.0)), "mul": float(dp.get("mul", 1.0)),
                                "shift": float(dp.get("shift", 0.0)), "edge": float(dp.get("edge", K_.DEPTH_EDGE))}
        if card.plane:
            n, d = card_plane(self.camera, card)
        else:
            from engine.cleanup3d.depthcard import approx_plane
            n, d = approx_plane(self.camera, int(dp["frame"]), np.load(p), dp)
        spec["n"], spec["d"] = [float(v) for v in n], float(d)
        self._depth_specs[card.id] = (sig, spec)
        return spec

    def depth_warp(self, frame: int, card: Card) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(mx, my, z) (h, w) float32 of plate frame `frame` through depth card `card`: the pixel of the card's
        reference frame each ray hits (MISS where the surface is not seen: occluded in the reference / off its edge)
        and the hit's camera-z depth (0 = none) – the dense correspondence of the depth reprojection."""
        card = self._card(card)
        if card.kind != "depth":
            raise ValueError(f"karta {card.id} nie jest kartą głębi")
        view, spec = self.frame_view(frame), self.card_spec(card)
        if self.device == "gpu":
            out = self._tmp("dwarp")
            self._worker().call({"op": "depth_warp", "tgt": _jsonable_view(view), "card": spec, "out": out})
            m = np.load(out)
            os.remove(out)
            return m[0], m[1], m[2]
        mx, my, z = K_.depth_warp(K_.NP, view, spec)
        return mx.astype(np.float32), my.astype(np.float32), z.astype(np.float32)

    def _card(self, card: Union[Card, str]) -> Card:
        cid = card if isinstance(card, str) else card.id
        for c in self.cards:
            if c.id == cid:
                return c
        raise KeyError(f"nieznana karta {cid}")

    def _frame_rays(self, view: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        sig = (np.asarray(view["K"]).tobytes(), None if view.get("dist") is None else np.asarray(view["dist"]).tobytes(),
               int(view["w"]), int(view["h"]))
        if sig not in self._rays:
            if len(self._rays) >= 2:
                self._rays.pop(next(iter(self._rays)))
            self._rays[sig] = K_.view_rays(K_.NP, view)
        return self._rays[sig]

    def _extent(self, card: Card) -> Optional[Dict[str, Any]]:
        """The card's extent for the kernel (canvas array + offset + the view of extent_frame), cached."""
        if card.id not in self._canvas:
            ext = None
            if card.extent:
                if card.extent_frame is None:
                    raise ValueError(f"karta {card.id}: zasięg bez klatki (extent_frame)")
                cv = extent_canvas(card, int(self.camera.width), int(self.camera.height), self.layers.get(card.id))
                canvas, ox, oy, base = cv
                ext = {"canvas_u8": canvas, "ox": ox, "oy": oy, "base": base,
                       "view": self.frame_view(int(card.extent_frame))}
            self._canvas[card.id] = ext
        return self._canvas[card.id]

    def extent_in_frame(self, card: Card, frame: int) -> np.ndarray:
        """(h, w) 0..1 the card's extent seen in `frame` (1 everywhere for a card without extent, 0 where the ray
        misses the card's plane)."""
        card = self._card(card)
        view = self.frame_view(frame)
        x, y = self._frame_rays(view)
        spec = self.card_spec(card)
        hit = K_.plane_hit(K_.NP, x, y, view, spec)
        e = self._extent(card)
        if e is None:
            return hit.astype(np.float32)
        ext = {**e, "canvas_arr": e["canvas_u8"].astype(np.float64) / 255.0}
        return np.where(hit, K_.extent_mask(K_.NP, x, y, view, spec, ext), 0.0).astype(np.float32)

    # ---------------------------------------------------------------- GPU plumbing
    def _worker(self) -> GeomGpuWorker:
        if self._gpu is None:
            self._gpu = GeomGpuWorker()
        return self._gpu

    def _tmp(self, tag: str) -> str:
        self._n += 1
        return str(self._worker().tmp / f"{tag}_{os.getpid()}_{self._n}.npy")

    def _gpu_map(self, tgt: Dict[str, Any], src: Dict[str, Any], card: Card) -> Tuple[np.ndarray, np.ndarray]:
        out = self._tmp("map")
        self._worker().call({"op": "map", "tgt": _jsonable_view(tgt), "src": _jsonable_view(src),
                             "card": self.card_spec(card), "out": out})
        m = np.load(out)
        os.remove(out)
        return m[0], m[1]

    def close(self) -> None:
        if self._depth_tmp is not None:
            import shutil
            shutil.rmtree(self._depth_tmp, ignore_errors=True)
            self._depth_tmp = None
            self._depth_specs.clear()
        if self._gpu is not None and self._own_gpu:
            self._gpu.close()
            from engine.gpu_manager import GPU
            GPU.forget(self._gpu)
            self._gpu = None

    # ---------------------------------------------------------------- the Geometry protocol
    def _map(self, tgt: Dict[str, Any], src: Dict[str, Any], card: Card, tgt_is_frame: bool) -> Tuple[np.ndarray, np.ndarray]:
        if self.device == "gpu":
            return self._gpu_map(tgt, src, card)
        x, y = self._frame_rays(tgt) if tgt_is_frame else K_.view_rays(K_.NP, tgt)
        mx, my, _ok = K_.project(K_.NP, x, y, tgt, src, self.card_spec(card))
        return mx.astype(np.float32), my.astype(np.float32)

    def frame_to_key(self, frame: int, key: Key, card: Card) -> Tuple[np.ndarray, np.ndarray]:
        """Maps (h, w) of plate frame `frame`: where to read in the key (through `card`)."""
        return self._map(self.frame_view(frame), self.key_view(key), card, True)

    def key_to_frame(self, key: Key, card: Card, frame: int) -> Tuple[np.ndarray, np.ndarray]:
        """Maps of the key's size: where to read in plate frame `frame` (through `card`)."""
        return self._map(self.key_view(key), self.frame_view(frame), card, key.canvas is None)

    def frame_to_frame(self, frame: int, src_frame: int, card: Card) -> Tuple[np.ndarray, np.ndarray]:
        """Maps of plate frame `frame`: where to read in plate frame `src_frame` (through `card`)."""
        return self._map(self.frame_view(frame), self.frame_view(src_frame), card, True)

    def points_frame_to_key(self, pts: np.ndarray, frame: int, key: Key, card: Card) -> np.ndarray:
        """(n, 2) plate pixels of `frame` -> (n, 2) key pixels (float64; NaN where the ray misses the card or the key)."""
        p = np.asarray(pts, np.float64).reshape(-1, 2)
        tgt, src = self.frame_view(frame), self.key_view(key)
        if self.device == "gpu":
            pin, out = self._tmp("pts"), self._tmp("ptsout")
            np.save(pin, p)
            self._worker().call({"op": "points", "pts": pin, "tgt": _jsonable_view(tgt), "src": _jsonable_view(src),
                                 "card": self.card_spec(card), "out": out})
            res = np.load(out)
            os.remove(pin)
            os.remove(out)
            return res
        x, y = K_.view_rays(K_.NP, tgt, p[:, 0], p[:, 1])
        mx, my, ok = K_.project(K_.NP, x, y, tgt, src, self.card_spec(card))
        res = np.stack([mx, my], 1)
        res[~ok] = np.nan
        return res

    def card_weights(self, frame: int) -> Dict[str, np.ndarray]:
        """card id -> (h, w) float32 weight in plate frame `frame` (a partition, see the module doc). Last frame cached."""
        if frame in self._weights:
            return self._weights[frame]
        view = self.frame_view(frame)
        specs = []
        for c in self.cards:
            specs.append({**self.card_spec(c), "seam_px": float(c.seam_px), "ext": self._extent(c)})
        if self.device == "gpu":
            req_cards = self._weights_cards(specs)
            out = self._tmp("w")
            self._worker().call({"op": "weights", "frame": _jsonable_view(view), "cards": req_cards, "out": out})
            arr = np.load(out)
            os.remove(out)
            ws = [arr[i] for i in range(len(self.cards))]
        else:
            for s in specs:
                if s["ext"] is not None:
                    s["ext"] = {**s["ext"], "canvas_arr": s["ext"]["canvas_u8"].astype(np.float64) / 255.0}
            x, y = self._frame_rays(view)
            ws = [w.astype(np.float32) for w in K_.weights(K_.NP, view, specs, x, y)]
        self._weights = {frame: {c.id: w for c, w in zip(self.cards, ws)}}
        return self._weights[frame]

    def _weights_cards(self, specs: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """The cards of the worker's "weights" / "frame_weights" request (extent canvases sent once as .npy)."""
        if specs is None:
            specs = [{**self.card_spec(c), "seam_px": float(c.seam_px), "ext": self._extent(c)} for c in self.cards]
        req_cards = []
        for c, s in zip(self.cards, specs):
            ext = s["ext"]
            if ext is not None:
                if "npy" not in ext:
                    ext["npy"] = self._tmp(f"ext_{c.id}")
                    np.save(ext["npy"], ext["canvas_u8"])
                ext = {"canvas": ext["npy"], "ox": ext["ox"], "oy": ext["oy"], "base": ext["base"],
                       "view": _jsonable_view(ext["view"])}
            req_cards.append({**s, "ext": ext})
        return req_cards

    def frame_weights(self, frame: int, roi: Sequence[int]) -> Dict[str, Tuple[bool, float]]:
        """device "gpu": the card weights of plate frame `frame` computed in the worker and kept resident there (the
        render's ops "project" / "motion" crop them, nothing travels back); card id -> (any weight > 1e-4 in `roi`,
        mean weight in `roi`)."""
        if self.device != "gpu":
            raise ValueError("frame_weights: tylko device gpu")
        resp = self._worker().call({"op": "frame_weights", "frame": _jsonable_view(self.frame_view(frame)),
                                    "cards": self._weights_cards(), "roi": [int(v) for v in roi]})
        return {c.id: (bool(a), float(m)) for c, (a, m) in zip(self.cards, resp["stats"])}

    def card_index(self, card: Union[Card, str]) -> int:
        """Index of a card in the worker's resident weights (the cards in order)."""
        cid = card if isinstance(card, str) else card.id
        return [c.id for c in self.cards].index(cid)

    def card_weight(self, card: Card, frame: int) -> np.ndarray:
        return self.card_weights(frame)[self._card(card).id]

    def coverage(self, frame: int, key: Key, card: Card, edge_px: float = 120.0) -> np.ndarray:
        """(h, w) 0..1 how far the key reaches in `frame`: 0 at the key's border, 1 from `edge_px` inside, 0 where the
        ray misses the card."""
        tgt, src = self.frame_view(frame), self.key_view(key)
        if self.device == "gpu":
            out = self._tmp("cov")
            self._worker().call({"op": "coverage", "tgt": _jsonable_view(tgt), "src": _jsonable_view(src),
                                 "card": self.card_spec(card), "edge_px": float(edge_px), "out": out})
            cov = np.load(out)
            os.remove(out)
            return cov
        mx, my = self.frame_to_key(frame, key, card)
        return K_.coverage(K_.NP, mx.astype(np.float64), my.astype(np.float64), int(src["w"]), int(src["h"]),
                           edge_px).astype(np.float32)
