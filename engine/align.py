# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/align.py - Auxiliary Plate Alignment & Registration Engine for MatteDesk.

Complies strictly with SPEC_INTERACTIVE_MATTE_TOOL.md (Section 4a, 1b, 10, 12: Stage E1b):
- Measures and aligns auxiliary plates (backlit, clear plate) to beauty reference.
- Evaluates static scene areas (excluding foreground/animated areas with dilation).
- Hierarchical global motion models:
    translation -> similarity -> affine -> homography.
  Chooses the simplest model achieving residual < 0.3 px.
- Local deformation model (SPEC 1b), method name `local_stmap` (field stored as a grid, engine.align_grid):
  Activated when the global model leaves residual >= 0.5 px.
  It is measured on the aux plate ALREADY WARPED by the frame's global model (`warp_global`, the same cv2 warp
  PlateWarper applies), so the grid is the residual over the global model (the frame keeps `transform_matrix` /
  dx, dy + `global_method`, and `grid_rel_path`); a clear plate shifted by more than the patch sees
  (SH010: 8.6 px median, 66 % of the points cut at +-8 px) is thus measured near zero.
  Grid points every ~48 px on static areas; aux<->beauty offsets measured by patch phase
  correlation (LOCAL_PATCH_PX = 128 px; user decision 2026-10-15: 64 px patches were noise on SH010 - median 2.1 px,
  16 % cut at +-8 px, 128 px: 0.5 px / 9 %) at anchor frames (every 24 fr + first/last); the same points are tracked in
  time separately in beauty and aux (engine.tracking: CoTracker3, LK only as a logged fallback) so
  the offsets are carried to every frame; per frame outliers are rejected (`reject_outliers`: weak
  correlation response, values sitting on the +-8 px clamp, inconsistency with the median of the 8 nearest
  neighbours), then a thin-plate-spline field clamped to [-8, +8] px, validated on held-out control points.
  The tracker that really ran is stored in the report.
- Composition (`compose_remap`, the one function every consumer uses: preview proxy, `_align`, STMap on request):
  aligned(x) = aux(M^-1 (x - d(x))) - global model first, then the grid, sampled once.
  Nuke: Transform / CornerPin2D (M) followed by the grid STMap - the same order.
- STMap Generation:
  Nuke-compliant float32 OpenEXR with UV coordinates computed from pixel centers:
    u = (x + disp_x + 0.5) / plate_w
    v = (y_nuke + (-disp_y) + 0.5) / plate_h
    where y_nuke = plate_h - 1 - y.
  Identity STMap produces bitwise identical sampling (diff < 1e-4).
- Nuke Group script generator:
  Single Group node Align_<plate> with 1 Input1, 1 Output1, Switch toggled by use_aligned,
  metadata tab 'Align' (method, residual_median, residual_max, align_json),
  containing Transform, CornerPin2D, or Read(STMap)+STMap.
- Warper:
  Generates <plate>_align/ OpenEXR sequence alongside source (only on user consent).
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
import scipy.interpolate as interp

from engine import align_grid
from engine.exr_io import imread_unchanged
from engine.procpool import run_frames
from engine.nuke_nk import (corner_pin_knobs, matrix_cv_to_nuke, scalar_knob, similarity_to_transform_knobs,
                            validate_nk, xy_knob)

logger = logging.getLogger("matte_desk.align")

LOCAL_PATCH_PX = 128   # patch of the local model's correlation (plate px; anchors are measured at work scale = half of it)


def phase_correlate(a: np.ndarray, b: np.ndarray, win: Optional[np.ndarray] = None) -> Tuple[Tuple[float, float], float]:
    """cv2.phaseCorrelate without side effects. With a window, OpenCV (4.11) multiplies BOTH inputs by it in place:
    whole feature maps came out windowed (ECC and residuals then ran on them) and views of overlapping patches
    corrupted their neighbours. Inputs are copied here; returns ((dx, dy), response) like OpenCV."""
    if win is None:
        return cv2.phaseCorrelate(a, b)
    return cv2.phaseCorrelate(np.array(a, np.float32), np.array(b, np.float32), win)


@dataclass
class AlignmentFrameResult:
    frame: int
    method: str  # "translation", "similarity", "affine", "homography", "local_stmap", "identity"
    residual_px: float  # median residual on static/control points
    residual_max_px: float
    dx: float = 0.0  # image pixels (positive = aux feature is right of beauty)
    dy: float = 0.0  # image pixels (positive = aux feature is below beauty)
    rotation_deg: float = 0.0
    scale: float = 1.0
    transform_matrix: Optional[List[List[float]]] = None
    grid_rel_path: Optional[str] = None   # grid_####.exr in <align>/grid_<plate>/ (engine.align_grid)
    global_method: str = ""   # local_stmap only: the global model the grid is the residual over (identity..homography)
    n_points: int = 0         # local_stmap: measured points of the frame, and how many reject_outliers dropped
    n_rejected: int = 0
    holdout_frac_ok: float = 0.0   # local_stmap: share of hold-out points under 0.3 px (SPEC 1b gate)
    status: str = "ok"  # "ok", "outlier", "uncertain"


@dataclass
class AlignmentReport:
    plate_name: str
    aux_plate_type: str  # "backlit", "cleanplate", or "aux"
    reference_plate: str
    frame_range: Tuple[int, int]
    method: str  # simplest model that worked across sequence
    residual_median: float
    residual_max: float
    needs_correction: bool  # True if residual_median > 0.3 or residual_max > 0.75
    recommendation: str  # Polish recommendation text
    frames: List[AlignmentFrameResult] = field(default_factory=list)
    nk_script_path: Optional[str] = None
    json_path: Optional[str] = None
    local_tracker_engine: str = ""  # tracker that actually ran for 1b ("cotracker3" / "lk_fallback"), "" if unused
    device: str = ""  # where it was computed: "cpu", "cuda" (engine.align_gpu) or "cuda+cpu" (GPU fell back mid-way)
    device_note: str = ""  # why not (only) the GPU: fallback reason, ComfyUI busy, ...
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AlignmentReport:
        frames = []
        for f in data.get("frames", []):
            f = dict(f)
            if f.get("method") == "cotracker_local":  # name used by the Gemini build
                f["method"] = "local_stmap"
            f.pop("stmap_rel_path", None)           # reports from before the grid (v0.15): full STMaps, re-measure
            if f.get("method") == "local_stmap" and "global_method" not in f:
                # grid measured on the raw aux (before 2026-10-15): not composable with the global model any more;
                # the global model stays, the grid is dropped (preview falls back to it) - re-measure
                f.pop("grid_rel_path", None)
                f["global_method"] = _global_method_of(f)
            frames.append(AlignmentFrameResult(**f))
        data_copy = dict(data)
        if data_copy.get("method") == "cotracker_local":
            data_copy["method"] = "local_stmap"
        data_copy["frames"] = frames
        return cls(**data_copy)


def stale_report_reason(data: Dict[str, Any]) -> str:
    """Why a saved Align_<plate>.json must be measured again ("" = current). Same rules as `AlignmentReport.from_dict`,
    which silently drops the local part of such reports (the preview then shows only the global model)."""
    frames = data.get("frames", []) or []
    if any("stmap_rel_path" in f for f in frames):
        return ("Stary pomiar (pełne STMapy sprzed siatki przesunięć) – dopasowanie lokalne pominięte, podgląd pokazuje sam "
                "model globalny. Zmierz ponownie.")
    if any(f.get("method") in ("local_stmap", "cotracker_local") and "global_method" not in f for f in frames):
        return ("Stary pomiar (siatka mierzona na surowym plate'cie, sprzed 2026-10-15) – dopasowanie lokalne pominięte, "
                "podgląd pokazuje sam model globalny. Zmierz ponownie.")
    return ""


def _global_method_of(f: Dict[str, Any]) -> str:
    """Name of the global model an old report's frame carries (by the shape of its data)."""
    m = f.get("transform_matrix")
    if m is not None:
        return "homography" if len(m) == 3 else "affine"
    if abs(float(f.get("dx", 0.0))) > 1e-9 or abs(float(f.get("dy", 0.0))) > 1e-9:
        return "translation"
    return "identity"


class AlignmentFeatureExtractor:
    """Extracts invariant features for registration across different lighting passes."""

    @staticmethod
    def extract(img: np.ndarray, mode: str = "backlit", scale: float = 0.5) -> np.ndarray:
        """Extract edge or luma feature map.
        
        Args:
            img: Float32 (ACES/linear) or uint8 image, shape (H, W, C) or (H, W).
            mode: 'backlit' uses log-luma Sobel magnitude; 'cleanplate' uses exposure-normalized log-luma.
            scale: Downscaling factor (default 0.5 for fast subpixel phase correlation).
        """
        # Convert to 2D luma
        if img.ndim == 3:
            # Mean across color channels
            luma = img.mean(axis=2).astype(np.float32)
        else:
            luma = img.astype(np.float32)

        # Scale down for speed and robustness
        if scale != 1.0:
            luma = cv2.resize(luma, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # Log-luma to handle high dynamic range and inverted contrast
        # L = log(max(luma, 1e-4) + 0.02)
        L = np.log(np.maximum(luma, 1e-4) + 0.02)

        if mode == "cleanplate":
            # similar appearance: exposure-normalized log-luma (zero mean, unit std)
            return ((L - float(L.mean())) / max(float(L.std()), 1e-6)).astype(np.float32)

        # Sobel gradients
        gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.hypot(gx, gy)

        # Robust clipping of hot pixels/extreme gradients
        clip_val = np.percentile(grad_mag, 99.5) if grad_mag.size > 0 else 1.0
        clipped = np.clip(grad_mag, 0, max(clip_val, 1e-5))

        # Normalize to 0..1
        max_v = clipped.max()
        if max_v > 1e-6:
            feature_map = clipped / max_v
        else:
            feature_map = clipped

        return feature_map.astype(np.float32)


def compute_static_mask(
    beauty: np.ndarray,
    aux: np.ndarray,
    mode: str = "backlit",
    scale: float = 0.5,
    user_mask: Optional[np.ndarray] = None,
    exclude_box: Optional[Tuple[int, int, int, int]] = None
) -> np.ndarray:
    """Computes a binary mask where 1.0 = static background region, 0.0 = excluded moving foreground.
    
    Args:
        beauty: Reference image.
        aux: Auxiliary image.
        mode: 'backlit' or 'cleanplate'.
        scale: Resolution scale factor.
        user_mask: Optional user mask (1 = include, 0 = exclude).
        exclude_box: Optional box [x0, y0, x1, y1] to mask out (e.g. animated puppet area).
    """
    h_orig, w_orig = beauty.shape[:2]
    h = int(round(h_orig * scale))
    w = int(round(w_orig * scale))

    mask = np.ones((h, w), dtype=np.float32)

    # 1. Edge margins (exclude 2% border to avoid boundary artifacts)
    border_y = max(4, int(round(h * 0.02)))
    border_x = max(4, int(round(w * 0.02)))
    mask[:border_y, :] = 0.0
    mask[-border_y:, :] = 0.0
    mask[:, :border_x] = 0.0
    mask[:, -border_x:] = 0.0

    # 2. Exclude specified foreground box
    if exclude_box is not None:
        bx0, by0, bx1, by1 = exclude_box
        sx0 = max(0, int(round(bx0 * scale)))
        sy0 = max(0, int(round(by0 * scale)))
        sx1 = min(w, int(round(bx1 * scale)))
        sy1 = min(h, int(round(by1 * scale)))
        mask[sy0:sy1, sx0:sx1] = 0.0

    # 3. Cleanplate difference masking: beauty - cleanplate highlights foreground object
    if mode == "cleanplate":
        b_luma = cv2.resize(beauty.mean(axis=2) if beauty.ndim == 3 else beauty, (w, h), interpolation=cv2.INTER_AREA)
        a_luma = cv2.resize(aux.mean(axis=2) if aux.ndim == 3 else aux, (w, h), interpolation=cv2.INTER_AREA)
        # Normalize exposure
        scale_exp = np.median(b_luma[mask > 0.5]) / max(1e-4, np.median(a_luma[mask > 0.5]))
        # Blur first: a few-px misalignment of textured background must not look like an object
        sig = max(2.0, 8.0 * scale)
        diff = np.abs(cv2.GaussianBlur(b_luma, (0, 0), sig) - cv2.GaussianBlur(a_luma * scale_exp, (0, 0), sig))
        # Robust outlier threshold (median + 6 sigma via MAD); a fixed percentile would always cut a
        # fixed share of the frame even when there is no object at all
        vals = diff[mask > 0.5] if (mask > 0.5).any() else diff.ravel()
        med = float(np.median(vals))
        mad = 1.4826 * float(np.median(np.abs(vals - med)))
        fg_thresh = max(med + 6.0 * mad, 0.05 * float(np.median(b_luma)))
        fg_mask = (diff > fg_thresh).astype(np.uint8)
        # Dilate foreground by 40px (scaled)
        kernel_size = max(5, int(round(40 * scale)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        fg_dilated = cv2.dilate(fg_mask, kernel)
        mask[fg_dilated > 0] = 0.0

    # 4. User mask if supplied
    if user_mask is not None:
        u_scaled = cv2.resize(user_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        mask *= (u_scaled > 0.5).astype(np.float32)

    return mask


class GlobalAligner:
    """Computes hierarchical global registration models: translation -> similarity -> affine -> homography."""

    def __init__(self, scale: float = 0.5):
        self.scale = scale

    def measure_translation(
        self,
        feat_aux: np.ndarray,
        feat_beauty: np.ndarray,
        static_mask: np.ndarray
    ) -> Tuple[float, float, float]:
        """Sub-pixel phase correlation.
        
        Returns (dx, dy, response) at full plate scale.
        dx > 0 means aux feature is to the right of beauty.
        dy > 0 means aux feature is below beauty.
        """
        h, w = feat_beauty.shape[:2]
        win = cv2.createHanningWindow((w, h), cv2.CV_32F) * static_mask

        # cv2.phaseCorrelate(src1, src2, window): computes shift that moves src1 onto src2
        (dx_s, dy_s), resp = phase_correlate(feat_aux, feat_beauty, win)

        # Scale back to full plate pixels
        dx = float(dx_s / self.scale)
        dy = float(dy_s / self.scale)
        return dx, dy, float(resp)

    def evaluate_model_residual(
        self,
        feat_aux: np.ndarray,
        feat_beauty: np.ndarray,
        static_mask: np.ndarray,
        model_type: str,
        params: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Evaluates residual registration error in full plate pixels.
        
        Returns (median_residual_px, max_residual_px).
        """
        h, w = feat_beauty.shape[:2]
        if model_type == "translation":
            dx_s = params["dx"] * self.scale
            dy_s = params["dy"] * self.scale
            M = np.float32([[1, 0, dx_s], [0, 1, dy_s]])
            warped_aux = cv2.warpAffine(feat_aux, M, (w, h), flags=cv2.INTER_CUBIC)
        elif model_type in ("similarity", "affine"):
            M = np.array(params["matrix"], dtype=np.float32)
            M_s = M.copy()
            M_s[0, 2] *= self.scale
            M_s[1, 2] *= self.scale
            warped_aux = cv2.warpAffine(feat_aux, M_s, (w, h), flags=cv2.INTER_CUBIC)
        elif model_type == "homography":
            H = np.array(params["matrix"], dtype=np.float32)
            S = np.diag([self.scale, self.scale, 1.0])
            S_inv = np.diag([1.0 / self.scale, 1.0 / self.scale, 1.0])
            H_s = S @ H @ S_inv
            warped_aux = cv2.warpPerspective(feat_aux, H_s, (w, h), flags=cv2.INTER_CUBIC)
        else:
            return 0.0, 0.0

        # Sample edge control points
        edge_mask = (static_mask > 0.5) & (feat_beauty > 0.05)
        step = max(12, int(round(24 * self.scale)))
        gy, gx = np.mgrid[step // 2:h:step, step // 2:w:step]
        valid = edge_mask[gy, gx]
        if not valid.any():
            return 0.0, 0.0

        pts = np.column_stack([gx[valid].ravel(), gy[valid].ravel()])
        residuals = []
        patch_r = 10
        for x, y in pts:
            if patch_r <= x < w - patch_r and patch_r <= y < h - patch_r:
                p_b = feat_beauty[y - patch_r:y + patch_r + 1, x - patch_r:x + patch_r + 1]
                p_w = warped_aux[y - patch_r:y + patch_r + 1, x - patch_r:x + patch_r + 1]
                if np.std(p_b) > 0.02:
                    (p_dx, p_dy), p_r = cv2.phaseCorrelate(p_w, p_b)
                    if p_r > 0.1 and not math.isnan(p_dx) and not math.isnan(p_dy):
                        res_px = math.hypot(p_dx, p_dy) / self.scale
                        residuals.append(res_px)

        if not residuals:
            # Fallback to feature pixel difference
            diff = float(np.abs(feat_beauty[edge_mask] - warped_aux[edge_mask]).mean())
            return diff, diff * 2.0

        res_arr = np.array(residuals, dtype=np.float32)
        med_err = float(np.median(res_arr))
        max_err = float(np.percentile(res_arr, 95))
        return med_err, max_err

    _ECC_MOTION = {"similarity": cv2.MOTION_EUCLIDEAN, "affine": cv2.MOTION_AFFINE, "homography": cv2.MOTION_HOMOGRAPHY}

    def fit_ecc(self, kind: str, feat_aux: np.ndarray, feat_beauty: np.ndarray, static_mask: np.ndarray,
                init_shift: Tuple[float, float] = (0.0, 0.0)) -> Optional[np.ndarray]:
        """ECC (direct alignment of the feature maps) for one model, started from the measured translation
        (full plate px) so it converges in a few iterations. Returns the full-scale matrix or None."""
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
        mask_u8 = (static_mask > 0.5).astype(np.uint8)
        sx, sy = init_shift[0] * self.scale, init_shift[1] * self.scale
        warp = np.eye(3, 3, dtype=np.float32) if kind == "homography" else np.eye(2, 3, dtype=np.float32)
        warp[0, 2], warp[1, 2] = sx, sy
        try:
            cc, M = cv2.findTransformECC(feat_aux, feat_beauty, warp, self._ECC_MOTION[kind], criteria, mask_u8)
        except cv2.error:
            return None
        if not (cc > 0.3) or np.isnan(M).any():
            return None
        if kind == "homography":
            return np.diag([1.0 / self.scale, 1.0 / self.scale, 1.0]) @ M @ np.diag([self.scale, self.scale, 1.0])
        M = M.copy()
        M[0, 2] /= self.scale
        M[1, 2] /= self.scale
        return M

    def fit_sift_models(self, feat_aux: np.ndarray, feat_beauty: np.ndarray, static_mask: np.ndarray) -> Dict[str, Any]:
        """Fallback when ECC fails: SIFT features + RANSAC for similarity, affine and homography."""
        models: Dict[str, Any] = {}
        mask_u8 = (static_mask > 0.5).astype(np.uint8)
        sift = cv2.SIFT_create(nfeatures=1500)
        kp_b, des_b = sift.detectAndCompute((feat_beauty * 255).astype(np.uint8), mask=mask_u8)
        kp_a, des_a = sift.detectAndCompute((feat_aux * 255).astype(np.uint8), mask=mask_u8)

        if des_b is None or des_a is None or len(kp_b) < 8 or len(kp_a) < 8:
            return models

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = sorted(bf.match(des_a, des_b), key=lambda x: x.distance)
        if len(matches) < 6:
            return models

        pts_a = np.float32([kp_a[m.queryIdx].pt for m in matches]) / self.scale
        pts_b = np.float32([kp_b[m.trainIdx].pt for m in matches]) / self.scale

        M_sim, _ = cv2.estimateAffinePartial2D(pts_a, pts_b, method=cv2.RANSAC, ransacReprojThreshold=2.0)
        if M_sim is not None:
            models["similarity"] = M_sim
        M_aff, _ = cv2.estimateAffine2D(pts_a, pts_b, method=cv2.RANSAC, ransacReprojThreshold=2.0)
        if M_aff is not None:
            models["affine"] = M_aff
        H, _ = cv2.findHomography(pts_a, pts_b, method=cv2.RANSAC, ransacReprojThreshold=2.0)
        if H is not None:
            models["homography"] = H
        return models

    def fit_feature_models(self, feat_aux: np.ndarray, feat_beauty: np.ndarray, static_mask: np.ndarray,
                           init_shift: Tuple[float, float] = (0.0, 0.0)) -> Dict[str, Any]:
        """All three models: ECC; when ECC similarity fails, SIFT + RANSAC fills the missing ones."""
        models = {k: M for k in ("similarity", "affine", "homography")
                  if (M := self.fit_ecc(k, feat_aux, feat_beauty, static_mask, init_shift)) is not None}
        if "similarity" not in models:
            for k, M in self.fit_sift_models(feat_aux, feat_beauty, static_mask).items():
                models.setdefault(k, M)
        return models

    def select_best_model(
        self,
        feat_aux: np.ndarray,
        feat_beauty: np.ndarray,
        static_mask: np.ndarray
    ) -> Tuple[str, Dict[str, Any], float, float]:
        """Hierarchical selection: Translation -> Similarity -> Affine -> Homography.

        Picks the simplest model with median residual < 0.3 px. Higher models are fitted only when the simpler
        one fails (ECC started from the measured translation; SIFT fallback when ECC similarity fails).
        Returns (method, params, residual_median, residual_max).
        """
        # Step 1: Translation
        dx, dy, resp = self.measure_translation(feat_aux, feat_beauty, static_mask)
        trans_params = {"dx": dx, "dy": dy, "response": resp}
        res_med, res_max = self.evaluate_model_residual(
            feat_aux, feat_beauty, static_mask, "translation", trans_params
        )

        if res_med < 0.3 and res_max < 0.75:
            return "translation", trans_params, res_med, res_max

        init = (dx, dy) if math.isfinite(dx) and math.isfinite(dy) else (0.0, 0.0)
        models: Dict[str, Any] = {}

        def model(kind: str) -> Optional[np.ndarray]:
            if kind not in models:
                models[kind] = self.fit_ecc(kind, feat_aux, feat_beauty, static_mask, init)
                if kind == "similarity" and models[kind] is None:
                    # same as fit_feature_models: ECC for all, SIFT fills what ECC could not
                    for k in ("affine", "homography"):
                        models[k] = self.fit_ecc(k, feat_aux, feat_beauty, static_mask, init)
                    for k, M in self.fit_sift_models(feat_aux, feat_beauty, static_mask).items():
                        if models.get(k) is None:
                            models[k] = M
            return models[kind]

        # Step 2: Similarity
        M = model("similarity")
        if M is not None:
            s_params = {
                "matrix": M.tolist(),
                "dx": float(M[0, 2]),
                "dy": float(M[1, 2]),
                "scale": float(math.hypot(M[0, 0], M[0, 1])),
                "rotation_deg": float(math.degrees(math.atan2(M[1, 0], M[0, 0])))
            }
            s_med, s_max = self.evaluate_model_residual(
                feat_aux, feat_beauty, static_mask, "similarity", s_params
            )
            if s_med < 0.3 and s_max < 0.75:
                return "similarity", s_params, s_med, s_max

        # Step 3: Affine
        M = model("affine")
        if M is not None:
            a_params = {"matrix": M.tolist(), "dx": float(M[0, 2]), "dy": float(M[1, 2])}
            a_med, a_max = self.evaluate_model_residual(
                feat_aux, feat_beauty, static_mask, "affine", a_params
            )
            if a_med < 0.3 and a_max < 0.75:
                return "affine", a_params, a_med, a_max

        # Step 4: Homography
        H = model("homography")
        if H is not None:
            h_params = {"matrix": H.tolist()}
            h_med, h_max = self.evaluate_model_residual(
                feat_aux, feat_beauty, static_mask, "homography", h_params
            )
            # returned even above the threshold (with the residual as the warning)
            return "homography", h_params, h_med, h_max

        # Default fallback to translation
        return "translation", trans_params, res_med, res_max


class LocalPatchAligner:
    """Single-frame local deformation: patch phase correlation on a static grid + thin-plate spline.

    This class does no temporal tracking; `LocalTemporalAligner` adds that (SPEC 1b)."""

    def __init__(self, plate_w: int, plate_h: int, grid_spacing: int = 48):
        self.plate_w = plate_w
        self.plate_h = plate_h
        self.grid_spacing = grid_spacing

    def generate_grid_points(
        self,
        static_mask: np.ndarray,
        scale: float = 0.5
    ) -> np.ndarray:
        """Generates points on a regular grid within the static mask at full plate scale."""
        h_s, w_s = static_mask.shape[:2]
        step = max(16, int(round(self.grid_spacing * scale)))
        gy, gx = np.mgrid[0:h_s + step:step, 0:w_s + step:step]
        gx = np.clip(gx, 0, w_s - 1)
        gy = np.clip(gy, 0, h_s - 1)

        valid = static_mask[gy, gx] > 0.5
        # Also always include corners / borders to anchor boundary
        corners_y = [0, h_s - 1, 0, h_s - 1]
        corners_x = [0, 0, w_s - 1, w_s - 1]
        for cy, cx in zip(corners_y, corners_x):
            valid[(gy == cy) & (gx == cx)] = True

        pts_s = np.unique(np.column_stack([gx[valid].ravel(), gy[valid].ravel()]), axis=0)
        # Convert to full plate coordinates
        pts_full = pts_s / scale
        return pts_full.astype(np.float32)

    def measure_local_displacements(
        self,
        beauty: np.ndarray,
        aux: np.ndarray,
        grid_pts: np.ndarray,
        patch_size: int = LOCAL_PATCH_PX,
        mode: str = "backlit"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Measures local subpixel displacements at each grid point using patch correlation.
        
        Returns (valid_points, displacements_dx_dy).
        """
        h, w = beauty.shape[:2]
        # Extract features for robust patch matching
        f_b = AlignmentFeatureExtractor.extract(beauty, mode=mode, scale=1.0)
        f_a = AlignmentFeatureExtractor.extract(aux, mode=mode, scale=1.0)

        valid_pts = []
        disps = []
        half_p = patch_size // 2

        for pt in grid_pts:
            x, y = int(round(pt[0])), int(round(pt[1]))
            if half_p <= x < w - half_p and half_p <= y < h - half_p:
                p_b = f_b[y - half_p:y + half_p, x - half_p:x + half_p]
                p_a = f_a[y - half_p:y + half_p, x - half_p:x + half_p]

                win = cv2.createHanningWindow((patch_size, patch_size), cv2.CV_32F)
                (dx, dy), resp = phase_correlate(p_a, p_b, win)
                # phase correlation = coarse init (its sub-pixel peak is biased ~0.1 px);
                # refine with windowed Lucas-Kanade on the patch (SPEC 1b "lokalne ECC na łatkach")
                if not (math.isnan(dx) or math.isnan(dy)) and abs(dx) <= 8 and abs(dy) <= 8:
                    dx, dy = self._refine_translation(f_a, p_b, x, y, dx, dy, win)
                if not math.isnan(dx) and not math.isnan(dy) and resp > 0.1:
                    # Clamp to max ±8 px as per spec
                    dx = float(np.clip(dx, -8.0, 8.0))
                    dy = float(np.clip(dy, -8.0, 8.0))
                    valid_pts.append([x, y])
                    disps.append([dx, dy])

        if len(valid_pts) < 4:
            # Fallback to zero displacement anchors
            return grid_pts, np.zeros_like(grid_pts)

        return np.array(valid_pts, dtype=np.float32), np.array(disps, dtype=np.float32)

    @staticmethod
    def _refine_translation(f_a: np.ndarray, p_b: np.ndarray, x: int, y: int, dx: float, dy: float,
                            win: np.ndarray, iters: int = 10) -> Tuple[float, float]:
        """Gauss-Newton refinement of a pure translation: aux sampled at (patch - d) ~ beauty patch."""
        ps = p_b.shape[0]
        cx, cy = x - 0.5, y - 0.5  # centre of the slice [x - ps/2, x + ps/2)
        gb_y, gb_x = np.gradient(p_b)
        for _ in range(iters):
            p_a = cv2.getRectSubPix(f_a, (ps, ps), (cx - dx, cy - dy))
            gy, gx = np.gradient(p_a)
            gx, gy = 0.5 * (gx + gb_x), 0.5 * (gy + gb_y)
            r = (p_b - p_a) * win
            # p_a(d + e) ~ p_a(d) - e . grad  ->  minimise |p_b - p_a + e.grad|
            a11, a12, a22 = float((win * gx * gx).sum()), float((win * gx * gy).sum()), float((win * gy * gy).sum())
            b1, b2 = float((r * gx).sum()), float((r * gy).sum())
            det = a11 * a22 - a12 * a12
            if det < 1e-9:
                break
            ex = -(a22 * b1 - a12 * b2) / det
            ey = -(a11 * b2 - a12 * b1) / det
            dx, dy = dx + ex, dy + ey
            if abs(ex) < 1e-3 and abs(ey) < 1e-3:
                break
        return float(dx), float(dy)

    def fit_rbf_grid(self, points: np.ndarray, displacements: np.ndarray,
                     step: int = align_grid.DEFAULT_STEP, resp: Optional[np.ndarray] = None,
                     reject: bool = True) -> Tuple[align_grid.AlignGrid, float, float]:
        """Thin-plate spline through the measured displacements, evaluated at the nodes of the alignment grid
        (engine.align_grid: one node per `step` px at the cell centres). Returns (grid, holdout_median, holdout_p95);
        `last_holdout_frac_ok` = share of hold-out points under 0.3 px (SPEC 1b gate).
        `reject`: drop outliers first (`reject_outliers`; `resp` = patch correlation response per point, NaN = not
        measured); `last_n_rejected` / `last_reject` say how many and why. Fewer than MIN_FIT_POINTS or than
        MIN_KEPT_FRAC of the points left, or a singular thin-plate system (collinear points) -> zero grid = the global
        model alone (frame `uncertain`)."""
        points = np.asarray(points, np.float64).reshape(-1, 2)
        displacements = np.asarray(displacements, np.float64).reshape(-1, 2)
        if reject:
            keep, info = reject_outliers(points, displacements, resp)
        else:
            keep, info = np.ones(len(points), bool), {"n": len(points), "resp": 0, "clip": 0, "nbr": 0}
        self.last_reject = info
        self.last_n_rejected = int((~keep).sum())
        n_all = len(points)
        points, displacements = points[keep], displacements[keep]
        n_pts = len(points)
        if n_pts < MIN_FIT_POINTS or n_pts < MIN_KEPT_FRAC * n_all:
            self.last_holdout_frac_ok = 0.0
            return align_grid.AlignGrid.zeros(self.plate_w, self.plate_h, step), 0.0, 0.0

        # Split 10% hold-out for validation
        np.random.seed(42)
        perm = np.random.permutation(n_pts)
        n_holdout = max(4, int(round(n_pts * 0.1)))
        train_idx = perm[n_holdout:]
        holdout_idx = perm[:n_holdout]

        train_pts = points[train_idx]
        train_disps = displacements[train_idx]
        holdout_pts = points[holdout_idx]
        holdout_disps = displacements[holdout_idx]

        # Fit Thin-Plate Spline RBF for X and Y components
        # (one interpolator for both components: the kernel system is solved once)
        try:
            rbf = interp.RBFInterpolator(train_pts, train_disps, kernel="thin_plate_spline", smoothing=0.1)
        except (np.linalg.LinAlgError, ValueError) as e:       # singular system: collinear / duplicate points
            logger.warning("thin-plate fit failed on %d points (%s) - zero grid", n_pts, e)
            self.last_holdout_frac_ok = 0.0
            return align_grid.AlignGrid.zeros(self.plate_w, self.plate_h, step), 0.0, 0.0

        # Validate on hold-out control points (not used in the fit)
        pred_ho = rbf(holdout_pts)
        err_ho = np.hypot(pred_ho[:, 0] - holdout_disps[:, 0], pred_ho[:, 1] - holdout_disps[:, 1])
        # SPEC 1b gate: residual < 0.3 px on >= 90% of control points
        self.last_holdout_frac_ok = float(np.mean(err_ho < 0.3))
        ho_median = float(np.median(err_ho))
        ho_max = float(np.percentile(err_ho, 95))

        nodes, (gh, gw) = align_grid.node_points(self.plate_w, self.plate_h, step)
        ev = rbf(nodes)
        gx = np.clip(ev[:, 0].reshape(gh, gw), -8.0, 8.0).astype(np.float32)
        gy = np.clip(ev[:, 1].reshape(gh, gw), -8.0, 8.0).astype(np.float32)
        return align_grid.AlignGrid(gx, gy, self.plate_w, self.plate_h), ho_median, ho_max

    def fit_rbf_deformation_field(
        self,
        points: np.ndarray,
        displacements: np.ndarray,
        eval_downscale: int = align_grid.DEFAULT_STEP
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """fit_rbf_grid + align_grid.upsample: dense (disp_x_full, disp_y_full, holdout_median, holdout_max)."""
        grid, ho_median, ho_max = self.fit_rbf_grid(points, displacements, step=eval_downscale)
        disp_x_full, disp_y_full = align_grid.upsample(grid)
        return disp_x_full, disp_y_full, ho_median, ho_max

    def generate_stmap(
        self,
        disp_x: np.ndarray,
        disp_y: np.ndarray,
        out_exr_path: Path
    ) -> None:
        """Generates Nuke-compliant float32 OpenEXR STMap.
        
        Formula:
            u = (x + disp_x + 0.5) / plate_w
            v = (y_nuke + (-disp_y) + 0.5) / plate_h
            where y_nuke = plate_h - 1 - y
        """
        import OpenEXR
        import Imath

        h, w = disp_x.shape
        Y, X = np.indices((h, w), dtype=np.float32)
        # When disp_x > 0, aux feature is to the right of beauty -> sample from X - disp_x
        write_stmap(X - disp_x, Y - disp_y, out_exr_path)


def write_stmap(map_x: np.ndarray, map_y: np.ndarray, out_exr_path: Path) -> None:
    """cv2.remap maps (source pixel of every output pixel, OpenCV coords) -> Nuke STMap EXR (float32 R = u, G = v):
        u = (map_x + 0.5) / w,   v = (h - 1 - map_y + 0.5) / h   (Nuke: pixel centre +0.5, origin bottom-left)."""
    import OpenEXR
    import Imath

    h, w = map_x.shape
    u = (np.asarray(map_x, np.float32) + 0.5) / float(w)
    v = ((h - 1) - np.asarray(map_y, np.float32) + 0.5) / float(h)
    u = np.clip(u, 0.0, 1.0).astype(np.float32)
    v = np.clip(v, 0.0, 1.0).astype(np.float32)
    out_exr_path = Path(out_exr_path)
    out_exr_path.parent.mkdir(parents=True, exist_ok=True)
    header = OpenEXR.Header(w, h)
    header["channels"] = {
        "R": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        "G": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    }

    exr_out = OpenEXR.OutputFile(str(out_exr_path), header)
    exr_out.writePixels({"R": u.tobytes(), "G": v.tobytes()})
    exr_out.close()


def refine_local_offsets(feat_beauty: np.ndarray, feat_aux: np.ndarray, pts: np.ndarray, disp: np.ndarray,
                         patch_size: int = LOCAL_PATCH_PX, max_change_px: float = 4.0) -> Tuple[np.ndarray, np.ndarray]:
    """Sub-pixel refinement of tracked offsets on full-resolution feature maps (SPEC 1b precision).

    Tracks carry the anchor offsets through time but come from a downscaled video (CoTracker3: longer side
    1024 px), so their offset can be off by a pixel or more on the plate. Per point: patch phase correlation
    around the tracked offset + Gauss-Newton (same as at the anchors). The tracked value is kept where the
    patch has no usable texture (weak correlation) or the correction exceeds max_change_px (wrong lock).
    pts, disp: (N, 2) plate px, disp = beauty point - aux point.
    Returns (disp_refined, refined_mask, response): the patch correlation response per point (NaN where the patch
    was not measured - outside the plate margin), for `reject_outliers`."""
    h, w = feat_beauty.shape[:2]
    half = patch_size // 2
    win = cv2.createHanningWindow((patch_size, patch_size), cv2.CV_32F)
    out = np.asarray(disp, np.float32).copy()
    refined = np.zeros(len(out), bool)
    response = np.full(len(out), np.nan, np.float64)
    for k, (p, d0) in enumerate(zip(pts, out)):
        x, y = int(round(float(p[0]))), int(round(float(p[1])))
        if not (half <= x < w - half and half <= y < h - half):
            continue
        p_b = feat_beauty[y - half:y + half, x - half:x + half]
        cx, cy = x - 0.5, y - 0.5  # centre of the slice, as in LocalPatchAligner._refine_translation
        p_a = cv2.getRectSubPix(feat_aux, (patch_size, patch_size), (cx - float(d0[0]), cy - float(d0[1])))
        (rx, ry), resp = phase_correlate(p_a, p_b, win)
        response[k] = resp
        if not (resp > 0.1) or not (math.isfinite(rx) and math.isfinite(ry)):
            continue
        dx, dy = LocalPatchAligner._refine_translation(feat_aux, p_b, x, y, float(d0[0]) + rx, float(d0[1]) + ry, win)
        if not (math.isfinite(dx) and math.isfinite(dy)) or math.hypot(dx - d0[0], dy - d0[1]) > max_change_px:
            continue
        out[k] = (np.clip(dx, -8.0, 8.0), np.clip(dy, -8.0, 8.0))
        refined[k] = True
    return out, refined, response


MIN_FIT_POINTS = 8       # fewer measured points left after rejection: no local field for the frame (global model only)
MIN_KEPT_FRAC = 0.2      # ... or fewer than this share of the frame's points: the measurement is noise, the survivors are
                         # chance agreement (SH010 clear plate 2026-10-15: 1-4 % kept gave fields with 38 px hold-out)
OUTLIER_RESP_MIN = 0.1   # patch correlation response at or below this: no usable texture under the patch
OUTLIER_CLIP_PX = 8.0    # SPEC 1b clamp: a value sitting on it was cut off, not measured
OUTLIER_NBR_K = 8        # neighbours whose median a point is compared with
OUTLIER_NBR_PX = 1.0     # deviation from that median above this: inconsistent with the surroundings


def reject_outliers(points: np.ndarray, displacements: np.ndarray, resp: Optional[np.ndarray] = None,
                    resp_min: float = OUTLIER_RESP_MIN, clip_px: float = OUTLIER_CLIP_PX,
                    nbr_k: int = OUTLIER_NBR_K, nbr_px: float = OUTLIER_NBR_PX) -> Tuple[np.ndarray, Dict[str, int]]:
    """Points to drop before the thin-plate fit. Three rules, in this order:
    1. `resp` (patch correlation response) at or below `resp_min` - NaN (not measured, tracked only) is kept;
    2. |dx| or |dy| on the +-clip_px clamp (within 1e-3 px);
    3. among the survivors, deviation above `nbr_px` from BOTH the median displacement of the `nbr_k` nearest
       other survivors (Euclidean, plate px) and a plane fitted through them (the median alone is off at the
       plate edge of a sloping field, the plane alone is pulled by a neighbouring outlier).
    Returns (keep (N,) bool, {"n": N, "resp": n1, "clip": n2, "nbr": n3}). The GPU worker (engine/align_gpu_worker.py,
    standalone) carries an identical copy - keep both the same (tests/test_align_gpu.py compares them)."""
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    d = np.asarray(displacements, np.float64).reshape(-1, 2)
    n = len(pts)
    keep = np.isfinite(d).all(axis=1) & np.isfinite(pts).all(axis=1)
    if resp is not None:
        r = np.asarray(resp, np.float64).reshape(-1)
        bad_resp = np.isfinite(r) & (r <= resp_min)
    else:
        bad_resp = np.zeros(n, bool)
    bad_clip = (np.abs(d) >= clip_px - 1e-3).any(axis=1) & ~bad_resp
    keep &= ~bad_resp & ~bad_clip
    n_nbr = 0
    idx = np.nonzero(keep)[0]
    if len(idx) > nbr_k:
        p, q = pts[idx], d[idx]
        dev = np.empty(len(idx), np.float64)
        for s in range(0, len(idx), 1024):
            blk = p[s:s + 1024]
            dist = ((blk[:, None, :] - p[None, :, :]) ** 2).sum(-1)
            dist[np.arange(len(blk)), s + np.arange(len(blk))] = np.inf      # not itself
            nb = np.argpartition(dist, nbr_k, axis=1)[:, :nbr_k]
            qb = q[s:s + 1024]
            med = np.median(q[nb], axis=1)
            dev_med = np.hypot(qb[:, 0] - med[:, 0], qb[:, 1] - med[:, 1])
            # a plane through the neighbours, evaluated at the point: at the plate edge the neighbour ring is
            # one-sided and the median of a sloping field is off, the plane is not; next to an outlier the plane
            # is pulled but the median is not - a point is dropped only when BOTH disagree with it
            rel = (p[nb] - blk[:, None, :]) / 100.0
            A = np.concatenate([np.ones(rel.shape[:2] + (1,)), rel], axis=2)
            AtA = A.transpose(0, 2, 1) @ A + 1e-6 * np.eye(3)
            beta = np.linalg.solve(AtA, A.transpose(0, 2, 1) @ q[nb])
            dev_plane = np.hypot(qb[:, 0] - beta[:, 0, 0], qb[:, 1] - beta[:, 0, 1])
            dev[s:s + 1024] = np.minimum(dev_med, dev_plane)
        bad = dev > nbr_px
        keep[idx[bad]] = False
        n_nbr = int(bad.sum())
    return keep, {"n": int(n), "resp": int(bad_resp.sum()), "clip": int(bad_clip.sum()), "nbr": n_nbr}


class LocalTemporalAligner:
    """SPEC 1b: local offsets measured at anchor frames, carried to every frame by point tracks.

    For each anchor a: grid point p (beauty) has offset d_a(p) to aux (patch correlation).
    Tracker follows p in beauty -> pb(t) and p + d_a in aux -> pa(t); offset at t = pa(t) - pb(t).
    Offsets of the two anchors bracketing t are blended linearly in time; occluded points
    (visible = False in either sequence) are skipped.
    """

    def __init__(self, plate_w: int, plate_h: int, grid_spacing: int = 48, anchor_every: int = 24,
                 work_scale: float = 0.5, tracker: Any = None, mode: str = "backlit"):
        self.mode = mode
        self.plate_w, self.plate_h = int(plate_w), int(plate_h)
        self.grid_spacing = grid_spacing
        self.anchor_every = anchor_every
        self.s = work_scale
        if tracker is None:
            from engine.tracking import make_point_tracker
            tracker = make_point_tracker()
        self.tracker = tracker
        self.engine = getattr(tracker, "engine", type(tracker).__name__)
        self.patch = LocalPatchAligner(plate_w, plate_h, grid_spacing=grid_spacing)
        # anchor offsets: LocalPatchAligner.measure_local_displacements, or its GPU twin (engine.align_gpu)
        self.measure_patches: Optional[Callable[..., Tuple[np.ndarray, np.ndarray]]] = None

    def anchors(self, n: int) -> List[int]:
        return sorted(set(range(0, n, self.anchor_every)) | {0, n - 1})

    def _track_segments(self, anchors: List[int], b_s: List[np.ndarray], a_s: List[np.ndarray],
                        q_b: np.ndarray, q_a: np.ndarray, owner: np.ndarray,
                        cancel_check: Optional[Callable[[], None]]) -> Tuple[Any, Any]:
        """Frame t only blends the two anchors bracketing it, so each anchor's points are tracked only over
        [previous anchor, next anchor] (<= 2 * anchor_every + 1 frames) - the whole sequence at once does not fit
        on the GPU (SH010 F1-193: CoTracker3 asked for 98-162 GiB). Returns full-length (T, N) results with
        visible = False outside each anchor's segment."""
        from engine.tracking import TrackResult, track_many
        T, N = len(b_s), len(q_b)
        videos, spans = [], []
        for ai, a in enumerate(anchors):
            idx = np.where(owner == ai)[0]
            if not len(idx):
                continue
            lo, hi = anchors[max(0, ai - 1)], anchors[min(len(anchors) - 1, ai + 1)]
            for seq, q in ((b_s, q_b), (a_s, q_a)):
                qq = q[idx].copy()
                qq[:, 0] -= lo
                videos.append((seq[lo:hi + 1], qq))
            spans.append((idx, lo, hi))
        res = track_many(self.tracker, videos, cancel_check)
        out = []
        for k in range(2):
            tracks = np.zeros((T, N, 2), np.float32)
            visible = np.zeros((T, N), bool)
            for (idx, lo, hi), r in zip(spans, res[k::2]):
                tracks[lo:hi + 1, idx] = r.tracks
                visible[lo:hi + 1, idx] = r.visible
            out.append(TrackResult(tracks=tracks, visible=visible, engine=res[k].engine if res else self.engine))
        return out[0], out[1]

    def work_size(self) -> Tuple[int, int]:
        return max(16, round(self.plate_w * self.s)), max(16, round(self.plate_h * self.s))

    def measure(self, beauty_frames: List[np.ndarray], aux_frames: List[np.ndarray],
                static_mask: Optional[np.ndarray] = None,
                cancel_check: Optional[Callable[[], None]] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Returns per frame (points_plate_px (N,2), displacements_plate_px (N,2)).
        Frames may be full plate size or already at work size (`work_size()`, no copy then)."""
        s = self.s
        T = len(beauty_frames)
        size = self.work_size()

        def at_work(f: np.ndarray) -> np.ndarray:
            return f if (f.shape[1], f.shape[0]) == size else cv2.resize(f, size, interpolation=cv2.INTER_AREA)

        b_s = [at_work(f) for f in beauty_frames]
        a_s = [at_work(f) for f in aux_frames]
        if static_mask is None:
            static_mask = np.ones((self.plate_h, self.plate_w), np.float32)
        mask_s = cv2.resize(static_mask.astype(np.float32), size, interpolation=cv2.INTER_NEAREST)

        # grid every `grid_spacing` plate px, expressed in work-scale pixels
        grid = self.patch.generate_grid_points(mask_s, scale=s) * s
        patch_px = max(16, int(round(LOCAL_PATCH_PX * s)) // 2 * 2)
        anchors = self.anchors(T)

        q_b, q_a, owner, gidx = [], [], [], []
        grid_key = {(round(float(p[0]), 3), round(float(p[1]), 3)): i for i, p in enumerate(grid)}
        for ai, a in enumerate(anchors):
            if cancel_check:
                cancel_check()
            measure = self.measure_patches or self.patch.measure_local_displacements
            pts, d = measure(b_s[a], a_s[a], grid, patch_size=patch_px, mode=self.mode)
            # phase correlation returns the shift d that maps aux onto beauty: aux point = p - d
            for p, dd in zip(pts, d):
                q_b.append([a, p[0], p[1]])
                q_a.append([a, p[0] - dd[0], p[1] - dd[1]])
                owner.append(ai)
                gidx.append(grid_key.get((round(float(p[0]), 3), round(float(p[1]), 3)), -1 - len(gidx)))
        if not q_b:
            return [(np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)) for _ in range(T)]
        q_b, q_a = np.array(q_b, np.float32), np.array(q_a, np.float32)
        owner, gidx = np.array(owner), np.array(gidx)

        tb, ta = self._track_segments(anchors, b_s, a_s, q_b, q_a, owner, cancel_check)

        out = []
        for t in range(T):
            hi = next((i for i, a in enumerate(anchors) if a >= t), len(anchors) - 1)
            lo = hi if anchors[hi] == t else max(0, hi - 1)
            if hi == lo:
                weights = {hi: 1.0}
            else:
                span = float(anchors[hi] - anchors[lo])
                weights = {lo: (anchors[hi] - t) / span, hi: (t - anchors[lo]) / span}
            acc: Dict[int, List[float]] = {}  # grid id -> [sum_w, sum_w*x, sum_w*y, sum_w*dx, sum_w*dy]
            for ai, wi in weights.items():
                if wi <= 0:
                    continue
                for j in np.where((owner == ai) & tb.visible[t] & ta.visible[t])[0]:
                    pb = tb.tracks[t, j]
                    d = pb - ta.tracks[t, j]
                    a_ = acc.setdefault(int(gidx[j]), [0.0, 0.0, 0.0, 0.0, 0.0])
                    a_[0] += wi; a_[1] += wi * pb[0]; a_[2] += wi * pb[1]; a_[3] += wi * d[0]; a_[4] += wi * d[1]
            if not acc:
                out.append((np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)))
                continue
            arr = np.array(list(acc.values()), np.float64)
            pts = arr[:, 1:3] / arr[:, :1]
            disp = arr[:, 3:5] / arr[:, :1]
            out.append(((pts / s).astype(np.float32), np.clip(disp / s, -8.0, 8.0).astype(np.float32)))
        return out


def _frame_matrix_cv(f: AlignmentFrameResult) -> np.ndarray:
    """Forward aux -> beauty transform of the frame's GLOBAL model in OpenCV pixel coords (3x3); for a local_stmap
    frame that is the model under the grid (`global_method`)."""
    method = f.global_method if f.method == "local_stmap" else f.method
    m = np.eye(3)
    if method in ("similarity", "affine", "homography") and f.transform_matrix is not None:
        t = np.asarray(f.transform_matrix, dtype=np.float64)
        m[: t.shape[0], :] = t
    elif method == "translation":
        m[0, 2], m[1, 2] = f.dx, f.dy
    return m


def _is_identity(m: np.ndarray) -> bool:
    return bool(np.all(np.abs(np.asarray(m, np.float64) - np.eye(3)) < 1e-12))


def warp_global(img: np.ndarray, m: np.ndarray, interpolation: int = cv2.INTER_CUBIC) -> np.ndarray:
    """The aux plate warped by a global model (3x3 forward aux -> beauty, OpenCV coords) - the one warp used both
    to pre-align the aux for the local model (CPU and GPU paths) and by PlateWarper for the global model alone:
    cv2.warpAffine for an affine matrix, warpPerspective otherwise, cubic, black outside. Identity = the image."""
    m = np.asarray(m, np.float64)
    if _is_identity(m):
        return img
    h, w = img.shape[:2]
    if np.all(np.abs(m[2] - [0.0, 0.0, 1.0]) < 1e-12):
        return cv2.warpAffine(img, m[:2].astype(np.float32), (w, h), flags=interpolation,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return cv2.warpPerspective(img, m.astype(np.float32), (w, h), flags=interpolation,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def compose_remap(frame: AlignmentFrameResult, grid: Optional[align_grid.AlignGrid], out_w: int, out_h: int,
                  plate_w: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """THE composition of a frame's two alignment models (SPEC 1b), used by every consumer (preview proxy, `_align`,
    STMap on request, tests): the global model M (`_frame_matrix_cv`: the model the grid was measured over) first,
    then the grid d (the residual):
        aux_global(x) = aux(M^-1 x)                       (global model alone, = warp_global)
        aligned(x)    = aux_global(x - d(x)) = aux(M^-1 (x - d(x)))
    Returns cv2.remap maps (float32 (out_h, out_w)) for ONE cubic resampling of the aux plate. Nuke: Transform /
    CornerPin2D (M) followed by the grid STMap - the same order, two resamplings.
    An output smaller than the plate (proxy) scales both models: M -> S M S^-1, d -> s * d (align_grid.upsample);
    `plate_w` is needed for that only when there is no grid. Grid None = the global model alone."""
    pw = grid.plate_w if grid is not None else (plate_w or out_w)
    sc = out_w / float(pw)
    m = _frame_matrix_cv(frame)
    if sc != 1.0:
        s_mat = np.diag([sc, sc, 1.0])
        m = s_mat @ m @ np.linalg.inv(s_mat)
    Y, X = np.indices((out_h, out_w), dtype=np.float32)
    if grid is not None:
        dx, dy = align_grid.upsample(grid, out_w, out_h, scale=sc)
        X = X - dx
        Y = Y - dy
    if _is_identity(m):
        return X, Y
    mi = np.linalg.inv(m)
    xd, yd = X.astype(np.float64), Y.astype(np.float64)
    u = mi[0, 0] * xd + mi[0, 1] * yd + mi[0, 2]
    v = mi[1, 0] * xd + mi[1, 1] * yd + mi[1, 2]
    wq = mi[2, 0] * xd + mi[2, 1] * yd + mi[2, 2]
    return (u / wq).astype(np.float32), (v / wq).astype(np.float32)


def stmap_for_frame(frame: AlignmentFrameResult, grid: Optional[align_grid.AlignGrid], out_exr_path: Path,
                    plate_w: Optional[int] = None, plate_h: Optional[int] = None) -> None:
    """Full-resolution STMap EXR of the frame's composed alignment (global model + grid) - only on request."""
    w = grid.plate_w if grid is not None else int(plate_w)
    h = grid.plate_h if grid is not None else int(plate_h)
    map_x, map_y = compose_remap(frame, grid, w, h, plate_w=w)
    write_stmap(map_x, map_y, Path(out_exr_path))


class NukeAlignGroupGenerator:
    """Generates standalone Nuke scripts containing a single Group node 'Align_<plate>'."""

    @staticmethod
    def generate_script(
        plate_name: str,
        report: AlignmentReport,
        plate_w: int,
        plate_h: int,
        output_nk_path: Path,
        output_json_path: Path
    ) -> str:
        """Generates Nuke script text with single Group node adhering to Nuke conventions.
        
        Outside: 1 input (Input1), 1 output (Output1).
        Inside: Dot1 -> Switch1 (inputs: unaligned input 0, aligned input 1), toggled by parent.use_aligned.
        """
        group_name = f"Align_{plate_name}"
        method = report.method
        res_med = round(report.residual_median, 3)
        res_max = round(report.residual_max, 3)
        json_str_path = str(output_json_path).replace("\\", "/")

        # Build animated curve strings for frames
        frames = report.frames
        first_frame = frames[0].frame if frames else 1
        last_frame = frames[-1].frame if frames else 100

        fr = [f.frame for f in frames]
        # Most general model present wins (frames may differ); matrices in Nuke coords. Under a local grid the
        # global model is the frame's `global_method` (the grid is the residual over it).
        order = ["identity", "translation", "similarity", "affine", "homography"]
        global_methods = [(f.global_method if f.method == "local_stmap" else f.method) for f in frames]
        global_methods = [m for m in global_methods if m in order]
        global_method = max(global_methods, key=order.index) if global_methods else "identity"
        node_method = "local_stmap" if method == "local_stmap" else global_method
        mats = [matrix_cv_to_nuke(_frame_matrix_cv(f), plate_h) for f in frames]
        global_node = NukeAlignGroupGenerator._global_node(global_method, fr, mats, plate_w, plate_h)

        # Construct internal node based on method (input 0 = top of the .nk stack = Dot1)
        if node_method == "local_stmap":
            # global model (Transform / CornerPin2D, M) FIRST, then the alignment grid (engine.align_grid) read at its
            # own size, scaled to the plate in Nuke and turned into an STMap: the same order as compose_remap
            grid_pat = str(output_nk_path.parent / f"grid_{plate_name}" / "grid_%04d.exr").replace("\\", "/")
            gw, gh = align_grid.grid_shape(plate_w, plate_h)
            if global_node:
                inner_node = (global_node + "\nset N_global [stack 0]\n"
                              + align_grid.nuke_nodes(grid_pat, first_frame, last_frame, plate_w, plate_h, gw, gh,
                                                      src_stack="$N_global"))
            else:
                inner_node = align_grid.nuke_nodes(grid_pat, first_frame, last_frame, plate_w, plate_h, gw, gh)
        elif global_node:
            inner_node = global_node
        else:
            # Identity / None
            inner_node = """ Dot {{
  name Dot_Aligned
  xpos 154
  ypos 60
 }}"""

        script = f"""Group {{
 name {group_name}
 tile_color 0x3d71ffff
 selected true
 xpos 0
 ypos 0
 addUserKnob {{20 AlignTab l Align}}
 addUserKnob {{6 use_aligned l "Use Aligned" +STARTLINE}}
 use_aligned true
 addUserKnob {{1 align_method l Method -STARTLINE}}
 align_method "{method}"
 addUserKnob {{7 residual_median l "Residual Median (px)"}}
 residual_median {res_med}
 addUserKnob {{7 residual_max l "Residual Max (px)"}}
 residual_max {res_max}
 addUserKnob {{2 align_json l "Config JSON"}}
 align_json "{json_str_path}"
 label "{method} (med: [value residual_median] px, max: [value residual_max] px)"
}}
 Input {{
  inputs 0
  name Input1
  xpos 0
  ypos -80
 }}
 Dot {{
  name Dot1
  xpos 34
  ypos -20
 }}
set N_dot1 [stack 0]
{inner_node}
push $N_dot1
 Switch {{
  inputs 2
  which {{{{parent.use_aligned}}}}
  name Switch1
  xpos 0
  ypos 140
 }}
 Output {{
  name Output1
  xpos 0
  ypos 200
 }}
end_group
"""
        # Save Nuke script and JSON
        output_nk_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_nk_path, "w", encoding="utf-8") as f:
            f.write(script)

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

        return script

    @staticmethod
    def _global_node(node_method: str, fr: List[int], mats: List[np.ndarray], plate_w: int, plate_h: int) -> str:
        """Transform (translation / similarity) or CornerPin2D (affine / homography) node text for the global model
        of every frame (matrices in Nuke coords); "" for identity. Takes the top of the .nk stack as its input."""
        if node_method in ("translation", "similarity"):
            center = (plate_w / 2.0, plate_h / 2.0)
            knobs = [similarity_to_transform_knobs(m, center) for m in mats]
            lines = [xy_knob("translate", fr, [k[0] for k in knobs], [k[1] for k in knobs])]
            if node_method == "similarity":
                lines.append(scalar_knob("rotate", fr, [k[2] for k in knobs]))
                lines.append(scalar_knob("scale", fr, [k[3] for k in knobs], nd=6))
            body = "\n".join("  " + ln for ln in lines)
            inner_node = f""" Transform {{
{body}
  center {{{center[0]} {center[1]}}}
  filter Cubic
  black_outside false
  name Transform1
  xpos 120
  ypos 60
 }}"""

        elif node_method in ("affine", "homography"):
            # CornerPin2D represents affine and projective maps exactly
            body = "\n".join("  " + ln for ln in corner_pin_knobs(fr, mats, plate_w, plate_h))
            inner_node = f""" CornerPin2D {{
{body}
  filter Cubic
  invert false
  name CornerPin2D_1
  xpos 120
  ypos 60
 }}"""
        else:
            inner_node = ""
        return inner_node

    @staticmethod
    def validate_nuke_syntax(script_text: str) -> Tuple[bool, str]:
        """Structural check (engine.nuke_nk.validate_nk) + the use_aligned switch."""
        problems = validate_nk(script_text)
        if "Switch1" not in script_text or "parent.use_aligned" not in script_text:
            problems.append("Missing Switch1 with parent.use_aligned")
        return (not problems), ("; ".join(problems) if problems else "Valid Nuke Group script")


class PlateWarper:
    """Warps auxiliary plate frames to create <plate>_align/ sequence."""

    @staticmethod
    def warp_frame(
        img: np.ndarray,
        result: AlignmentFrameResult,
        field_path: Optional[Path] = None,
        interpolation: int = cv2.INTER_CUBIC,
        plate_w: Optional[int] = None
    ) -> np.ndarray:
        """Warps a single frame according to AlignmentFrameResult. local_stmap: `compose_remap` (global model, then
        the grid at `field_path`; a missing grid = the global model alone), one resampling; an image smaller than
        the plate (proxy) gets both models scaled (`plate_w` needed only when there is no grid)."""
        h, w = img.shape[:2]

        if result.method == "identity" or (abs(result.dx) < 1e-4 and abs(result.dy) < 1e-4 and result.method == "translation"):
            return img.copy()

        if result.method == "translation":
            # Image translation: positive dx moves aux right onto beauty -> warpAffine with dx, dy
            M = np.float32([[1, 0, result.dx], [0, 1, result.dy]])
            return cv2.warpAffine(img, M, (w, h), flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        elif result.method in ("similarity", "affine"):
            M = np.array(result.transform_matrix, dtype=np.float32)
            return cv2.warpAffine(img, M, (w, h), flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        elif result.method == "homography":
            H = np.array(result.transform_matrix, dtype=np.float32)
            return cv2.warpPerspective(img, H, (w, h), flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        elif result.method == "local_stmap":
            grid = align_grid.read_grid(Path(field_path)) if field_path is not None and Path(field_path).exists() else None
            if grid is None and _is_identity(_frame_matrix_cv(result)):
                return img.copy()
            map_x, map_y = compose_remap(result, grid, w, h, plate_w=plate_w)
            return cv2.remap(img, map_x, map_y, interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        return img.copy()

    @classmethod
    def warp_sequence(
        cls,
        aux_pattern: str,
        report: AlignmentReport,
        dest_dir: Path,
        grid_dir: Optional[Path] = None,
        progress_cb: Optional[Any] = None
    ) -> List[Path]:
        """Warps entire sequence and writes OpenEXR frames to dest_dir with a README.md."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        written_files: List[Path] = []
        total = len(report.frames)

        for idx, f_res in enumerate(report.frames):
            f_num = f_res.frame
            src_path_str = aux_pattern.replace("####", f"{f_num:04d}").replace("%04d", f"{f_num:04d}")
            src_file = Path(src_path_str)
            if not src_file.exists():
                continue

            dest_file = dest_dir / src_file.name

            # If 0 displacement, we can copy or hardlink
            if f_res.method == "identity" or (abs(f_res.dx) < 1e-4 and abs(f_res.dy) < 1e-4 and f_res.method == "translation"):
                try:
                    os.link(str(src_file), str(dest_file))
                except (OSError, AttributeError):
                    import shutil
                    shutil.copyfile(str(src_file), str(dest_file))
                written_files.append(dest_file)
                if progress_cb:
                    progress_cb(idx + 1, total, f_num)
                continue

            # Otherwise warp and write
            img = imread_unchanged(src_file)
            grid_f = (grid_dir / (f_res.grid_rel_path or f"grid_{f_num:04d}.exr")) if grid_dir else None
            warped = cls.warp_frame(img, f_res, field_path=grid_f)

            # Write OpenEXR float32
            cv2.imwrite(str(dest_file), warped)
            written_files.append(dest_file)

            if progress_cb:
                progress_cb(idx + 1, total, f_num)

        # Write README.md in dest_dir
        readme_path = dest_dir / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(f"""# Aligned Plate: {report.plate_name}
Wygenerowano przez MatteDesk (Stage E1b) · {datetime.now().isoformat()}

- Plate pomocniczy: `{report.plate_name}` ({report.aux_plate_type})
- Plate referencyjny: `{report.reference_plate}`
- Zakres klatek: F{report.frame_range[0]}–F{report.frame_range[1]}
- Zastosowana metoda: `{report.method}`
- Reszta błędu mediana: {report.residual_median:.3f} px
- Reszta błędu max: {report.residual_max:.3f} px
- Status korekty: {report.recommendation}
- Plik Nuke: `Align_{report.plate_name}.nk`
""")

        return written_files


def _frame_path(pattern: str, f: int) -> str:
    return pattern.replace("####", f"{f:04d}").replace("%04d", f"{f:04d}")


def _global_frame_job(job: Tuple[int, str, str, str, float, Any]) -> Optional[AlignmentFrameResult]:
    """One frame of the global model (runs in a pool process). None when a file is missing."""
    f_num, b_path, a_path, mode, scale, exclude_box = job
    if not os.path.exists(b_path) or not os.path.exists(a_path):
        return None
    img_b = imread_unchanged(b_path)
    img_a = imread_unchanged(a_path)
    feat_b = AlignmentFeatureExtractor.extract(img_b, mode=mode, scale=scale)
    feat_a = AlignmentFeatureExtractor.extract(img_a, mode=mode, scale=scale)
    static_mask = compute_static_mask(img_b, img_a, mode=mode, scale=scale, exclude_box=exclude_box)
    method, params, med_err, max_err = GlobalAligner(scale=scale).select_best_model(feat_a, feat_b, static_mask)
    return AlignmentFrameResult(
        frame=f_num, method=method, residual_px=med_err, residual_max_px=max_err,
        dx=params.get("dx", 0.0), dy=params.get("dy", 0.0), rotation_deg=params.get("rotation_deg", 0.0),
        scale=params.get("scale", 1.0), transform_matrix=params.get("matrix"))


def _load_pair_job(job: Tuple[str, str, Tuple[int, int], Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Beauty + aux frame at the local model's work size (pool process: decode + downscale off the main process);
    the aux is first warped by the frame's global model `m` (3x3, full resolution, `warp_global`)."""
    b_path, a_path, size, m = job
    b = cv2.resize(imread_unchanged(b_path), size, interpolation=cv2.INTER_AREA)
    a = cv2.resize(warp_global(imread_unchanged(a_path), m), size, interpolation=cv2.INTER_AREA)
    return b, a


def aux_luma_global(a_path: str, m: Any) -> np.ndarray:
    """Full-resolution luma of an aux frame warped by its global model (what the local model refines on; the GPU
    path decodes with the same function, engine.align_gpu.decode_local_job)."""
    img = imread_unchanged(a_path)
    luma = img.mean(axis=2).astype(np.float32) if img.ndim == 3 else img.astype(np.float32)
    return warp_global(luma, m)


def _local_frame_job(job: Tuple[Any, ...]) -> Optional[Dict[str, Any]]:
    """One frame of the local model: full-res refinement of the tracked offsets on the aux pre-aligned by the
    global model, outlier rejection, thin-plate field -> grid EXR."""
    b_path, a_path, mode, plate_w, plate_h, grid_spacing, pts, disp, grid_path, m = job
    if len(pts) < 8:
        return None
    feat_b = AlignmentFeatureExtractor.extract(imread_unchanged(b_path), mode=mode, scale=1.0)
    feat_a = AlignmentFeatureExtractor.extract(aux_luma_global(a_path, m), mode=mode, scale=1.0)
    disp, refined, resp = refine_local_offsets(feat_b, feat_a, pts, disp)
    patch = LocalPatchAligner(plate_w, plate_h, grid_spacing=grid_spacing)
    grid, ho_med, ho_max = patch.fit_rbf_grid(pts, disp, resp=resp)
    align_grid.write_grid(Path(grid_path), grid)
    return {"ho_med": ho_med, "ho_max": ho_max, "ho_frac_ok": getattr(patch, "last_holdout_frac_ok", 0.0),
            "refined_frac": float(refined.mean()) if len(refined) else 0.0, "grid": grid_path,
            "n_points": int(len(pts)), "n_rejected": int(patch.last_n_rejected), "reject": patch.last_reject}


def _warped_coverage(mats: List[np.ndarray], plate_w: int, plate_h: int, margin: int) -> np.ndarray:
    """(plate_h, plate_w) float32 mask of the beauty pixels that the warped aux covers on EVERY frame, eroded by
    `margin` px, so no measurement patch sees the black border the global pre-warp leaves outside the plate."""
    cover = np.ones((plate_h, plate_w), np.float32)
    ones = np.ones((plate_h, plate_w), np.float32)
    seen = set()
    for m in mats:
        key = tuple(np.round(np.asarray(m, np.float64).ravel(), 6))
        if key in seen:
            continue
        seen.add(key)
        if not _is_identity(m):
            cover = np.minimum(cover, warp_global(ones, m, interpolation=cv2.INTER_NEAREST))
    if margin > 0 and (cover < 1.0).any():
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * margin + 1, 2 * margin + 1))
        cover = cv2.erode((cover >= 1.0).astype(np.float32), k)
    return cover


class AlignmentPipeline:
    """Coordinates measurement, temporal smoothing, Nuke export, and plate warping."""

    @classmethod
    def measure_sequence(
        cls,
        beauty_pattern: str,
        aux_pattern: str,
        first_frame: int,
        last_frame: int,
        plate_name: str,
        aux_plate_type: str,
        plate_w: int,
        plate_h: int,
        exclude_box: Optional[Tuple[int, int, int, int]] = None,
        progress_cb: Optional[Any] = None,
        grid_dir: Optional[Path] = None,
        tracker: Any = None,
        workers: Optional[int] = None,
        step_cb: Optional[Callable[[str], None]] = None,
        cancel_check: Optional[Callable[[], None]] = None,
        device: Optional[str] = None,
        gpu_python: Optional[str] = None,
        gpu_env: Optional[Dict[str, str]] = None,
    ) -> AlignmentReport:
        """Measures full sequence alignment between beauty and auxiliary plate.

        When the best global model leaves a residual >= 0.5 px (or max > 0.5 px) and `grid_dir` is given, the local
        model (SPEC 1b, `LocalTemporalAligner`) runs and per-frame alignment grids (engine.align_grid) are written
        to `grid_dir`.
        Frames are measured in parallel processes (`engine.procpool`, `workers`; 1 = in this process, same result).
        `device` auto | cpu | gpu (None = config `align_device`): gpu = the same models and thresholds computed by
        engine/align_gpu_worker.py in ComfyUI's Python (engine.align_gpu); auto falls back to this CPU code, the
        report says where it ran (`device`, `device_note`). gpu_python / gpu_env: tests.
        progress_cb(done, total, frame) per frame, step_cb(text) at each phase; both, and cancel_check(), may raise
        to cancel ("Przerwij").
        """
        from engine import align_gpu
        scale = 0.5
        total_frames = last_frame - first_frame + 1
        jobs = [(f, _frame_path(beauty_pattern, f), _frame_path(aux_pattern, f), aux_plate_type, scale, exclude_box)
                for f in range(first_frame, last_frame + 1)]

        def on_done(n: int, i: int) -> None:
            if progress_cb:
                progress_cb(n, total_frames, jobs[i][0])

        if device is None:
            device = os.environ.get("MD_ALIGN_DEVICE")      # tests pin "cpu" (tests/conftest.py)
        if device is None:
            from engine.config import load_config
            device = str(load_config().get("align_device") or "auto")
        strict = device.lower() == "gpu"
        dev, note = align_gpu.resolve_device(device, gpu_python, gpu_env)
        gpu = align_gpu.open_aligner(gpu_python, gpu_env) if dev == "gpu" else None
        n_dec = 1 if workers == 1 else min(align_gpu.decode_workers(), workers or 64)
        used: List[str] = []
        try:
            if step_cb:
                step_cb("Wyrównanie: model globalny" + (" (GPU)" if gpu else ""))
            results: Optional[List[Any]] = None
            if gpu is not None:
                try:
                    results = align_gpu.global_frames(gpu, jobs, n_dec, on_done, cancel_check)
                    used.append(gpu.device)
                except align_gpu.GpuWorkerError as e:
                    if strict:
                        raise
                    logger.warning("alignment GPU failed, global model on CPU: %s", e)
                    note = f"GPU zawiodło ({e}) – liczone na CPU"
                    align_gpu.close_aligner(gpu)
                    gpu = None
            if results is None:
                results = run_frames(_global_frame_job, jobs, workers, on_done)
                used.append("cpu")
            frame_results: List[AlignmentFrameResult] = [r for r in results if r is not None]

            if not frame_results:
                raise ValueError(f"No frames could be read for alignment from {aux_pattern}")

            # Temporal trajectory smoothing (median filter radius 2 on shifts)
            cls._smooth_trajectory(frame_results)

            local_engine = ""
            # global model insufficient (SPEC 4a pkt 1): residual > 0.5 px or uneven residual map (max > 0.5 px)
            if grid_dir is not None and (float(np.median([f.residual_px for f in frame_results])) >= 0.5
                                          or float(np.median([f.residual_max_px for f in frame_results])) > 0.5):
                run_local = lambda g: cls._run_local_model(  # noqa: E731
                    beauty_pattern, aux_pattern, frame_results, aux_plate_type, plate_w, plate_h,
                    exclude_box, Path(grid_dir), tracker, workers=workers, progress_cb=progress_cb, step_cb=step_cb,
                    cancel_check=cancel_check, gpu=g, decode_workers=n_dec)
                local_engine = ""
                if gpu is not None:
                    try:
                        local_engine = run_local(gpu)
                        used.append(gpu.device)
                    except align_gpu.GpuWorkerError as e:
                        if strict:
                            raise
                        logger.warning("alignment GPU failed, local model on CPU: %s", e)
                        note = f"GPU zawiodło w modelu lokalnym ({e}) – liczone na CPU"
                        align_gpu.close_aligner(gpu)
                        gpu = None
                if not local_engine:
                    local_engine = run_local(None)
                    used.append("cpu")
        finally:
            align_gpu.close_aligner(gpu)
        used_dev = "+".join(dict.fromkeys(used))

        # Sequence-level statistics
        all_meds = [f.residual_px for f in frame_results]
        all_maxs = [f.residual_max_px for f in frame_results]
        seq_med = float(np.median(all_meds))
        seq_max = float(np.max(all_maxs))

        # Determine dominant method
        methods = [f.method for f in frame_results]
        dominant_method = max(set(methods), key=methods.count)

        # Thresholds per SPEC 4a: median > 0.3 px or max > 0.75 px
        needs_correction = (seq_med > 0.3) or (seq_max > 0.75)

        if not needs_correction:
            recommendation = f"{aux_plate_type.capitalize()} zgodny (max {seq_max:.2f} px) – nic nie rób"
        else:
            recommendation = (
                f"{aux_plate_type.capitalize()} przesunięty: max {seq_max:.2f} px, "
                f"F{first_frame}–{last_frame} (zalecana korekta metodą {dominant_method})"
            )

        report = AlignmentReport(
            plate_name=plate_name,
            aux_plate_type=aux_plate_type,
            reference_plate="beauty",
            frame_range=(first_frame, last_frame),
            method=dominant_method,
            residual_median=seq_med,
            residual_max=seq_max,
            needs_correction=needs_correction,
            recommendation=recommendation,
            frames=frame_results,
            local_tracker_engine=local_engine,
            device=used_dev,
            device_note=note if "cpu" in used or dev == "cpu" else "",
        )
        if local_engine:
            report.recommendation += f" · dopasowanie lokalne (śledzenie: {local_engine})"
        return report

    @staticmethod
    def _run_local_model(beauty_pattern: str, aux_pattern: str, frame_results: List[AlignmentFrameResult],
                         aux_plate_type: str, plate_w: int, plate_h: int,
                         exclude_box: Optional[Tuple[int, int, int, int]], grid_dir: Path, tracker: Any,
                         workers: Optional[int] = None, progress_cb: Optional[Any] = None,
                         step_cb: Optional[Callable[[str], None]] = None,
                         cancel_check: Optional[Callable[[], None]] = None, gpu: Any = None,
                         decode_workers: Optional[int] = None) -> str:
        """Local model; gpu (engine.align_gpu.GpuAligner) computes the anchor offsets and per-frame fields there,
        with `decode_workers` processes for decoding instead of `workers`."""
        from engine import align_gpu
        frames = [r.frame for r in frame_results]
        paths = [(_frame_path(beauty_pattern, f), _frame_path(aux_pattern, f)) for f in frames]

        def on_done(n: int, i: int) -> None:
            if progress_cb:
                progress_cb(n, len(frames), frames[i])

        def step(text: str) -> None:
            if step_cb:
                step_cb(text)
            if cancel_check:
                cancel_check()

        # the local model is measured on the aux pre-aligned by each frame's GLOBAL model (the grid = the residual)
        mats = [_frame_matrix_cv(r) for r in frame_results]
        local = LocalTemporalAligner(plate_w, plate_h, tracker=tracker, mode=aux_plate_type)
        aux0 = warp_global(imread_unchanged(paths[0][1]), mats[0])
        mask_s = compute_static_mask(imread_unchanged(paths[0][0]), aux0, mode=aux_plate_type,
                                     scale=0.5, exclude_box=exclude_box)
        mask = cv2.resize(mask_s.astype(np.float32), (plate_w, plate_h), interpolation=cv2.INTER_NEAREST)
        mask *= _warped_coverage(mats, plate_w, plate_h, margin=LOCAL_PATCH_PX // 2 + 8)
        del aux0

        load_workers = workers
        if gpu is not None:
            local.measure_patches = align_gpu.make_patch_measure(gpu, cancel_check)
            load_workers = decode_workers
        step("Dopasowanie lokalne: wczytywanie klatek")
        small = run_frames(_load_pair_job, [(b, a, local.work_size(), m) for (b, a), m in zip(paths, mats)],
                           load_workers, on_done)
        step(f"Dopasowanie lokalne: śledzenie punktów ({local.engine})")
        fields = local.measure([b for b, _ in small], [a for _, a in small], mask, cancel_check=cancel_check)
        del small

        step("Dopasowanie lokalne: doprecyzowanie w pełnej rozdzielczości + siatka" + (" (GPU)" if gpu else ""))
        grid_dir.mkdir(parents=True, exist_ok=True)
        grids = [str(grid_dir / f"grid_{f:04d}.exr") for f in frames]
        if gpu is not None:
            results = align_gpu.local_frames(gpu, paths, fields, aux_plate_type, plate_w, plate_h, grids,
                                             decode_workers or 1, on_done, cancel_check, mats=mats,
                                             patch=LOCAL_PATCH_PX)
        else:
            jobs = [(b, a, aux_plate_type, plate_w, plate_h, local.grid_spacing, pts, disp, sp, m)
                    for (b, a), (pts, disp), sp, m in zip(paths, fields, grids, mats)]
            results = run_frames(_local_frame_job, jobs, workers, on_done)
        for r, res in zip(frame_results, results):
            if res is None:
                r.status = "uncertain"
                continue
            r.global_method = r.method          # the model the grid is the residual over (matrix / dx, dy stay)
            r.method = "local_stmap"
            r.residual_px, r.residual_max_px = res["ho_med"], res["ho_max"]
            r.holdout_frac_ok = float(res["ho_frac_ok"])
            r.n_points, r.n_rejected = int(res.get("n_points", 0)), int(res.get("n_rejected", 0))
            r.grid_rel_path = Path(res["grid"]).name
            r.status = "ok" if res["ho_frac_ok"] >= 0.9 else "uncertain"
        return local.engine

    @staticmethod
    def _smooth_trajectory(frames: List[AlignmentFrameResult]) -> None:
        """Applies median filter (window 5 / radius 2) to translation trajectories, flagging outliers."""
        if len(frames) < 5:
            return

        dxs = np.array([f.dx for f in frames])
        dys = np.array([f.dy for f in frames])

        # Median filter
        from scipy.ndimage import median_filter
        smooth_dx = median_filter(dxs, size=5, mode="nearest")
        smooth_dy = median_filter(dys, size=5, mode="nearest")

        for idx, f in enumerate(frames):
            orig_dist = math.hypot(f.dx - smooth_dx[idx], f.dy - smooth_dy[idx])
            if orig_dist > 1.5:  # Significant outlier
                f.status = "outlier"
            else:
                f.dx = float(smooth_dx[idx])
                f.dy = float(smooth_dy[idx])
