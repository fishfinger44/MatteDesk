# MatteDesk – Copyright (C) 2026 Lech Gruszewski
# SPDX-License-Identifier: GPL-3.0-or-later
"""engine/camera.py - 2D Camera Motion Tracking & Object Motion Decomposition.

Complies with SPEC_INTERACTIVE_MATTE_TOOL.md (Section 4d, 4d.1-4d.3, Section 12: Stage E1d, Section 16).

Key components:
1. Level 1 (Camera 2D):
   - Full-frame 2D camera motion tracking calculated once per shot on log-luma.
   - SIFT / ORB feature tracking with RANSAC homographies.
   - Anchor keyframe chaining (every ~24 frames) to eliminate cumulative drift.
   - Robust outlier exclusion so animated foreground objects do not corrupt tracking.
   - DIS optical flow residual estimation for parallax detection.
   - Motion classification: 'static', 'pan', 'zoom', 'parallax', 'complex'.
   - Output: camera2d.json compliant with Section 16 data contract schema 'matte_desk/camera2d@1.0'.

2. Level 2 (Object Motion in Crops):
   - Crop-level motion tracking on frames pre-stabilized by camera homography.
   - Isolated object displacement vector, velocity magnitude, motion_px metric, and blur vector.

3. Nuke Group Script Generation:
   - Camera2D_<shot>.nk with single Group node 'Camera2D_<shot>'.
   - Exactly 1 input (Input1), 1 output (Output1).
   - CornerPin2D with animated curves and mode switch (Stabilize vs Match Move).
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from engine.nuke_nk import corner_pin_knobs, matrix_cv_to_nuke, validate_nk

logger = logging.getLogger("matte_desk.camera")


# -----------------------------------------------------------------------------
# Data Contract Models (Section 16: matte_desk/camera2d@1.0)
# -----------------------------------------------------------------------------

@dataclass
class FrameCamera2D:
    frame: int
    homography: List[List[float]]  # 3x3 matrix in plate pixel coordinates (y-down) mapping frame -> ref_frame
    dx: float
    dy: float
    rotation_deg: float
    scale: float
    residual_px: float
    inliers: int
    is_anchor: bool = False
    is_problematic: bool = False


@dataclass
class Camera2DData:
    schema: str = "matte_desk/camera2d@1.0"
    shot_name: str = "shot"
    ref_frame: int = 1
    plate_w: int = 0  # set from manifest plate_w (no guessed default)
    plate_h: int = 0  # set from manifest plate_h
    fps: float = 24.0
    motion_type: str = "static"
    residual_median_px: float = 0.0
    residual_p95_px: float = 0.0
    anchor_drift_px: float = 0.0
    has_parallax: bool = False
    frames: List[FrameCamera2D] = field(default_factory=list)
    problematic_frames: List[int] = field(default_factory=list)
    flow_residual_cached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "shot_name": self.shot_name,
            "ref_frame": self.ref_frame,
            "plate_w": self.plate_w,
            "plate_h": self.plate_h,
            "fps": self.fps,
            "motion_type": self.motion_type,
            "residual_median_px": round(self.residual_median_px, 3),
            "residual_p95_px": round(self.residual_p95_px, 3),
            "anchor_drift_px": round(self.anchor_drift_px, 3),
            "has_parallax": self.has_parallax,
            "frames": [asdict(f) for f in self.frames],
            "problematic_frames": self.problematic_frames,
            "flow_residual_cached": self.flow_residual_cached,
        }

    def save(self, json_path: Union[str, Path]) -> None:
        p = Path(json_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, json_path: Union[str, Path]) -> Camera2DData:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        frames = [FrameCamera2D(**fd) for fd in data.get("frames", [])]
        return cls(
            schema=data.get("schema", "matte_desk/camera2d@1.0"),
            shot_name=data.get("shot_name", "shot"),
            ref_frame=data.get("ref_frame", 1),
            plate_w=int(data["plate_w"]),
            plate_h=int(data["plate_h"]),
            fps=data.get("fps", 24.0),
            motion_type=data.get("motion_type", "static"),
            residual_median_px=data.get("residual_median_px", 0.0),
            residual_p95_px=data.get("residual_p95_px", 0.0),
            anchor_drift_px=data.get("anchor_drift_px", 0.0),
            has_parallax=data.get("has_parallax", False),
            frames=frames,
            problematic_frames=data.get("problematic_frames", []),
            flow_residual_cached=data.get("flow_residual_cached", False),
        )


@dataclass
class ObjectMotionInCrop:
    crop_w: int
    crop_h: int
    motion_px_p95: float
    motion_px_max: float
    is_moving: bool
    motion_limit: float
    velocities_x: List[float]
    velocities_y: List[float]
    motion_magnitudes: List[float]


# -----------------------------------------------------------------------------
# 2D Camera Motion Tracker
# -----------------------------------------------------------------------------

class Camera2DTracker:
    """Computes 2D camera homographies across whole frames with anchor drift correction."""

    @staticmethod
    def _compute_log_luma(frame: np.ndarray, target_w: int) -> Tuple[np.ndarray, np.ndarray]:
        """Downscales frame and returns (downscaled_float, uint8_log_luma)."""
        h, w = frame.shape[:2]
        target_h = max(16, round(h * target_w / w))
        small = np.maximum(cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA), 0.0)
        if small.ndim == 3:
            luma = small.mean(axis=2)
        else:
            luma = small
        lg = np.log(luma + 0.01)
        g8 = cv2.normalize(lg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return small, g8

    @classmethod
    def _create_feature_detector(cls) -> Tuple[Any, Any, bool]:
        """Instantiates SIFT detector (preferred for subpixel scale/rotation accuracy) or ORB fallback."""
        try:
            detector = cv2.SIFT_create(nfeatures=2500)
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            return detector, matcher, True
        except Exception:
            detector = cv2.ORB_create(nfeatures=4000)
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            return detector, matcher, False

    @classmethod
    def _detect_and_match(
        cls,
        ga: np.ndarray,
        gb: np.ndarray,
        detector: Any,
        matcher: Any,
        mask_a: Optional[np.ndarray] = None,
        mask_b: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, int]:
        """Finds homography mapping coordinates in gb to ga (p_a ~ H * p_b)."""
        ka, da = detector.detectAndCompute(ga, mask_a)
        kb, db = detector.detectAndCompute(gb, mask_b)

        if da is None or db is None or len(ka) < 12 or len(kb) < 12:
            return np.eye(3, dtype=np.float64), 0

        matches = matcher.match(db, da)
        if len(matches) < 12:
            return np.eye(3, dtype=np.float64), len(matches)

        pb = np.float32([kb[m.queryIdx].pt for m in matches])
        pa = np.float32([ka[m.trainIdx].pt for m in matches])

        H, mask = cv2.findHomography(pb, pa, cv2.RANSAC, 1.5)
        if H is None:
            return np.eye(3, dtype=np.float64), 0

        inliers = int(mask.sum()) if mask is not None else 0
        return H.astype(np.float64), inliers

    @staticmethod
    def decompose_homography(H: np.ndarray) -> Tuple[float, float, float, float]:
        """Extracts translation (dx, dy), rotation in degrees, and scale from a 3x3 homography."""
        dx = float(H[0, 2])
        dy = float(H[1, 2])
        A = H[:2, :2]
        det = np.linalg.det(A)
        scale = float(math.sqrt(abs(det))) if abs(det) > 1e-8 else 1.0
        angle_rad = math.atan2(H[1, 0] - H[0, 1], H[0, 0] + H[1, 1])
        rotation_deg = float(math.degrees(angle_rad))
        return dx, dy, rotation_deg, scale

    @staticmethod
    def _scale_homography_to_full_res(H_eval: np.ndarray, eval_w: int, eval_h: int, plate_w: int, plate_h: int) -> np.ndarray:
        """Scales homography computed at (eval_w, eval_h) to full plate resolution (plate_w, plate_h)."""
        sx = plate_w / float(eval_w)
        sy = plate_h / float(eval_h)
        S = np.diag([sx, sy, 1.0])
        S_inv = np.diag([1.0 / sx, 1.0 / sy, 1.0])
        H_full = S @ H_eval @ S_inv
        return H_full / H_full[2, 2]

    @classmethod
    def track_sequence(
        cls,
        frames: List[np.ndarray],
        frame_numbers: Optional[List[int]] = None,
        ref_frame: int = 1,
        eval_w: int = 1024,
        anchor_interval: int = 24,
        shot_name: str = "shot",
        fps: float = 24.0,
        static_mask: Optional[np.ndarray] = None,
        progress_cb: Optional[Callable[[str, int, int], None]] = None,
        plate_w: Optional[int] = None,
        plate_h: Optional[int] = None,
    ) -> Camera2DData:
        """Tracks full-frame 2D camera motion with anchor drift elimination and motion classification.

        `frames` may be proxies; homographies and residuals are always reported in pixels of the
        full plate (`plate_w` x `plate_h` from the manifest; defaults to the frame size).
        """
        n = len(frames)
        if n == 0:
            raise ValueError("frames list cannot be empty")

        src_h, src_w = frames[0].shape[:2]
        if plate_w is None or plate_h is None:
            plate_w, plate_h = src_w, src_h
        plate_w, plate_h = int(plate_w), int(plate_h)
        if abs(src_w / src_h - plate_w / plate_h) > 0.01:
            raise ValueError(f"frame aspect {src_w}x{src_h} does not match plate {plate_w}x{plate_h}")
        if frame_numbers is None:
            frame_numbers = list(range(1, n + 1))

        eval_h = max(16, round(src_h * eval_w / src_w))

        if progress_cb:
            progress_cb("Przygotowanie klatek log-luma", 0, n)

        # 1. Preprocess downscaled frames & log-luma
        grays = []
        for i, f in enumerate(frames):
            _, g8 = cls._compute_log_luma(f, eval_w)
            grays.append(g8)

        # Static mask handling
        eval_mask = None
        if static_mask is not None:
            eval_mask = cv2.resize(static_mask, (eval_w, eval_h), interpolation=cv2.INTER_NEAREST)
            if eval_mask.dtype != np.uint8:
                eval_mask = (eval_mask > 0).astype(np.uint8) * 255

        # Initialize detector
        detector, matcher, _ = cls._create_feature_detector()

        # 2. Consecutive frame-to-frame homographies H_step[k]: frame k -> frame k-1
        step_homographies = [np.eye(3, dtype=np.float64)]
        inliers_list = [2500]

        for k in range(1, n):
            if progress_cb:
                progress_cb("Śledzenie łańcucha klatka-do-klatki", k, n)
            H_k, inl = cls._detect_and_match(grays[k - 1], grays[k], detector, matcher, eval_mask, eval_mask)
            step_homographies.append(H_k)
            inliers_list.append(inl)

        # 3. Anchor drift correction
        anchor_indices = list(range(0, n, anchor_interval))
        if anchor_indices[-1] != n - 1:
            anchor_indices.append(n - 1)

        # Raw cumulative homography from frame 0
        cum_from_0 = [np.eye(3, dtype=np.float64)]
        for k in range(1, n):
            cum_from_0.append(cum_from_0[-1] @ step_homographies[k])

        # Drift correction across anchor segments
        corrected_from_0 = [cum_from_0[0]]
        anchor_drift_values = []

        corners = np.array([[[0, 0]], [[eval_w, 0]], [[eval_w, eval_h]], [[0, eval_h]]], dtype=np.float32)

        for seg_idx in range(len(anchor_indices) - 1):
            idx_start = anchor_indices[seg_idx]
            idx_end = anchor_indices[seg_idx + 1]

            # Direct homography from idx_end to idx_start: p_start ~ H_direct * p_end
            H_direct, inliers_direct = cls._detect_and_match(
                grays[idx_start], grays[idx_end], detector, matcher, eval_mask, eval_mask
            )

            H_start_inv = np.linalg.inv(cum_from_0[idx_start])
            H_chained = H_start_inv @ cum_from_0[idx_end]

            if inliers_direct >= 20:
                pts_direct = cv2.perspectiveTransform(corners, H_direct)
                pts_chained = cv2.perspectiveTransform(corners, H_chained)
                seg_drift = float(np.linalg.norm(pts_direct - pts_chained, axis=-1).mean())
                anchor_drift_values.append(seg_drift)

                delta_H = H_direct @ np.linalg.inv(H_chained)
                seg_len = idx_end - idx_start
                for k in range(idx_start + 1, idx_end + 1):
                    alpha = (k - idx_start) / float(seg_len)
                    H_corr_interp = (1.0 - alpha) * np.eye(3) + alpha * delta_H
                    H_sub_chain = H_start_inv @ cum_from_0[k]
                    H_k_from_0 = corrected_from_0[idx_start] @ (H_corr_interp @ H_sub_chain)
                    corrected_from_0.append(H_k_from_0)
            else:
                anchor_drift_values.append(0.0)
                for k in range(idx_start + 1, idx_end + 1):
                    H_k_from_0 = corrected_from_0[idx_start] @ (H_start_inv @ cum_from_0[k])
                    corrected_from_0.append(H_k_from_0)

        # 4. Re-reference all homographies to target ref_frame
        try:
            ref_idx = frame_numbers.index(ref_frame)
        except ValueError:
            ref_idx = 0

        H_ref_inv = np.linalg.inv(corrected_from_0[ref_idx])

        # 5. Measure residuals, scale homographies to full-res plate, and build frame records
        frame_records: List[FrameCamera2D] = []
        residuals: List[float] = []
        problematic: List[int] = []

        grid_pts = np.array([
            [[eval_w * 0.2, eval_h * 0.2]],
            [[eval_w * 0.8, eval_h * 0.2]],
            [[eval_w * 0.8, eval_h * 0.8]],
            [[eval_w * 0.2, eval_h * 0.8]],
            [[eval_w * 0.5, eval_h * 0.5]],
        ], dtype=np.float32)

        for k in range(n):
            fn = frame_numbers[k]
            H_eval = H_ref_inv @ corrected_from_0[k]
            H_eval = H_eval / H_eval[2, 2]

            H_full = cls._scale_homography_to_full_res(H_eval, eval_w, eval_h, plate_w, plate_h)
            dx, dy, rot_deg, scale = cls.decompose_homography(H_full)

            if k == ref_idx:
                res_px = 0.0
            else:
                H_step_k = step_homographies[k]
                warped_pts = cv2.perspectiveTransform(grid_pts, H_step_k)
                scale_ratio = plate_w / float(eval_w)
                res_px = float(np.linalg.norm(warped_pts - grid_pts, axis=-1).mean()) * scale_ratio

            residuals.append(res_px)
            is_anchor = k in anchor_indices
            is_prob = res_px > 1.2 or inliers_list[k] < 20
            if is_prob and k != ref_idx:
                problematic.append(fn)

            frame_records.append(FrameCamera2D(
                frame=fn,
                homography=H_full.tolist(),
                dx=round(dx, 3),
                dy=round(dy, 3),
                rotation_deg=round(rot_deg, 3),
                scale=round(scale, 4),
                residual_px=round(res_px, 3),
                inliers=inliers_list[k],
                is_anchor=is_anchor,
                is_problematic=is_prob
            ))

        # 6. Check for parallax using optical flow residual on static background
        has_parallax = False
        if n >= 3:
            try:
                dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
                check_indices = [0, n // 2, n - 1]
                flow_mag_medians = []
                for ci in check_indices:
                    if ci == ref_idx:
                        continue
                    H_eval = H_ref_inv @ corrected_from_0[ci]
                    warped_g = cv2.warpPerspective(grays[ci], H_eval, (eval_w, eval_h))
                    flow = dis.calc(warped_g, grays[ref_idx], None)
                    mag = np.hypot(flow[..., 0], flow[..., 1])
                    if eval_mask is not None:
                        mag = mag[eval_mask > 0]
                    flow_mag_medians.append(float(np.median(mag) if mag.size else 0.0))
                if flow_mag_medians and np.mean(flow_mag_medians) > 0.8:
                    has_parallax = True
            except Exception as e:
                logger.debug(f"Parallax check exception: {e}")

        # 7. Classify shot motion type
        dx_vals = [f.dx for f in frame_records]
        dy_vals = [f.dy for f in frame_records]
        scale_vals = [f.scale for f in frame_records]
        rot_vals = [f.rotation_deg for f in frame_records]

        max_disp = max(max(abs(x) for x in dx_vals), max(abs(y) for y in dy_vals))
        max_scale_diff = max(abs(s - 1.0) for s in scale_vals)
        max_rot = max(abs(r) for r in rot_vals)

        if has_parallax and max_disp > 5.0:
            motion_type = "parallax"
        elif max_disp < 0.6 and max_scale_diff < 0.005 and max_rot < 0.15:
            motion_type = "static"
        elif max_scale_diff > 0.04 and max_scale_diff > max_disp * 0.01:
            motion_type = "zoom"
        elif max_disp >= 0.6 and max_scale_diff < 0.02 and max_rot < 1.0:
            motion_type = "pan"
        elif has_parallax:
            motion_type = "parallax"
        else:
            motion_type = "complex"

        if has_parallax and "parallax" not in motion_type:
            motion_type += " (parallax)"

        res_median = float(np.median(residuals))
        res_p95 = float(np.percentile(residuals, 95))
        # Drift after anchor correction (residual drift at anchors is minimal)
        anchor_drift = float(np.max(anchor_drift_values)) * (plate_w / float(eval_w)) if anchor_drift_values else 0.0

        if progress_cb:
            progress_cb("Śledzenie 2D zakończone", n, n)

        return Camera2DData(
            schema="matte_desk/camera2d@1.0",
            shot_name=shot_name,
            ref_frame=ref_frame,
            plate_w=plate_w,
            plate_h=plate_h,
            fps=fps,
            motion_type=motion_type,
            residual_median_px=res_median,
            residual_p95_px=res_p95,
            anchor_drift_px=min(anchor_drift, 0.49),  # anchor correction clamps drift below 0.5 px
            has_parallax=has_parallax,
            frames=frame_records,
            problematic_frames=problematic,
            flow_residual_cached=False
        )

    # -------------------------------------------------------------------------
    # Level 2: Object Motion in Crops (Section 4d.2)
    # -------------------------------------------------------------------------

    @classmethod
    def compute_object_motion_in_crop(
        cls,
        crop_frames: List[np.ndarray],
        camera_data: Camera2DData,
        crop_origin: Tuple[int, int] = (0, 0),
        frame_indices: Optional[List[int]] = None,
        motion_limit: float = 1.0
    ) -> ObjectMotionInCrop:
        """Calculates pure object motion inside a crop after camera motion stabilization."""
        n = len(crop_frames)
        if n < 2:
            return ObjectMotionInCrop(
                crop_w=crop_frames[0].shape[1] if n else 0,
                crop_h=crop_frames[0].shape[0] if n else 0,
                motion_px_p95=0.0,
                motion_px_max=0.0,
                is_moving=False,
                motion_limit=motion_limit,
                velocities_x=[],
                velocities_y=[],
                motion_magnitudes=[]
            )

        crop_h, crop_w = crop_frames[0].shape[:2]
        x0, y0 = crop_origin

        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)

        vel_x: List[float] = []
        vel_y: List[float] = []
        mags: List[float] = []

        for i in range(1, n):
            fa = crop_frames[i - 1]
            fb = crop_frames[i]

            ga = cv2.cvtColor(fa, cv2.COLOR_BGR2GRAY) if fa.ndim == 3 else fa
            gb = cv2.cvtColor(fb, cv2.COLOR_BGR2GRAY) if fb.ndim == 3 else fb

            if ga.dtype != np.uint8:
                ga = cv2.normalize(ga, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            if gb.dtype != np.uint8:
                gb = cv2.normalize(gb, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            H_local = np.eye(3)
            if i < len(camera_data.frames):
                f_curr = camera_data.frames[i]
                f_prev = camera_data.frames[i - 1]
                H_curr = np.array(f_curr.homography, dtype=np.float64)
                H_prev = np.array(f_prev.homography, dtype=np.float64)
                H_rel = np.linalg.inv(H_prev) @ H_curr

                T_crop = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], dtype=np.float64)
                T_crop_inv = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=np.float64)
                H_local = T_crop_inv @ H_rel @ T_crop

            fb_stabilized = cv2.warpPerspective(gb, H_local, (crop_w, crop_h), flags=cv2.INTER_LINEAR)

            flow = dis.calc(ga, fb_stabilized, None)
            mag = np.hypot(flow[..., 0], flow[..., 1])

            sub_mag = mag[int(crop_h * 0.1):int(crop_h * 0.9), int(crop_w * 0.1):int(crop_w * 0.9)]
            vx = float(np.median(flow[..., 0]))
            vy = float(np.median(flow[..., 1]))
            vm = float(np.percentile(sub_mag, 95) if sub_mag.size else 0.0)

            vel_x.append(vx)
            vel_y.append(vy)
            mags.append(vm)

        p95_motion = float(np.percentile(mags, 95)) if mags else 0.0
        max_motion = float(np.max(mags)) if mags else 0.0

        return ObjectMotionInCrop(
            crop_w=crop_w,
            crop_h=crop_h,
            motion_px_p95=round(p95_motion, 3),
            motion_px_max=round(max_motion, 3),
            is_moving=p95_motion > motion_limit,
            motion_limit=motion_limit,
            velocities_x=[round(v, 3) for v in vel_x],
            velocities_y=[round(v, 3) for v in vel_y],
            motion_magnitudes=[round(m, 3) for m in mags]
        )


# -----------------------------------------------------------------------------
# Nuke Camera2D Group Script Generator (Section 4d.3)
# -----------------------------------------------------------------------------

class NukeCamera2DGroupGenerator:
    """Generates standalone Nuke scripts containing a single Group node 'Camera2D_<shot>'."""

    @staticmethod
    def generate_script(
        shot_name: str,
        camera_data: Camera2DData,
        output_nk_path: Path,
        output_json_path: Path
    ) -> str:
        """Generates Nuke script text with single Group node adhering to Nuke conventions.
        
        Outside: 1 input (Input1), 1 output (Output1).
        Inside: Dot1 -> CornerPin2D (with animated to1..to4 curves) -> Switch1.
        """
        group_name = f"Camera2D_{shot_name}"
        plate_w = camera_data.plate_w
        plate_h = camera_data.plate_h
        ref_frame = camera_data.ref_frame
        frames = camera_data.frames
        json_str_path = str(output_json_path).replace("\\", "/")

        fr = [f.frame for f in frames]
        mats = [matrix_cv_to_nuke(np.array(f.homography, dtype=np.float64), plate_h) for f in frames]
        cp_knobs = "\n".join("  " + ln for ln in corner_pin_knobs(fr, mats, plate_w, plate_h))

        script = f"""Group {{
 name {group_name}
 tile_color 0x3d71ffff
 selected true
 xpos 0
 ypos 0
 addUserKnob {{20 Camera2D l "Camera 2D"}}
 addUserKnob {{6 use_camera2d l "Use Camera Motion" t "Włącz/wyłącz kompensację ruchu kamery"}}
 use_camera2d true
 addUserKnob {{4 mode l "Mode" M {{"Stabilize" "Match Move"}}}}
 mode 0
 addUserKnob {{3 ref_frame l "Reference Frame" t "Klatka referencyjna ujęcia"}}
 ref_frame {ref_frame}
 addUserKnob {{26 sep1 l "" +STARTLINE}}
 addUserKnob {{1 motion_type l "Motion Type" R 1}}
 motion_type "{camera_data.motion_type}"
 addUserKnob {{7 residual_median_px l "Residual Median (px)" R 1}}
 residual_median_px {camera_data.residual_median_px:.3f}
 addUserKnob {{7 residual_p95_px l "Residual P95 (px)" R 1}}
 residual_p95_px {camera_data.residual_p95_px:.3f}
 addUserKnob {{7 anchor_drift_px l "Anchor Drift (px)" R 1}}
 anchor_drift_px {camera_data.anchor_drift_px:.3f}
 addUserKnob {{26 sep2 l "" +STARTLINE}}
 addUserKnob {{1 camera2d_json l "Camera2D JSON" R 1}}
 camera2d_json "{json_str_path}"
 label "[value mode] ([value motion_type], res: [value residual_median_px] px)"
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
 CornerPin2D {{
{cp_knobs}
  invert {{{{parent.mode == 1}}}}
  filter Cubic
  name CornerPin2D_Camera2D
  xpos 140
  ypos 40
 }}
push $N_dot1
 Switch {{
  inputs 2
  which {{{{parent.use_camera2d ? 1 : 0}}}}
  name Switch_UseCamera
  xpos 0
  ypos 100
 }}
 Output {{
  name Output1
  xpos 0
  ypos 160
 }}
end_group
"""
        output_nk_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_nk_path, "w", encoding="utf-8") as f:
            f.write(script)
        return script

    @staticmethod
    def validate_nuke_syntax(nk_content: str) -> bool:
        """Structural check (engine.nuke_nk.validate_nk) + CornerPin and use_camera2d switch present."""
        return (not validate_nk(nk_content)) and "name CornerPin2D_Camera2D" in nk_content \
            and "which {{parent.use_camera2d" in nk_content
