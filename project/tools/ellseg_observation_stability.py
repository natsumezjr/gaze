from __future__ import annotations

"""
Quantify EllSeg observation stability (pupil/iris geometry jitter).

Goal:
- Under a single gaze point (user keeps still), measure across frames:
  - pupil center frame-to-frame jitter (std of cx/cy)
  - iris center frame-to-frame jitter (std of cx/cy)
  - pupil-iris relative position stability (std of dx/dy, std of distance)
  - ellipse axes/angle jitter (std of major/minor/angle)

This tool is intentionally standalone:
- It uses EyeRoiModel + EllSeg native outputs only.
- It does NOT change any ROI strategies in the main pipeline.

Signal layers (per frame in records.jsonl):
- Raw: pupil_center_roi / pupil_center_network, pupil_mask_centroid, geometry from EllSeg (no EMA).
- Analysis: pupil_center_analysis, analysis_nx/ny, analysis_frame_accept (strict gate, no fallback).
- Display: display_pupil_center_roi / display_iris_center_roi (EMA-smoothed when frame_accept; may reuse previous via fallback).
"""

import argparse
import json
import os
import time
from dataclasses import replace
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from project.core.recognition.eye_model_ellseg import EllSegResult, EyeRoiModel
from project.core.recognition.pupil_tracking import (
    LocalPupilTracker,
    TRACK_MAX_APPLY_SHIFT,
    shift_ellipse_center,
)
from project.core.recognition.roi.geometry import \
    check_observation_plausibility
from project.core.recognition.roi.observation import build_roi_observation
from project.core.recognition.runtime_visualization import (
    ellseg_component_debug_mask_to_bgr, ellseg_segmentation_mask_to_bgr)

FRAME_ACCEPT_DIST_MAX = 4.0
FRAME_ACCEPT_PUPIL_JUMP_MAX = 10.0
FRAME_ACCEPT_IRIS_JUMP_MAX = 10.0
FRAME_ACCEPT_DIST_JUMP_MAX = 4.5
FALLBACK_MAX_CONSECUTIVE = 3
EMA_ALPHA = 0.7
BASELINE_IRIS_EDGE_MARGIN_MIN = 2.0
BASELINE_PUPIL_CENTER_DISAGREEMENT_MAX = 4.5
BASELINE_WARMUP_ACCEPTS = 3

# Analysis signal (strict gate for oculomotor analysis; no EMA / no fallback here).
ANALYSIS_DISAGREEMENT_MAX_PX = 3.5
ANALYSIS_PUPIL_IRIS_DIST_RATIO_MAX = 0.24
ANALYSIS_CLEAN_AREA_RATIO_MIN = 0.6
ANALYSIS_IRIS_EDGE_MARGIN_MIN = 2
ANALYSIS_EPS = 1e-6


def clamp_roi(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    x = max(0, min(int(x), img_w - 1))
    y = max(0, min(int(y), img_h - 1))
    w = max(1, min(int(w), img_w - x))
    h = max(1, min(int(h), img_h - y))
    return x, y, w, h


def fit_circular_std_deg(angles_deg: np.ndarray) -> float:
    """
    Circular std for ellipse angle (0..180 can wrap).
    OpenCV fitEllipse returns angle in degrees in [0,180).
    We map to unit circle at 2x angle to make 0/180 consistent.
    """
    if angles_deg.size == 0:
        return 0.0
    ang = np.deg2rad(angles_deg.astype(np.float64))
    # Use doubled angle so that theta and theta+90 (180 span) are handled more consistently.
    z = np.exp(1j * 2.0 * ang)
    R = np.abs(z.mean())
    if R <= 1e-12:
        return float(np.sqrt(-2.0 * np.log(max(R, 1e-12))))
    # Equivalent circular std estimate on the circle.
    return float(np.sqrt(-2.0 * np.log(R)))


def mean_std(x: np.ndarray) -> Dict[str, float]:
    if x.size == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "rmse": 0.0}
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=0))
    rmse = float(np.sqrt(np.mean((x - mean) ** 2)))
    return {"n": int(x.size), "mean": mean, "std": std, "rmse": rmse}


def dist_stats(x: np.ndarray) -> Dict[str, float]:
    if x.size == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=0)),
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _pair_dist(a: Optional[List[float]], b: Optional[List[float]]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(np.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def _ema_pair(current: List[float], previous: Optional[List[float]], alpha: float) -> List[float]:
    if previous is None:
        return [float(current[0]), float(current[1])]
    return [
        float(alpha * current[0] + (1.0 - alpha) * previous[0]),
        float(alpha * current[1] + (1.0 - alpha) * previous[1]),
    ]


def _ema_scalar(current: float, previous: Optional[float], alpha: float) -> float:
    if previous is None:
        return float(current)
    return float(alpha * float(current) + (1.0 - alpha) * float(previous))


def _ema_angle_deg(current: float, previous: Optional[float], alpha: float) -> float:
    if previous is None:
        return float(current)
    # Ellipse angle is periodic over 180 deg.
    delta = ((float(current) - float(previous) + 90.0) % 180.0) - 90.0
    return float((float(previous) + alpha * delta) % 180.0)


def _compute_pupil_center_analysis(
    network: Optional[List[float]],
    mask_centroid: Optional[List[float]],
    *,
    mask_ok: bool,
    pupil_mask_area: int,
    pupil_component_count: int,
    pupil_clean_area_ratio: Optional[float],
) -> Tuple[Optional[List[float]], str]:
    """Good clean mask: centroid only; else if network/centroid close: blend 0.5/0.5."""
    if network is None or mask_centroid is None:
        return None, "none"
    cr = float(pupil_clean_area_ratio) if pupil_clean_area_ratio is not None else 0.0
    mask_good = (
        bool(mask_ok)
        and int(pupil_mask_area) > 0
        and int(pupil_component_count) >= 1
        and cr >= ANALYSIS_CLEAN_AREA_RATIO_MIN
    )
    if mask_good:
        return [float(mask_centroid[0]), float(mask_centroid[1])], "centroid"
    d = float(np.hypot(float(network[0]) - float(mask_centroid[0]), float(network[1]) - float(mask_centroid[1])))
    if d <= ANALYSIS_DISAGREEMENT_MAX_PX:
        return [
            0.5 * (float(network[0]) + float(mask_centroid[0])),
            0.5 * (float(network[1]) + float(mask_centroid[1])),
        ], "blend"
    return None, "none"


def _fill_analysis_signal_fields(record: Dict[str, Any]) -> None:
    """Analysis layer: no EMA, no fallback. Half-axis normalized nx/ny."""
    record["timestamp"] = float(time.time())
    record["iris_center"] = record.get("iris_center_roi")

    network = record.get("pupil_center_raw_roi")
    if network is None:
        network = record.get("pupil_center_roi")
    if isinstance(network, list) and len(network) >= 2:
        record["pupil_center_network"] = [float(network[0]), float(network[1])]
    else:
        record["pupil_center_network"] = None

    mask_c = record.get("pupil_mask_centroid")
    if isinstance(mask_c, list) and len(mask_c) >= 2:
        record["pupil_center_mask_centroid"] = [float(mask_c[0]), float(mask_c[1])]
    else:
        record["pupil_center_mask_centroid"] = None

    disagree: Optional[float] = None
    if record["pupil_center_network"] is not None and record["pupil_center_mask_centroid"] is not None:
        a, b = record["pupil_center_network"], record["pupil_center_mask_centroid"]
        disagree = float(np.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))
    record["pupil_center_disagreement"] = disagree

    fused = record.get("pupil_center_fused_roi")
    if isinstance(fused, list) and len(fused) >= 2:
        pca = [float(fused[0]), float(fused[1])]
    else:
        pca, _mode = _compute_pupil_center_analysis(
            record["pupil_center_network"],
            record["pupil_center_mask_centroid"],
            mask_ok=bool(record.get("mask_ok", False)),
            pupil_mask_area=int(record.get("pupil_mask_area", 0) or 0),
            pupil_component_count=int(record.get("pupil_component_count", 0) or 0),
            pupil_clean_area_ratio=record.get("pupil_clean_area_ratio"),
        )
    record["pupil_center_analysis"] = pca

    reasons: List[str] = []
    if not bool(record.get("valid", False)):
        reasons.append("valid_false")
    if not bool(record.get("mask_ok", False)):
        reasons.append("mask_ok_false")
    if not bool(record.get("geometry_ok", False)):
        reasons.append("geometry_ok_false")
    if int(record.get("pupil_component_count", 0) or 0) < 1:
        reasons.append("pupil_component_count_lt_1")
    if int(record.get("pupil_mask_area", 0) or 0) <= 0:
        reasons.append("pupil_mask_area_zero")
    if disagree is not None and disagree > ANALYSIS_DISAGREEMENT_MAX_PX:
        reasons.append("pupil_center_disagreement_too_large")
    cr = record.get("pupil_clean_area_ratio")
    if cr is None or float(cr) < ANALYSIS_CLEAN_AREA_RATIO_MIN:
        reasons.append("pupil_clean_area_ratio_too_low")
    iris_m = record.get("iris_edge_margin_min")
    if iris_m is None or int(iris_m) < ANALYSIS_IRIS_EDGE_MARGIN_MIN:
        reasons.append("iris_edge_margin_too_small")

    ic = record.get("iris_center_roi")
    imaj = record.get("iris_major")
    imin = record.get("iris_minor")
    if pca is not None and isinstance(ic, list) and len(ic) >= 2 and imaj is not None:
        dist_analysis = float(np.hypot(float(pca[0]) - float(ic[0]), float(pca[1]) - float(ic[1])))
        if dist_analysis > float(imaj) * ANALYSIS_PUPIL_IRIS_DIST_RATIO_MAX:
            reasons.append("pupil_iris_dist_ratio_too_large")
    elif pca is None:
        reasons.append("pupil_center_analysis_unavailable")

    accept = len(reasons) == 0 and pca is not None
    record["analysis_frame_accept"] = bool(accept)
    record["analysis_reject_reasons"] = reasons

    record["analysis_nx"] = None
    record["analysis_ny"] = None
    if accept and pca is not None and isinstance(ic, list) and len(ic) >= 2 and imaj is not None and imin is not None:
        rx = max(float(imaj) * 0.5, ANALYSIS_EPS)
        ry = max(float(imin) * 0.5, ANALYSIS_EPS)
        record["analysis_nx"] = float((float(pca[0]) - float(ic[0])) / rx)
        record["analysis_ny"] = float((float(pca[1]) - float(ic[1])) / ry)


class EllSegStabilityRunner:
    def __init__(
        self,
        device_index: int,
        side: str,
        split_half: bool,
        autofocus: bool,
        backend_dshow: bool,
        num_frames: int,
        interval_ms: float,
        output_root: str,
        roi: Optional[Tuple[int, int, int, int]],
        roi_zoom: int,
    ) -> None:
        if side not in ("left", "right"):
            raise ValueError("side must be left/right")
        self.side = side
        self.split_half = split_half
        self.num_frames = int(num_frames)
        self.interval_ms = float(interval_ms)
        self.roi_zoom = int(max(1, roi_zoom))
        self.roi = roi
        self.output_root = output_root

        backend = cv2.CAP_DSHOW if backend_dshow else None
        if backend is not None:
            self.cap = cv2.VideoCapture(device_index, backend)
            if not self.cap.isOpened():
                print("[WARN] CAP_DSHOW failed to open camera. Falling back to default backend...")
                self.cap = cv2.VideoCapture(device_index)
        else:
            self.cap = cv2.VideoCapture(device_index)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera index={device_index}")

        try:
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0)
        except Exception:
            pass

        self.eye_model = EyeRoiModel()
        self._pupil_tracker = LocalPupilTracker()

    def _split_left_right(self, frame: np.ndarray) -> np.ndarray:
        if not self.split_half:
            if self.side != "left":
                raise ValueError("split_half must be enabled for side=right")
            return frame
        h, w = frame.shape[:2]
        half = w // 2
        if self.side == "left":
            return frame[:, :half]
        return frame[:, half:]

    def _select_roi_interactively(self, eye_img: np.ndarray) -> Tuple[int, int, int, int]:
        win = f"Select ROI ({self.side}) - drag, press ENTER/SPACE"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        roi = cv2.selectROI(win, eye_img, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(win)
        x, y, w, h = roi
        if w <= 0 or h <= 0:
            raise RuntimeError("ROI selection invalid (w/h <= 0)")
        img_h, img_w = eye_img.shape[:2]
        x, y, w, h = clamp_roi(x, y, w, h, img_w=img_w, img_h=img_h)
        return int(x), int(y), int(w), int(h)

    def _draw_roi_and_geometry(self, eye_img: np.ndarray, roi: Tuple[int, int, int, int], result: Optional[EllSegResult]) -> np.ndarray:
        x, y, w, h = roi
        vis = eye_img.copy()
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)

        if result is None or not result.valid:
            return vis

        # Draw ellipses on full eye half.
        if result.iris_ellipse is not None:
            el = result.iris_ellipse
            cx = int(round(el.cx + x))
            cy = int(round(el.cy + y))
            ax1 = int(round((el.major_axis * 0.5)))
            ax2 = int(round((el.minor_axis * 0.5)))
            cv2.ellipse(vis, (cx, cy), (max(1, ax1), max(1, ax2)), float(el.angle_deg), 0, 360, (0, 200, 255), 1)

        if result.pupil_ellipse is not None:
            el = result.pupil_ellipse
            cx = int(round(el.cx + x))
            cy = int(round(el.cy + y))
            ax1 = int(round((el.major_axis * 0.5)))
            ax2 = int(round((el.minor_axis * 0.5)))
            cv2.ellipse(vis, (cx, cy), (max(1, ax1), max(1, ax2)), float(el.angle_deg), 0, 360, (0, 255, 0), 1)

        if result.pupil_center is not None:
            pc = result.pupil_center
            cv2.circle(vis, (int(round(pc.x + x)), int(round(pc.y + y))), 3, (0, 255, 0), -1)

        return vis

    def _draw_display_geometry(
        self,
        eye_img: np.ndarray,
        roi: Tuple[int, int, int, int],
        output_state: Optional[Dict[str, Any]],
        fallback_used: bool,
    ) -> np.ndarray:
        x, y, w, h = roi
        vis = eye_img.copy()
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)
        if output_state is None:
            return vis

        iris_color = (0, 220, 255) if not fallback_used else (0, 140, 200)
        pupil_color = (0, 255, 0) if not fallback_used else (0, 180, 0)

        if output_state.get("smoothed_iris_center_roi") is not None:
            c = output_state["smoothed_iris_center_roi"]
            cx = int(round(float(c[0]) + x))
            cy = int(round(float(c[1]) + y))
            ax1 = int(round(float(output_state.get("smoothed_iris_major", 0.0)) * 0.5))
            ax2 = int(round(float(output_state.get("smoothed_iris_minor", 0.0)) * 0.5))
            if ax1 > 0 and ax2 > 0:
                cv2.ellipse(
                    vis,
                    (cx, cy),
                    (ax1, ax2),
                    float(output_state.get("smoothed_iris_angle_deg", 0.0)),
                    0,
                    360,
                    iris_color,
                    1,
                )

        if output_state.get("smoothed_pupil_center_roi") is not None:
            c = output_state["smoothed_pupil_center_roi"]
            cx = int(round(float(c[0]) + x))
            cy = int(round(float(c[1]) + y))
            ax1 = int(round(float(output_state.get("smoothed_pupil_major", 0.0)) * 0.5))
            ax2 = int(round(float(output_state.get("smoothed_pupil_minor", 0.0)) * 0.5))
            if ax1 > 0 and ax2 > 0:
                cv2.ellipse(
                    vis,
                    (cx, cy),
                    (ax1, ax2),
                    float(output_state.get("smoothed_pupil_angle_deg", 0.0)),
                    0,
                    360,
                    pupil_color,
                    1,
                )
            cv2.circle(vis, (cx, cy), 3, pupil_color, -1)

        return vis

    def run(self) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(self.output_root, f"ellseg_stability_{self.side}_{ts}")
        mask_dir = os.path.join(out_dir, "mask_color")
        mask_component_dir = os.path.join(out_dir, "mask_component_debug")
        vis_dir = os.path.join(out_dir, "roi_vis")
        ensure_dir(out_dir)
        ensure_dir(mask_dir)
        ensure_dir(mask_component_dir)
        ensure_dir(vis_dir)

        records_path = os.path.join(out_dir, "records.jsonl")
        summary_path = os.path.join(out_dir, "summary.json")
        config_path = os.path.join(out_dir, "config.json")

        win_eye = f"EllSeg Stability Viewer ({self.side})"
        cv2.namedWindow(win_eye, cv2.WINDOW_NORMAL)

        # Grab first frame for ROI selection.
        while True:
            ok, frame = self.cap.read()
            if ok and frame is not None:
                eye_img = self._split_left_right(frame)
                break
            time.sleep(0.01)

        eye_h, eye_w = eye_img.shape[:2]
        if self.roi is None:
            self.roi = self._select_roi_interactively(eye_img)
        else:
            x, y, w, h = self.roi
            self.roi = clamp_roi(x, y, w, h, img_w=eye_w, img_h=eye_h)

        roi = self.roi

        # Save config.
        config = {
            "side": self.side,
            "split_half": self.split_half,
            "device_index": None,
            "num_frames": self.num_frames,
            "interval_ms": self.interval_ms,
            "roi": {"x": roi[0], "y": roi[1], "w": roi[2], "h": roi[3]},
            "roi_zoom": self.roi_zoom,
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        print(f"[INFO] Start capture: side={self.side}, roi={roi}, frames={self.num_frames}")

        all_records = []
        n_total = 0
        n_valid = 0
        interrupted = False
        accepted_count = 0
        fallback_count = 0
        reject_count = 0
        jump_reject_count = 0
        consecutive_fallback = 0
        prev_output: Optional[Dict[str, Any]] = None
        prev_accepted_raw_state: Optional[Dict[str, Any]] = None
        self._pupil_tracker.reset()

        with open(records_path, "w", encoding="utf-8") as rec_f:
            try:
                while n_total < self.num_frames:
                    ok, frame = self.cap.read()
                    if not ok or frame is None:
                        time.sleep(0.01)
                        continue
                    n_total += 1

                    eye_img = self._split_left_right(frame)
                    x, y, w, h = roi
                    crop_bgr = eye_img[y : y + h, x : x + w]
                    if crop_bgr is None or crop_bgr.size == 0:
                        continue

                    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                    result = self.eye_model.infer(crop_rgb)
                    effective_result = result

                    sharp = float(cv2.Laplacian(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())

                    record: Dict[str, Any] = {
                        "frame_idx": n_total,
                        "valid": bool(
                            result is not None
                            and result.valid
                            and result.iris_ellipse is not None
                            and result.pupil_ellipse is not None
                            and result.pupil_center is not None
                        ),
                        "sharp_laplacian_var": sharp,
                        "pupil_center_roi": None,
                        "pupil_major": None,
                        "pupil_minor": None,
                        "pupil_angle_deg": None,
                        "iris_center_roi": None,
                        "iris_center": None,
                        "iris_major": None,
                        "iris_minor": None,
                        "iris_angle_deg": None,
                        "pupil_minus_iris_dx": None,
                        "pupil_minus_iris_dy": None,
                        "pupil_minus_iris_dist": None,
                        "mask_exists": bool(result is not None and result.segmentation_mask is not None),
                        "roi_xywh": [int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])],
                        "crop_size_hw": [int(h), int(w)],
                        "pupil_mask_area": 0,
                        "iris_mask_area": 0,
                        "pupil_component_count": 0,
                        "iris_component_count": 0,
                        "pupil_main_component_area": 0,
                        "iris_main_component_area": 0,
                        "pupil_main_component_area_ratio": 0.0,
                        "iris_main_component_area_ratio": 0.0,
                        "pupil_mask_centroid": None,
                        "iris_mask_centroid": None,
                        "pupil_mask_to_iris_center_dist": None,
                        "iris_mask_to_pupil_center_dist": None,
                        "iris_edge_margin_left": None,
                        "iris_edge_margin_right": None,
                        "iris_edge_margin_top": None,
                        "iris_edge_margin_bottom": None,
                        "iris_edge_margin_min": None,
                        "pupil_edge_margin_min": None,
                        "pupil_bbox_xywh": None,
                        "iris_bbox_xywh": None,
                        "geometry_ok": False,
                        "geometry_reject_reason": "invalid_result",
                        "mask_ok": False,
                        "mask_reject_reason": "mask_not_available",
                        "mask_warning_reasons": [],
                        "mask_warning_count": 0,
                        "pupil_clean_area_ratio": None,
                        "raw_pupil_component_count": 0,
                        "clean_pupil_component_count": 0,
                        "raw_iris_component_count": 0,
                        "clean_iris_component_count": 0,
                        "raw_pupil_mask_area": 0,
                        "clean_pupil_mask_area": 0,
                        "raw_iris_mask_area": 0,
                        "clean_iris_mask_area": 0,
                        "frame_accept": False,
                        "baseline_accept": False,
                        "frame_reject_reasons": [],
                        "jump_reject_reason": "",
                        "pupil_center_jump_from_prev": None,
                        "iris_center_jump_from_prev": None,
                        "dist_jump_from_prev": None,
                        "used_previous_frame_fallback": False,
                        "fallback_reason": "",
                        "output_pupil_center_roi": None,
                        "output_iris_center_roi": None,
                        "output_pupil_major": None,
                        "output_pupil_minor": None,
                        "output_pupil_angle_deg": None,
                        "output_iris_major": None,
                        "output_iris_minor": None,
                        "output_iris_angle_deg": None,
                        "output_pupil_minus_iris_dx": None,
                        "output_pupil_minus_iris_dy": None,
                        "output_pupil_minus_iris_dist": None,
                        "smoothed_pupil_center_roi": None,
                        "smoothed_iris_center_roi": None,
                        "smoothed_pupil_major": None,
                        "smoothed_pupil_minor": None,
                        "smoothed_pupil_angle_deg": None,
                        "smoothed_iris_major": None,
                        "smoothed_iris_minor": None,
                        "smoothed_iris_angle_deg": None,
                        "timestamp": None,
                        "pupil_center_network": None,
                        "pupil_center_raw_roi": None,
                        "pupil_center_tracked_roi": None,
                        "pupil_center_fused_roi": None,
                        "pupil_center_source": "",
                        "pupil_tracking_score": None,
                        "pupil_tracking_shift_px": None,
                        "pupil_center_mask_centroid": None,
                        "pupil_center_analysis": None,
                        "pupil_center_disagreement": None,
                        "analysis_frame_accept": False,
                        "analysis_reject_reasons": [],
                        "analysis_nx": None,
                        "analysis_ny": None,
                        "display_pupil_center_roi": None,
                        "display_iris_center_roi": None,
                    }

                    obs = None
                    raw_obs = None
                    if result is not None and result.valid and result.pupil_center is not None and result.segmentation_mask is not None and result.segmentation_mask.size > 0:
                        try:
                            raw_obs = build_roi_observation(result=result, roi_xywh=roi, side=self.side)
                            fused_center, tracking_debug = self._pupil_tracker.update(
                                crop_bgr,
                                network_center=result.pupil_center,
                                mask_centroid=raw_obs.pupil_mask_centroid,
                                iris_ellipse=result.iris_ellipse,
                                mask_ok=raw_obs.mask_ok,
                                pupil_mask_area=raw_obs.clean_pupil_mask_area,
                                pupil_clean_area_ratio=raw_obs.pupil_clean_area_ratio,
                            )
                            record["pupil_center_tracked_roi"] = tracking_debug.get("tracking_center_roi")
                            record["pupil_tracking_score"] = tracking_debug.get("tracking_score")
                            record["pupil_center_source"] = str(tracking_debug.get("fused_source") or "")
                            record["pupil_center_fused_roi"] = tracking_debug.get("fused_center_roi")
                            if fused_center is not None:
                                record["pupil_tracking_shift_px"] = float(
                                    np.hypot(float(fused_center.x) - float(result.pupil_center.x), float(fused_center.y) - float(result.pupil_center.y))
                                )
                            if (
                                fused_center is not None
                                and record["pupil_tracking_shift_px"] is not None
                                and float(record["pupil_tracking_shift_px"]) <= TRACK_MAX_APPLY_SHIFT
                            ):
                                effective_result = replace(
                                    result,
                                    pupil_center=fused_center,
                                    pupil_ellipse=shift_ellipse_center(result.pupil_ellipse, fused_center),
                                )
                        except Exception as exc:
                            record["pupil_center_source"] = f"tracking_error:{type(exc).__name__}"

                    if effective_result is not None and effective_result.segmentation_mask is not None and effective_result.segmentation_mask.size > 0:
                        try:
                            obs = build_roi_observation(result=effective_result, roi_xywh=roi, side=self.side)
                            geometry_ok, geometry_reason = check_observation_plausibility(obs)
                            record["geometry_ok"] = bool(geometry_ok)
                            record["geometry_reject_reason"] = str(geometry_reason)
                            record["mask_ok"] = bool(obs.mask_ok)
                            record["mask_reject_reason"] = str(obs.mask_reject_reason)
                            record["mask_warning_reasons"] = list(obs.mask_warning_reasons or [])
                            record["mask_warning_count"] = int(obs.mask_warning_count)
                            record["pupil_clean_area_ratio"] = (
                                float(obs.pupil_clean_area_ratio) if obs.pupil_clean_area_ratio is not None else None
                            )
                            record["pupil_mask_area"] = int(obs.clean_pupil_mask_area)
                            record["iris_mask_area"] = int(obs.clean_iris_mask_area)
                            record["pupil_component_count"] = int(obs.clean_pupil_component_count)
                            record["iris_component_count"] = int(obs.clean_iris_component_count)
                            record["pupil_main_component_area"] = int(obs.pupil_main_component_area)
                            record["iris_main_component_area"] = int(obs.iris_main_component_area)
                            record["pupil_main_component_area_ratio"] = float(obs.pupil_main_component_area_ratio)
                            record["iris_main_component_area_ratio"] = float(obs.iris_main_component_area_ratio)
                            record["pupil_mask_centroid"] = (
                                [float(obs.pupil_mask_centroid[0]), float(obs.pupil_mask_centroid[1])]
                                if obs.pupil_mask_centroid is not None
                                else None
                            )
                            record["iris_mask_centroid"] = (
                                [float(obs.iris_mask_centroid[0]), float(obs.iris_mask_centroid[1])]
                                if obs.iris_mask_centroid is not None
                                else None
                            )
                            record["pupil_mask_to_iris_center_dist"] = (
                                float(obs.pupil_mask_to_iris_center_dist)
                                if obs.pupil_mask_to_iris_center_dist is not None
                                else None
                            )
                            record["iris_mask_to_pupil_center_dist"] = (
                                float(obs.iris_mask_to_pupil_center_dist)
                                if obs.iris_mask_to_pupil_center_dist is not None
                                else None
                            )
                            record["iris_edge_margin_left"] = obs.iris_edge_margin_left
                            record["iris_edge_margin_right"] = obs.iris_edge_margin_right
                            record["iris_edge_margin_top"] = obs.iris_edge_margin_top
                            record["iris_edge_margin_bottom"] = obs.iris_edge_margin_bottom
                            record["iris_edge_margin_min"] = obs.iris_edge_margin_min
                            record["pupil_edge_margin_min"] = obs.pupil_edge_margin_min
                            record["pupil_bbox_xywh"] = list(obs.pupil_mask_bbox) if obs.pupil_mask_bbox is not None else None
                            record["iris_bbox_xywh"] = list(obs.iris_mask_bbox) if obs.iris_mask_bbox is not None else None
                            record["raw_pupil_component_count"] = int(obs.raw_pupil_component_count)
                            record["clean_pupil_component_count"] = int(obs.clean_pupil_component_count)
                            record["raw_iris_component_count"] = int(obs.raw_iris_component_count)
                            record["clean_iris_component_count"] = int(obs.clean_iris_component_count)
                            record["raw_pupil_mask_area"] = int(obs.raw_pupil_mask_area)
                            record["clean_pupil_mask_area"] = int(obs.clean_pupil_mask_area)
                            record["raw_iris_mask_area"] = int(obs.raw_iris_mask_area)
                            record["clean_iris_mask_area"] = int(obs.clean_iris_mask_area)
                        except Exception as exc:
                            record["mask_reject_reason"] = f"mask_metrics_error:{type(exc).__name__}"

                    if record["valid"]:
                        assert effective_result is not None
                        assert result is not None
                        assert effective_result.pupil_center is not None
                        assert effective_result.iris_ellipse is not None
                        assert effective_result.pupil_ellipse is not None

                        pc = effective_result.pupil_center
                        ic = effective_result.iris_ellipse
                        record["pupil_center_raw_roi"] = [float(result.pupil_center.x), float(result.pupil_center.y)]

                        record["pupil_center_roi"] = [float(pc.x), float(pc.y)]
                        record["pupil_major"] = float(effective_result.pupil_ellipse.major_axis)
                        record["pupil_minor"] = float(effective_result.pupil_ellipse.minor_axis)
                        record["pupil_angle_deg"] = float(effective_result.pupil_ellipse.angle_deg)

                        record["iris_center_roi"] = [float(ic.cx), float(ic.cy)]
                        record["iris_major"] = float(ic.major_axis)
                        record["iris_minor"] = float(ic.minor_axis)
                        record["iris_angle_deg"] = float(ic.angle_deg)

                        dx = float(pc.x - ic.cx)
                        dy = float(pc.y - ic.cy)
                        dist = float(np.hypot(dx, dy))
                        record["pupil_minus_iris_dx"] = dx
                        record["pupil_minus_iris_dy"] = dy
                        record["pupil_minus_iris_dist"] = dist

                        # Save mask_color if available.
                        mask_img = ellseg_segmentation_mask_to_bgr(effective_result.segmentation_mask, dst_h=h, dst_w=w)
                        if mask_img is not None:
                            mask_path = os.path.join(mask_dir, f"frame_{n_total:06d}.jpg")
                            cv2.imwrite(mask_path, mask_img)
                        if obs is not None:
                            mask_component_img = ellseg_component_debug_mask_to_bgr(
                                effective_result.segmentation_mask,
                                obs.cleaned_pupil_mask,
                                obs.discarded_pupil_mask,
                                obs.cleaned_iris_mask,
                                obs.discarded_iris_mask,
                                dst_h=h,
                                dst_w=w,
                            )
                            if mask_component_img is not None:
                                comp_path = os.path.join(mask_component_dir, f"frame_{n_total:06d}.jpg")
                                cv2.imwrite(comp_path, mask_component_img)

                        n_valid += 1

                    _fill_analysis_signal_fields(record)

                    frame_reject_reasons: List[str] = []
                    if not bool(record["valid"]):
                        frame_reject_reasons.append("invalid_result")
                    if bool(record["valid"]) and not bool(record.get("mask_ok", False)):
                        frame_reject_reasons.append(f"mask_bad:{record.get('mask_reject_reason', '')}")
                    if bool(record["valid"]) and not bool(record.get("geometry_ok", False)):
                        frame_reject_reasons.append(f"geometry_bad:{record.get('geometry_reject_reason', '')}")
                    if record.get("pupil_minus_iris_dist") is not None and float(record["pupil_minus_iris_dist"]) > FRAME_ACCEPT_DIST_MAX:
                        frame_reject_reasons.append("dist_too_large")

                    baseline_accept = bool(
                        record["valid"]
                        and record.get("mask_ok", False)
                        and record.get("geometry_ok", False)
                        and record.get("iris_edge_margin_min") is not None
                        and float(record["iris_edge_margin_min"]) >= BASELINE_IRIS_EDGE_MARGIN_MIN
                        and record.get("pupil_center_disagreement") is not None
                        and float(record["pupil_center_disagreement"]) <= BASELINE_PUPIL_CENTER_DISAGREEMENT_MAX
                        and record.get("pupil_minus_iris_dist") is not None
                        and float(record["pupil_minus_iris_dist"]) <= FRAME_ACCEPT_DIST_MAX
                    )
                    record["baseline_accept"] = baseline_accept

                    jump_enabled = bool(
                        record["valid"]
                        and prev_accepted_raw_state is not None
                        and accepted_count >= BASELINE_WARMUP_ACCEPTS
                    )
                    if jump_enabled:
                        pupil_jump = _pair_dist(record.get("pupil_center_roi"), prev_accepted_raw_state.get("pupil_center_roi"))
                        iris_jump = _pair_dist(record.get("iris_center_roi"), prev_accepted_raw_state.get("iris_center_roi"))
                        dist_jump = (
                            abs(float(record["pupil_minus_iris_dist"]) - float(prev_accepted_raw_state["pupil_minus_iris_dist"]))
                            if record.get("pupil_minus_iris_dist") is not None
                            and prev_accepted_raw_state.get("pupil_minus_iris_dist") is not None
                            else None
                        )
                        record["pupil_center_jump_from_prev"] = pupil_jump
                        record["iris_center_jump_from_prev"] = iris_jump
                        record["dist_jump_from_prev"] = dist_jump
                        if pupil_jump is not None and pupil_jump > FRAME_ACCEPT_PUPIL_JUMP_MAX:
                            frame_reject_reasons.append("pupil_center_jump_too_large")
                            record["jump_reject_reason"] = "pupil_center_jump_too_large"
                        elif iris_jump is not None and iris_jump > FRAME_ACCEPT_IRIS_JUMP_MAX:
                            frame_reject_reasons.append("iris_center_jump_too_large")
                            record["jump_reject_reason"] = "iris_center_jump_too_large"
                        elif dist_jump is not None and dist_jump > FRAME_ACCEPT_DIST_JUMP_MAX:
                            frame_reject_reasons.append("dist_jump_too_large")
                            record["jump_reject_reason"] = "dist_jump_too_large"
                    else:
                        record["pupil_center_jump_from_prev"] = None
                        record["iris_center_jump_from_prev"] = None
                        record["dist_jump_from_prev"] = None

                    record["frame_reject_reasons"] = frame_reject_reasons
                    frame_accept = len(frame_reject_reasons) == 0 and bool(record["valid"]) and baseline_accept
                    record["frame_accept"] = bool(frame_accept)
                    output_state: Optional[Dict[str, Any]] = None

                    if frame_accept:
                        smoothed_pupil = _ema_pair(record["pupil_center_roi"], prev_output.get("smoothed_pupil_center_roi") if prev_output else None, EMA_ALPHA)
                        smoothed_iris = _ema_pair(record["iris_center_roi"], prev_output.get("smoothed_iris_center_roi") if prev_output else None, EMA_ALPHA)
                        smoothed_pupil_major = _ema_scalar(float(record["pupil_major"]), prev_output.get("smoothed_pupil_major") if prev_output else None, EMA_ALPHA)
                        smoothed_pupil_minor = _ema_scalar(float(record["pupil_minor"]), prev_output.get("smoothed_pupil_minor") if prev_output else None, EMA_ALPHA)
                        smoothed_pupil_angle = _ema_angle_deg(float(record["pupil_angle_deg"]), prev_output.get("smoothed_pupil_angle_deg") if prev_output else None, EMA_ALPHA)
                        smoothed_iris_major = _ema_scalar(float(record["iris_major"]), prev_output.get("smoothed_iris_major") if prev_output else None, EMA_ALPHA)
                        smoothed_iris_minor = _ema_scalar(float(record["iris_minor"]), prev_output.get("smoothed_iris_minor") if prev_output else None, EMA_ALPHA)
                        smoothed_iris_angle = _ema_angle_deg(float(record["iris_angle_deg"]), prev_output.get("smoothed_iris_angle_deg") if prev_output else None, EMA_ALPHA)
                        out_dx = float(smoothed_pupil[0] - smoothed_iris[0])
                        out_dy = float(smoothed_pupil[1] - smoothed_iris[1])
                        output_state = {
                            "output_pupil_center_roi": list(record["pupil_center_roi"]),
                            "output_iris_center_roi": list(record["iris_center_roi"]),
                            "output_pupil_major": float(record["pupil_major"]),
                            "output_pupil_minor": float(record["pupil_minor"]),
                            "output_pupil_angle_deg": float(record["pupil_angle_deg"]),
                            "output_iris_major": float(record["iris_major"]),
                            "output_iris_minor": float(record["iris_minor"]),
                            "output_iris_angle_deg": float(record["iris_angle_deg"]),
                            "smoothed_pupil_center_roi": smoothed_pupil,
                            "smoothed_iris_center_roi": smoothed_iris,
                            "output_pupil_minus_iris_dx": out_dx,
                            "output_pupil_minus_iris_dy": out_dy,
                            "output_pupil_minus_iris_dist": float(np.hypot(out_dx, out_dy)),
                            "smoothed_pupil_major": smoothed_pupil_major,
                            "smoothed_pupil_minor": smoothed_pupil_minor,
                            "smoothed_pupil_angle_deg": smoothed_pupil_angle,
                            "smoothed_iris_major": smoothed_iris_major,
                            "smoothed_iris_minor": smoothed_iris_minor,
                            "smoothed_iris_angle_deg": smoothed_iris_angle,
                        }
                        accepted_count += 1
                        consecutive_fallback = 0
                        prev_accepted_raw_state = {
                            "pupil_center_roi": list(record["pupil_center_roi"]),
                            "iris_center_roi": list(record["iris_center_roi"]),
                            "pupil_minus_iris_dist": float(record["pupil_minus_iris_dist"]),
                        }
                    elif prev_output is not None and consecutive_fallback < FALLBACK_MAX_CONSECUTIVE:
                        output_state = dict(prev_output)
                        record["used_previous_frame_fallback"] = True
                        record["fallback_reason"] = "|".join(frame_reject_reasons) if frame_reject_reasons else "reuse_previous_output"
                        fallback_count += 1
                        consecutive_fallback += 1
                    else:
                        reject_count += 1
                        if record.get("jump_reject_reason"):
                            jump_reject_count += 1
                        consecutive_fallback = 0

                    if output_state is not None:
                        record.update(output_state)
                        prev_output = output_state
                    record["display_pupil_center_roi"] = record.get("smoothed_pupil_center_roi")
                    record["display_iris_center_roi"] = record.get("smoothed_iris_center_roi")

                    display_vis = self._draw_display_geometry(
                        eye_img,
                        roi,
                        output_state if output_state is not None else prev_output,
                        bool(record.get("used_previous_frame_fallback", False)),
                    )
                    vis_zoom = cv2.resize(
                        display_vis[y : y + h, x : x + w],
                        (max(1, w * self.roi_zoom), max(1, h * self.roi_zoom)),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    cv2.putText(
                        vis_zoom,
                        f"frame={n_total} sharp={sharp:.1f}",
                        (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    out_vis_path = os.path.join(vis_dir, f"frame_{n_total:06d}.jpg")
                    cv2.imwrite(out_vis_path, vis_zoom)

                    all_records.append(record)
                    rec_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    # Live viewer + early quit.
                    cv2.imshow(win_eye, vis_zoom)
                    k = cv2.waitKey(1) & 0xFF
                    if k in (ord("q"), 27):
                        break

                    if self.interval_ms > 0:
                        time.sleep(self.interval_ms / 1000.0)
            except KeyboardInterrupt:
                interrupted = True
                print("\n[INFO] Ctrl+C received, finalizing current capture...")

        # Final statistics on valid frames only.
        def arr(key: str) -> np.ndarray:
            vals = [r.get(key) for r in all_records]
            return np.array([v for v in vals if v is not None], dtype=np.float64)

        output_records = [r for r in all_records if r.get("smoothed_pupil_center_roi") is not None and r.get("smoothed_iris_center_roi") is not None]
        pupil_cx = np.array([r["smoothed_pupil_center_roi"][0] for r in output_records], dtype=np.float64) if output_records else np.array([], dtype=np.float64)
        pupil_cy = np.array([r["smoothed_pupil_center_roi"][1] for r in output_records], dtype=np.float64) if output_records else np.array([], dtype=np.float64)
        iris_cx = np.array([r["smoothed_iris_center_roi"][0] for r in output_records], dtype=np.float64) if output_records else np.array([], dtype=np.float64)
        iris_cy = np.array([r["smoothed_iris_center_roi"][1] for r in output_records], dtype=np.float64) if output_records else np.array([], dtype=np.float64)

        dx = arr("output_pupil_minus_iris_dx")
        dy = arr("output_pupil_minus_iris_dy")
        dist = arr("output_pupil_minus_iris_dist")

        pupil_major = arr("output_pupil_major")
        pupil_minor = arr("output_pupil_minor")
        pupil_angle = arr("output_pupil_angle_deg")
        iris_major = arr("output_iris_major")
        iris_minor = arr("output_iris_minor")
        iris_angle = arr("output_iris_angle_deg")
        pupil_mask_area = arr("pupil_mask_area")
        iris_mask_area = arr("iris_mask_area")
        iris_edge_margin_min = arr("iris_edge_margin_min")
        pupil_clean_area_ratio = arr("pupil_clean_area_ratio")
        analysis_nx = arr("analysis_nx")
        analysis_ny = arr("analysis_ny")
        pupil_center_disagreement = arr("pupil_center_disagreement")
        pupil_tracking_score = arr("pupil_tracking_score")
        pupil_tracking_shift_px = arr("pupil_tracking_shift_px")
        n_analysis_accept = int(sum(1 for r in all_records if bool(r.get("analysis_frame_accept", False))))
        source_counts: Dict[str, int] = {}
        for r in all_records:
            src = str(r.get("pupil_center_source") or "none")
            source_counts[src] = int(source_counts.get(src, 0)) + 1

        summary = {
            "n_total_frames": int(n_total),
            "n_valid_frames": int(n_valid),
            "n_output_frames": int(len(output_records)),
            "accepted_frames": int(accepted_count),
            "fallback_frames": int(fallback_count),
            "rejected_without_output_frames": int(reject_count),
            "jump_reject_frames": int(jump_reject_count),
            "interrupted_by_ctrl_c": bool(interrupted),
            "pupil_center_cx": mean_std(pupil_cx),
            "pupil_center_cy": mean_std(pupil_cy),
            "iris_center_cx": mean_std(iris_cx),
            "iris_center_cy": mean_std(iris_cy),
            "pupil_minus_iris_dx": mean_std(dx),
            "pupil_minus_iris_dy": mean_std(dy),
            "pupil_minus_iris_dist": mean_std(dist),
            "pupil_major": mean_std(pupil_major),
            "pupil_minor": mean_std(pupil_minor),
            "pupil_angle_circular_std_deg": float(fit_circular_std_deg(pupil_angle)),
            "iris_major": mean_std(iris_major),
            "iris_minor": mean_std(iris_minor),
            "iris_angle_circular_std_deg": float(fit_circular_std_deg(iris_angle)),
            "frames_pupil_component_gt1": int(sum(1 for r in all_records if (r.get("pupil_component_count") or 0) > 1)),
            "frames_iris_component_gt1": int(sum(1 for r in all_records if (r.get("iris_component_count") or 0) > 1)),
            "frames_iris_edge_margin_lt_2": int(
                sum(1 for r in all_records if r.get("iris_edge_margin_min") is not None and float(r["iris_edge_margin_min"]) < 2.0)
            ),
            "frames_pupil_edge_margin_lt_2": int(
                sum(1 for r in all_records if r.get("pupil_edge_margin_min") is not None and float(r["pupil_edge_margin_min"]) < 2.0)
            ),
            "frames_dist_gt_4": int(sum(1 for r in output_records if r.get("output_pupil_minus_iris_dist") is not None and float(r["output_pupil_minus_iris_dist"]) > 4.0)),
            "frames_dist_gt_6": int(sum(1 for r in output_records if r.get("output_pupil_minus_iris_dist") is not None and float(r["output_pupil_minus_iris_dist"]) > 6.0)),
            "raw_frames_dist_gt_4": int(sum(1 for r in all_records if r.get("pupil_minus_iris_dist") is not None and float(r["pupil_minus_iris_dist"]) > 4.0)),
            "frames_mask_bad": int(sum(1 for r in all_records if not bool(r.get("mask_ok", False)))),
            "frames_geometry_bad": int(sum(1 for r in all_records if not bool(r.get("geometry_ok", False)))),
            "frames_baseline_accept": int(sum(1 for r in all_records if bool(r.get("baseline_accept", False)))),
            "frames_frame_accept": int(sum(1 for r in all_records if bool(r.get("frame_accept", False)))),
            "frames_used_previous_frame_fallback": int(sum(1 for r in all_records if bool(r.get("used_previous_frame_fallback", False)))),
            "frames_pupil_component_eq_0": int(sum(1 for r in all_records if (r.get("pupil_component_count") or 0) == 0)),
            "frames_pupil_clean_area_ratio_lt_0_6": int(
                sum(1 for r in all_records if r.get("pupil_clean_area_ratio") is not None and float(r["pupil_clean_area_ratio"]) < 0.6)
            ),
            "pupil_mask_area": dist_stats(pupil_mask_area),
            "iris_mask_area": dist_stats(iris_mask_area),
            "iris_edge_margin_min": dist_stats(iris_edge_margin_min),
            "pupil_clean_area_ratio": dist_stats(pupil_clean_area_ratio),
            "frames_analysis_accept": int(n_analysis_accept),
            "analysis_accept_rate": float(n_analysis_accept) / float(max(int(n_total), 1)),
            "analysis_invalid_frames": int(int(n_total) - n_analysis_accept),
            "analysis_nx": dist_stats(analysis_nx),
            "analysis_ny": dist_stats(analysis_ny),
            "pupil_center_disagreement": dist_stats(pupil_center_disagreement),
            "pupil_tracking_score": dist_stats(pupil_tracking_score),
            "pupil_tracking_shift_px": dist_stats(pupil_tracking_shift_px),
            "frames_source_track_mask": int(source_counts.get("track_mask", 0)),
            "frames_source_track_network": int(source_counts.get("track_network", 0)),
            "frames_source_track_only": int(source_counts.get("track_only", 0)),
            "frames_source_mask_network": int(source_counts.get("mask_network", 0)),
            "frames_source_mask_network_prev_select": int(source_counts.get("mask_network_prev_select", 0)),
            "frames_source_mask_only": int(source_counts.get("mask_only", 0)),
            "frames_source_network_only": int(source_counts.get("network_only", 0)),
            "frames_source_tracking_error": int(sum(v for k, v in source_counts.items() if k.startswith("tracking_error:"))),
            "frames_source_none": int(source_counts.get("none", 0)),
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("[RESULT] EllSeg observation stability summary")
        print(json.dumps(summary, ensure_ascii=False, indent=2))

        self.cap.release()
        cv2.destroyAllWindows()


def parse_roi(s: str) -> Tuple[int, int, int, int]:
    # "x,y,w,h"
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("roi must be 'x,y,w,h'")
    return int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantify EllSeg pupil/iris observation jitter across frames.")
    parser.add_argument("--device-index", type=int, default=1)
    parser.add_argument("--side", type=str, default="left", choices=["left", "right"])
    parser.add_argument("--split-half", action="store_true", help="Split 3840x1080 side-by-side into left/right halves.")
    parser.add_argument("--no-split-half", dest="split_half", action="store_false")
    parser.set_defaults(split_half=True)

    parser.add_argument("--roi-xywh", type=str, default=None, help="Fixed ROI 'x,y,w,h' (skip interactive selection).")
    parser.add_argument("--roi-zoom", type=int, default=4, help="Zoom factor for saved ROI vis images.")
    parser.add_argument("--num-frames", type=int, default=200)
    parser.add_argument("--interval-ms", type=float, default=0.0)
    parser.add_argument("--autofocus", action="store_true", help="Enable autofocus at start.")
    parser.add_argument("--no-autofocus", dest="autofocus", action="store_false")
    parser.set_defaults(autofocus=False)
    parser.add_argument("--backend-dshow", action="store_true", help="Use cv2.CAP_DSHOW first (Windows).")
    parser.add_argument("--output-root", type=str, default=None, help="Output root dir.")

    args = parser.parse_args()

    if args.output_root is None:
        # project/debug/... (works from repo root)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project/
        args.output_root = os.path.join(base_dir, "debug", "ellseg_stability")

    roi = parse_roi(args.roi_xywh) if args.roi_xywh else None

    runner = EllSegStabilityRunner(
        device_index=args.device_index,
        side=args.side,
        split_half=args.split_half,
        autofocus=args.autofocus,
        backend_dshow=args.backend_dshow,
        num_frames=args.num_frames,
        interval_ms=args.interval_ms,
        output_root=args.output_root,
        roi=roi,
        roi_zoom=args.roi_zoom,
    )
    runner.run()


if __name__ == "__main__":
    main()
