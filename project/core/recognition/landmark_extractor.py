from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from project.config.logging_config import setup_logging
from project.core.recognition.compat_points import CANONICAL_ORDER
from project.core.recognition.eye_model_ellseg import EyeRoiModel, EllSegResult, ellipse_to_cardinal_points
from project.core.recognition.pupil_tracking import (
    LocalPupilTracker,
    TRACK_MAX_APPLY_SHIFT,
    shift_ellipse_center,
)
from project.core.recognition.roi.constants import MODE_DETECT, MODE_TRACK, FACE_BBOX_STABLE_MAX_SHIFT_RATIO
from project.core.recognition.roi.debug import build_roi_debug_record
from project.core.recognition.roi.geometry import (
    RoiGeometry,
    check_observation_plausibility,
    compute_roi_geometry,
    should_switch_to_detect,
)
from project.core.recognition.roi.observation import build_roi_observation
from project.core.recognition.roi.state_machine import RoiStateMachine
from project.core.recognition.roi.types import RoiBox
from project.data.data_models import BGRImage, Ellipse2D, Point2D

logger = setup_logging(__name__)


def _points_roi_to_image(eye_dict: Dict[str, List[Point2D]], roi_x: int, roi_y: int) -> Dict[str, List[Point2D]]:
    """将单眼 ROI 内点转为图像坐标。"""
    out: Dict[str, List[Point2D]] = {}
    for k, pts in eye_dict.items():
        out[k] = [Point2D(x=p.x + roi_x, y=p.y + roi_y) for p in pts]
    return out


def validate_single_eye_core_dict(eye_dict: Dict[str, List[Any]]) -> bool:
    """单眼核心点校验：仅检查 pupil=1、iris=4。"""
    if not eye_dict:
        return False
    core_required = {"pupil": 1, "iris": 4}
    for ft, count in core_required.items():
        pts = eye_dict.get(ft, [])
        if len(pts) != count:
            return False
    return True


def _compute_face_bbox_stability(
    current_face_bbox: Optional[Tuple[int, int, int, int]],
    last_face_bbox: Optional[Tuple[int, int, int, int]],
    img_w: int,
    img_h: int,
) -> Tuple[bool, float]:
    """
    Face bbox stability:
    - compute corner max shift ratio between last and current bboxes
    - stable if max_shift <= FACE_BBOX_STABLE_MAX_SHIFT_RATIO
    """

    if current_face_bbox is None or last_face_bbox is None:
        return False, 1.0

    cx, cy, cw, ch = [float(v) for v in current_face_bbox]
    lx, ly, lw, lh = [float(v) for v in last_face_bbox]

    cur = [(cx, cy), (cx + cw, cy), (cx, cy + ch), (cx + cw, cy + ch)]
    last = [(lx, ly), (lx + lw, ly), (lx, ly + lh), (lx + lw, ly + lh)]
    norm = float(max(1.0, min(img_w, img_h)))

    shifts = [float(np.hypot(ax - bx, ay - by)) / norm for (ax, ay), (bx, by) in zip(cur, last)]
    max_shift = float(max(shifts))
    return max_shift <= FACE_BBOX_STABLE_MAX_SHIFT_RATIO, max_shift


def _shift_ellipse(el: Optional[Ellipse2D], dx: float, dy: float) -> Optional[Ellipse2D]:
    if el is None:
        return None
    return Ellipse2D(
        cx=el.cx + dx,
        cy=el.cy + dy,
        major_axis=el.major_axis,
        minor_axis=el.minor_axis,
        angle_deg=el.angle_deg,
        confidence=el.confidence,
    )


class EyeLandmarkExtractor:
    """
    单眼关键点提取器：
    - 调用 `RoiStateMachine.predict()` 获取当前帧 ROI（DETECT/TRACK/RECOVER）。
    - ROI 内运行 EllSeg(EyeRoiModel)。
    - 基于 EllSegResult + 当前 ROI 构造 RoiObservation，计算 RoiGeometry，并调用 `commit_success/commit_failure` 更新状态机。
    - 返回 minimal debug payload（native_geometry + roi_debug）。
    """

    def __init__(
        self,
        side: str,
        roi_state_machine: RoiStateMachine,
        eye_model: EyeRoiModel,
    ):
        if side not in ("left", "right"):
            raise ValueError(f"invalid side for EyeLandmarkExtractor: {side}")
        self._side = side
        self._roi_state_machine = roi_state_machine
        self._eye_model = eye_model
        self._pupil_tracker = LocalPupilTracker()

    def _build_fail_debug(
        self,
        *,
        frame_id: int,
        current_mode: str,
        face_bbox: Optional[Tuple[int, int, int, int]],
        final_roi: Optional[RoiBox],
        track_from_previous_bbox_applied: bool,
    ) -> Dict[str, Any]:
        geom = RoiGeometry()
        roi_tuple = final_roi.as_tuple() if final_roi is not None else None
        dbg_rec = build_roi_debug_record(
            frame_id=frame_id,
            eye=self._side,
            current_mode=current_mode,
            face_bbox=face_bbox,
            final_roi=roi_tuple,
            union_mask_bbox=None,
            track_from_previous_bbox_applied=track_from_previous_bbox_applied,
            geometry=geom,
        )
        return {
            "side": self._side,
            "roi": roi_tuple,
            "native_geometry": {},
            "roi_mode": current_mode,
            "roi_debug": dbg_rec.to_log_dict(),
        }

    def extract_one(
        self,
        bgr_image: BGRImage,
        face_bbox: Optional[Tuple[int, int, int, int]],
        frame_id: int = -1,
        face_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Dict[str, List[Point2D]], Optional[RoiBox], Dict[str, Any]]:
        img = bgr_image.data
        if img is None or img.size == 0:
            self._roi_state_machine.commit_failure("empty_image")
            return (
                False,
                {k: [] for k in CANONICAL_ORDER},
                None,
                self._build_fail_debug(
                    frame_id=frame_id,
                    current_mode=MODE_DETECT,
                    face_bbox=face_bbox,
                    final_roi=None,
                    track_from_previous_bbox_applied=False,
                ),
            )

        # ROI prediction (this decides DETECT/TRACK/RECOVER for this frame)
        roi, roi_mode = self._roi_state_machine.predict(img, face_bbox, face_info)
        if roi is None or not roi.valid:
            self._roi_state_machine.commit_failure("roi_fail")
            return (
                False,
                {k: [] for k in CANONICAL_ORDER},
                roi,
                self._build_fail_debug(
                    frame_id=frame_id,
                    current_mode=roi_mode,
                    face_bbox=face_bbox,
                    final_roi=roi,
                    track_from_previous_bbox_applied=(roi_mode == MODE_TRACK),
                ),
            )

        x, y, w, h = roi.as_tuple()
        if roi_mode == MODE_DETECT:
            self._pupil_tracker.reset()
        patch_bgr = img[y : y + h, x : x + w]
        if patch_bgr.size == 0:
            self._roi_state_machine.commit_failure("roi_out_of_image")
            return (
                False,
                {k: [] for k in CANONICAL_ORDER},
                roi,
                self._build_fail_debug(
                    frame_id=frame_id,
                    current_mode=roi_mode,
                    face_bbox=face_bbox,
                    final_roi=roi,
                    track_from_previous_bbox_applied=(roi_mode == MODE_TRACK),
                ),
            )

        patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
        result: Optional[EllSegResult] = self._eye_model.infer(patch_rgb)
        if result is None or not result.valid:
            self._roi_state_machine.commit_failure("ellseg_invalid")
            return (
                False,
                {k: [] for k in CANONICAL_ORDER},
                roi,
                self._build_fail_debug(
                    frame_id=frame_id,
                    current_mode=roi_mode,
                    face_bbox=face_bbox,
                    final_roi=roi,
                    track_from_previous_bbox_applied=(roi_mode == MODE_TRACK),
                ),
            )

        raw_result = result
        raw_obs = build_roi_observation(result=raw_result, roi_xywh=roi.as_tuple(), side=self._side)
        fused_pupil_center_roi, tracking_debug = self._pupil_tracker.update(
            patch_bgr,
            network_center=raw_result.pupil_center,
            mask_centroid=raw_obs.pupil_mask_centroid,
            iris_ellipse=raw_result.iris_ellipse,
            mask_ok=raw_obs.mask_ok,
            pupil_mask_area=raw_obs.clean_pupil_mask_area,
            pupil_clean_area_ratio=raw_obs.pupil_clean_area_ratio,
        )
        tracking_shift_px = None
        if fused_pupil_center_roi is not None and raw_result.pupil_center is not None:
            tracking_shift_px = float(
                np.hypot(
                    float(fused_pupil_center_roi.x) - float(raw_result.pupil_center.x),
                    float(fused_pupil_center_roi.y) - float(raw_result.pupil_center.y),
                )
            )
        if (
            fused_pupil_center_roi is not None
            and raw_result.pupil_center is not None
            and tracking_shift_px is not None
            and tracking_shift_px <= TRACK_MAX_APPLY_SHIFT
        ):
            result = replace(
                raw_result,
                pupil_center=fused_pupil_center_roi,
                pupil_ellipse=shift_ellipse_center(raw_result.pupil_ellipse, fused_pupil_center_roi),
            )
        else:
            result = raw_result

        # compatibility export: pupil=1, iris=4 points
        eye_dict_roi: Dict[str, List[Point2D]] = {k: [] for k in CANONICAL_ORDER}
        eye_dict_roi["pupil"] = [result.pupil_center] if result.pupil_center else []
        if result.iris_ellipse:
            left_pt, right_pt, top_pt, bottom_pt = ellipse_to_cardinal_points(result.iris_ellipse)
            eye_dict_roi["iris"] = [left_pt, right_pt, top_pt, bottom_pt]

        if not validate_single_eye_core_dict(eye_dict_roi):
            self._roi_state_machine.commit_failure("core_schema_incomplete")
            return (
                False,
                {k: [] for k in CANONICAL_ORDER},
                roi,
                self._build_fail_debug(
                    frame_id=frame_id,
                    current_mode=roi_mode,
                    face_bbox=face_bbox,
                    final_roi=roi,
                    track_from_previous_bbox_applied=(roi_mode == MODE_TRACK),
                ),
            )

        eye_dict_img = _points_roi_to_image(eye_dict_roi, x, y)

        # ------------------------------------------------------------------
        # New main chain: RoiObservation + RoiGeometry + DETECT gating
        # ------------------------------------------------------------------
        obs = build_roi_observation(result=result, roi_xywh=roi.as_tuple(), side=self._side)
        plausible, plausibility_reason = check_observation_plausibility(obs)
        if not plausible:
            self._roi_state_machine.commit_failure(f"implausible_geometry:{plausibility_reason}")
            return (
                False,
                {k: [] for k in CANONICAL_ORDER},
                roi,
                self._build_fail_debug(
                    frame_id=frame_id,
                    current_mode=roi_mode,
                    face_bbox=face_bbox,
                    final_roi=roi,
                    track_from_previous_bbox_applied=(roi_mode == MODE_TRACK),
                ),
            )

        geom = compute_roi_geometry(obs)

        img_h, img_w = img.shape[:2]
        last_face_bbox = self._roi_state_machine.get_previous_face_bbox()
        face_bbox_is_stable, face_bbox_corner_max_shift = _compute_face_bbox_stability(
            current_face_bbox=face_bbox,
            last_face_bbox=last_face_bbox,
            img_w=img_w,
            img_h=img_h,
        )
        geom.face_bbox_is_stable = bool(face_bbox_is_stable)
        geom.face_bbox_corner_max_shift = float(face_bbox_corner_max_shift)

        should_switch = should_switch_to_detect(
            geometry_confidence=geom.geometry_confidence,
            face_bbox_is_stable=geom.face_bbox_is_stable,
        )
        self._roi_state_machine.commit_success(
            roi=roi,
            observation=obs,
            geometry=geom,
            should_switch_to_detect=should_switch,
        )

        track_from_previous_bbox_applied = (roi_mode == MODE_TRACK)
        dbg_rec = build_roi_debug_record(
            frame_id=frame_id,
            eye=self._side,
            current_mode=roi_mode,
            face_bbox=face_bbox,
            final_roi=roi.as_tuple(),
            union_mask_bbox=obs.union_mask_bbox,
            track_from_previous_bbox_applied=track_from_previous_bbox_applied,
            geometry=geom,
        )

        native_geometry = {
            "segmentation_mask": result.segmentation_mask,
            "pupil_center": (
                Point2D(x=result.pupil_center.x + x, y=result.pupil_center.y + y) if result.pupil_center else None
            ),
            "raw_pupil_center": (
                Point2D(x=raw_result.pupil_center.x + x, y=raw_result.pupil_center.y + y)
                if raw_result.pupil_center
                else None
            ),
            "tracked_pupil_center": (
                Point2D(x=fused_pupil_center_roi.x + x, y=fused_pupil_center_roi.y + y)
                if fused_pupil_center_roi is not None
                else None
            ),
            "pupil_tracking_source": tracking_debug.get("fused_source"),
            "pupil_tracking_score": tracking_debug.get("tracking_score"),
            "pupil_tracking_shift_px": tracking_shift_px,
            "pupil_ellipse": _shift_ellipse(result.pupil_ellipse, x, y),
            "iris_ellipse": _shift_ellipse(result.iris_ellipse, x, y),
        }

        debug: Dict[str, Any] = {
            "side": self._side,
            "roi": roi.as_tuple(),
            "native_geometry": native_geometry,
            "roi_mode": roi_mode,
            "roi_debug": dbg_rec.to_log_dict(),
        }
        return True, eye_dict_img, roi, debug


def extract_fitting_dict_for_pair(
    left: BGRImage,
    right: BGRImage,
    left_extractor: EyeLandmarkExtractor,
    right_extractor: EyeLandmarkExtractor,
    left_bbox: Optional[Tuple[int, int, int, int]],
    right_bbox: Optional[Tuple[int, int, int, int]],
    frame_id: int = -1,
    left_face_info: Optional[Dict[str, Any]] = None,
    right_face_info: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Dict[str, Dict[str, List[Point2D]]], Dict[str, Any]]:
    ok_ll, dict_ll, roi_ll, debug_ll = left_extractor.extract_one(
        left, left_bbox, frame_id=frame_id, face_info=left_face_info
    )
    ok_rr, dict_rr, roi_rr, debug_rr = right_extractor.extract_one(
        right, right_bbox, frame_id=frame_id, face_info=right_face_info
    )

    debug: Dict[str, Any] = {
        "left": {
            "roi": roi_ll.as_tuple() if roi_ll else None,
            "debug": debug_ll,
            "debug_right_eye": {},
            "points": {"left": dict_ll, "right": {}},
        },
        "right": {
            "roi": roi_rr.as_tuple() if roi_rr else None,
            "debug": {},
            "debug_right_eye": debug_rr,
            "points": {"left": {}, "right": dict_rr},
        },
    }

    if not ok_ll:
        logger.warning("left eye extraction fail (left image)")
        debug["reason"] = "left_eye_fail"
        return False, {"left": {}, "right": {}}, debug
    if not ok_rr:
        logger.warning("right eye extraction fail (right image)")
        debug["reason"] = "right_eye_fail"
        return False, {"left": {}, "right": {}}, debug

    result: Dict[str, Dict[str, List[Point2D]]] = {
        "left": {"left": dict_ll, "right": {}},
        "right": {"left": {}, "right": dict_rr},
    }
    return True, result, debug


__all__ = [
    "EyeLandmarkExtractor",
    "extract_fitting_dict_for_pair",
    "validate_single_eye_core_dict",
]

