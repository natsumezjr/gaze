from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from project.config.logging_config import setup_logging
from project.core.recognition.face_detector import YUNET_CASE_0_ALL_5_IN_BOUNDS
from .constants import (
    FIXED_RATIO,
    INIT_BAND_H_RATIO,
    INIT_BAND_LEFT_TL_X_RATIO,
    INIT_BAND_RIGHT_TL_X_RATIO,
    INIT_BAND_TL_Y_RATIO,
    INIT_BAND_W_RATIO,
    INIT_LEFT_EYE_DX_RATIO,
    INIT_LEFT_EYE_DY_RATIO,
    INIT_RIGHT_EYE_DX_RATIO,
    INIT_RIGHT_EYE_DY_RATIO,
    INIT_ROI_EXPAND_RATIO,
    MODE_DETECT,
    MODE_RECOVER,
    MODE_TRACK,
    MAX_ASPECT_RATIO,
    MIN_ASPECT_RATIO,
    RECOVER_MASK_OFFSET_DX_RATIO_TH,
    RECOVER_MASK_OFFSET_DY_RATIO_TH,
    ROI_MAX_H_FACE_RATIO,
    ROI_MAX_W_FACE_RATIO,
    ROI_MIN_H_FACE_RATIO,
    ROI_MIN_W_FACE_RATIO,
    UNION_MASK_BBOX_H_EXPAND_RATIO,
    UNION_MASK_BBOX_W_EXPAND_RATIO,
)
from .observation import RoiObservation
from .geometry import RoiGeometry
from .types import RoiBox, RoiMachineState

logger = setup_logging(__name__, logging.DEBUG)


class RoiStateMachine:
    """
    Minimal ROI state machine:
    - predict(...) decides which ROI to use for current frame (DETECT/TRACK/RECOVER).
    - commit_success(...) updates internal state after successful EllSeg+geometry decision.
    - commit_failure(...) updates internal state after ROI-internal reject.
    - reset(...) clears state.
    """

    def __init__(self, side: str):
        if side not in ("left", "right"):
            raise ValueError(f"invalid side for RoiStateMachine: {side}")
        self._side = side
        self._state = RoiMachineState()

        # previous face bbox (used by geometry stability in extractor)
        self._previous_face_bbox: Optional[Tuple[int, int, int, int]] = None

        # last observation/geometry used by TRACK/RECOVER predict
        self._last_observation: Optional[RoiObservation] = None
        self._last_geometry: Optional[RoiGeometry] = None

    def get_previous_face_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        return self._previous_face_bbox

    # ------------------------------------------------------------------
    # DETECT / TRACK / RECOVER predict
    # ------------------------------------------------------------------
    def predict(
        self,
        image: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]],
        face_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[RoiBox], str]:
        """
        Return (roi_for_current_frame, roi_mode).
        """

        # Update face priors for stability computations.
        if face_bbox is not None:
            self._previous_face_bbox = self._state.last_face_bbox
            self._state.last_face_bbox = face_bbox
            self._state.last_face_info = face_info

        mode = self._state.mode

        # DETECT
        if mode == MODE_DETECT:
            if face_bbox is None:
                # Wait for face re-detection; do not reuse old ROI.
                return None, MODE_DETECT

            roi = self.build_detect_roi_from_face(
                face_bbox=face_bbox,
                image_shape=image.shape,
                face_info=face_info,
            )
            self._state.last_roi = roi
            return roi, MODE_DETECT

        # TRACK
        if mode == MODE_TRACK:
            if self._last_observation is None or self._last_geometry is None:
                self._state.mode = MODE_DETECT
                return self.predict(image=image, face_bbox=face_bbox, face_info=face_info)

            if self._state.last_roi is None:
                self._state.mode = MODE_DETECT
                return self.predict(image=image, face_bbox=face_bbox, face_info=face_info)

            union_bbox = self._last_observation.union_mask_bbox
            if union_bbox is None:
                return self._state.last_roi, MODE_TRACK

            track_roi = self.build_track_roi_from_last_bbox(
                union_bbox_local=union_bbox,
                base_roi=self._state.last_roi,
                image_shape=image.shape,
            )
            if track_roi is not None and track_roi.valid:
                return track_roi, MODE_TRACK
            return self._state.last_roi, MODE_TRACK

        # RECOVER
        if mode == MODE_RECOVER:
            if self._last_observation is None or self._last_geometry is None:
                self._state.mode = MODE_DETECT
                return self.predict(image=image, face_bbox=face_bbox, face_info=face_info)

            if self._state.last_roi is None:
                self._state.mode = MODE_DETECT
                return self.predict(image=image, face_bbox=face_bbox, face_info=face_info)

            union_bbox = self._last_observation.union_mask_bbox
            if union_bbox is None:
                self._state.mode = MODE_DETECT
                return self.predict(image=image, face_bbox=face_bbox, face_info=face_info)

            candidate_roi = self.build_track_roi_from_last_bbox(
                union_bbox_local=union_bbox,
                base_roi=self._state.last_roi,
                image_shape=image.shape,
            )
            if candidate_roi is None:
                return self._state.last_roi, MODE_RECOVER

            recover_roi = self.build_recover_roi_from_previous_and_current_bbox(
                previous_roi=self._state.last_roi,
                current_roi=candidate_roi,
                image_shape=image.shape,
            )
            if recover_roi is not None:
                return recover_roi, MODE_RECOVER

            return candidate_roi, MODE_RECOVER

        # Defensive fallback
        self._state.mode = MODE_DETECT
        return self.predict(image=image, face_bbox=face_bbox, face_info=face_info)

    # ------------------------------------------------------------------
    # Update after success
    # ------------------------------------------------------------------
    def commit_success(
        self,
        roi: RoiBox,
        observation: RoiObservation,
        geometry: RoiGeometry,
        should_switch_to_detect: bool,
    ) -> None:
        """
        Update state after EllSeg succeeded and geometry decision is computed.
        """

        self._last_observation = observation
        self._last_geometry = geometry

        mode = self._state.mode  # mode used for current frame

        if should_switch_to_detect:
            next_mode = MODE_DETECT
        else:
            offsets = self.compute_bbox_offset_ratio(prev_roi=roi)
            if offsets is None:
                next_mode = MODE_TRACK
            else:
                dx_ratio, dy_ratio = offsets
                if dx_ratio > RECOVER_MASK_OFFSET_DX_RATIO_TH or dy_ratio > RECOVER_MASK_OFFSET_DY_RATIO_TH:
                    next_mode = MODE_RECOVER
                else:
                    next_mode = MODE_TRACK

        logger.debug(
            "[ROI_DECISION] side=%s mode=%s roi=%s geo=%.3f face_stable=%s next_mode=%s",
            self._side,
            mode,
            roi.as_tuple(),
            geometry.geometry_confidence,
            geometry.face_bbox_is_stable,
            next_mode,
        )

        self._state.last_roi = roi
        self._state.mode = next_mode

    # ------------------------------------------------------------------
    # Update after reject
    # ------------------------------------------------------------------
    def commit_failure(self, reason: str) -> None:
        """
        ROI-internal reject occurred for this frame.
        """
        logger.debug(
            "RoiStateMachine.commit_failure side=%s reason=%s mode=%s",
            self._side,
            reason,
            self._state.mode,
        )
        # On any ROI-internal reject, request fresh YuNet face bbox next frame.
        self._state.mode = MODE_DETECT

    def reset(self) -> None:
        logger.debug("RoiStateMachine.reset side=%s", self._side)
        self._state = RoiMachineState()
        self._previous_face_bbox = None
        self._last_observation = None
        self._last_geometry = None

    # ------------------------------------------------------------------
    # Geometry ROI builders (track/recover/detect)
    # ------------------------------------------------------------------
    def build_detect_roi_from_face(
        self,
        face_bbox: Tuple[int, int, int, int],
        image_shape: Tuple[int, ...],
        face_info: Optional[Dict[str, Any]] = None,
    ) -> RoiBox:
        """
        Build DETECT initial ROI from face bbox + YuNet eye center + fixed ratio.
        """

        fx, fy, fw, fh = face_bbox
        img_h, img_w = int(image_shape[0]), int(image_shape[1])

        # case id: idx in [0..3]
        case_id = int(face_info.get("case_id", YUNET_CASE_0_ALL_5_IN_BOUNDS)) if face_info else YUNET_CASE_0_ALL_5_IN_BOUNDS
        idx = int(case_id) if 0 <= int(case_id) <= 3 else 0

        if self._side == "left":
            band_tl_x = fx + float(INIT_BAND_LEFT_TL_X_RATIO[idx]) * fw
        else:
            band_tl_x = fx + float(INIT_BAND_RIGHT_TL_X_RATIO[idx]) * fw
        band_tl_y = fy + float(INIT_BAND_TL_Y_RATIO[idx]) * fh
        band_w = float(INIT_BAND_W_RATIO[idx]) * fw
        band_h = float(INIT_BAND_H_RATIO[idx]) * fh

        band_cx = band_tl_x + 0.5 * band_w
        band_cy = band_tl_y + 0.5 * band_h

        eye_key = "left_eye" if self._side == "left" else "right_eye"
        eye_pt = face_info.get(eye_key) if face_info else None

        if eye_pt is not None:
            ex, ey = float(eye_pt[0]), float(eye_pt[1])
            if self._side == "left":
                dx_r = float(INIT_LEFT_EYE_DX_RATIO[idx])
                dy_r = float(INIT_LEFT_EYE_DY_RATIO[idx])
            else:
                dx_r = float(INIT_RIGHT_EYE_DX_RATIO[idx])
                dy_r = float(INIT_RIGHT_EYE_DY_RATIO[idx])

            eye_cx = ex - dx_r * band_w
            eye_cy = ey - dy_r * band_h
            eye_w = band_w
            eye_h = band_h
        else:
            eye_cx, eye_cy, eye_w, eye_h = band_cx, band_cy, band_w, band_h

        fix_ratio = FIXED_RATIO[idx]
        final_cx = fix_ratio * band_cx + (1.0 - fix_ratio) * eye_cx
        final_cy = fix_ratio * band_cy + (1.0 - fix_ratio) * eye_cy
        final_w = fix_ratio * band_w + (1.0 - fix_ratio) * eye_w
        final_h = fix_ratio * band_h + (1.0 - fix_ratio) * eye_h

        # extra expand before clip
        final_w = float(final_w + INIT_ROI_EXPAND_RATIO * band_w)
        final_h = float(final_h + INIT_ROI_EXPAND_RATIO * band_h)

        limits = self._get_roi_size_limits(face_bbox=face_bbox, image_shape=image_shape)
        if limits is None:
            w = max(1, min(int(round(final_w)), img_w))
            h = max(1, min(int(round(final_h)), img_h))
            x = int(final_cx - 0.5 * w)
            y = int(final_cy - 0.5 * h)
            x = max(0, min(x, img_w - w))
            y = max(0, min(y, img_h - h))
            return RoiBox(x=x, y=y, w=w, h=h, side=self._side, valid=False)

        min_w, min_h, max_w, max_h = limits
        w = int(np.clip(final_w, min_w, max_w))
        h = int(np.clip(final_h, min_h, max_h))

        x = int(final_cx - 0.5 * w)
        y = int(final_cy - 0.5 * h)
        x = max(0, min(x, img_w - w))
        y = max(0, min(y, img_h - h))

        roi = RoiBox(x=x, y=y, w=w, h=h, side=self._side, valid=True)
        roi.valid = self._validate_roi(roi)
        return roi

    def build_track_roi_from_last_bbox(
        self,
        union_bbox_local: Tuple[int, int, int, int],
        base_roi: RoiBox,
        image_shape: Tuple[int, ...],
    ) -> Optional[RoiBox]:
        """
        TRACK ROI update:
        ROI center is set on previous union_mask bbox center,
        and ROI size is expanded by fixed UNION_MASK_BBOX_*_EXPAND_RATIO.
        """

        ubx, uby, ubw, ubh = union_bbox_local
        if ubw <= 0 or ubh <= 0:
            return None

        img_h, img_w = int(image_shape[0]), int(image_shape[1])
        union_cx = base_roi.x + ubx + 0.5 * ubw
        union_cy = base_roi.y + uby + 0.5 * ubh

        new_w = float(ubw) * UNION_MASK_BBOX_W_EXPAND_RATIO
        new_h = float(ubh) * UNION_MASK_BBOX_H_EXPAND_RATIO

        limits = self._get_roi_size_limits(face_bbox=self._state.last_face_bbox, image_shape=image_shape)
        if limits is None:
            return None

        min_w, min_h, max_w, max_h = limits
        w_i = max(min_w, min(int(round(new_w)), max_w, img_w))
        h_i = max(min_h, min(int(round(new_h)), max_h, img_h))

        x_i = max(0, min(int(round(union_cx - 0.5 * w_i)), img_w - w_i))
        y_i = max(0, min(int(round(union_cy - 0.5 * h_i)), img_h - h_i))

        return RoiBox(x=x_i, y=y_i, w=w_i, h=h_i, side=self._side, valid=True)

    def build_recover_roi_from_previous_and_current_bbox(
        self,
        previous_roi: RoiBox,
        current_roi: RoiBox,
        image_shape: Tuple[int, ...],
    ) -> Optional[RoiBox]:
        """
        RECOVER ROI update:
        Use the 8 points (corners) from previous_roi and current_roi
        to compute a maximum bounding ROI.
        """

        if not previous_roi.valid or not current_roi.valid:
            return None

        img_h, img_w = int(image_shape[0]), int(image_shape[1])
        if img_h <= 0 or img_w <= 0:
            return None

        corners_x = [previous_roi.x, previous_roi.x + previous_roi.w, current_roi.x, current_roi.x + current_roi.w]
        corners_y = [previous_roi.y, previous_roi.y + previous_roi.h, current_roi.y, current_roi.y + current_roi.h]
        min_x, max_x = min(corners_x), max(corners_x)
        min_y, max_y = min(corners_y), max(corners_y)

        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(img_w, max_x)
        max_y = min(img_h, max_y)

        w = int(round(max_x - min_x))
        h = int(round(max_y - min_y))
        if w <= 0 or h <= 0:
            return None

        x = int(round(min_x))
        y = int(round(min_y))
        x = max(0, min(x, img_w - w))
        y = max(0, min(y, img_h - h))

        roi = RoiBox(x=x, y=y, w=w, h=h, side=self._side, valid=True)
        roi.valid = self._validate_roi(roi)
        return roi if roi.valid else None

    # ------------------------------------------------------------------
    # State update helpers
    # ------------------------------------------------------------------
    def compute_bbox_offset_ratio(self, prev_roi: RoiBox) -> Optional[Tuple[float, float]]:
        if self._last_observation is None:
            return None
        union_bbox = self._last_observation.union_mask_bbox
        if union_bbox is None:
            return None
        if prev_roi.w <= 0 or prev_roi.h <= 0:
            return None

        ubx, uby, ubw, ubh = union_bbox
        mask_cx = prev_roi.x + ubx + 0.5 * ubw
        mask_cy = prev_roi.y + uby + 0.5 * ubh
        roi_cx = prev_roi.x + 0.5 * prev_roi.w
        roi_cy = prev_roi.y + 0.5 * prev_roi.h
        dx_ratio = abs(mask_cx - roi_cx) / max(prev_roi.w, 1e-6)
        dy_ratio = abs(mask_cy - roi_cy) / max(prev_roi.h, 1e-6)
        return dx_ratio, dy_ratio

    # ------------------------------------------------------------------
    # ROI constraints helpers (size/aspect validation)
    # ------------------------------------------------------------------
    def _validate_roi(self, roi: RoiBox, prev_roi: Optional[RoiBox] = None) -> bool:
        limits = self._get_roi_size_limits_no_image_clip(face_bbox=self._state.last_face_bbox)
        if limits is not None:
            min_w, min_h, max_w, max_h = limits
            if roi.w < min_w or roi.h < min_h:
                return False
            if roi.w > max_w or roi.h > max_h:
                return False

        aspect = roi.w / float(max(1, roi.h))
        if aspect < MIN_ASPECT_RATIO or aspect > MAX_ASPECT_RATIO:
            return False

        if prev_roi is not None and prev_roi.w > 0 and prev_roi.h > 0:
            # reject ROI size jump
            max_ratio = 1.5
            if roi.w > prev_roi.w * max_ratio or roi.w < prev_roi.w / max_ratio:
                return False
            if roi.h > prev_roi.h * max_ratio or roi.h < prev_roi.h / max_ratio:
                return False

        return True

    def _get_roi_size_limits(
        self,
        face_bbox: Optional[Tuple[int, int, int, int]],
        image_shape: Tuple[int, ...],
    ) -> Optional[Tuple[int, int, int, int]]:
        img_h, img_w = int(image_shape[0]), int(image_shape[1])
        if face_bbox is None:
            return None
        _, _, fw, fh = [int(v) for v in face_bbox]
        if fw <= 0 or fh <= 0:
            return None

        min_w = max(1, int(round(fw * ROI_MIN_W_FACE_RATIO)))
        min_h = max(1, int(round(fh * ROI_MIN_H_FACE_RATIO)))
        max_w = max(min_w, int(round(fw * ROI_MAX_W_FACE_RATIO)))
        max_h = max(min_h, int(round(fh * ROI_MAX_H_FACE_RATIO)))

        min_w = min(min_w, img_w)
        min_h = min(min_h, img_h)
        max_w = min(max_w, img_w)
        max_h = min(max_h, img_h)
        if min_w > max_w or min_h > max_h:
            return None

        return int(min_w), int(min_h), int(max_w), int(max_h)

    def _get_roi_size_limits_no_image_clip(
        self,
        face_bbox: Optional[Tuple[int, int, int, int]],
    ) -> Optional[Tuple[int, int, int, int]]:
        if face_bbox is None:
            return None
        _, _, fw, fh = [int(v) for v in face_bbox]
        if fw <= 0 or fh <= 0:
            return None
        min_w = max(1, int(round(fw * ROI_MIN_W_FACE_RATIO)))
        min_h = max(1, int(round(fh * ROI_MIN_H_FACE_RATIO)))
        max_w = max(min_w, int(round(fw * ROI_MAX_W_FACE_RATIO)))
        max_h = max(min_h, int(round(fh * ROI_MAX_H_FACE_RATIO)))
        if min_w > max_w or min_h > max_h:
            return None
        return int(min_w), int(min_h), int(max_w), int(max_h)


__all__ = ["RoiStateMachine"]

