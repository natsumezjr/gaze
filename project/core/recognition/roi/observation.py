from __future__ import annotations

import cv2
import numpy as np

from dataclasses import dataclass
from typing import Optional, Tuple

from project.data.data_models import Ellipse2D, Point2D
from project.core.recognition.eye_model_ellseg import EllSegResult
from .utils import (
    compute_pupil_center_inside_iris_score,
    points_inside_ellipse_ratio,
    sample_ellipse_points,
)


@dataclass
class RoiObservation:
    """
    ROI-local observation for geometry / state machine decision.

    All *_roi fields are in ROI-local coordinates.
    All *_img fields are in image coordinates.
    """

    side: str
    roi_xywh: Tuple[int, int, int, int]
    roi_shape_hw: Tuple[int, int]

    pupil_center_img: Optional[Point2D]
    iris_center_img: Optional[Point2D]

    pupil_ellipse_roi: Optional[Ellipse2D]
    iris_ellipse_roi: Optional[Ellipse2D]
    pupil_ellipse_img: Optional[Ellipse2D]
    iris_ellipse_img: Optional[Ellipse2D]

    pupil_mask_bbox: Optional[Tuple[int, int, int, int]]
    iris_mask_bbox: Optional[Tuple[int, int, int, int]]
    union_mask_bbox: Optional[Tuple[int, int, int, int]]

    pupil_mask_centroid: Optional[Tuple[float, float]]
    iris_mask_centroid: Optional[Tuple[float, float]]
    union_mask_centroid: Optional[Tuple[float, float]]

    pupil_overlap_score: float
    iris_overlap_score: float

    pupil_inside_iris_ratio: float
    pupil_center_inside_iris_score: float
    pupil_center_inside_iris_d2: float


def _shift_ellipse(el: Optional[Ellipse2D], dx: float, dy: float) -> Optional[Ellipse2D]:
    if el is None:
        return None
    return Ellipse2D(
        cx=float(el.cx) + dx,
        cy=float(el.cy) + dy,
        major_axis=float(el.major_axis),
        minor_axis=float(el.minor_axis),
        angle_deg=float(el.angle_deg),
        confidence=float(getattr(el, "confidence", 0.0) or 0.0),
    )


def _bbox_from_bool(m: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    if m is None or not np.any(m):
        return None
    rows, cols = np.where(m)
    if rows.size == 0 or cols.size == 0:
        return None
    y_min, y_max = int(rows.min()), int(rows.max())
    x_min, x_max = int(cols.min()), int(cols.max())
    return (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)


def _centroid_from_bool(m: np.ndarray) -> Optional[Tuple[float, float]]:
    if m is None or not np.any(m):
        return None
    mom = cv2.moments(m.astype(np.uint8))
    if mom.get("m00", 0.0) <= 0:
        return None
    return (float(mom["m10"] / mom["m00"]), float(mom["m01"] / mom["m00"]))


def build_roi_observation(
    result: EllSegResult,
    roi_xywh: Tuple[int, int, int, int],
    side: str,
) -> RoiObservation:
    """
    Build RoiObservation directly from EllSegResult + current RoiBox.

    """

    roi_x, roi_y, roi_w, roi_h = roi_xywh
    roi_shape_hw = (int(roi_h), int(roi_w))

    pupil_ellipse_roi = result.pupil_ellipse
    iris_ellipse_roi = result.iris_ellipse

    pupil_center_roi = result.pupil_center

    pupil_ellipse_img = _shift_ellipse(pupil_ellipse_roi, roi_x, roi_y)
    iris_ellipse_img = _shift_ellipse(iris_ellipse_roi, roi_x, roi_y)

    pupil_center_img = (
        Point2D(x=float(pupil_center_roi.x) + roi_x, y=float(pupil_center_roi.y) + roi_y)
        if pupil_center_roi is not None
        else None
    )
    iris_center_img = (
        Point2D(x=float(iris_ellipse_roi.cx) + roi_x, y=float(iris_ellipse_roi.cy) + roi_y)
        if iris_ellipse_roi is not None
        else None
    )

    seg_map = result.segmentation_mask
    pupil_mask_bbox: Optional[Tuple[int, int, int, int]] = None
    iris_mask_bbox: Optional[Tuple[int, int, int, int]] = None
    union_mask_bbox: Optional[Tuple[int, int, int, int]] = None
    pupil_mask_centroid: Optional[Tuple[float, float]] = None
    iris_mask_centroid: Optional[Tuple[float, float]] = None
    union_mask_centroid: Optional[Tuple[float, float]] = None

    pupil_overlap_score = 0.0
    iris_overlap_score = 0.0

    if seg_map is not None and seg_map.size > 0:
        seg_u8 = seg_map.astype(np.uint8)
        pupil_mask = seg_u8 == 2
        iris_mask = seg_u8 == 1
        union_mask = np.logical_or(pupil_mask, iris_mask)

        pupil_mask_bbox = _bbox_from_bool(pupil_mask)
        iris_mask_bbox = _bbox_from_bool(iris_mask)
        union_mask_bbox = _bbox_from_bool(union_mask)

        pupil_mask_centroid = _centroid_from_bool(pupil_mask)
        iris_mask_centroid = _centroid_from_bool(iris_mask)
        union_mask_centroid = _centroid_from_bool(union_mask)

        def _overlap(el: Optional[Ellipse2D], target_bool: np.ndarray) -> float:
            if el is None:
                return 0.0
            h, w = target_bool.shape[:2]
            ell_mask = np.zeros((h, w), dtype=np.uint8)
            ax1 = int(round(float(el.major_axis) * 0.5))
            ax2 = int(round(float(el.minor_axis) * 0.5))
            if ax1 <= 0 or ax2 <= 0:
                return 0.0
            c = (int(round(float(el.cx))), int(round(float(el.cy))))
            cv2.ellipse(ell_mask, c, (ax1, ax2), float(el.angle_deg), 0, 360, 1, thickness=-1)
            n_ell = int(np.count_nonzero(ell_mask))
            if n_ell <= 0:
                return 0.0
            inter = np.logical_and(ell_mask.astype(bool), target_bool)
            return float(np.count_nonzero(inter)) / float(n_ell)

        pupil_overlap_score = _overlap(pupil_ellipse_roi, pupil_mask)
        iris_overlap_score = _overlap(iris_ellipse_roi, iris_mask)

    # pupil inside iris ratio must be derived from ellipse points
    pupil_inside_iris_ratio = 0.0
    if pupil_ellipse_roi is not None and iris_ellipse_roi is not None:
        pts = sample_ellipse_points(pupil_ellipse_roi, num_points=72)
        pupil_inside_iris_ratio = points_inside_ellipse_ratio(pts, iris_ellipse_roi)

    pupil_center_inside_iris_score = 0.0
    pupil_center_inside_iris_d2 = 0.0
    if pupil_center_roi is not None and iris_ellipse_roi is not None:
        pupil_center_inside_iris_score, pupil_center_inside_iris_d2 = compute_pupil_center_inside_iris_score(
            pupil_center_roi,
            iris_ellipse_roi,
        )

    return RoiObservation(
        side=side,
        roi_xywh=(int(roi_x), int(roi_y), int(roi_w), int(roi_h)),
        roi_shape_hw=(int(roi_shape_hw[0]), int(roi_shape_hw[1])),
        pupil_center_img=pupil_center_img,
        iris_center_img=iris_center_img,
        pupil_ellipse_roi=pupil_ellipse_roi,
        iris_ellipse_roi=iris_ellipse_roi,
        pupil_ellipse_img=pupil_ellipse_img,
        iris_ellipse_img=iris_ellipse_img,
        pupil_mask_bbox=pupil_mask_bbox,
        iris_mask_bbox=iris_mask_bbox,
        union_mask_bbox=union_mask_bbox,
        pupil_mask_centroid=pupil_mask_centroid,
        iris_mask_centroid=iris_mask_centroid,
        union_mask_centroid=union_mask_centroid,
        pupil_overlap_score=float(pupil_overlap_score),
        iris_overlap_score=float(iris_overlap_score),
        pupil_inside_iris_ratio=float(pupil_inside_iris_ratio),
        pupil_center_inside_iris_score=float(pupil_center_inside_iris_score),
        pupil_center_inside_iris_d2=float(pupil_center_inside_iris_d2),
    )


__all__ = ["RoiObservation", "build_roi_observation"]

