from __future__ import annotations

import numpy as np

from dataclasses import dataclass
from typing import Optional, Tuple

from project.core.recognition.roi.observation import RoiObservation
from .constants import (
    ROI_GEOMETRY_CONFIDENCE_THRESHOLD,
    ROI_PUPIL_CENTER_INSIDE_IRIS_D2_MAX,
    ROI_PUPIL_INSIDE_IRIS_RATIO_MIN,
    ROI_PUPIL_IRIS_CENTER_DIST_RATIO_MAX,
    ROI_PUPIL_TO_IRIS_MAJOR_RATIO_MAX,
    ROI_PUPIL_TO_IRIS_MAJOR_RATIO_MIN,
    ROI_PUPIL_TO_IRIS_MINOR_RATIO_MAX,
    ROI_PUPIL_TO_IRIS_MINOR_RATIO_MIN,
)


@dataclass
class RoiGeometry:
    """
    Geometry confidence and related constraints.

    This dataclass intentionally contains only core geometry terms.
    """

    contain_score: float = 0.0
    center_score: float = 0.0
    overlap_score: float = 0.0

    geometry_confidence: float = 0.0

    face_bbox_corner_max_shift: float = 0.0
    face_bbox_is_stable: bool = False


def compute_roi_geometry(obs: RoiObservation) -> RoiGeometry:
    """
    Compute pure ROI geometry confidence.

    Notes:
    - geometry_confidence is derived only from core ROI geometry signals.
    - mapping is kept aligned with the validated v2 continuous implementation.
    """

    # raw values (kept un-clipped for mapping)
    contain_raw = float(obs.pupil_inside_iris_ratio)
    center_raw = float(obs.pupil_center_inside_iris_score)

    overlap_pupil_raw = float(obs.pupil_overlap_score)
    overlap_iris_raw = float(obs.iris_overlap_score)
    overlap_mean_raw = 0.5 * overlap_pupil_raw + 0.5 * overlap_iris_raw

    # base scores (clipped)
    contain_score = float(np.clip(contain_raw, 0.0, 1.0))
    center_score = float(np.clip(center_raw, 0.0, 1.0))
    overlap_score = float(
        0.5 * np.clip(overlap_pupil_raw, 0.0, 1.0) + 0.5 * np.clip(overlap_iris_raw, 0.0, 1.0)
    )

    # 1) contain mapping
    contain_mapped = float(max(contain_raw, 0.0) ** 2.5)

    # 2) center mapping (continuous normalized-exp style)
    c0 = 0.30
    k_center = 4.0
    if center_raw <= c0:
        center_mapped = 0.0
    else:
        n_center = float(np.clip((center_raw - c0) / max(1.0 - c0, 1e-6), 0.0, 1.0))
        denom_center = 1.0 - float(np.exp(-k_center))
        if denom_center <= 0.0:
            center_mapped = float(n_center)
        else:
            center_mapped = float((1.0 - np.exp(-k_center * n_center)) / denom_center)

    # 3) overlap mapping (continuous normalized-exp style)
    o0 = 0.20
    k_overlap = 3.0
    overlap_raw = overlap_mean_raw
    if overlap_raw <= o0:
        overlap_mapped = 0.0
    else:
        n_overlap = float(np.clip((overlap_raw - o0) / max(1.0 - o0, 1e-6), 0.0, 1.0))
        denom_overlap = 1.0 - float(np.exp(-k_overlap))
        if denom_overlap <= 0.0:
            overlap_mapped = float(n_overlap)
        else:
            overlap_mapped = float((1.0 - np.exp(-k_overlap * n_overlap)) / denom_overlap)

    combined_pupil_conf = float(np.sqrt(max(contain_mapped, 0.0) * max(center_mapped, 0.0)))
    geometry_confidence = float(np.clip(combined_pupil_conf * (0.7 + 0.3 * overlap_mapped), 0.0, 1.0))

    return RoiGeometry(
        contain_score=contain_score,
        center_score=center_score,
        overlap_score=overlap_score,
        geometry_confidence=geometry_confidence,
    )


def should_switch_to_detect(
    geometry_confidence: float,
    face_bbox_is_stable: bool,
) -> bool:
    """
    DETECT gating:
    - if face bbox is not stable -> switch to DETECT
    - else if geometry_confidence < ROI_GEOMETRY_CONFIDENCE_THRESHOLD -> switch to DETECT
    """

    if not face_bbox_is_stable:
        return True
    return float(geometry_confidence) < float(ROI_GEOMETRY_CONFIDENCE_THRESHOLD)


def check_observation_plausibility(obs: RoiObservation) -> Tuple[bool, str]:
    """
    Reject obviously implausible pupil/iris geometry before it is exported.

    These checks intentionally focus on hard physical constraints instead of
    soft confidence scores so gross outliers do not leak into 3D fitting.
    """

    pupil = obs.pupil_ellipse_roi
    iris = obs.iris_ellipse_roi
    pupil_center = obs.pupil_center_img
    if pupil is None or iris is None or pupil_center is None:
        return False, "missing_core_geometry"

    if pupil.major_axis <= 0.0 or pupil.minor_axis <= 0.0 or iris.major_axis <= 0.0 or iris.minor_axis <= 0.0:
        return False, "non_positive_axis"

    if pupil.major_axis >= iris.major_axis or pupil.minor_axis >= iris.minor_axis:
        return False, "pupil_axis_not_smaller_than_iris"

    major_ratio = float(pupil.major_axis) / float(max(iris.major_axis, 1e-6))
    minor_ratio = float(pupil.minor_axis) / float(max(iris.minor_axis, 1e-6))
    if not (ROI_PUPIL_TO_IRIS_MAJOR_RATIO_MIN <= major_ratio <= ROI_PUPIL_TO_IRIS_MAJOR_RATIO_MAX):
        return False, "pupil_iris_major_ratio_out_of_range"
    if not (ROI_PUPIL_TO_IRIS_MINOR_RATIO_MIN <= minor_ratio <= ROI_PUPIL_TO_IRIS_MINOR_RATIO_MAX):
        return False, "pupil_iris_minor_ratio_out_of_range"

    if float(obs.pupil_inside_iris_ratio) < float(ROI_PUPIL_INSIDE_IRIS_RATIO_MIN):
        return False, "pupil_boundary_outside_iris"
    if float(obs.pupil_center_inside_iris_d2) > float(ROI_PUPIL_CENTER_INSIDE_IRIS_D2_MAX):
        return False, "pupil_center_outside_iris"

    dx = float(obs.pupil_center_img.x) - float(obs.iris_center_img.x)
    dy = float(obs.pupil_center_img.y) - float(obs.iris_center_img.y)
    center_dist = float(np.hypot(dx, dy))
    max_iris_axis = float(max(iris.major_axis, iris.minor_axis, 1e-6))
    if center_dist > float(ROI_PUPIL_IRIS_CENTER_DIST_RATIO_MAX) * max_iris_axis:
        return False, "pupil_iris_center_offset_too_large"

    return True, ""


__all__ = [
    "RoiGeometry",
    "compute_roi_geometry",
    "check_observation_plausibility",
    "should_switch_to_detect",
]

