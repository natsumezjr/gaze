from __future__ import annotations

import numpy as np

from dataclasses import dataclass
from typing import Optional, Tuple

from project.core.recognition.roi.observation import RoiObservation
from .constants import ROI_GEOMETRY_CONFIDENCE_THRESHOLD


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


__all__ = [
    "RoiGeometry",
    "compute_roi_geometry",
    "should_switch_to_detect",
]

