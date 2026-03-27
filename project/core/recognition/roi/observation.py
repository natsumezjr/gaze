from __future__ import annotations

import cv2
import numpy as np

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from project.data.data_models import Ellipse2D, Point2D
from project.core.recognition.eye_model_ellseg import EllSegResult
from .utils import (
    compute_pupil_center_inside_iris_score,
    points_inside_ellipse_ratio,
    sample_ellipse_points,
)
from .mask_metrics import (
    check_mask_quality,
    clean_iris_mask,
    clean_pupil_mask,
    compute_edge_margins,
    extract_connected_components,
    largest_component_stats,
    mask_bbox,
    split_pupil_iris_masks,
    select_main_iris_component,
    select_main_pupil_component,
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

    raw_pupil_component_count: int = 0
    raw_iris_component_count: int = 0
    clean_pupil_component_count: int = 0
    clean_iris_component_count: int = 0

    raw_pupil_mask_area: int = 0
    raw_iris_mask_area: int = 0
    clean_pupil_mask_area: int = 0
    clean_iris_mask_area: int = 0

    pupil_main_component_area: int = 0
    iris_main_component_area: int = 0
    pupil_main_component_area_ratio: float = 0.0
    iris_main_component_area_ratio: float = 0.0

    pupil_edge_margin_left: Optional[int] = None
    pupil_edge_margin_right: Optional[int] = None
    pupil_edge_margin_top: Optional[int] = None
    pupil_edge_margin_bottom: Optional[int] = None
    pupil_edge_margin_min: Optional[int] = None
    iris_edge_margin_left: Optional[int] = None
    iris_edge_margin_right: Optional[int] = None
    iris_edge_margin_top: Optional[int] = None
    iris_edge_margin_bottom: Optional[int] = None
    iris_edge_margin_min: Optional[int] = None

    pupil_mask_to_iris_center_dist: Optional[float] = None
    iris_mask_to_pupil_center_dist: Optional[float] = None
    pupil_centroid_inside_iris: Optional[bool] = None

    mask_ok: bool = False
    mask_reject_reason: str = ""
    mask_warning_reasons: Optional[List[str]] = None
    mask_warning_count: int = 0
    pupil_clean_area_ratio: Optional[float] = None
    cleaned_pupil_mask: Optional[np.ndarray] = None
    cleaned_iris_mask: Optional[np.ndarray] = None
    discarded_pupil_mask: Optional[np.ndarray] = None
    discarded_iris_mask: Optional[np.ndarray] = None


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


def _centroid_from_bool(m: np.ndarray) -> Optional[Tuple[float, float]]:
    if m is None or not np.any(m):
        return None
    mom = cv2.moments(m.astype(np.uint8))
    if mom.get("m00", 0.0) <= 0:
        return None
    return (float(mom["m10"] / mom["m00"]), float(mom["m01"] / mom["m00"]))


def analyze_segmentation_masks(
    seg_map: Optional[np.ndarray],
    iris_center: Optional[Tuple[float, float]] = None,
    pupil_center: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    if seg_map is None or np.asarray(seg_map).size == 0:
        return {}
    masks = split_pupil_iris_masks(seg_map)
    raw_pupil = masks["pupil_mask"]
    raw_iris = masks["iris_mask"]
    h, w = raw_pupil.shape[:2] if raw_pupil.size else (0, 0)
    if h <= 0 or w <= 0:
        return {}

    raw_pupil_stats = largest_component_stats(raw_pupil)
    raw_iris_stats = largest_component_stats(raw_iris)

    cleaned_pupil = clean_pupil_mask(raw_pupil)
    cleaned_iris = clean_iris_mask(raw_iris)
    clean_iris_selected, clean_iris_stats = select_main_iris_component(cleaned_iris)
    clean_pupil_selected, clean_pupil_stats = select_main_pupil_component(cleaned_pupil, iris_center=iris_center)

    clean_union = np.logical_or(clean_pupil_selected, clean_iris_selected)
    pupil_bbox = mask_bbox(clean_pupil_selected)
    iris_bbox = mask_bbox(clean_iris_selected)

    pupil_centroid = _centroid_from_bool(clean_pupil_selected)
    iris_centroid = _centroid_from_bool(clean_iris_selected)
    union_centroid = _centroid_from_bool(clean_union)

    pupil_margins = compute_edge_margins(pupil_bbox, w=w, h=h)
    iris_margins = compute_edge_margins(iris_bbox, w=w, h=h)

    pupil_mask_to_iris_center_dist = None
    iris_mask_to_pupil_center_dist = None
    if pupil_centroid is not None and iris_center is not None:
        pupil_mask_to_iris_center_dist = float(
            np.hypot(float(pupil_centroid[0]) - float(iris_center[0]), float(pupil_centroid[1]) - float(iris_center[1]))
        )
    if iris_centroid is not None and pupil_center is not None:
        iris_mask_to_pupil_center_dist = float(
            np.hypot(float(iris_centroid[0]) - float(pupil_center[0]), float(iris_centroid[1]) - float(pupil_center[1]))
        )

    pupil_centroid_inside_iris = None
    if pupil_centroid is not None and clean_iris_selected.size > 0:
        px = int(round(float(pupil_centroid[0])))
        py = int(round(float(pupil_centroid[1])))
        if 0 <= py < clean_iris_selected.shape[0] and 0 <= px < clean_iris_selected.shape[1]:
            pupil_centroid_inside_iris = bool(clean_iris_selected[py, px])
        else:
            pupil_centroid_inside_iris = False

    out = {
        "raw_pupil_component_count": int(raw_pupil_stats["component_count"]),
        "raw_iris_component_count": int(raw_iris_stats["component_count"]),
        "clean_pupil_component_count": int(clean_pupil_stats["component_count"]),
        "clean_iris_component_count": int(clean_iris_stats["component_count"]),
        "raw_pupil_mask_area": int(np.count_nonzero(raw_pupil)),
        "raw_iris_mask_area": int(np.count_nonzero(raw_iris)),
        "clean_pupil_mask_area": int(np.count_nonzero(clean_pupil_selected)),
        "clean_iris_mask_area": int(np.count_nonzero(clean_iris_selected)),
        "pupil_main_component_area": int(clean_pupil_stats["main_component_area"]),
        "iris_main_component_area": int(clean_iris_stats["main_component_area"]),
        "pupil_main_component_area_ratio": float(clean_pupil_stats["main_component_area_ratio"]),
        "iris_main_component_area_ratio": float(clean_iris_stats["main_component_area_ratio"]),
        "pupil_bbox_xywh": pupil_bbox,
        "iris_bbox_xywh": iris_bbox,
        "pupil_mask_centroid": pupil_centroid,
        "iris_mask_centroid": iris_centroid,
        "union_mask_centroid": union_centroid,
        "pupil_edge_margin_left": pupil_margins["left"],
        "pupil_edge_margin_right": pupil_margins["right"],
        "pupil_edge_margin_top": pupil_margins["top"],
        "pupil_edge_margin_bottom": pupil_margins["bottom"],
        "pupil_edge_margin_min": pupil_margins["min"],
        "iris_edge_margin_left": iris_margins["left"],
        "iris_edge_margin_right": iris_margins["right"],
        "iris_edge_margin_top": iris_margins["top"],
        "iris_edge_margin_bottom": iris_margins["bottom"],
        "iris_edge_margin_min": iris_margins["min"],
        "pupil_mask_to_iris_center_dist": pupil_mask_to_iris_center_dist,
        "iris_mask_to_pupil_center_dist": iris_mask_to_pupil_center_dist,
        "pupil_centroid_inside_iris": pupil_centroid_inside_iris,
        "cleaned_pupil_mask": clean_pupil_selected,
        "cleaned_iris_mask": clean_iris_selected,
        "discarded_pupil_mask": np.logical_and(cleaned_pupil, np.logical_not(clean_pupil_selected)),
        "discarded_iris_mask": np.logical_and(cleaned_iris, np.logical_not(clean_iris_selected)),
        "raw_pupil_components": extract_connected_components(raw_pupil),
        "raw_iris_components": extract_connected_components(raw_iris),
    }
    out["pupil_clean_area_ratio"] = float(out["clean_pupil_mask_area"]) / float(max(out["raw_pupil_mask_area"], 1))
    metrics_for_quality = {
        "pupil_mask_area": out["clean_pupil_mask_area"],
        "iris_mask_area": out["clean_iris_mask_area"],
        "pupil_component_count": out["clean_pupil_component_count"],
        "pupil_main_component_area_ratio": out["pupil_main_component_area_ratio"],
        "pupil_centroid_inside_iris": out["pupil_centroid_inside_iris"],
        "iris_edge_margin_min": out["iris_edge_margin_min"],
        "pupil_edge_margin_min": out["pupil_edge_margin_min"],
        "pupil_clean_area_ratio": out["pupil_clean_area_ratio"],
    }
    mask_ok, mask_reason = check_mask_quality(metrics_for_quality)
    out["mask_ok"] = bool(mask_ok)
    out["mask_reject_reason"] = str(mask_reason)
    out["mask_warning_reasons"] = list(metrics_for_quality.get("mask_warning_reasons", []))
    out["mask_warning_count"] = int(metrics_for_quality.get("mask_warning_count", 0))
    return out


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

    mask_metrics: Dict[str, Any] = {}
    if seg_map is not None and seg_map.size > 0:
        h, w = int(seg_map.shape[0]), int(seg_map.shape[1])
        iris_center_roi = (float(iris_ellipse_roi.cx), float(iris_ellipse_roi.cy)) if iris_ellipse_roi is not None else None
        pupil_center_xy = (float(pupil_center_roi.x), float(pupil_center_roi.y)) if pupil_center_roi is not None else None
        mask_metrics = analyze_segmentation_masks(seg_map, iris_center=iris_center_roi, pupil_center=pupil_center_xy)
        clean_pupil_mask = mask_metrics.get("cleaned_pupil_mask")
        clean_iris_mask = mask_metrics.get("cleaned_iris_mask")
        union_mask = (
            np.logical_or(clean_pupil_mask, clean_iris_mask)
            if clean_pupil_mask is not None and clean_iris_mask is not None
            else None
        )

        pupil_mask_bbox = mask_metrics.get("pupil_bbox_xywh")
        iris_mask_bbox = mask_metrics.get("iris_bbox_xywh")
        union_mask_bbox = mask_bbox(union_mask) if union_mask is not None else None

        pupil_mask_centroid = mask_metrics.get("pupil_mask_centroid")
        iris_mask_centroid = mask_metrics.get("iris_mask_centroid")
        union_mask_centroid = mask_metrics.get("union_mask_centroid")

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

        pupil_overlap_score = _overlap(pupil_ellipse_roi, clean_pupil_mask if clean_pupil_mask is not None else np.zeros((h, w), dtype=bool))
        iris_overlap_score = _overlap(iris_ellipse_roi, clean_iris_mask if clean_iris_mask is not None else np.zeros((h, w), dtype=bool))

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
        raw_pupil_component_count=int(mask_metrics.get("raw_pupil_component_count", 0)),
        raw_iris_component_count=int(mask_metrics.get("raw_iris_component_count", 0)),
        clean_pupil_component_count=int(mask_metrics.get("clean_pupil_component_count", 0)),
        clean_iris_component_count=int(mask_metrics.get("clean_iris_component_count", 0)),
        raw_pupil_mask_area=int(mask_metrics.get("raw_pupil_mask_area", 0)),
        raw_iris_mask_area=int(mask_metrics.get("raw_iris_mask_area", 0)),
        clean_pupil_mask_area=int(mask_metrics.get("clean_pupil_mask_area", 0)),
        clean_iris_mask_area=int(mask_metrics.get("clean_iris_mask_area", 0)),
        pupil_main_component_area=int(mask_metrics.get("pupil_main_component_area", 0)),
        iris_main_component_area=int(mask_metrics.get("iris_main_component_area", 0)),
        pupil_main_component_area_ratio=float(mask_metrics.get("pupil_main_component_area_ratio", 0.0)),
        iris_main_component_area_ratio=float(mask_metrics.get("iris_main_component_area_ratio", 0.0)),
        pupil_edge_margin_left=mask_metrics.get("pupil_edge_margin_left"),
        pupil_edge_margin_right=mask_metrics.get("pupil_edge_margin_right"),
        pupil_edge_margin_top=mask_metrics.get("pupil_edge_margin_top"),
        pupil_edge_margin_bottom=mask_metrics.get("pupil_edge_margin_bottom"),
        pupil_edge_margin_min=mask_metrics.get("pupil_edge_margin_min"),
        iris_edge_margin_left=mask_metrics.get("iris_edge_margin_left"),
        iris_edge_margin_right=mask_metrics.get("iris_edge_margin_right"),
        iris_edge_margin_top=mask_metrics.get("iris_edge_margin_top"),
        iris_edge_margin_bottom=mask_metrics.get("iris_edge_margin_bottom"),
        iris_edge_margin_min=mask_metrics.get("iris_edge_margin_min"),
        pupil_mask_to_iris_center_dist=mask_metrics.get("pupil_mask_to_iris_center_dist"),
        iris_mask_to_pupil_center_dist=mask_metrics.get("iris_mask_to_pupil_center_dist"),
        pupil_centroid_inside_iris=mask_metrics.get("pupil_centroid_inside_iris"),
        mask_ok=bool(mask_metrics.get("mask_ok", False)),
        mask_reject_reason=str(mask_metrics.get("mask_reject_reason", "")),
        mask_warning_reasons=list(mask_metrics.get("mask_warning_reasons", [])),
        mask_warning_count=int(mask_metrics.get("mask_warning_count", 0)),
        pupil_clean_area_ratio=(
            float(mask_metrics.get("pupil_clean_area_ratio"))
            if mask_metrics.get("pupil_clean_area_ratio") is not None
            else None
        ),
        cleaned_pupil_mask=mask_metrics.get("cleaned_pupil_mask"),
        cleaned_iris_mask=mask_metrics.get("cleaned_iris_mask"),
        discarded_pupil_mask=mask_metrics.get("discarded_pupil_mask"),
        discarded_iris_mask=mask_metrics.get("discarded_iris_mask"),
    )


__all__ = ["RoiObservation", "build_roi_observation", "analyze_segmentation_masks"]

