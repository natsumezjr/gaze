from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


MaskBBox = Tuple[int, int, int, int]

PUPIL_MIN_MAIN_AREA = 8
IRIS_MIN_MAIN_AREA = 16
PUPIL_MAX_COMPONENTS_FOR_LOW_RATIO = 2
PUPIL_MAIN_RATIO_MIN = 0.7
PUPIL_CENTER_TO_IRIS_DIST_MAX_RATIO = 0.55
PUPIL_TO_IRIS_AREA_RATIO_MIN = 0.03
PUPIL_TO_IRIS_AREA_RATIO_MAX = 0.45

PUPIL_KERNEL_SIZE = 3
IRIS_KERNEL_SIZE = 3


def _to_bool_mask(mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return np.zeros((0, 0), dtype=bool)
    m = np.asarray(mask)
    if m.ndim != 2:
        if m.ndim == 3 and m.shape[2] >= 1:
            m = m[:, :, 0]
        else:
            return np.zeros((0, 0), dtype=bool)
    return m.astype(bool)


def compute_mask_area(mask: np.ndarray) -> int:
    m = _to_bool_mask(mask)
    if m.size == 0:
        return 0
    return int(np.count_nonzero(m))


def compute_mask_centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    m = _to_bool_mask(mask)
    if m.size == 0 or not np.any(m):
        return None
    mom = cv2.moments(m.astype(np.uint8))
    m00 = float(mom.get("m00", 0.0))
    if m00 <= 0.0:
        return None
    return (float(mom["m10"] / m00), float(mom["m01"] / m00))


def mask_bbox(mask: np.ndarray) -> Optional[MaskBBox]:
    m = _to_bool_mask(mask)
    if m.size == 0 or not np.any(m):
        return None
    rows, cols = np.where(m)
    if rows.size == 0 or cols.size == 0:
        return None
    y0 = int(rows.min())
    y1 = int(rows.max())
    x0 = int(cols.min())
    x1 = int(cols.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def extract_connected_components(mask: np.ndarray) -> List[Dict[str, Any]]:
    m = _to_bool_mask(mask)
    if m.size == 0 or not np.any(m):
        return []
    m_u8 = m.astype(np.uint8)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(m_u8, connectivity=8)
    comps: List[Dict[str, Any]] = []
    for label in range(1, int(n_labels)):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        cx = float(centroids[label, 0])
        cy = float(centroids[label, 1])
        comp_mask = labels == label
        comps.append(
            {
                "label": int(label),
                "area": area,
                "bbox": (x, y, w, h),
                "centroid": (cx, cy),
                "mask": comp_mask,
            }
        )
    comps.sort(key=lambda c: int(c["area"]), reverse=True)
    return comps


def largest_component_stats(mask: np.ndarray) -> Dict[str, Any]:
    total = compute_mask_area(mask)
    comps = extract_connected_components(mask)
    if not comps:
        return {
            "component_count": 0,
            "main_component_area": 0,
            "main_component_area_ratio": 0.0,
            "main_component_bbox": None,
            "main_component_centroid": None,
            "main_component_mask": np.zeros_like(_to_bool_mask(mask), dtype=bool),
        }
    main = comps[0]
    main_area = int(main["area"])
    ratio = float(main_area / max(total, 1))
    return {
        "component_count": int(len(comps)),
        "main_component_area": main_area,
        "main_component_area_ratio": ratio,
        "main_component_bbox": main["bbox"],
        "main_component_centroid": main["centroid"],
        "main_component_mask": main["mask"].astype(bool),
    }


def compute_edge_margins(mask_or_bbox: Any, w: int, h: int) -> Dict[str, Optional[int]]:
    bbox: Optional[MaskBBox]
    if isinstance(mask_or_bbox, np.ndarray):
        bbox = mask_bbox(mask_or_bbox)
    elif isinstance(mask_or_bbox, (tuple, list)) and len(mask_or_bbox) == 4:
        bbox = (
            int(mask_or_bbox[0]),
            int(mask_or_bbox[1]),
            int(mask_or_bbox[2]),
            int(mask_or_bbox[3]),
        )
    else:
        bbox = None
    if bbox is None:
        return {"left": None, "right": None, "top": None, "bottom": None, "min": None}
    x, y, bw, bh = bbox
    left = int(x)
    right = int(int(w) - (x + bw))
    top = int(y)
    bottom = int(int(h) - (y + bh))
    return {
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
        "min": int(min(left, right, top, bottom)),
    }


def split_pupil_iris_masks(segmentation_mask: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    if segmentation_mask is None:
        z = np.zeros((0, 0), dtype=bool)
        return {"pupil_mask": z, "iris_mask": z}
    seg = np.asarray(segmentation_mask)
    if seg.ndim != 2:
        if seg.ndim == 3 and seg.shape[2] >= 1:
            seg = seg[:, :, 0]
        else:
            z = np.zeros((0, 0), dtype=bool)
            return {"pupil_mask": z, "iris_mask": z}
    seg_u8 = np.rint(seg).astype(np.uint8)
    pupil = seg_u8 == 2
    iris = seg_u8 == 1
    # Legacy fallback: foreground-only 0/1 mask treated as iris-like region.
    if not np.any(pupil) and not np.any(iris):
        iris = seg_u8 > 0
    return {"pupil_mask": pupil.astype(bool), "iris_mask": iris.astype(bool)}


def _ellipse_kernel(size: int) -> np.ndarray:
    s = int(max(1, size))
    if s % 2 == 0:
        s += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s, s))


def clean_pupil_mask(mask: np.ndarray) -> np.ndarray:
    m = _to_bool_mask(mask).astype(np.uint8)
    if m.size == 0:
        return m.astype(bool)
    k = _ellipse_kernel(PUPIL_KERNEL_SIZE)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m.astype(bool)


def clean_iris_mask(mask: np.ndarray) -> np.ndarray:
    m = _to_bool_mask(mask).astype(np.uint8)
    if m.size == 0:
        return m.astype(bool)
    k = _ellipse_kernel(IRIS_KERNEL_SIZE)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m.astype(bool)


def select_main_iris_component(iris_mask: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    stats = largest_component_stats(iris_mask)
    selected = stats["main_component_mask"].astype(bool)
    ok = bool(stats["component_count"] > 0 and stats["main_component_area"] >= IRIS_MIN_MAIN_AREA)
    out = {
        "component_count": int(stats["component_count"]),
        "main_component_area": int(stats["main_component_area"]),
        "main_component_area_ratio": float(stats["main_component_area_ratio"]),
        "main_component_centroid": stats["main_component_centroid"],
        "main_component_bbox": stats["main_component_bbox"],
        "ok": ok,
        "reject_reason": "" if ok else "iris_main_component_too_small_or_missing",
    }
    return selected, out


def select_main_pupil_component(
    pupil_mask: np.ndarray,
    iris_center: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    m = _to_bool_mask(pupil_mask)
    comps = extract_connected_components(m)
    total_area = int(np.count_nonzero(m))
    if not comps:
        z = np.zeros_like(m, dtype=bool)
        return z, {
            "component_count": 0,
            "main_component_area": 0,
            "main_component_area_ratio": 0.0,
            "main_component_centroid": None,
            "main_component_bbox": None,
            "ok": False,
            "reject_reason": "pupil_no_component",
        }

    if iris_center is not None and len(comps) >= 3:
        icx, icy = float(iris_center[0]), float(iris_center[1])
        comps = sorted(
            comps,
            key=lambda c: (
                -int(c["area"]),
                float(np.hypot(float(c["centroid"][0]) - icx, float(c["centroid"][1]) - icy)),
            ),
        )
    main = comps[0]
    selected = main["mask"].astype(bool)
    main_area = int(main["area"])
    ratio = float(main_area / max(total_area, 1))
    ok = True
    reason = ""
    if main_area < PUPIL_MIN_MAIN_AREA:
        ok = False
        reason = "pupil_main_component_too_small"
    if ok and len(comps) >= 3 and iris_center is not None:
        cx, cy = float(main["centroid"][0]), float(main["centroid"][1])
        dist = float(np.hypot(cx - float(iris_center[0]), cy - float(iris_center[1])))
        h, w = m.shape[:2]
        scale = float(max(1.0, min(w, h)))
        if dist > PUPIL_CENTER_TO_IRIS_DIST_MAX_RATIO * scale:
            ok = False
            reason = "pupil_main_component_far_from_iris_center"
    return selected, {
        "component_count": int(len(comps)),
        "main_component_area": int(main_area),
        "main_component_area_ratio": float(ratio),
        "main_component_centroid": main["centroid"],
        "main_component_bbox": main["bbox"],
        "ok": bool(ok),
        "reject_reason": reason,
    }


def check_mask_quality(metrics: Dict[str, Any]) -> Tuple[bool, str]:
    warnings: List[str] = []
    pupil_area = int(metrics.get("pupil_mask_area", 0) or 0)
    iris_area = int(metrics.get("iris_mask_area", 0) or 0)
    if pupil_area <= 0:
        return False, "pupil_mask_area_zero"
    if iris_area <= 0:
        return False, "iris_mask_area_zero"
    comp_count = int(metrics.get("pupil_component_count", 0) or 0)
    main_ratio = float(metrics.get("pupil_main_component_area_ratio", 0.0) or 0.0)
    if comp_count > PUPIL_MAX_COMPONENTS_FOR_LOW_RATIO and main_ratio < PUPIL_MAIN_RATIO_MIN:
        warnings.append("pupil_fragmented_main_ratio_low")
    area_ratio = float(pupil_area) / float(max(iris_area, 1))
    if area_ratio < PUPIL_TO_IRIS_AREA_RATIO_MIN:
        warnings.append("pupil_to_iris_area_ratio_too_small")
    if area_ratio > PUPIL_TO_IRIS_AREA_RATIO_MAX:
        warnings.append("pupil_to_iris_area_ratio_too_large")

    pupil_inside = metrics.get("pupil_centroid_inside_iris")
    if pupil_inside is False:
        warnings.append("pupil_centroid_outside_iris_mask")

    iris_margin_min = metrics.get("iris_edge_margin_min")
    pupil_margin_min = metrics.get("pupil_edge_margin_min")
    if iris_margin_min is not None and int(iris_margin_min) < 0:
        warnings.append("iris_edge_margin_negative")
    if pupil_margin_min is not None and int(pupil_margin_min) < 0:
        warnings.append("pupil_edge_margin_negative")
    clean_ratio = metrics.get("pupil_clean_area_ratio")
    if clean_ratio is not None and float(clean_ratio) < 0.6:
        warnings.append("pupil_clean_area_ratio_lt_0.6")
    metrics["mask_warning_reasons"] = warnings
    metrics["mask_warning_count"] = int(len(warnings))
    return True, ""


__all__ = [
    "extract_connected_components",
    "largest_component_stats",
    "compute_mask_area",
    "compute_mask_centroid",
    "compute_edge_margins",
    "split_pupil_iris_masks",
    "select_main_pupil_component",
    "select_main_iris_component",
    "clean_pupil_mask",
    "clean_iris_mask",
    "check_mask_quality",
    "mask_bbox",
]
