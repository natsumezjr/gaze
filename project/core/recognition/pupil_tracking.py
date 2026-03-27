from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from project.data.data_models import Ellipse2D, Point2D

TRACK_TEMPLATE_HALF_MIN = 6
TRACK_TEMPLATE_HALF_MAX = 16
TRACK_SEARCH_RADIUS = 16
TRACK_MIN_SCORE = 0.45
TRACK_STRONG_SCORE = 0.62
TRACK_TEMPLATE_UPDATE_SCORE = 0.55
TRACK_TEMPLATE_UPDATE_STRONG_SCORE = 0.70
TRACK_MASK_AGREE_MAX = 5.0
TRACK_NETWORK_AGREE_MAX = 6.0
TRACK_MAX_APPLY_SHIFT = 10.0


def _to_xy(center: Optional[object]) -> Optional[Tuple[float, float]]:
    if center is None:
        return None
    if isinstance(center, Point2D):
        return float(center.x), float(center.y)
    if isinstance(center, (tuple, list)) and len(center) >= 2:
        return float(center[0]), float(center[1])
    return None


def _dist(a: Optional[Tuple[float, float]], b: Optional[Tuple[float, float]]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(np.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def _weighted_point(points: List[Tuple[Tuple[float, float], float]]) -> Optional[Point2D]:
    if not points:
        return None
    weight_sum = float(sum(max(0.0, w) for _, w in points))
    if weight_sum <= 1e-6:
        return None
    x = sum(float(pt[0]) * float(w) for pt, w in points) / weight_sum
    y = sum(float(pt[1]) * float(w) for pt, w in points) / weight_sum
    return Point2D(x=float(x), y=float(y))


def shift_ellipse_center(el: Optional[Ellipse2D], center: Optional[Point2D]) -> Optional[Ellipse2D]:
    if el is None or center is None:
        return el
    return Ellipse2D(
        cx=float(center.x),
        cy=float(center.y),
        major_axis=float(el.major_axis),
        minor_axis=float(el.minor_axis),
        angle_deg=float(el.angle_deg),
        confidence=el.confidence,
    )


class LocalPupilTracker:
    """
    Lightweight local tracker for pupil center.

    Strategy:
    - Keep a tiny grayscale template around the previous accepted pupil center.
    - Search locally in the current ROI with template matching.
    - Fuse tracker / mask centroid / network center conservatively.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._template: Optional[np.ndarray] = None
        self._last_center: Optional[Tuple[float, float]] = None
        self._template_half: int = TRACK_TEMPLATE_HALF_MIN

    def _compute_half_size(
        self,
        *,
        iris_ellipse: Optional[Ellipse2D],
        pupil_mask_area: int,
    ) -> int:
        by_area = int(round(max(0.0, np.sqrt(max(1.0, float(pupil_mask_area))) * 0.6)))
        by_iris = 0
        if iris_ellipse is not None:
            by_iris = int(round(max(float(iris_ellipse.minor_axis) * 0.22, float(iris_ellipse.major_axis) * 0.16)))
        half = max(TRACK_TEMPLATE_HALF_MIN, by_area, by_iris)
        return int(min(TRACK_TEMPLATE_HALF_MAX, half))

    def _extract_patch(self, gray: np.ndarray, center: Tuple[float, float], half: int) -> Optional[np.ndarray]:
        h, w = gray.shape[:2]
        cx = int(round(float(center[0])))
        cy = int(round(float(center[1])))
        x0 = cx - half
        y0 = cy - half
        x1 = cx + half + 1
        y1 = cy + half + 1
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            return None
        patch = gray[y0:y1, x0:x1]
        if patch.size == 0 or patch.shape[0] != 2 * half + 1 or patch.shape[1] != 2 * half + 1:
            return None
        return patch.copy()

    def _track(self, gray: np.ndarray) -> Tuple[Optional[Tuple[float, float]], Optional[float]]:
        if self._template is None or self._last_center is None:
            return None, None
        tmpl = self._template
        th, tw = tmpl.shape[:2]
        if th <= 0 or tw <= 0:
            return None, None
        h, w = gray.shape[:2]
        cx, cy = self._last_center
        half_w = tw // 2
        half_h = th // 2
        sr = TRACK_SEARCH_RADIUS
        x0 = max(0, int(round(cx)) - half_w - sr)
        y0 = max(0, int(round(cy)) - half_h - sr)
        x1 = min(w, int(round(cx)) + half_w + sr + 1)
        y1 = min(h, int(round(cy)) + half_h + sr + 1)
        search = gray[y0:y1, x0:x1]
        if search.shape[0] < th or search.shape[1] < tw:
            return None, None
        score_map = cv2.matchTemplate(search, tmpl, cv2.TM_CCOEFF_NORMED)
        _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(score_map)
        center = (float(x0 + max_loc[0] + half_w), float(y0 + max_loc[1] + half_h))
        return center, float(max_val)

    def _fuse(
        self,
        *,
        track_xy: Optional[Tuple[float, float]],
        track_score: Optional[float],
        mask_xy: Optional[Tuple[float, float]],
        network_xy: Optional[Tuple[float, float]],
        mask_ok: bool,
        pupil_mask_area: int,
        pupil_clean_area_ratio: Optional[float],
    ) -> Tuple[Optional[Point2D], str]:
        mask_good = bool(mask_ok) and int(pupil_mask_area) > 0 and (
            pupil_clean_area_ratio is None or float(pupil_clean_area_ratio) >= 0.5
        )
        if track_xy is not None and track_score is not None and track_score >= TRACK_MIN_SCORE:
            if mask_good and mask_xy is not None:
                d_tm = _dist(track_xy, mask_xy)
                if d_tm is not None and d_tm <= TRACK_MASK_AGREE_MAX:
                    pts: List[Tuple[Tuple[float, float], float]] = [(track_xy, 0.55), (mask_xy, 0.35)]
                    if network_xy is not None and (_dist(network_xy, track_xy) or 999.0) <= TRACK_NETWORK_AGREE_MAX:
                        pts.append((network_xy, 0.10))
                    return _weighted_point(pts), "track_mask"
            if network_xy is not None:
                d_tn = _dist(track_xy, network_xy)
                if d_tn is not None and d_tn <= TRACK_NETWORK_AGREE_MAX:
                    return _weighted_point([(track_xy, 0.7), (network_xy, 0.3)]), "track_network"
        if track_score >= TRACK_STRONG_SCORE:
                return Point2D(x=float(track_xy[0]), y=float(track_xy[1])), "track_only"
        if mask_good and mask_xy is not None and network_xy is not None:
            d_nm = _dist(mask_xy, network_xy)
            if d_nm is not None and d_nm <= TRACK_MASK_AGREE_MAX:
                if self._last_center is not None:
                    d_mask_prev = _dist(mask_xy, self._last_center)
                    d_net_prev = _dist(network_xy, self._last_center)
                    if d_mask_prev is not None and d_net_prev is not None:
                        chosen = mask_xy if d_mask_prev <= d_net_prev else network_xy
                        return Point2D(x=float(chosen[0]), y=float(chosen[1])), "mask_network_prev_select"
                return _weighted_point([(mask_xy, 0.7), (network_xy, 0.3)]), "mask_network"
        if mask_good and mask_xy is not None:
            return Point2D(x=float(mask_xy[0]), y=float(mask_xy[1])), "mask_only"
        if network_xy is not None:
            return Point2D(x=float(network_xy[0]), y=float(network_xy[1])), "network_only"
        return None, "none"

    def update(
        self,
        eye_patch_bgr: np.ndarray,
        *,
        network_center: Optional[object],
        mask_centroid: Optional[object],
        iris_ellipse: Optional[Ellipse2D],
        mask_ok: bool,
        pupil_mask_area: int,
        pupil_clean_area_ratio: Optional[float],
    ) -> Tuple[Optional[Point2D], Dict[str, Any]]:
        gray = cv2.cvtColor(eye_patch_bgr, cv2.COLOR_BGR2GRAY)
        network_xy = _to_xy(network_center)
        mask_xy = _to_xy(mask_centroid)
        track_xy, track_score = self._track(gray)
        fused_center, fused_source = self._fuse(
            track_xy=track_xy,
            track_score=track_score,
            mask_xy=mask_xy,
            network_xy=network_xy,
            mask_ok=mask_ok,
            pupil_mask_area=pupil_mask_area,
            pupil_clean_area_ratio=pupil_clean_area_ratio,
        )

        should_refresh_template = False
        if fused_center is not None:
            if fused_source in {"track_mask", "track_network"} and track_score is not None and track_score >= TRACK_TEMPLATE_UPDATE_SCORE:
                should_refresh_template = True
            elif fused_source == "track_only" and track_score is not None and track_score >= TRACK_TEMPLATE_UPDATE_STRONG_SCORE:
                should_refresh_template = True

        if should_refresh_template:
            center_xy = (float(fused_center.x), float(fused_center.y))
            half = self._compute_half_size(iris_ellipse=iris_ellipse, pupil_mask_area=pupil_mask_area)
            patch = self._extract_patch(gray, center_xy, half)
            if patch is not None:
                self._template = patch
                self._last_center = center_xy
                self._template_half = half
            elif network_xy is not None:
                self._last_center = network_xy
        debug = {
            "tracking_center_roi": [float(track_xy[0]), float(track_xy[1])] if track_xy is not None else None,
            "tracking_score": float(track_score) if track_score is not None else None,
            "fused_source": fused_source,
            "fused_center_roi": [float(fused_center.x), float(fused_center.y)] if fused_center is not None else None,
            "template_refresh": bool(should_refresh_template),
        }
        return fused_center, debug


__all__ = [
    "LocalPupilTracker",
    "TRACK_MAX_APPLY_SHIFT",
    "shift_ellipse_center",
]
