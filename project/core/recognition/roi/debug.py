from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple

from .geometry import RoiGeometry


@dataclass
class RoiDebugRecord:
    """
    Minimal ROI debug record.

    This record is intentionally aligned with roi_debug_analyzer consumption.
    """

    frame_id: int
    eye: str

    current_mode: str
    face_bbox: Optional[Tuple[int, int, int, int]]

    final_roi: Optional[Tuple[int, int, int, int]]
    union_mask_bbox: Optional[Tuple[int, int, int, int]]

    geometry_confidence: float
    face_bbox_is_stable: bool
    face_bbox_corner_max_shift: float

    track_from_previous_bbox_applied: bool

    def to_log_dict(self) -> dict:
        return asdict(self)


def build_roi_debug_record(
    *,
    frame_id: int,
    eye: str,
    current_mode: str,
    face_bbox: Optional[Tuple[int, int, int, int]],
    final_roi: Optional[Tuple[int, int, int, int]],
    union_mask_bbox: Optional[Tuple[int, int, int, int]],
    track_from_previous_bbox_applied: bool,
    geometry: RoiGeometry,
) -> RoiDebugRecord:
    return RoiDebugRecord(
        frame_id=frame_id,
        eye=eye,
        current_mode=current_mode,
        face_bbox=face_bbox,
        final_roi=final_roi,
        union_mask_bbox=union_mask_bbox,
        geometry_confidence=float(geometry.geometry_confidence),
        face_bbox_is_stable=bool(geometry.face_bbox_is_stable),
        face_bbox_corner_max_shift=float(geometry.face_bbox_corner_max_shift),
        track_from_previous_bbox_applied=bool(track_from_previous_bbox_applied),
    )


__all__ = ["RoiDebugRecord", "build_roi_debug_record"]

