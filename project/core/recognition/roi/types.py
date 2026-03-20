from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .constants import MODE_DETECT


@dataclass
class RoiBox:
    """
    ROI box definition in image coordinates (pixels).
    """

    x: int
    y: int
    w: int
    h: int
    side: str
    valid: bool = True

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return int(self.x), int(self.y), int(self.w), int(self.h)


@dataclass
class RoiMachineState:
    """
    Minimal state for ROI state machine.
    """

    mode: str = MODE_DETECT
    last_roi: Optional[RoiBox] = None
    last_face_bbox: Optional[Tuple[int, int, int, int]] = None
    last_face_info: Optional[Dict[str, Any]] = None

