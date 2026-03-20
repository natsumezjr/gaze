"""
ROI recognition state machine + geometry + observation + debug record.

This package provides a minimal closed-loop implementation:
- ROI State Machine decides DETECT/TRACK/RECOVER and outputs RoiBox for current frame.
- ROI Observation builds geometry-relevant data directly from EllSegResult + current RoiBox.
- ROI Geometry computes geometry_confidence and DETECT switching decision.
- ROI Debug Record stores only fields consumed by roi_debug_analyzer / runtime overlay.
"""

from .constants import (  # noqa: F401
    MODE_DETECT,
    MODE_TRACK,
    MODE_RECOVER,
    ROI_GEOMETRY_CONFIDENCE_THRESHOLD,
)
from .types import RoiBox, RoiMachineState  # noqa: F401
from .state_machine import RoiStateMachine  # noqa: F401
from .observation import RoiObservation, build_roi_observation  # noqa: F401
from .geometry import RoiGeometry, compute_roi_geometry, should_switch_to_detect  # noqa: F401
from .debug import RoiDebugRecord, build_roi_debug_record  # noqa: F401

