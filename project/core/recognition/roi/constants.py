"""
ROI state machine constants.

These values are treated as validated strategy parameters:
DETECT / TRACK / RECOVER thresholds, ROI size limits, and initialization ratios.
"""

# ---------------------------------------------------------------------------
# State modes
# ---------------------------------------------------------------------------
MODE_DETECT = "detect"
MODE_TRACK = "track"
MODE_RECOVER = "recover"

# ---------------------------------------------------------------------------
# size limit (face-ratio based)
# ---------------------------------------------------------------------------
ROI_MIN_W_FACE_RATIO = 0.205
ROI_MIN_H_FACE_RATIO = 0.061
ROI_MAX_W_FACE_RATIO = 0.303
ROI_MAX_H_FACE_RATIO = 0.156

# ---------------------------------------------------------------------------
# initialization (YuNet band + fixed ratio + eye center)
# ---------------------------------------------------------------------------
# band top-left x-ratio inside face bbox
INIT_BAND_LEFT_TL_X_RATIO = [0.600, 0.643, 0.639, 0.596]
INIT_BAND_RIGHT_TL_X_RATIO = [0.197, 0.192, 0.175, 0.170]
INIT_BAND_TL_Y_RATIO = [0.380, 0.400, 0.482, 0.604]
INIT_BAND_W_RATIO = [0.230, 0.230, 0.230, 0.260]
INIT_BAND_H_RATIO = [0.120, 0.120, 0.120, 0.150]

# eye-to-band ratios (eye center vs band geometry)
INIT_RIGHT_EYE_DX_RATIO = [0.032, 0.047, 0.155, 0.040]
INIT_RIGHT_EYE_DY_RATIO = [-0.389, 0.07, 0.19, -0.764]
INIT_LEFT_EYE_DX_RATIO = [0.060, 0.176, 0.077, 0.223]
INIT_LEFT_EYE_DY_RATIO = [0.037, 0.07, 0.19, -0.601]

# initial ROI extra expand (before clip)
INIT_ROI_EXPAND_RATIO = 0.20
FIXED_RATIO = [0.10, 0.60, 0.20, 0.50]

# ---------------------------------------------------------------------------
# track / recover update
# ---------------------------------------------------------------------------
UNION_MASK_BBOX_W_EXPAND_RATIO = 1.8
UNION_MASK_BBOX_H_EXPAND_RATIO = 1.6

# recover trigger: union mask center offset vs previous ROI center
RECOVER_MASK_OFFSET_DX_RATIO_TH = 0.3
RECOVER_MASK_OFFSET_DY_RATIO_TH = 0.25

# ---------------------------------------------------------------------------
# thresholds
# ---------------------------------------------------------------------------
# Geometry confidence threshold used by DETECT/TRACK gating.
ROI_GEOMETRY_CONFIDENCE_THRESHOLD = 0.80

# Hard plausibility gates for EllSeg native geometry.
# These are used to reject geometrically impossible observations before
# they enter triangulation / eyeball fitting.
ROI_PUPIL_INSIDE_IRIS_RATIO_MIN = 0.60
ROI_PUPIL_CENTER_INSIDE_IRIS_D2_MAX = 1.15
ROI_PUPIL_IRIS_CENTER_DIST_RATIO_MAX = 0.35
ROI_PUPIL_TO_IRIS_MAJOR_RATIO_MIN = 0.08
ROI_PUPIL_TO_IRIS_MAJOR_RATIO_MAX = 0.80
ROI_PUPIL_TO_IRIS_MINOR_RATIO_MIN = 0.08
ROI_PUPIL_TO_IRIS_MINOR_RATIO_MAX = 0.85

# ---------------------------------------------------------------------------
# additional stability parameters (kept from validated implementation)
# ---------------------------------------------------------------------------
# face bbox stable threshold (corner max shift ratio) is implemented in geometry.py
FACE_BBOX_STABLE_MAX_SHIFT_RATIO = 0.05

# ROI aspect ratio constraints (safety checks)
MIN_ASPECT_RATIO = 0.6
MAX_ASPECT_RATIO = 4.0

