"""Configuration for fitting module (screen geometry, constants)."""

from .models import *
from .fitting_strategy import *
from .fitting_state import *
from .initial_weight import *
from .settings import *

__all__ = [
    # 从models导入
    "EllipsoidParams", "Plane", "KeyCoordinates", "SingleEyeKeyCoordinates", "EYEBALL_RADIUS",
    
    # 从fitting_strategy导入
    "ANATOMICAL_CONSTRAINTS", "MEDIAPIPE_QUALITY_THRESHOLDS", 
    "FittingStrategy", "FittingStrategyConfig", "FITTING_STRATEGIES", "DEFAULT_FITTING_STRATEGY",
    
    # 从fitting_state导入
    "FittingResult", "FittingStatus", "ConvergenceStep", "QualityMetrics", 
    "WeightResult", "SamplingResult", "ConstraintValidationResult",
    
    # 从initial_weight导入
    "ANATOMICAL_WEIGHT_PARAMS", "GEOMETRIC_WEIGHT_PARAMS", "GLOBAL_WEIGHT_CONFIG",
    "AnatomicalWeightParams", "GeometricWeightParams", "GlobalWeightConfig",
    
    # 从settings导入
    "ScreenConfig", "DEFAULT_SCREEN_WIDTH_M", "DEFAULT_SCREEN_HEIGHT_M"
]


