"""Minimal constants for fitting module."""
from typing import Dict, List, Union
import numpy as np


KeyCoordinates = Dict[str, Dict[str, Union[np.ndarray, List[np.ndarray], None]]]

# 单眼关键点坐标类型（给CenterFitter使用）
SingleEyeKeyCoordinates = Dict[str, Union[np.ndarray, List[np.ndarray], None]]


