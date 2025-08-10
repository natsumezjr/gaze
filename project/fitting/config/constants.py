"""Minimal constants for fitting module."""
from typing import Dict, List, Union
import numpy as np

# 状态码（最小集）
STATUS_SUCCESS = 0
STATUS_CALIBRATION_REQUIRED = 10

KeyCoordinates = Dict[str, Dict[str, Union[np.ndarray, List[np.ndarray], None]]]


