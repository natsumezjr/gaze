"""Estimate Kappa parameters from calibration samples (placeholders).

Each sample should contain:
- c_eye: np.ndarray (3,)
- c_pupil: np.ndarray (3,)
- target_point: np.ndarray (3,)  # known fixation point on the screen plane in camera coords
"""

from typing import Dict, Iterable, Tuple


def estimate_kappa(samples: Iterable[Dict]) -> Tuple[object, float, Dict]:
    """Estimate kappa_axis (unit vector) and kappa_angle (radians).

    Returns (kappa_axis, kappa_angle, summary)
    """
    pass


