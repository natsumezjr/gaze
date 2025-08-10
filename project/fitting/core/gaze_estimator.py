"""Compute theoretical gaze and apply Kappa compensation (placeholders)."""

from typing import Dict
import numpy as np
from project.fitting.utils.geometry import vector_of_2_points,normalize_vector,rotate_vector


def compute_theoretical_gaze(c_eye:np.ndarray, c_pupil:np.ndarray)->np.ndarray:
    """Return unit vector from eye center to pupil center.

    Args:
        c_eye: np.ndarray (3,)
        c_pupil: np.ndarray (3,)
    """
    gaze_vec = vector_of_2_points(c_eye, c_pupil)
    gaze_vec = normalize_vector(gaze_vec)
    return gaze_vec



def apply_kappa_compensation(gaze_vec:np.ndarray, kappa_axis:np.ndarray, kappa_angle:float)->np.ndarray:
    """Rotate gaze_vec around kappa_axis by kappa_angle (axis-angle)."""
    return rotate_vector(gaze_vec, kappa_axis, kappa_angle)


def estimate_final_gaze(c_eye, c_pupil, kappa_params: Dict):
    """Convenience wrapper: theory gaze then apply Kappa.

    kappa_params: {"axis": np.ndarray(3,), "angle": float} or compatible
    """
    raise NotImplementedError("Implement final gaze estimation")


