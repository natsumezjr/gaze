from __future__ import annotations

"""
End-to-end backend test for the calibration pipeline:
1) Run initial fitting to populate SESSION_MANAGER.last_fitting
2) Push 9-point calibration samples (using camera resolution or 1920x1080 fallback)
3) Fit kappa model
4) Compute compensated gaze and print metrics

Note: This script expects that recognition pipeline has already filled
      RECG_FIT_DATA_MANAGER with camera-derived 3D points (esp. pupil).
      If not present, it will abort with a clear message.
"""

import time
import numpy as np

from project.fitting.main import main as fit_main
from project.fitting.app.api import start_session, push_sample, fit_kappa, compute_gaze
from project.fitting.app.state import SESSION_MANAGER
from project.fitting.core.kappa_calibrator_pro import (
    build_samples_from_arrays,
    estimate_kappa as estimate_kappa_pro,
    evaluate_fit,
)
from project.recognition.utils.camera_data_manager import get_resolution
from project.recg_fit_data.data_manager import RECG_FIT_DATA_MANAGER


def ensure_pupil_available(eye: str = "left") -> None:
    """Ensure we have pupil data from recognition (camera source)."""
    pupils = RECG_FIT_DATA_MANAGER.get_coordinate_point(eye, "pupil")
    if not pupils:
        raise RuntimeError(
            "Pupil data not found from RECG_FIT_DATA_MANAGER.\n"
            "Please run the recognition pipeline to feed camera-derived 3D points first."
        )


def nine_point_grid(resolution: tuple[int, int] | None) -> list[tuple[int, int]]:
    if resolution is None:
        w, h = 1920, 1080
    else:
        w, h = resolution
    xs = [int(w * r) for r in (0.1, 0.5, 0.9)]
    ys = [int(h * r) for r in (0.1, 0.5, 0.9)]
    points = [(x, y) for y in ys for x in xs]
    return points


def main() -> None:
    print("== Step 1: Initial fitting ==")
    # Run fitting to populate SESSION_MANAGER.last_fitting
    fit_main()

    # Verify last fitting exists
    if SESSION_MANAGER.get_session() is None or not SESSION_MANAGER.get_session().last_fitting:
        # If fitting didn't create a session automatically, create one now and require last_fitting separately
        start_session("autostart")
    if not SESSION_MANAGER.get_session().last_fitting:
        print("[ERROR] No last_fitting found. Ensure fitting wrote results to SESSION_MANAGER.")
        return

    # Ensure pupil exists from recognition
    try:
        ensure_pupil_available("left")
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    print("== Step 2: Push 9-point samples ==")
    start_session("pipeline-test-session")
    res = get_resolution()
    points = nine_point_grid(res)
    for uv in points:
        push_sample({"timestamp": time.time(), "target_pixel": uv, "eye": "left"})
    print(f"Pushed {len(points)} samples.")

    print("== Step 3: Fit kappa ==")
    out = fit_kappa()
    print("Kappa model:", out["kappa_model"])  # axis, angle_deg, samples_count, quality

    print("== Step 4: Compute compensated gaze ==")
    center_uv = (points[4] if len(points) >= 5 else (960, 540))
    gaze = compute_gaze("left", center_uv)
    print("Gaze (compensated):", gaze["gaze"])  # origin, direction, target_pixel

    # Optional: report evaluation metrics on current samples
    print("== Step 5: Evaluate fit on current samples ==")
    session = SESSION_MANAGER.get_session()
    eyes = np.vstack([s.eye_center.reshape(1, 3) for s in session.samples])
    pupils = np.vstack([s.pupil_center.reshape(1, 3) for s in session.samples])
    target_pixels = np.vstack([np.array(s.target_pixel).reshape(1, 2) for s in session.samples])
    K = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]], dtype=float)
    samples_pro = build_samples_from_arrays(eyes, pupils, target_pixels=target_pixels, K=K)
    # Use kappa from fit_kappa()
    # Reconstruct kappa vector from model (axis * angle_rad)
    angle_rad = np.radians(out["kappa_model"]["angle_deg"])\
        if isinstance(out, dict) and "kappa_model" in out else 0.0
    axis = np.array(out["kappa_model"]["axis"], dtype=float)\
        if isinstance(out, dict) and "kappa_model" in out else np.array([0, 0, 1], dtype=float)
    kappa_vec = axis * angle_rad
    metrics = evaluate_fit(samples_pro, kappa_vec, K=K, return_pixel_err=True)
    print("Evaluation:", metrics)


if __name__ == "__main__":
    main()


