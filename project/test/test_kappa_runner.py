
"""
test_kappa_runner.py
A small, reproducible test program for kappa_calibrator.py.

It runs two test cases:
A) 3D target points (VR/AR-like setup)
B) Screen pixels + intrinsics K (monitor-like setup)

For each case, it:
  - synthesizes N samples with a known true kappa
  - adds optional noise to simulate measurement errors
  - estimates kappa with the provided solver
  - reports angular and (if applicable) pixel errors
"""

import numpy as np
import sys
sys.path.append("/mnt/data")
from kappa_calibrator_pro import (
    build_samples_from_arrays, estimate_kappa, evaluate_fit,
    rodrigues, normalize, unproject_pixels_to_rays
)

def deg(x): return np.degrees(x)

def synth_case_points(N=60, kappa_true_deg=(2.0, -1.5, 0.0), noise_std_deg=0.5, seed=42):
    rng = np.random.default_rng(seed)
    kappa_true = np.radians(kappa_true_deg)

    # optical axes (unit)
    V = rng.normal(size=(N,3))
    V = V / np.linalg.norm(V, axis=1, keepdims=True)

    # ground-truth target directions = R(kappa_true) @ V
    D = (rodrigues(kappa_true) @ V.T).T

    # add angular noise on D (simulate target direction uncertainty)
    if noise_std_deg and noise_std_deg > 0:
        ang = np.radians(noise_std_deg) * rng.normal(size=(N,3))
        noise_R = np.array([rodrigues(a) for a in ang])
        D = np.einsum('nij,nj->ni', noise_R, D)
        D = D / np.linalg.norm(D, axis=1, keepdims=True)

    # eye/pupil so that (pupil - eye) == V (only direction matters)
    eyes = np.zeros_like(V)
    pupils = V.copy()

    # Use D as "3D points" on unit sphere (direction); estimator normalizes anyway.
    samples = build_samples_from_arrays(eyes, pupils, target_points=D)
    kappa_est, info = estimate_kappa(samples, lock_roll=True)
    metrics = evaluate_fit(samples, kappa_est)

    return {
        "true_kappa_deg": kappa_true_deg,
        "est_kappa_deg": list(np.round(deg(kappa_est), 5)),
        "fit_metrics": metrics,
        "solver_info": {"num_samples": int(info["num_samples"]), "kappa_rad": list(info["kappa_rad"]), "kappa_deg": list(info["kappa_deg"]), "mean_ang_err_deg": float(info["mean_ang_err_deg"]), "median_ang_err_deg": float(info["median_ang_err_deg"]), "max_ang_err_deg": float(info["max_ang_err_deg"]), "per_sample_err_deg": list(info["per_sample_err_deg"])}
    }

def synth_case_pixels(N=60, kappa_true_deg=(2.0, -1.5, 0.0), noise_pix=0.5, seed=7):
    rng = np.random.default_rng(seed)
    kappa_true = np.radians(kappa_true_deg)

    # Intrinsics
    fx, fy = 800.0, 800.0
    cx, cy = 640.0, 360.0
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=float)

    # optical axes
    V = rng.normal(size=(N,3))
    V = V / np.linalg.norm(V, axis=1, keepdims=True)

    # rotate V to get target rays
    D = (rodrigues(kappa_true) @ V.T).T

    # project to pixels
    rays = D / (D[:,2:3] + 1e-12)
    pix = (K @ rays.T).T[:, :2]

    # add pixel noise
    if noise_pix and noise_pix > 0:
        pix += rng.normal(scale=noise_pix, size=pix.shape)

    eyes = np.zeros_like(V)
    pupils = V.copy()

    samples = build_samples_from_arrays(eyes, pupils, target_pixels=pix, K=K)
    kappa_est, info = estimate_kappa(samples, lock_roll=True)
    metrics = evaluate_fit(samples, kappa_est, K=K, return_pixel_err=True)

    return {
        "true_kappa_deg": kappa_true_deg,
        "est_kappa_deg": list(np.round(deg(kappa_est), 5)),
        "fit_metrics": metrics,
        "solver_info": {"num_samples": int(info["num_samples"]), "kappa_rad": list(info["kappa_rad"]), "kappa_deg": list(info["kappa_deg"]), "mean_ang_err_deg": float(info["mean_ang_err_deg"]), "median_ang_err_deg": float(info["median_ang_err_deg"]), "max_ang_err_deg": float(info["max_ang_err_deg"]), "per_sample_err_deg": list(info["per_sample_err_deg"])},
        "K": K.tolist()
    }

def main():
    out = {}
    out["case_A_points"] = synth_case_points()
    out["case_B_pixels"] = synth_case_pixels()
    import json
    print("=== Kappa Calibration Test Results ===")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
