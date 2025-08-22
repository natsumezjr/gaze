"""
Kappa (κ) compensation: estimate a small, constant rotation that maps the
optical axis to the visual axis for a given user, using calibration samples.

This module is **self-contained** (no external project imports) and provides:
- `estimate_kappa(...)`        -> Solve for kappa as a small-angle rotation α=[αx,αy,αz]
- `apply_kappa(v, kappa)`      -> Rotate 3D vectors by κ (Rodrigues)
- `evaluate_fit(...)`          -> Angular / pixel errors after applying κ
- `incremental_update(...)`    -> Lightweight recursive update for online tuning
- `unproject_pixels_to_rays(...)` -> Convert pixel coords (u,v) to camera rays via K^-1

Typical pipeline per sample i:
  optical axis v_i = normalize(c_pupil - c_eye)
  target ray d_i   = normalize(K^-1 [u_i, v_i, 1]^T)  or  normalize(target_point_i - C_cam)
Solve α so that R(α) v_i ≈ d_i in least squares sense (small-angle linearization).

Author: ChatGPT
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Dict
import math
import numpy as np


# ---------- Utilities ----------

def normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v, dtype=float)
    return v / n


def hat(w: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix [w]_x such that [w]_x x = w × x."""
    wx, wy, wz = w
    return np.array([[0.0, -wz,  wy],
                     [wz,   0.0, -wx],
                     [-wy,  wx,  0.0]], dtype=float)


def rodrigues(w: np.ndarray) -> np.ndarray:
    """Rodrigues' rotation from axis-angle vector w (radians)."""
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3, dtype=float)
    k = w / theta
    K = hat(k)
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def project_pixel_to_ray(u: float, v: float, K: np.ndarray) -> np.ndarray:
    """Back-project pixel (u,v) using intrinsics K to a **direction ray** in camera coords."""
    K = np.asarray(K, dtype=float)
    uv1 = np.array([u, v, 1.0], dtype=float)
    ray = np.linalg.inv(K) @ uv1
    return normalize(ray)


def unproject_pixels_to_rays(pixels: np.ndarray, K: np.ndarray) -> np.ndarray:
    """pixels: (N,2), return (N,3) unit rays."""
    pixels = np.asarray(pixels, dtype=float)
    rays = np.stack([project_pixel_to_ray(u, v, K) for u, v in pixels], axis=0)
    return rays


# ---------- Data Model ----------

@dataclass
class Sample:
    """One calibration sample.
    Provide either (target_ray) OR (target_pixel, K). If both provided, target_ray is used.
    """
    c_eye: np.ndarray           # (3,)
    c_pupil: np.ndarray         # (3,)
    target_point: Optional[np.ndarray] = None  # (3,), in camera coords (optional)
    target_ray: Optional[np.ndarray] = None    # (3,), unit direction ray to target
    target_pixel: Optional[Tuple[float, float]] = None  # (u,v) in pixels
    K: Optional[np.ndarray] = None             # Camera intrinsics (3x3) if using target_pixel
    weight: float = 1.0

    def optical_axis(self) -> np.ndarray:
        return normalize(np.asarray(self.c_pupil) - np.asarray(self.c_eye))

    def target_dir(self) -> Optional[np.ndarray]:
        if self.target_ray is not None:
            return normalize(self.target_ray)
        if self.target_point is not None:
            return normalize(np.asarray(self.target_point))  # camera at origin
        if self.target_pixel is not None and self.K is not None:
            u, v = self.target_pixel
            return project_pixel_to_ray(u, v, self.K)
        return None


# ---------- Core Estimation ----------

def estimate_kappa(
    samples: Iterable[Sample],
    lock_roll: bool = True,
    robust_iters: int = 3,
    huber_delta_deg: float = 5.0,
) -> Tuple[np.ndarray, Dict]:
    """
    Estimate small-angle κ = [αx, αy, αz] (radians) via linearization:
      R(α) v ≈ v + α × v  ≈ d
      => (-[v]_x) α = (d - v)
    Stack all samples and solve weighted least squares (optionally lock αz=0).

    Returns:
        kappa: (3,) axis-angle in radians
        info:  dict with residual stats (mean/median angular error in deg, weights, etc.)
    """
    Vs: List[np.ndarray] = []
    Ds: List[np.ndarray] = []
    Ws: List[float] = []

    for s in samples:
        v = s.optical_axis()
        d = s.target_dir()
        if d is None:
            continue
        if np.linalg.norm(v) < 1e-12 or np.linalg.norm(d) < 1e-12:
            continue
        Vs.append(v)
        Ds.append(d)
        Ws.append(float(s.weight))

    if not Vs:
        raise ValueError("No valid samples provided for kappa estimation.")

    V = np.stack(Vs, 0)  # (N,3)
    D = np.stack(Ds, 0)  # (N,3)
    w = np.asarray(Ws, dtype=float)  # (N,)

    # Build linear system: M α = b
    # α × v = -[v]_x α
    # v + α×v ≈ d  -> -[v]_x α = d - v
    N = V.shape[0]
    M = np.zeros((3*N, 3), dtype=float)
    b = np.zeros((3*N,), dtype=float)
    for i in range(N):
        vx = hat(V[i])  # [v]_x
        M[3*i:3*i+3, :] = -vx
        b[3*i:3*i+3] = D[i] - V[i]

    # Weights -> expand to block-diagonal: simple way is to multiply rows by sqrt(w_i)
    Wrows = np.repeat(np.sqrt(w), 3)
    M_w = M * Wrows[:, None]
    b_w = b * Wrows

    # Optionally lock roll (αz=0) to fit only αx, αy
    if lock_roll:
        M_w = M_w[:, :2]  # columns for αx, αy
        # robust reweighting (Huber on angular residuals)
        alpha_xy = np.linalg.lstsq(M_w, b_w, rcond=None)[0]  # (2,)
        alpha = np.array([alpha_xy[0], alpha_xy[1], 0.0], dtype=float)
    else:
        alpha = np.linalg.lstsq(M_w, b_w, rcond=None)[0]  # (3,)

    # Iteratively reweight using Huber on angular residuals to suppress outliers
    def angle_err_deg(a: np.ndarray) -> np.ndarray:
        R = rodrigues(a)
        Vrot = (R @ V.T).T
        dots = np.clip(np.sum(Vrot * D, axis=1), -1.0, 1.0)
        return np.degrees(np.arccos(dots))

    delta = float(huber_delta_deg)
    for _ in range(max(0, robust_iters)):
        ang = angle_err_deg(alpha)  # (N,)
        # Huber weights
        r = ang
        w_h = np.ones_like(r)
        mask = np.abs(r) > delta
        w_h[mask] = delta / (np.abs(r[mask]) + 1e-12)
        # combine with original weights
        w_tot = w * w_h
        Wrows = np.repeat(np.sqrt(w_tot), 3)
        Mwr = M * Wrows[:, None]
        bwr = b * Wrows
        if lock_roll:
            Mwr2 = Mwr[:, :2]
            alpha_xy = np.linalg.lstsq(Mwr2, bwr, rcond=None)[0]
            alpha = np.array([alpha_xy[0], alpha_xy[1], 0.0], dtype=float)
        else:
            alpha = np.linalg.lstsq(Mwr, bwr, rcond=None)[0]

    # Final metrics
    err_deg = angle_err_deg(alpha)
    info = {
        "num_samples": int(N),
        "kappa_rad": alpha.copy(),
        "kappa_deg": np.degrees(alpha),
        "mean_ang_err_deg": float(np.mean(err_deg)),
        "median_ang_err_deg": float(np.median(err_deg)),
        "max_ang_err_deg": float(np.max(err_deg)),
        "per_sample_err_deg": err_deg.tolist(),
    }
    return alpha, info


def apply_kappa(vectors: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """Rotate a set of vectors (N,3) by κ (3,) using Rodrigues."""
    V = np.asarray(vectors, dtype=float)
    R = rodrigues(np.asarray(kappa, dtype=float))
    return (R @ V.T).T


def evaluate_fit(
    samples: Iterable[Sample],
    kappa: np.ndarray,
    K: Optional[np.ndarray] = None,
    return_pixel_err: bool = False
) -> Dict:
    """Compute angular error for provided samples after applying κ.
    If K and target_pixel exist, also report pixel reprojection error (approx)."""
    Vs, Ds, pix_targets = [], [], []
    for s in samples:
        v = s.optical_axis()
        d = s.target_dir()
        if d is None:
            continue
        Vs.append(v); Ds.append(d)
        if s.target_pixel is not None:
            pix_targets.append(s.target_pixel)
        else:
            pix_targets.append(None)

    V = np.stack(Vs, 0)
    D = np.stack(Ds, 0)
    Vrot = apply_kappa(V, kappa)
    dots = np.clip(np.sum(Vrot * D, axis=1), -1.0, 1.0)
    ang_err = np.degrees(np.arccos(dots))

    out = {
        "mean_ang_err_deg": float(np.mean(ang_err)),
        "median_ang_err_deg": float(np.median(ang_err)),
        "max_ang_err_deg": float(np.max(ang_err)),
        "N": int(len(ang_err)),
    }

    if return_pixel_err and K is not None and all(p is not None for p in pix_targets):
        # Project rays to pixels using K (assume z=1 normalization)
        rays = Vrot / (Vrot[:, 2:3] + 1e-12)
        pred_pix = (K @ rays.T).T  # homogeneous with z normalized to 1 → K*[x,y,1]^T
        pred_uv = pred_pix[:, :2]
        tgt_uv = np.array(pix_targets, dtype=float)
        pix_err = np.linalg.norm(pred_uv - tgt_uv, axis=1)
        out.update({
            "mean_pix_err": float(np.mean(pix_err)),
            "median_pix_err": float(np.median(pix_err)),
            "max_pix_err": float(np.max(pix_err)),
        })

    return out


# ---------- Online / Incremental Update ----------

def incremental_update(kappa_prev: np.ndarray, v: np.ndarray, d: np.ndarray, beta: float = 0.1) -> np.ndarray:
    """One-sample gradient-like update on κ (small-angle), minimizing angular error.
    beta: learning rate in [0,1]."""
    v = normalize(v); d = normalize(d)
    # approximate gradient of angle error ~ - (d × v_rot) wrt α
    v_rot = rodrigues(kappa_prev) @ v
    g = np.cross(d, v_rot)  # 3D error axis (direction to rotate v_rot towards d)
    # update κ in the direction of g
    kappa_new = kappa_prev + beta * g
    # clamp magnitude to reasonable range (±15 deg)
    max_rad = np.radians(15.0)
    if np.linalg.norm(kappa_new) > max_rad:
        kappa_new = kappa_new * (max_rad / np.linalg.norm(kappa_new))
    return kappa_new


# ---------- Convenience: build samples from arrays ----------

def build_samples_from_arrays(
    eyes: np.ndarray, pupils: np.ndarray,
    target_pixels: Optional[np.ndarray] = None,
    K: Optional[np.ndarray] = None,
    target_points: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
) -> List[Sample]:
    """Helper to construct Sample list.
    Exactly one of (target_pixels,K) or (target_points) should be provided.
    """
    eyes = np.asarray(eyes, dtype=float)
    pupils = np.asarray(pupils, dtype=float)
    N = eyes.shape[0]
    if weights is None:
        weights = np.ones((N,), dtype=float)
    samples: List[Sample] = []

    for i in range(N):
        s = Sample(c_eye=eyes[i], c_pupil=pupils[i], weight=float(weights[i]))
        if target_points is not None:
            s.target_point = target_points[i]
        elif target_pixels is not None and K is not None:
            s.target_pixel = (float(target_pixels[i,0]), float(target_pixels[i,1]))
            s.K = K
        else:
            raise ValueError("Provide either target_points (3D) or (target_pixels, K).")
        samples.append(s)
    return samples


# ---------- Demo / CLI ----------

if __name__ == "__main__":
    # Minimal synthetic demo
    np.random.seed(0)
    # True kappa (yaw, pitch, roll) in degrees (for synthetic test)
    kappa_true_deg = np.array([2.0, -1.0, 0.0])
    kappa_true = np.radians(kappa_true_deg)

    # Generate random optical axes and targets by rotating with true kappa
    N = 40
    V = np.random.randn(N, 3)
    V = np.array([normalize(v) for v in V])
    D = (rodrigues(kappa_true) @ V.T).T  # targets
    eyes = np.zeros_like(V)              # camera origin → only direction matters
    pupils = V                           # so that (pupil - eye) == V

    samples = build_samples_from_arrays(eyes, pupils, target_points=D)  # using rays as "points"
    kappa_est, info = estimate_kappa(samples, lock_roll=True)
    print("[Demo] True κ (deg):", kappa_true_deg)
    print("[Demo] Est.  κ (deg):", np.degrees(kappa_est))
    print("[Demo] Fit info:", info)
