"""
Kappa (κ) compensation: estimate a small, constant rotation that maps the
optical axis to the visual axis for a given user, using calibration samples.

This module is **self-contained** (no external project imports) and provides:
- `estimate_kappa(...)`        -> Solve for kappa as a small-angle rotation α=[αx,αy,αz]
- `apply_kappa(v, kappa)`      -> Rotate 3D vectors by κ (Rodrigues)
- `evaluate_fit(...)`          -> Angular errors after applying κ
- `incremental_update(...)`    -> Lightweight recursive update for online tuning

Typical pipeline per sample i:
  optical axis v_i = normalize(c_pupil - c_eye)
  target ray d_i   = normalize(K^-1 [u_i, v_i, 1]^T)  or  normalize(target_point_i - C_cam)
Solve α so that R(α) v_i ≈ d_i in least squares sense (small-angle linearization).

Author: ChatGPT
"""

from typing import Iterable, List, Optional, Tuple, Dict
import math
import threading
import numpy as np
from project.data.data_models import GazeSamples, Point2D, Point3D


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




# ---------- Data Model ----------




# ---------- Core Estimation ----------

def estimate_kappa(
    samples: Iterable[GazeSamples],
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
    samples: Iterable[GazeSamples],
    kappa: np.ndarray,
) -> Dict:
    """Compute angular error for provided samples after applying κ.
    """
    Vs, Ds = [], []
    for s in samples:
        v = s.optical_axis()
        d = s.target_dir()
        if d is None:
            continue
        Vs.append(v); Ds.append(d)

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

    return out





# ---------- Convenience: build samples from arrays ----------

def build_samples_from_arrays(
    eyes: List[Point3D], pupils: List[Point3D],
    target_points: List[Point3D],
    weights: Optional[np.ndarray] = None,
) -> List[GazeSamples]:
    """Helper to construct GazeSamples list.
    Exactly one of (target_points) should be provided.
    """
    eyes = np.array([eye.to_ndarray() for eye in eyes], dtype=float)
    pupils = np.array([pupil.to_ndarray() for pupil in pupils], dtype=float)
    target_points = np.array([target_point.to_ndarray() for target_point in target_points], dtype=float)
    N = eyes.shape[0]
    if weights is None:
        weights = np.ones((N,), dtype=float)
    samples: List[GazeSamples] = []

    for i in range(N):
        s = GazeSamples(c_eye=eyes[i], c_pupil=pupils[i], weight=float(weights[i]))
        if target_points is not None:
            s.target_point = target_points[i]
        else:
            raise ValueError("Provide either target_points (3D).")
        samples.append(s)
    return samples


class KappaStorage:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.kappa_storage: Dict[str, Point3D] = {
            "left": None,
            "right": None
        }
        self.pixel_storage: Dict[str, List[Point2D]] = {
            "left": None,
            "right": None
        }
        
        self.center_storage: Dict[str, Dict[str, List[Point3D]]] = {
            "left": {
                "eyeball": [],
                "pupil": []
            },
            "right": {
                "eyeball": [],
                "pupil": []
            }
        }

    def set_kappa(self, kappa_left: Point3D, kappa_right: Point3D):
        self.kappa_storage["left"] = kappa_left
        self.kappa_storage["right"] = kappa_right
    
    def set_pixel(self, pixel_left: List[Point2D], pixel_right: List[Point2D]):
        self.pixel_storage["left"] = pixel_left
        self.pixel_storage["right"] = pixel_right
    
    def append_center(self, eye: str, type: str, center: Point3D):
        self.center_storage[eye][type].append(center)
    
    def get_center(self, eye: str, type: str) -> List[Point3D]:
        return self.center_storage[eye][type]
    
    def is_center_valid(self) -> bool:
        return all(self.center_storage[eye] is not None for eye in self.center_storage)
    
    def reset_center(self):
        self.center_storage = {
            "left": {
                "eyeball": [],
                "pupil": []
            },
            "right": {
                "eyeball": [],
                "pupil": []
            }
        }

    
    def get_kappa(self, eye: str) -> Point3D:
        return self.kappa_storage[eye]
    
    def get_pixel(self, eye: str) -> List[Point2D]:
        return self.pixel_storage[eye]
    
    def is_kappa_valid(self) -> bool:
        return all(self.kappa_storage[eye] is not None for eye in self.kappa_storage)
    
    def is_pixel_valid(self) -> bool:
        return all(self.pixel_storage[eye] is not None for eye in self.pixel_storage)
    
    def reset_kappa(self):
        self.kappa_storage = {
            "left": None,
            "right": None
        }

KAPPA_STORAGE = KappaStorage()

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
