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

from typing import Iterable, List, Optional, Tuple, Dict, Union
import math
import threading
import numpy as np
from project.data.data_models import GazeSamples, Point2D, Point3D, Kappa, KappaEstimationResult, KappaFitEvaluation


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
) -> Tuple[Kappa, KappaEstimationResult]:
    """
    Estimate small-angle κ = [αx, αy, αz] (radians) via linearization:
      R(α) v ≈ v + α × v  ≈ d
      => (-[v]_x) α = (d - v)
    Stack all samples and solve weighted least squares (optionally lock αz=0).

    Returns:
        kappa: Kappa object (axis-angle in radians)
        result: KappaEstimationResult with statistics
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
    kappa = Kappa.from_ndarray(alpha)
    result = KappaEstimationResult(
        kappa=kappa,
        num_samples=int(N),
        mean_ang_err_deg=float(np.mean(err_deg)),
        median_ang_err_deg=float(np.median(err_deg)),
        max_ang_err_deg=float(np.max(err_deg)),
        per_sample_err_deg=err_deg.tolist(),
    )
    return kappa, result


def apply_kappa(vectors: np.ndarray, kappa: Union[Kappa, np.ndarray]) -> np.ndarray:
    """Rotate a set of vectors (N,3) by κ using Rodrigues.
    
    Args:
        vectors: (N,3) array of vectors to rotate
        kappa: Kappa object or (3,) array (axis-angle in radians)
    
    Returns:
        (N,3) array of rotated vectors
    """
    V = np.asarray(vectors, dtype=float)
    if isinstance(kappa, Kappa):
        kappa_arr = kappa.to_ndarray()
    else:
        kappa_arr = np.asarray(kappa, dtype=float)
    R = rodrigues(kappa_arr)
    return (R @ V.T).T


def evaluate_fit(
    samples: Iterable[GazeSamples],
    kappa: Union[Kappa, np.ndarray],
) -> KappaFitEvaluation:
    """Compute angular error for provided samples after applying κ.
    
    Args:
        samples: Iterable of GazeSamples
        kappa: Kappa object or (3,) array (axis-angle in radians)
    
    Returns:
        KappaFitEvaluation with error statistics
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

    return KappaFitEvaluation(
        mean_ang_err_deg=float(np.mean(ang_err)),
        median_ang_err_deg=float(np.median(ang_err)),
        max_ang_err_deg=float(np.max(ang_err)),
        num_samples=int(len(ang_err)),
    )





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


class SamplesStorage:
    """标定样本存储 - 最简数据结构: {frame_id: {eye: {pupils: [], eyeball: [], pixel: []}}}"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        # {frame_id: {eye: {pupils: [], eyeball: [], pixel: []}}}
        self.samples: Dict[int, Dict[str, Dict[str, List]]] = {}
        self._initialized = True
    
    def append(self, frame_id: int, eye: str, pupil: Point3D, eyeball: Point3D, pixel: Point2D):
        """添加一个标定样本"""
        if frame_id not in self.samples:
            self.samples[frame_id] = {"left": {"pupils": [], "eyeball": [], "pixel": []},
                                      "right": {"pupils": [], "eyeball": [], "pixel": []}}
        self.samples[frame_id][eye]["pupils"].append(pupil)
        self.samples[frame_id][eye]["eyeball"].append(eyeball)
        self.samples[frame_id][eye]["pixel"].append(pixel)
    
    def get_all_samples(self) -> Dict[int, Dict[str, Dict[str, List]]]:
        """获取所有样本"""
        return self.samples
    
    def clear(self):
        """清空所有样本"""
        self.samples = {}


class KappaStorage:
    """Kappa 存储 - 只存储已计算出的 kappa"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self.kappa_storage: Dict[str, Optional[Kappa]] = {
            "left": None,
            "right": None
        }
        self._initialized = True

    def set_kappa(self, eye: str, kappa: Kappa):
        """设置指定眼睛的 kappa"""
        self.kappa_storage[eye] = kappa
    
    def get_kappa(self, eye: str) -> Optional[Kappa]:
        """获取指定眼睛的 kappa"""
        return self.kappa_storage[eye]
    
    def is_kappa_valid(self) -> bool:
        """检查 kappa 是否有效（左右眼都有）"""
        return all(self.kappa_storage[eye] is not None for eye in self.kappa_storage)
    
    def reset_kappa(self):
        """重置 kappa"""
        self.kappa_storage = {
            "left": None,
            "right": None
        }

KAPPA_STORAGE = KappaStorage()
SAMPLES_STORAGE = SamplesStorage()

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
    kappa_est, result = estimate_kappa(samples, lock_roll=True)
    print("[Demo] True κ (deg):", kappa_true_deg)
    print("[Demo] Est.  κ (deg):", kappa_est.to_degrees())
    print("[Demo] Fit result:", result)
