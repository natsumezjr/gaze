from __future__ import annotations

"""
提供最小API，供前端或其他模块调用：
- start_session(session_id)
- push_sample(payload)
- fit_kappa(session_id)
- compute_gaze(eye, target_pixel)

与现有项目结构对齐，尽可能保持命名与类型一致；
必要处做了轻量封装，便于后续扩展为HTTP或本地调用。
"""

from typing import Dict, Tuple, Optional
import numpy as np

from project.fitting.app.state import (
    SESSION_MANAGER,
    CalibrationSample,
    KappaModel,
    GazeVector,
)
from project.fitting.core.gaze_estimator import compute_theoretical_gaze
from project.fitting.core.kappa_calibrator_pro import (
    build_samples_from_arrays,
    estimate_kappa as estimate_kappa_pro,
    apply_kappa,
)
from project.recg_fit_data.data_manager import RECG_FIT_DATA_MANAGER


def start_session(session_id: str) -> Dict:
    session = SESSION_MANAGER.start_session(session_id=session_id)
    return {
        "session_id": session.session_id,
        "stage": session.stage,
        "samples": len(session.samples),
        "has_kappa": session.kappa_model is not None,
        "has_fitting": len(session.last_fitting) > 0,
    }


def push_sample(payload: Dict) -> Dict:
    """
    输入payload建议包含：
    {
      "timestamp": float,
      "target_pixel": [x, y],
      "eye": "left"|"right",
      "frame_id": Optional[int]
    }

    其余字段（eye_center / pupil_center / theoretical_gaze / quality）由后端补齐：
    - eye_center: 使用最近一次拟合结果中心
    - pupil_center: 目前采用与eye_center相邻的一个合理近似或后续从识别模块补齐
    - theoretical_gaze: 由 compute_theoretical_gaze(eye_center, pupil_center) 计算
    - quality: 先用拟合confidence，后续可融合更多质量信号
    """
    eye = payload.get("eye", "left")
    timestamp = float(payload.get("timestamp", 0.0))
    target_pixel = tuple(payload.get("target_pixel", (0, 0)))  # type: ignore
    frame_id = payload.get("frame_id")

    session = SESSION_MANAGER.get_session()
    if session is None or eye not in session.last_fitting:
        raise RuntimeError("未找到最近拟合结果，请先执行拟合以生成 last_fitting。")

    last_fit = session.last_fitting[eye]
    eye_center = last_fit.center.astype(float)

    # 从识别模块的实时数据管理器获取瞳孔中心（摄像头来源）
    pupil_points = RECG_FIT_DATA_MANAGER.get_coordinate_point(eye, "pupil")
    if not pupil_points:
        raise RuntimeError("未获取到瞳孔中心数据，请确保识别模块已将pupil写入数据管理器。")
    # 取第一点或对多点求均值（保留可扩展性）
    if isinstance(pupil_points, list) and len(pupil_points) > 1:
        pts = np.vstack([np.asarray(p)[:3].reshape(1, 3) for p in pupil_points if isinstance(p, np.ndarray)])
        pupil_center = np.mean(pts, axis=0)
    else:
        p = pupil_points[0]
        pupil_center = np.asarray(p)[:3].astype(float)

    theoretical_gaze = compute_theoretical_gaze(eye_center, pupil_center)
    quality = float(last_fit.confidence)

    sample = CalibrationSample(
        timestamp=timestamp,
        target_pixel=(int(target_pixel[0]), int(target_pixel[1])),
        eye_center=eye_center,
        pupil_center=pupil_center,
        theoretical_gaze=theoretical_gaze,
        quality=quality,
        frame_id=frame_id,
    )
    SESSION_MANAGER.add_sample(sample)

    return {"ok": True, "samples": len(SESSION_MANAGER.get_session().samples)}


def fit_kappa(session_id: Optional[str] = None, K: Optional[np.ndarray] = None) -> Dict:
    """
    使用当前会话的样本估计Kappa模型。
    - K: 相机内参矩阵(3x3)。若为空，可用默认内参或从配置加载。
    """
    session = SESSION_MANAGER.get_session()
    if session is None or len(session.samples) == 0:
        raise RuntimeError("无可用样本，请先 push_sample。")

    # 组装 arrays 以使用专业版校准器
    eyes = np.vstack([s.eye_center.reshape(1, 3) for s in session.samples])
    pupils = np.vstack([s.pupil_center.reshape(1, 3) for s in session.samples])
    target_pixels = np.vstack([np.array(s.target_pixel).reshape(1, 2) for s in session.samples])

    if K is None:
        # 采用简易默认内参（可替换为从配置加载）
        K = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]], dtype=float)

    samples_pro = build_samples_from_arrays(eyes, pupils, target_pixels=target_pixels, K=K)
    kappa_vec, info = estimate_kappa_pro(samples_pro, lock_roll=True)

    # 将轴角向量转换为轴+角度（度）
    angle_deg = float(np.degrees(np.linalg.norm(kappa_vec)))
    axis = kappa_vec / (np.linalg.norm(kappa_vec) + 1e-9)

    model = KappaModel(axis=axis.astype(float), angle_deg=angle_deg, samples_count=len(session.samples), quality=float(info.get("quality", 1.0)) if isinstance(info, dict) else 1.0)
    SESSION_MANAGER.set_kappa_model(model)

    return {
        "kappa_model": {
            "axis": model.axis.tolist(),
            "angle_deg": model.angle_deg,
            "samples_count": model.samples_count,
            "quality": model.quality,
        }
    }


def compute_gaze(eye: str, target_pixel: Optional[Tuple[int, int]] = None) -> Dict:
    """
    返回当前帧（或最近）补偿后的视线向量。
    - 使用会话中的 last_fitting 与 kappa_model。
    - 如果缺少 kappa_model，退化为理论视线（未补偿）。
    """
    session = SESSION_MANAGER.get_session()
    if session is None or eye not in session.last_fitting:
        raise RuntimeError("未找到最近拟合结果，请先执行拟合。")

    last_fit = session.last_fitting[eye]
    eye_center = last_fit.center.astype(float)
    pupil_center = (eye_center + np.array([0.0, 0.0, -1.0], dtype=float))
    v = compute_theoretical_gaze(eye_center, pupil_center)

    if session.kappa_model is not None:
        # 将轴角模型转回向量表达，用现有 apply_kappa 接口
        axis = session.kappa_model.axis
        angle_rad = np.radians(session.kappa_model.angle_deg)
        kappa_vec = axis * angle_rad
        compensated = apply_kappa(v.reshape(1, 3), kappa_vec)[0]
        d = compensated
    else:
        d = v

    gaze = GazeVector(eye=eye, origin=eye_center, direction=d, target_pixel=tuple(target_pixel) if target_pixel else None)

    return {
        "gaze": {
            "eye": gaze.eye,
            "origin": gaze.origin.tolist(),
            "direction": gaze.direction.tolist(),
            "target_pixel": list(gaze.target_pixel) if gaze.target_pixel else None,
        }
    }


