from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

# 与现有代码保持一致的命名与类型
Eye = Literal["left", "right"]


@dataclass
class FittingResultLite:
    """
    轻量版拟合结果，仅保存前端/会话需要的关键字段。
    与 project.fitting.core.fitting_strategy.FittingResult 对应，便于在会话中存取。
    """

    center: np.ndarray  # (3,), mm
    radius: float  # mm
    confidence: float  # 0~1
    converged: bool
    final_residual_norm: float
    strategy_used: str


@dataclass
class CalibrationSample:
    """
    标定采样单元。与现有Kappa校准流程的数据需求兼容。
    - target_pixel: 前端标定点的屏幕像素坐标 (x, y)
    - eye_center / pupil_center: 当前帧的三维坐标（毫米）
    - theoretical_gaze: 当前帧根据几何计算得到的理论视线向量（单位向量）
    - quality: 质量分数（0~1），可由检测置信度与稳定性综合而来
    """

    timestamp: float
    target_pixel: Tuple[int, int]
    eye_center: np.ndarray  # (3,), mm
    pupil_center: np.ndarray  # (3,), mm
    theoretical_gaze: np.ndarray  # (3,), unit vector
    quality: float
    frame_id: Optional[int] = None


@dataclass
class KappaModel:
    """
    Kappa补偿模型。
    - axis: 单位向量
    - angle_deg: 角度（度）
    - samples_count: 用于估计的样本数
    - quality: 模型质量（0~1）
    """

    axis: np.ndarray  # (3,), unit vector
    angle_deg: float
    samples_count: int
    quality: float


@dataclass
class GazeVector:
    """
    最终（补偿后）视线向量。
    - origin: 视线起点，通常为眼球中心
    - direction: 单位向量，已进行Kappa补偿
    - target_pixel: 可选，便于回显对应屏幕点
    """

    eye: Eye
    origin: np.ndarray  # (3,), mm
    direction: np.ndarray  # (3,), unit vector
    target_pixel: Optional[Tuple[int, int]] = None


@dataclass
class CalibrationSession:
    """
    标定会话：贯穿一次标定流程的状态容器。
    stage: pre_fitting | collecting | calibrated
    """

    session_id: str
    stage: str = "pre_fitting"
    samples: List[CalibrationSample] = field(default_factory=list)
    kappa_model: Optional[KappaModel] = None
    last_fitting: Dict[Eye, FittingResultLite] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: time.time())


class CalibrationSessionManager:
    """
    标定会话单例管理器：用于跨模块保存与读取会话与采样数据。
    - 与 RECG_FIT_DATA_MANAGER 并行：后者负责三维关键点；本管理器负责标定/补偿的会话级数据。
    """

    _instance: Optional["CalibrationSessionManager"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # 当前活跃会话
        self.current: Optional[CalibrationSession] = None

    # -------------------- 会话控制 --------------------
    def start_session(self, session_id: str) -> CalibrationSession:
        """创建或重置当前会话。"""
        self.current = CalibrationSession(session_id=session_id)
        return self.current

    def get_session(self) -> Optional[CalibrationSession]:
        return self.current

    # -------------------- 结果与样本写入 --------------------
    def set_last_fitting(self, eye: Eye, result: FittingResultLite) -> None:
        """保存最近一次拟合结果，用于后续样本补齐与显示。"""
        if self.current is None:
            # 自动创建默认会话，便于在未显式开启时也能工作
            self.start_session(session_id="default")
        self.current.last_fitting[eye] = result

    def add_sample(self, sample: CalibrationSample) -> None:
        if self.current is None:
            self.start_session(session_id="default")
        self.current.samples.append(sample)
        # 进入采集阶段
        if self.current.stage == "pre_fitting":
            self.current.stage = "collecting"

    # -------------------- Kappa模型 --------------------
    def set_kappa_model(self, model: KappaModel) -> None:
        if self.current is None:
            self.start_session(session_id="default")
        self.current.kappa_model = model
        self.current.stage = "calibrated"


# 全局单例实例
SESSION_MANAGER = CalibrationSessionManager()


