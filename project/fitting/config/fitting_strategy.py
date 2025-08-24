"""
拟合策略配置模块
分层设计：
  1. 基础配置（共用参数，收敛/权重/约束）
  2. 算法配置（LM / RANSAC / 权重）
  3. 策略配置（组合不同算法，适应不同数据质量）
"""

from dataclasses import dataclass, field
from typing import Dict

# ==================== 解剖学约束参数 ====================
ANATOMICAL_CONSTRAINTS = {
    # 瞳孔到眼球表面的距离范围（0.1-0.3mm）
    "PUPIL_TO_SURFACE_MIN": 0.0001,      # 0.1mm
    "PUPIL_TO_SURFACE_MAX": 0.0003,      # 0.3mm
    
    # 虹膜边界到眼球表面的距离范围（0.5-1.2mm）
    "IRIS_TO_SURFACE_MIN": 0.0005,       # 0.5mm
    "IRIS_TO_SURFACE_MAX": 0.0012,       # 1.2mm
    
    # 眼睛轮廓到眼球表面的距离范围（1.0-2.0mm）
    "CONTOUR_TO_SURFACE_MIN": 0.001,     # 1.0mm
    "CONTOUR_TO_SURFACE_MAX": 0.004,     # 4.0mm
    
    # 眼球半径范围（8-20mm）
    "EYEBALL_RADIUS_MIN": 0.008,         # 8mm
    "EYEBALL_RADIUS_MAX": 0.020,         # 20mm
    "EYEBALL_RADIUS_DEFAULT": 0.012,     # 12mm
    
    # 生理偏差中值 - 用于残差计算
    # 这些值基于解剖学约束范围计算得出，用于在拟合过程中校正个体差异
    # 残差公式：f(θ) = ||p_i - c|| - (r + δ_i)，其中δ_i为生理偏差
    "PHYSIOLOGICAL_BIAS_MEDIAN": {
        "pupil_center": 0.0002,           # 瞳孔中心偏差中值 0.2mm (0.1-0.3mm范围的中值)
        "iris_boundaries": 0.00085,       # 虹膜边界偏差中值 0.85mm (0.5+1.2)/2
        "eye_contours": 0.0025            # 眼睛轮廓偏差中值 2.5mm (1.0+4.0)/2
    }
}

# ==================== 数据质量阈值 ====================
MEDIAPIPE_QUALITY_THRESHOLDS = {
    "HIGH": {"min_points": 8, "confidence": 0.9},   # 高质量：≥8点 & 置信度≥0.9
    "MEDIUM": {"min_points": 4, "confidence": 0.7}, # 中质量：≥4点 & 置信度≥0.7
    "LOW": {"max_points": 3}                        # 低质量：≤3点
}

# ==================== 基础配置 ====================
@dataclass
class BaseFittingConfig:
    """所有拟合策略共用的参数"""
    
    max_trials: int = 100                 # 半径拟合阈值，大于100次默认为半径已经趋于稳定，直接使用已有半径
    min_samples: int = 8                  # 最少需要的点数（例：低于8点直接失败）
    confidence_threshold: float = 0.8     # 数据整体置信度下限
    
    # 权重配置
    min_weight: float = 0.1               # 权重下限（防止点被完全丢弃）
    geometric_weight_sigma: float = 0.001 # 几何权重高斯σ（例：σ=0.001时，残差>0.003权重接近0）


# ==================== LM配置 ====================
@dataclass
class LMAlgorithmConfig:
    """Levenberg-Marquardt算法配置"""
    
    lambda_lm: float = 0.01               # 初始阻尼因子
    lambda_factor: float = 10.0           # 调整倍率（例：残差变差时 *10）
    min_lambda: float = 1e-8              # 阻尼下限
    max_lambda: float = 1e8               # 阻尼上限
    max_iterations: int = 30              # 最大迭代次数
    
    # 收敛判定
    param_tolerance: float = 5e-6         # 参数变化阈值（例：小于5e-6认为收敛）
    residual_tolerance: float = 5e-6      # 残差变化阈值（例：小于5e-6认为收敛）
    max_step: float = 0.01                # 单次迭代最大步长（避免跳跃过大）
    geometric_weight_factor: float = 10   # 启用几何权重的最低阈值因子
    
    # 误差容忍度
    residual_error_tolerance: float = 0.0005       # 残差误差容忍度（例：0.5 mm）

# ==================== 深度约束配置 ====================
@dataclass
class ConstraintConfig:
    """深度约束参数"""
    
    lambda_depth: float = 1000.0          # 深度惩罚系数 λ_depth
    lambda_radius: float = 20.0         # 半径惩罚系数 λ_radius
    radius_prior: float = 0.012         # 半径先验值（12 mm）
    depth_offset_epsilon: float = 0.0002  # 偏移ε（0.2 mm，避免数值抖动）
    softplus_tau: float = 0.001           # softplus平滑参数τ
    constraint_penalty_factor: float = 1000.0  # 约束惩罚因子

# ==================== RANSAC配置 ====================
@dataclass
class RANSACConfig:
    """RANSAC采样配置"""
    
    max_iterations: int = 50              # 最大迭代次数
    threshold: float = 0.0005             # 内点阈值（残差 < 0.5 mm 认为内点）
    min_inlier_ratio: float = 0.7         # 内点比例要求（例：≥70%为有效模型）
    
    # 点数限制
    min_iris_points: int = 3              # 至少需要3个虹膜点
    max_iris_points: int = 4             # 最多使用4个虹膜点（避免过拟合）
    min_contour_points: int = 4           # 至少需要4个轮廓点
    max_contour_points: int = 10          # 最多使用10个轮廓点（避免过拟合）
    
    # 收敛判定
    inlier_ratio_threshold: float = 0.9   # 内点比例阈值（例：≥90%可以直接判断为收敛）

# ==================== 策略配置 ====================
@dataclass
class ConstrainedLMConfig(BaseFittingConfig):
    """高质量数据：直接使用约束LM"""
    
    strategy_name: str = "constrained_lm"
    lm_config: LMAlgorithmConfig = field(default_factory=LMAlgorithmConfig)
    constraint_config: ConstraintConfig = field(default_factory=ConstraintConfig)

@dataclass
class ConstrainedRANSACLMConfig(BaseFittingConfig):
    """中质量数据：RANSAC初筛 + 约束LM"""
    
    strategy_name: str = "constrained_ransac_lm"
    lm_config: LMAlgorithmConfig = field(default_factory=LMAlgorithmConfig)
    constraint_config: ConstraintConfig = field(default_factory=ConstraintConfig)
    ransac_config: RANSACConfig = field(default_factory=RANSACConfig)

@dataclass
class EllipsoidFallbackConfig(BaseFittingConfig):
    """低质量数据：回退到椭球模型"""
    
    strategy_name: str = "ellipsoid_fallback"
    ransac_config: RANSACConfig = field(default_factory=RANSACConfig)

@dataclass
class FailConfig(BaseFittingConfig):
    """极低质量数据：直接失败"""
    
    strategy_name: str = "fail"
    action: str = "return_failure"        # 直接返回失败
    reason: str = "数据质量过低"

# ==================== 权重配置类 ====================
@dataclass
class WeightConfig:
    """权重管理专用配置"""
    
    # === 权重融合参数 ===
    gaussian_fusion_sigma: float = 0.5    # 高斯融合标准差
    weight_fusion_enabled: bool = True    # 启用权重融合
    min_weight: float = 0.1               # 最小权重值
    normalization: bool = False            # 是否启用权重归一化

# ==================== 策略映射 ====================
FITTING_STRATEGIES = {
    "constrained_lm": ConstrainedLMConfig(),
    "constrained_ransac_lm": ConstrainedRANSACLMConfig(),
    "ellipsoid_fallback": EllipsoidFallbackConfig(),
    "fail": FailConfig()
}

DEFAULT_FITTING_STRATEGY = ConstrainedRANSACLMConfig()
