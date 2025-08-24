"""
拟合策略配置模块
整合所有现有的配置定义，避免重复
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import numpy as np

# 解剖学约束参数（基于main.py中的科学阈值）
ANATOMICAL_CONSTRAINTS = {
    # 瞳孔到眼球表面的距离范围（基于main.py中的0.1-0.3mm）
    "PUPIL_TO_SURFACE_MIN": 0.0001,      # 0.1mm
    "PUPIL_TO_SURFACE_MAX": 0.0003,      # 0.3mm
    
    # 虹膜边界到眼球表面的距离范围（基于main.py中的0.5-1.2mm）
    "IRIS_TO_SURFACE_MIN": 0.0005,       # 0.5mm
    "IRIS_TO_SURFACE_MAX": 0.0012,       # 1.2mm
    
    # 眼睛轮廓到眼球表面的距离范围（基于main.py中的1.0-2.0mm）
    "CONTOUR_TO_SURFACE_MIN": 0.001,     # 1.0mm
    "CONTOUR_TO_SURFACE_MAX": 0.002,     # 2.0mm
    
    # 眼球半径范围（基于解剖学标准）
    "EYEBALL_RADIUS_MIN": 0.008,         # 8mm（最小半径）
    "EYEBALL_RADIUS_MAX": 0.020,         # 20mm（最大半径）
    "EYEBALL_RADIUS_DEFAULT": 0.012      # 12mm（默认半径）
}

# 数据质量阈值（基于Mediapipe特点）
MEDIAPIPE_QUALITY_THRESHOLDS = {
    "HIGH_QUALITY_MIN_POINTS": 8,         # 高质量最小点数
    "MEDIUM_QUALITY_MIN_POINTS": 4,       # 中质量最小点数
    "LOW_QUALITY_MAX_POINTS": 3,          # 低质量最大点数
    "HIGH_QUALITY_CONFIDENCE": 0.9,       # 高质量置信度阈值
    "MEDIUM_QUALITY_CONFIDENCE": 0.7,     # 中质量置信度阈值
}

@dataclass
class BaseFittingConfig:
    """基础拟合配置类"""
    # === 算法参数 ===
    max_trials: int = 100
    max_ransac_iterations: int = 10
    ransac_threshold: float = 0.0015
    min_samples: int = 8
    confidence_threshold: float = 0.8
    
    # === 采样策略 ===
    min_iris_points: int = 3
    min_contour_points: int = 4
    max_contour_points: int = 10
    
    # === 优化参数 ===
    max_iterations: int = 30
    param_tolerance: float = 1e-6
    residual_tolerance: float = 1e-8
    max_step: float = 0.01
    
    # === 权重参数 ===
    min_weight: float = 0.1
    normalization: bool = True
    
    # === 残差驱动权重调整参数 ===
    residual_cutoff_threshold: float = 1e-6
    residual_protection_factor: float = 10.0
    geometric_weight_cap: float = 0.8
    
    # === 自适应阈值参数 ===
    threshold_adjustment_factor: float = 1.5
    geometric_residual_threshold: float = 0.0008
    
    # === 参数融合参数 ===
    fusion_weight_sigma: float = 0.5
    outlier_rejection_ratio: float = 0.2
    
    # === Softplus平滑配置 ===
    softplus_tau: float = 0.001                   # 平滑参数，控制平滑程度
    softplus_epsilon: float = 1e-8                # 数值保护参数，防止除零错误
    
    # === LM算法配置 ===
    lambda_lm: float = 0.01                       # LM阻尼因子初始值
    lambda_factor: float = 10.0                   # LM阻尼因子调整倍数
    min_lambda: float = 1e-8                      # 最小阻尼因子
    max_lambda: float = 1e8                       # 最大阻尼因子
    trust_region_factor: float = 0.1              # 信赖域因子
    convergence_ratio_threshold: float = 0.75     # 收敛比例阈值
    max_lm_iterations: int = 50                   # LM最大迭代次数
    lm_param_tolerance: float = 1e-6              # LM参数收敛阈值
    lm_residual_tolerance: float = 1e-6           # LM残差收敛阈值
    
    # === 约束配置 ===
    constraint_penalty_factor_1: float = 1000.0   # 点约束惩罚因子λ1
    constraint_penalty_factor_2: float = 1000.0   # 深度约束惩罚因子λ2
    depth_offset_epsilon: float = 0.0002          # 深度偏移量ε（0.2mm）
    constraint_dimension: int = 1                 # 约束维度
    
    # === 权重配置 ===
    geometric_weight_factor: float = 10.0         # 几何权重启用因子
    weight_fusion_enabled: bool = True            # 启用权重融合
    gaussian_fusion_sigma: float = 0.5            # 高斯融合标准差
    
    # === RANSAC配置 ===
    samples_tolerance_ratio: float = 0.7          # RANSAC内点比例阈值

@dataclass
class ConstrainedLMConfig(BaseFittingConfig):
    """约束LM拟合策略配置 - 高质量数据"""
    strategy_name: str = "constrained_lm"
    sampling_method: str = "full"                 # 全部采样
    optimization_method: str = "lm"                # 直接LM优化
    constraint_enforcement: str = "strict"         # 严格约束
    max_lm_iterations: int = 100                  # 更多迭代次数
    lm_param_tolerance: float = 1e-6              # 更高精度
    use_sphere_only: bool = True                  # 仅使用球体拟合
    fallback_to_ellipsoid: bool = False

@dataclass
class ConstrainedRANSACLMConfig(BaseFittingConfig):
    """约束RANSAC+LM拟合策略配置 - 中质量数据"""
    strategy_name: str = "constrained_ransac_lm"
    sampling_method: str = "ransac"               # RANSAC采样
    optimization_method: str = "ransac_lm"         # RANSAC + LM
    constraint_enforcement: str = "required"       # 必须约束
    max_ransac_iterations: int = 50               # RANSAC最大迭代次数
    max_lm_iterations: int = 50                   # LM最大迭代次数
    use_sphere_only: bool = True                  # 先尝试球体拟合
    fallback_to_ellipsoid: bool = True            # 拟合效果差时使用椭球

@dataclass
class EllipsoidFallbackConfig(BaseFittingConfig):
    """椭球回退策略配置"""
    strategy_name: str = "ellipsoid_fallback"
    sampling_method: str = "ransac"               # RANSAC采样
    optimization_method: str = "ransac_lm"         # RANSAC + LM
    constraint_enforcement: str = "required"       # 必须约束
    use_sphere_only: bool = False                 # 直接使用椭球拟合
    fallback_to_ellipsoid: bool = False

@dataclass
class FailConfig(BaseFittingConfig):
    """失败策略配置"""
    strategy_name: str = "fail"
    sampling_method: str = "none"
    optimization_method: str = "none"
    constraint_enforcement: str = "none"
    max_iterations: int = 0
    action: str = "return_failure"
    reason: str = "数据质量过低"

@dataclass
class WeightConfig:
    """权重配置参数"""
    gaussian_fusion_sigma: float = 0.5            # 高斯融合标准差
    min_weight: float = 0.1                       # 最小权重值
    normalization: bool = False                    # 是否启用权重归一化
    weight_fusion_enabled: bool = True            # 启用权重融合

# 策略配置表
FITTING_STRATEGIES = {
    "constrained_lm": ConstrainedLMConfig(),
    "constrained_ransac_lm": ConstrainedRANSACLMConfig(),
    "ellipsoid_fallback": EllipsoidFallbackConfig(),
    "fail": FailConfig()
}

# 默认配置
DEFAULT_FITTING_STRATEGY = ConstrainedRANSACLMConfig()