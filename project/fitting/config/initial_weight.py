"""
初始权重配置模块
基于 zjr/_int_center_fitter.md 中的解剖学先验权重表
"""
from dataclasses import dataclass

@dataclass
class AnatomicalWeightParams:
    """解剖学权重参数"""
    offset_range: tuple[float, float]
    median_distance: float
    sigma: float
    w_anatomy: float


@dataclass
class GeometricWeightParams:
    """几何权重参数"""
    initial_sigma: float
    threshold: float
    min_weight: float


@dataclass
class GlobalWeightConfig:
    """全局权重配置"""
    min_weight: float = 0.1
    snr_weight_default: float = 1.0
    normalization: bool = True
    sigma_global: float = 1.5


# 解剖学先验权重参数表（基于 zjr/_int_center_fitter.md）
ANATOMICAL_WEIGHT_PARAMS = {
    "pupil_center": AnatomicalWeightParams(
        offset_range=(0.0001, 0.0003),
        median_distance=0.0002,
        sigma=0.0015,
        w_anatomy=0.9911505004882849
    ),
    "iris_boundary": AnatomicalWeightParams(
        offset_range=(0.0005, 0.0012),
        median_distance=0.00085,
        sigma=0.0015,
        w_anatomy=0.851670507229441
    ),
    "eye_contour": AnatomicalWeightParams(
        offset_range=(0.001, 0.004),
        median_distance=0.0025,
        sigma=0.0015,
        w_anatomy=0.607
    )
}

# 几何残差权重的初始化标准差（基于各解剖结构的中值距离）
GEOMETRIC_WEIGHT_PARAMS = {
    "pupil_center": GeometricWeightParams(
        initial_sigma=0.0001,
        threshold=0.0010,
        min_weight=0.3
    ),
    "iris_boundary": GeometricWeightParams(
        initial_sigma=0.00035,
        threshold=0.0010,
        min_weight=0.2
    ),
    "eye_contour": GeometricWeightParams(
        initial_sigma=0.0015,
        threshold=0.0010,
        min_weight=0.2
    )
}

# 全局权重配置
GLOBAL_WEIGHT_CONFIG = GlobalWeightConfig()