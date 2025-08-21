"""
初始权重配置模块
基于 zjr/_int_center_fitter.md 中的解剖学先验权重表
"""

# 解剖学先验权重参数表（基于 zjr/_int_center_fitter.md）
ANATOMICAL_WEIGHT_PARAMS = {
    # 解剖结构: (偏移范围(mm), 中值d_i(mm), σ(mm), w_anatomy)
    "pupil_center": {
        "offset_range": (0.1, 0.3),      # 偏移范围(mm)
        "median_distance": 0.2,          # 中值d_i(mm)
        "sigma": 1.5,                    # σ(mm)
        "w_anatomy": 0.991               # 预计算的解剖学权重
    },
    "iris_boundary": {
        "offset_range": (0.5, 1.2),
        "median_distance": 0.85,
        "sigma": 1.5,
        "w_anatomy": 0.839
    },
    "eye_contour": {
        "offset_range": (1.0, 2.0),
        "median_distance": 1.5,
        "sigma": 1.5,
        "w_anatomy": 0.607
    }
}

# 几何残差权重的初始化标准差（基于各解剖结构的中值距离）
GEOMETRIC_WEIGHT_PARAMS = {
    "pupil_center": {
        "initial_sigma": 0.2,            # mm，用于几何残差权重计算
        "threshold": 1.0,                # mm，距离阈值
        "min_weight": 0.3                # 最小权重保护
    },
    "iris_boundary": {
        "initial_sigma": 0.85,
        "threshold": 1.5,
        "min_weight": 0.2
    },
    "eye_contour": {
        "initial_sigma": 1.5,
        "threshold": 2.0,
        "min_weight": 0.2
    }
}




# 全局权重配置
GLOBAL_WEIGHT_CONFIG = {
    "min_weight": 0.1,                   # 全局最小权重保护
    "snr_weight_default": 1.0,           # 默认信号质量权重
    "normalization": True,               # 是否进行权重归一化
    "sigma_global": 1.5                  # 全局标准差参数
}
