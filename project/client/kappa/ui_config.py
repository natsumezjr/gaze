"""UI配色配置模块 - 统一管理所有颜色和样式配置"""
from enum import Enum
from typing import Dict, Tuple, List

class BackgroundColor(Enum):
    """背景色枚举"""
    BLACK = "black"
    GRAY = "gray"
    WHITE = "white"

# 背景色配置
BACKGROUND_COLOR_MAP = {
    BackgroundColor.BLACK: "#000000",
    BackgroundColor.GRAY: "#808080",
    BackgroundColor.WHITE: "#FFFFFF"
}

# 文字颜色配置（适配不同背景）
TEXT_COLOR_MAP = {
    BackgroundColor.BLACK: "#FFFFFF",
    BackgroundColor.GRAY: "#000000",
    BackgroundColor.WHITE: "#000000"
}

# 校准点颜色配置（9个点）
CALIBRATION_POINT_COLORS = [
    ("#DC3545", "#C82333"),  # 左上角
    ("#20C997", "#1E7E34"),  # 上边中点
    ("#0D6EFD", "#0A58CA"),  # 右上角
    ("#198754", "#155724"),  # 左边中点
    ("#FFC107", "#FF9800"),  # 屏幕中心
    ("#FD7E14", "#E8590C"),  # 右边中点
    ("#6610F2", "#520DC2"),  # 左下角
    ("#E91E63", "#C2185B"),  # 下边中点
    ("#009688", "#00796B")   # 右下角
]

# 校准点配置 - 9点布局：四个角 + 四条边的中点 + 屏幕正中间
CALIBRATION_POINTS = [
    (0, 0),      # 左上角
    (50, 0),     # 上边中点
    (100, 0),    # 右上角
    (0, 50),     # 左边中点
    (50, 50),    # 屏幕正中间
    (100, 50),   # 右边中点
    (0, 100),    # 左下角
    (50, 100),   # 下边中点
    (100, 100)   # 右下角
]

# 动态效果配置
ANIMATION_CONFIG = {
    "pulse_min_scale": 0.8,      # 脉冲最小缩放比例
    "pulse_max_scale": 1.2,      # 脉冲最大缩放比例
    "pulse_duration": 1000,      # 脉冲周期（毫秒）
    "glow_intensity_min": 0.5,    # 发光最小强度
    "glow_intensity_max": 1.0,   # 发光最大强度
    "glow_color": "#FFFF00",     # 发光颜色（黄色）
    "detection_range_px": 50,    # 检测范围（像素），可配置
    "detection_range_ratio": 0.05 # 检测范围（相对屏幕比例），可配置
}
