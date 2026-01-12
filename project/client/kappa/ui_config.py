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

# 校准点颜色配置（9个点）- 使用提供的配色方案
CALIBRATION_POINT_COLORS = [
    ("#E67DAF", "#F9CCE0"),  # 左上角 - 粉色系
    ("#20AEDD", "#9ACAE8"),  # 上边中点 - 蓝色系
    ("#F9DF91", "#FFE8BF"),  # 右上角 - 黄色系
    ("#FCB2AF", "#FEF1F3"),  # 左边中点 - 红色系
    ("#75C8AE", "#D2E9CB"),  # 屏幕中心 - 绿色系
    ("#9BDFFD", "#E1F3FB"),  # 右边中点 - 浅蓝色系
    ("#FC9871", "#FFE2CE"),  # 左下角 - 橙色系
    ("#4C9BE6", "#A8CFE8"),  # 下边中点 - 深蓝色系
    ("#B0DC66", "#E8F3E1")   # 右下角 - 浅绿色系
]

# 屏幕边距配置
SCREEN_MARGIN_RATIO = 0.02  # 2%边距，确保点在屏幕内可见

# 圆圈大小配置（相对于屏幕宽度）
CIRCLE_SIZE_RATIO = 0.005  # 圆圈大小为屏幕宽度的1%（更小更精致）

# 校准点配置 - 9点布局：四个角 + 四条边的中点 + 屏幕正中间
# 使用5%-95%范围，向中心靠拢
CALIBRATION_POINTS = [
    (5, 5),      # 左上角
    (50, 5),     # 上边中点
    (95, 5),     # 右上角
    (5, 50),     # 左边中点
    (50, 50),    # 屏幕正中间
    (95, 50),    # 右边中点
    (5, 90),     # 左下角
    (50, 90),    # 下边中点
    (95, 90)     # 右下角
]

# 动态效果配置
ANIMATION_CONFIG = {
    "pulse_min_scale": 0.7,      # 脉冲最小缩放比例（减小变化幅度）
    "pulse_max_scale": 1.1,      # 脉冲最大缩放比例（减小变化幅度）
    "pulse_duration": 1500,      # 脉冲周期（毫秒）- 更慢的律动
    "glow_intensity_min": 0.5,    # 发光最小强度
    "glow_intensity_max": 1.0,   # 发光最大强度
    "glow_color": "#FFFF00",     # 发光颜色（黄色）
    "detection_range_px": 50,    # 检测范围（像素），可配置
    "detection_range_ratio": 0.05 # 检测范围（相对屏幕比例），可配置
}
