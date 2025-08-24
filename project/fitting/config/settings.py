"""
使用 Windows 的 pywin32 获取主显示器物理尺寸（毫米），并获取当前屏幕分辨率，
将分辨率直接写入 SCREEN_WITH_RGBD 的一个字段（resolution_px）。

约定：右手系，Z 轴指向前方，X 轴向右，Y 轴向下；屏幕中心位于原点，屏幕平面位于 z=0。
若无法获取物理尺寸，则回退到默认宽 0.6 m、高 0.34 m。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np


DEFAULT_SCREEN_WIDTH_M: float = 0.60
DEFAULT_SCREEN_HEIGHT_M: float = 0.34


@dataclass
class ScreenConfig:
    """屏幕配置类"""
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    point: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    x_axis: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    y_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    width_m: float = 0.60
    height_m: float = 0.34
    resolution_px: tuple[int, int] = (0, 0)
    
    def __post_init__(self):
        """初始化后计算屏幕四角坐标"""
        half_w = self.width_m / 2.0
        half_h = self.height_m / 2.0
        cap_x_offset = 0.0
        cap_y_offset = 0.005
        
        self.top_left = np.array([-half_w + cap_x_offset, -half_h + cap_y_offset, 0.0])
        self.top_right = np.array([half_w + cap_x_offset, -half_h + cap_y_offset, 0.0])
        self.bottom_left = np.array([-half_w + cap_x_offset, half_h + cap_y_offset, 0.0])
        self.bottom_right = np.array([half_w + cap_x_offset, half_h + cap_y_offset, 0.0])
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式以保持向后兼容"""
        return {
            "normal": self.normal,
            "point": self.point,
            "x_axis": self.x_axis,
            "y_axis": self.y_axis,
            "top_left": self.top_left,
            "top_right": self.top_right,
            "bottom_left": self.bottom_left,
            "bottom_right": self.bottom_right,
            "width_m": self.width_m,
            "height_m": self.height_m,
            "resolution_px": self.resolution_px,
        }


def _get_primary_screen_physical_size_m() -> tuple[float, float]:
    """返回主显示器物理宽高（单位：米）。

    优先通过 pywin32 的设备能力接口读取 HORZSIZE/VERTSIZE（毫米）。
    若 pywin32 不可用或读取失败，回退到默认值。
    """
    try:
        import win32gui  # type: ignore
        import win32print  # type: ignore
    except Exception:
        return DEFAULT_SCREEN_WIDTH_M, DEFAULT_SCREEN_HEIGHT_M

    hdc = None
    try:
        hdc = win32gui.GetDC(0)
        if not hdc:
            return DEFAULT_SCREEN_WIDTH_M, DEFAULT_SCREEN_HEIGHT_M

        HORZSIZE = 4  # 设备以毫米表示的物理宽度
        VERTSIZE = 6  # 设备以毫米表示的物理高度

        width_mm = float(win32print.GetDeviceCaps(hdc, HORZSIZE))
        height_mm = float(win32print.GetDeviceCaps(hdc, VERTSIZE))
    except Exception:
        return DEFAULT_SCREEN_WIDTH_M, DEFAULT_SCREEN_HEIGHT_M
    finally:
        if hdc is not None:
            try:
                import win32gui as _w32g  # type: ignore
                _w32g.ReleaseDC(0, hdc)
            except Exception:
                pass

    if width_mm <= 0 or height_mm <= 0 or not np.isfinite(width_mm) or not np.isfinite(height_mm):
        return DEFAULT_SCREEN_WIDTH_M, DEFAULT_SCREEN_HEIGHT_M

    return width_mm / 1000.0, height_mm / 1000.0


def _get_current_resolution_px() -> tuple[int, int]:
    """获取当前主显示器分辨率（像素）。优先使用 pywin32，失败回退到 ctypes。"""
    # 方案一：pywin32
    try:
        import win32api  # type: ignore
        width = int(win32api.GetSystemMetrics(0))  # SM_CXSCREEN
        height = int(win32api.GetSystemMetrics(1))  # SM_CYSCREEN
        if width > 0 and height > 0:
            return width, height
    except Exception:
        pass
    # 方案二：ctypes（不依赖 pywin32）
    try:
        import ctypes
        user32 = ctypes.windll.user32
        width = int(user32.GetSystemMetrics(0))
        height = int(user32.GetSystemMetrics(1))
        if width > 0 and height > 0:
            return width, height
    except Exception:
        pass
    return (0, 0)


# 获取屏幕配置
_SCREEN_WIDTH_M, _SCREEN_HEIGHT_M = _get_primary_screen_physical_size_m()
_CURRENT_RES_W, _CURRENT_RES_H = _get_current_resolution_px()

# 创建屏幕配置实例
SCREEN_CONFIG = ScreenConfig(
    width_m=_SCREEN_WIDTH_M,
    height_m=_SCREEN_HEIGHT_M,
    resolution_px=(_CURRENT_RES_W, _CURRENT_RES_H)
)



# 向后兼容的字典格式
SCREEN_WITH_RGBD = SCREEN_CONFIG.to_dict()


def main() -> None:
    print("SCREEN_CONFIG:", SCREEN_CONFIG)



if __name__ == "__main__":
    main()