"""
使用 Windows 的 pywin32 获取主显示器物理尺寸（毫米），并获取当前屏幕分辨率，
将分辨率直接写入 SCREEN_WITH_RGBD 的一个字段（resolution_px）。

约定：右手系，Z 轴指向前方，X 轴向右，Y 轴向下；屏幕中心位于原点，屏幕平面位于 z=0。
若无法获取物理尺寸，则回退到默认宽 0.6 m、高 0.34 m。
"""

from __future__ import annotations

import numpy as np


DEFAULT_SCREEN_WIDTH_M: float = 0.60
DEFAULT_SCREEN_HEIGHT_M: float = 0.34


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


_SCREEN_WIDTH_M, _SCREEN_HEIGHT_M = _get_primary_screen_physical_size_m()
_HALF_W = _SCREEN_WIDTH_M / 2.0
_HALF_H = _SCREEN_HEIGHT_M / 2.0

# 可选：通过“米”为单位对屏幕平面整体进行平移
_CAP_X_OFFSET_M = 0.0
_CAP_Y_OFFSET_M = 0.005

# 当前分辨率（像素）
_CURRENT_RES_W, _CURRENT_RES_H = _get_current_resolution_px()

# 屏幕几何（中心为原点、屏幕平面 z=0）
SCREEN_WITH_RGBD = {
    "normal": np.array([0.0, 0.0, 1.0]),
    "point": np.array([0.0, 0.0, 0.0]),
    "x_axis": np.array([1.0, 0.0, 0.0]),
    "y_axis": np.array([0.0, 1.0, 0.0]),
    "top_left": np.array([-_HALF_W + _CAP_X_OFFSET_M, -_HALF_H + _CAP_Y_OFFSET_M, 0.0]),
    "top_right": np.array([_HALF_W + _CAP_X_OFFSET_M, -_HALF_H + _CAP_Y_OFFSET_M, 0.0]),
    "bottom_left": np.array([-_HALF_W + _CAP_X_OFFSET_M, _HALF_H + _CAP_Y_OFFSET_M, 0.0]),
    "bottom_right": np.array([_HALF_W + _CAP_X_OFFSET_M, _HALF_H + _CAP_Y_OFFSET_M, 0.0]),
    "width_m": float(_SCREEN_WIDTH_M),
    "height_m": float(_SCREEN_HEIGHT_M),
    "resolution_px": (_CURRENT_RES_W, _CURRENT_RES_H),
}


# 拟合算法参数配置
FITTING_ALGORITHM_CONFIG = {
    # RANSAC参数
    "max_trials": 100,                    # 最大采样次数
    "ransac_threshold": 0.0015,          # 初始RANSAC距离阈值(m)
    "min_inlier_ratio": 0.7,              # 最小内点比例
    "min_samples": 4,  
    "ransac_max_iterations": 50,         # 最大迭代次数
    
    # 牛顿-高斯算法参数
    "convergence_tol": 1e-6,              # 参数收敛容差
    "residual_tol": 1e-8,                 # 残差收敛容差
    "newton_max_iterations": 50,         # 牛顿-高斯最大迭代次数
    
    # 自适应阈值参数
    "threshold_adjustment_factor": 1.5,   # 阈值调整因子
    "geometric_residual_threshold": 0.0008, # 几何残差阈值(m)
    
    # 采样策略参数
    "points_per_eye": 4,                  # 每只眼睛采样点数
    "sampling_interval_ms": 33,           # 采样间隔(ms)
    
    # 参数融合参数
    "fusion_weight_sigma": 0.5,           # 融合权重标准差
    "outlier_rejection_ratio": 0.2,       # 异常值剔除比例
}

def main() -> None:
    print("SCREEN_WITH_RGBD:", SCREEN_WITH_RGBD)


if __name__ == "__main__":
    main()



