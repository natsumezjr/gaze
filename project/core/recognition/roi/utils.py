"""
ROI 几何工具函数：与具体语义无关的椭圆采样与几何关系计算。

注意：
- 仅包含纯几何运算，不依赖上层语义/质量模块。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from project.data.data_models import Ellipse2D, Point2D


def sample_ellipse_points(el: Ellipse2D, num_points: int = 72) -> np.ndarray:
    """
    在椭圆周线上均匀采样 num_points 个点，返回形状 (N,2)，每行 [x, y]。

    椭圆参数：
    - el.cx, el.cy: 椭圆中心
    - el.major_axis, el.minor_axis: 全长轴
    - el.angle_deg: 椭圆旋转角度（度）
    """
    import math

    a = float(el.major_axis) * 0.5
    b = float(el.minor_axis) * 0.5
    if a <= 0 or b <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    theta = np.linspace(0.0, 2.0 * math.pi, num_points, endpoint=False)
    xs = a * np.cos(theta)
    ys = b * np.sin(theta)

    ang = math.radians(float(el.angle_deg))
    c, s = math.cos(ang), math.sin(ang)

    gx = float(el.cx) + xs * c - ys * s
    gy = float(el.cy) + xs * s + ys * c
    return np.stack([gx, gy], axis=1).astype(np.float32)


def points_inside_ellipse_ratio(pts: np.ndarray, el: Ellipse2D) -> float:
    """
    计算 pts 中落在椭圆 el 内部的比例。

    - pts: 形状 (N,2) 的点集，坐标系与 el 一致。
    - el: 目标椭圆。
    """
    import math

    if pts.size == 0:
        return 0.0

    a = float(el.major_axis) * 0.5
    b = float(el.minor_axis) * 0.5
    if a <= 0 or b <= 0:
        return 0.0

    ang = math.radians(float(el.angle_deg))
    c, s = math.cos(ang), math.sin(ang)

    dx = pts[:, 0] - float(el.cx)
    dy = pts[:, 1] - float(el.cy)
    x_p = dx * c + dy * s
    y_p = -dx * s + dy * c

    val = (x_p / (a + 1e-6)) ** 2 + (y_p / (b + 1e-6)) ** 2
    inside = val <= 1.0 + 1e-3
    return float(np.count_nonzero(inside)) / float(len(pts))


def compute_pupil_center_inside_iris_score(
    pupil_center: Point2D,
    iris_ellipse: Ellipse2D,
) -> Tuple[float, float]:
    """
    计算 pupil_center 在 iris_ellipse 主轴坐标系下的归一化二次型值 d2 以及对应得分。

    定义：
        d2 = (x/a)^2 + (y/b)^2
      - d2 <= 1 说明在椭圆内部；
      - score = max(0, 1 - d2)，d2=0 时得 1，d2=1 时得 0。
    """
    import math

    a = float(iris_ellipse.major_axis) * 0.5
    b = float(iris_ellipse.minor_axis) * 0.5
    if a <= 0 or b <= 0:
        return 0.0, 1e3

    ang = math.radians(float(iris_ellipse.angle_deg))
    c, s = math.cos(ang), math.sin(ang)

    dx = float(pupil_center.x) - float(iris_ellipse.cx)
    dy = float(pupil_center.y) - float(iris_ellipse.cy)

    x_p = dx * c + dy * s
    y_p = -dx * s + dy * c

    d2 = float((x_p / (a + 1e-6)) ** 2 + (y_p / (b + 1e-6)) ** 2)
    score = float(max(0.0, 1.0 - d2))
    return score, d2


__all__ = [
    "sample_ellipse_points",
    "points_inside_ellipse_ratio",
    "compute_pupil_center_inside_iris_score",
]