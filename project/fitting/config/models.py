"""数据模型定义模块"""
from typing import Dict, List, Union, Optional
from dataclasses import dataclass
import numpy as np


# 数据模型定义
class EllipsoidParams:
    """
    椭球参数类型：
    - axes: 三主轴长度 (a, b, c)，单位 m，np.ndarray(3,)
    - rotation: 世界到椭球主轴坐标的旋转矩阵 R，np.ndarray(3,3)
    - center: 椭球中心坐标，np.ndarray(3,)
    """
    def __init__(self):
        self.axes = np.array([0.012, 0.012, 0.012])
        self.rotation = np.eye(3)
        self.center = np.array([0.0, 0.0, 0.0])
    
    def get_axes(self) -> np.ndarray:
        return self.axes
    
    def get_rotation(self) -> np.ndarray:
        return self.rotation
    
    def get_center(self) -> np.ndarray:
        return self.center
    
    def __str__(self) -> str:
        """格式化输出椭球参数信息"""
        # 格式化主轴长度
        axes_str = f"({self.axes[0]:.3f}, {self.axes[1]:.3f}, {self.axes[2]:.3f})"
        
        # 格式化中心点
        center_str = f"({self.center[0]:.3f}, {self.center[1]:.3f}, {self.center[2]:.3f})"
        
        # 检查是否为球体（三个轴长度相等）
        is_sphere = np.allclose(self.axes, self.axes[0], atol=1e-6)
        shape_type = "球体" if is_sphere else "椭球"
        
        # 构建椭球参数字符串
        ellipsoid_str = (
            f"椭球参数 ({shape_type}):\n"
            f"  📏 主轴长度: {axes_str} m\n"
            f"  📍 中心坐标: {center_str} m\n"
            f"  🔄 旋转矩阵: {self.rotation.shape[0]}x{self.rotation.shape[1]} 单位矩阵" if np.allclose(self.rotation, np.eye(3)) else f"  🔄 旋转矩阵: {self.rotation.shape[0]}x{self.rotation.shape[1]} 非单位矩阵"
        )
        
        return ellipsoid_str

class Plane:
    """
    平面类型：
    - normal: 平面法向量，np.ndarray(3,)
    - point: 平面坐标系的原点在3D世界坐标系中的坐标，np.ndarray(3,)
    - x_axis: 平面坐标系的x轴向量，np.ndarray(3,)
    - y_axis: 平面坐标系的y轴向量，np.ndarray(3,)
    """
    def __init__(self, rgb_d=False):
        from project.fitting.config.settings import SCREEN_WITH_RGBD
        
        if rgb_d:
            self.normal = SCREEN_WITH_RGBD["normal"]
            self.point = SCREEN_WITH_RGBD["point"]
            self.x_axis = SCREEN_WITH_RGBD["x_axis"]
            self.y_axis = SCREEN_WITH_RGBD["y_axis"]
            self.top_left = SCREEN_WITH_RGBD["top_left"]
            self.top_right = SCREEN_WITH_RGBD["top_right"]
            self.bottom_left = SCREEN_WITH_RGBD["bottom_left"]
            self.bottom_right = SCREEN_WITH_RGBD["bottom_right"]
            # 补充物理尺寸（米）
            self.width_m = float(SCREEN_WITH_RGBD.get("width_m", 0.0))
            self.height_m = float(SCREEN_WITH_RGBD.get("height_m", 0.0))
        else:
            self.normal = np.array([0,0,1])
            self.point = np.array([0,0,0])
            self.x_axis = np.array([1,0,0])
            self.y_axis = np.array([0,1,0])
            self.top_left = np.array([-0.30, 0.005, 0])
            self.top_right = np.array([0.30, 0.005, 0])
            self.bottom_left = np.array([-0.30, 0.345, 0])
            self.bottom_right = np.array([0.30, 0.345, 0])
            # 默认物理尺寸（米）
            self.width_m = 0.60
            self.height_m = 0.34
    
    def _intersection_of_vector_and_plane(self, vector: np.ndarray) -> Optional[np.ndarray]:
        '''
        计算向量与平面的交点（假设向量起点为世界原点）。
        Args:
            vector: 方向向量 (3,)
        Returns:
            交点 np.ndarray(3,)，若无唯一交点（平行或共面）返回 None
        '''
        v = np.asarray(vector, dtype=float).reshape(3)
        n = np.asarray(self.normal, dtype=float).reshape(3)
        p0 = np.asarray(self.point, dtype=float).reshape(3)

        denom = float(np.dot(n, v))
        if np.isclose(denom, 0.0):
            # 平行：若原点在平面上则无唯一交点（共面），否则无交点
            return None

        t = float(np.dot(n, p0) / denom)
        return v * t

    def _point_3d_to_2d(self, point: np.ndarray) -> np.ndarray:
        '''
        将3D点转换成在平面2D坐标系下的坐标
        Args:
            point: 3D点
        Returns:
            2D坐标
        '''
        x = np.dot(point - self.point, self.x_axis)
        y = np.dot(point - self.point, self.y_axis)
        return np.array([x, y])
    
    def intersection_on_plane(self, vector: np.ndarray) -> Optional[np.ndarray]:
        '''
        计算向量与平面的交点
        Args:
            vector: 向量
        Returns:
            交点（在平面2D坐标系下）
        '''
        p3 = self._intersection_of_vector_and_plane(vector)
        if p3 is None:
            return None
        return self._point_3d_to_2d(p3)
