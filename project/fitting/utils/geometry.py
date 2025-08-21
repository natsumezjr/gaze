"""Geometry helpers (placeholders)."""

from typing import TypedDict, Tuple
import numpy as np
from project.fitting.config.constants import KeyCoordinates
from project.fitting.config.settings import SCREEN_WITH_RGBD

class EllipsoidParams(TypedDict):
    """
    椭球参数占位类型（仅类型标注用）：
    - axes: 三主轴长度 (a, b, c)，单位 m，np.ndarray(3,)
    - rotation: 世界到椭球主轴坐标的旋转矩阵 R，np.ndarray(3,3)
    """
    axes: np.ndarray           # (3,)
    rotation: np.ndarray       # (3, 3)
    center: np.ndarray        # (3,)
    
    def get_axes(self)->np.ndarray:
        return self["axes"]
    
    def get_rotation(self)->np.ndarray:
        return self["rotation"]
    
    def get_center(self)->np.ndarray:
        return self["center"]

class Plane(TypedDict, total=False):
    """
    平面类型（仅类型标注用）：
    - normal: 平面法向量，np.ndarray(3,)
    - point: 平面坐标系的原点在3D世界坐标系中的坐标，np.ndarray(3,)
    - x_axis: 平面坐标系的x轴向量，np.ndarray(3,)
    - y_axis: 平面坐标系的y轴向量，np.ndarray(3,)
    - top_left/top_right/bottom_left/bottom_right: 屏幕四角（3D）
    - width_m/height_m: 屏幕物理宽高（米）
    """
    normal: np.ndarray
    point: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    
    def __init__(self,rgb_d=False):
        if rgb_d:
            self["normal"] = SCREEN_WITH_RGBD["normal"]
            self["point"] = SCREEN_WITH_RGBD["point"]
            self["x_axis"] = SCREEN_WITH_RGBD["x_axis"]
            self["y_axis"] = SCREEN_WITH_RGBD["y_axis"]
            self["top_left"] = SCREEN_WITH_RGBD["top_left"]
            self["top_right"] = SCREEN_WITH_RGBD["top_right"]
            self["bottom_left"] = SCREEN_WITH_RGBD["bottom_left"]
            self["bottom_right"] = SCREEN_WITH_RGBD["bottom_right"]
            # 补充物理尺寸（米）
            self["width_m"] = float(SCREEN_WITH_RGBD.get("width_m", 0.0))
            self["height_m"] = float(SCREEN_WITH_RGBD.get("height_m", 0.0))
        else:
            self["normal"] = np.array([0,0,1])
            self["point"] = np.array([0,0,0])
            self["x_axis"] = np.array([1,0,0])
            self["y_axis"] = np.array([0,1,0])
            self["top_left"] = np.array([-0.30, 0.005, 0])
            self["top_right"] = np.array([0.30, 0.005, 0])
            self["bottom_left"] = np.array([-0.30, 0.345, 0])
            self["bottom_right"] = np.array([0.30, 0.345, 0])
            # 默认物理尺寸（米）
            self["width_m"] = 0.60
            self["height_m"] = 0.34
            
            
    def _intersection_of_vector_and_plane(self,vector: np.ndarray) -> np.ndarray | None:
        '''
        计算向量与平面的交点（假设向量起点为世界原点）。
        Args:
            vector: 方向向量 (3,)
            plane: 平面（包含 'normal'(3,), 'point'(3,)）
        Returns:
            交点 np.ndarray(3,)，若无唯一交点（平行或共面）返回 None
        '''
        v = np.asarray(vector, dtype=float).reshape(3)
        n = np.asarray(self["normal"], dtype=float).reshape(3)
        p0 = np.asarray(self["point"], dtype=float).reshape(3)

        denom = float(np.dot(n, v))
        if np.isclose(denom, 0.0):
            # 平行：若原点在平面上则无唯一交点（共面），否则无交点
            return None

        t = float(np.dot(n, p0) / denom)
        return v * t

    def _point_3d_to_2d(self,point: np.ndarray) -> np.ndarray:
        '''
        将3D点转换成在平面2D坐标系下的坐标
        Args:
            point: 3D点
            plane: 平面
        Returns:
            2D坐标
        '''
        x = np.dot(point - self["point"],self["x_axis"])
        y = np.dot(point - self["point"],self["y_axis"])
        return np.array([x,y])
    
    def intersection_on_plane(self,vector:np.ndarray)->np.ndarray | None:
        '''
        计算点与平面的交点
        Args:
            point: 点
        Returns:
            交点（在平面2D坐标系下）
        '''
        p3 = self._intersection_of_vector_and_plane(vector)
        if p3 is None:
            return None
        return self._point_3d_to_2d(p3)
        
            
class CenterFitter:
    """
    中心拟合器类型（仅类型标注用）：
    - center_fitter: 中心拟合器函数，签名统一为：
        center_fitter(key_coordinates: KeyCoordinates, threshold: float, trials_time: int, max_trials: int)
        -> Tuple[EllipsoidParams, np.ndarray]
    """
    def __init__(self):
        self._data = []
        
    def center_fitter(self,key_coordinates:KeyCoordinates,threshold:float,trials_times:int,max_trials:int)->Tuple[EllipsoidParams, np.ndarray]:
        if trials_times >= max_trials:
            return self._int_center_fitter(key_coordinates,threshold)
        else:
            self._data.append(key_coordinates)
            return self._int_center_fitter(key_coordinates,threshold)
        
    def _int_center_fitter(self,key_coordinates:KeyCoordinates,threshold:float)->Tuple[EllipsoidParams, np.ndarray]:
        '''
        函数功能：对于传入的key_coordinates，进行椭球或者球体拟合，返回（椭）球心坐标
        算法：目前采用的算法RANSAC与加权最小二乘法结合
        '''
        pass
    
def vector_of_2_points(point1:np.ndarray,point2:np.ndarray)->np.ndarray:
    return point2 - point1



def normalize_vector(vector:np.ndarray)->np.ndarray:
    '''
    归一化向量
    Args:
        vector: 向量
    Returns:
        归一化后的向量
    '''
    return vector / np.linalg.norm(vector)

def angle_between_vectors(vector1:np.ndarray,vector2:np.ndarray)->float:
    '''
    计算两个向量之间的夹角
    Args:
        vector1: 向量1
        vector2: 向量2
    Returns:
        夹角（弧度）
    '''
    return np.arccos(np.dot(vector1,vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2)))

# 对外接口

def intersect_pixel_on_screen(vector:np.ndarray,rgb_d=False)->np.ndarray:
    '''
    计算向量与屏幕的交点（像素坐标,(w,h)）
    Args:
        vector: 向量
        screen: 屏幕
    Returns:
        交点
    '''
    screen = Plane(rgb_d)
    point_2d_with_camera_as_origin = screen.intersection_on_plane(vector)
    if point_2d_with_camera_as_origin is None:
        return np.array([np.nan, np.nan])

    # 统一到以左上角为原点的2D坐标
    top_left_2d = screen._point_3d_to_2d(screen["top_left"])
    point_2d_with_top_left_as_origin = point_2d_with_camera_as_origin - top_left_2d

    x,y = point_2d_with_top_left_as_origin
    screen_width,screen_height = screen["width_m"],screen["height_m"]

    # 使用实际分辨率（来自设置）
    w,h = SCREEN_WITH_RGBD["resolution_px"]
    if screen_width <= 0 or screen_height <= 0 or w <= 0 or h <= 0:
        return np.array([np.nan, np.nan])

    return np.array([w*x/screen_width, h*y/screen_height])

def center_fitter(key_coordinates:KeyCoordinates,threshold:float,trials_times:int,max_trials:int)->Tuple[EllipsoidParams, np.ndarray]:
    '''
    中心拟合器
    Args:
        key_coordinates: 关键点坐标
        threshold: 阈值
        trials_times: 尝试次数
        max_trials: 最大尝试次数
    Returns:
        中心拟合结果
    '''
    return CenterFitter().center_fitter(key_coordinates,threshold,trials_times,max_trials)

def rotate_vector(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    '''
    使用罗德里格斯旋转公式绕轴旋转向量
    
    Args:
        vector: 要旋转的向量 (3,)
        axis: 旋转轴单位向量 (3,)
        angle: 旋转角度（弧度）
    
    Returns:
        旋转后的向量 (3,)
    '''
    # 确保轴向量是单位向量
    axis = normalize_vector(axis)
    
    # 罗德里格斯旋转公式：v' = v*cos(θ) + (k×v)*sin(θ) + k(k·v)(1-cos(θ))
    # 其中 k 是旋转轴，θ 是旋转角度
    
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    # 计算各个分量
    v_parallel = np.dot(vector, axis) * axis  # 平行于轴的分量
    v_perpendicular = vector - v_parallel     # 垂直于轴的分量
    
    # 计算旋转后的向量
    rotated_vector = (v_parallel + 
                     v_perpendicular * cos_angle + 
                     np.cross(axis, vector) * sin_angle)
    
    return rotated_vector
        
        






