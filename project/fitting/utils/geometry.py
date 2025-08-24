import numpy as np
from project.fitting.config.models import Plane
from project.fitting.config.settings import SCREEN_WITH_RGBD

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
    top_left_2d = screen._point_3d_to_2d(screen.top_left)
    point_2d_with_top_left_as_origin = point_2d_with_camera_as_origin - top_left_2d

    x,y = point_2d_with_top_left_as_origin
    screen_width,screen_height = screen.width_m,screen.height_m

    # 使用实际分辨率（来自设置）
    w,h = SCREEN_WITH_RGBD["resolution_px"]
    if screen_width <= 0 or screen_height <= 0 or w <= 0 or h <= 0:
        return np.array([np.nan, np.nan])

    return np.array([w*x/screen_width, h*y/screen_height])



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