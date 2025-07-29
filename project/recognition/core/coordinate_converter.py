import numpy as np
from typing import List, Tuple

def normalized_to_pixel(norm_coords: Tuple[float, float], image_shape: Tuple[int, int]) -> Tuple[int, int]:
    """
    归一化坐标转像素坐标
    :param norm_coords: (x_norm, y_norm)
    :param image_shape: (height, width)
    :return: (x_pixel, y_pixel)
    可能用到的库函数：无
    """
    pass

def pixel_to_3d(pixel_coords: Tuple[int, int], depth_map: np.ndarray, camera_params: dict) -> np.ndarray:
    """
    像素坐标结合深度转三维坐标
    :param pixel_coords: (x_pixel, y_pixel)
    :param depth_map: 深度图
    :param camera_params: 相机内参
    :return: 三维坐标 (np.ndarray)
    可能用到的库函数：numpy
    """
    pass

def batch_convert_landmarks(landmarks: List, depth_map: np.ndarray, camera_params: dict) -> List[np.ndarray]:
    """
    批量转换关键点为三维坐标
    :param landmarks: 关键点列表
    :param depth_map: 深度图
    :param camera_params: 相机内参
    :return: 三维坐标列表
    可能用到的库函数：numpy
    """
    pass

def validate_3d_coordinates(coords_3d: List[np.ndarray]) -> bool:
    """
    验证三维坐标有效性
    :param coords_3d: 三维坐标列表
    :return: 是否有效
    可能用到的库函数：numpy
    """
    pass
