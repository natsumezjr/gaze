"""Compute theoretical gaze and apply Kappa compensation (placeholders)."""

from typing import Dict, Optional, Tuple
import numpy as np
from project.fitting.utils.geometry import vector_of_2_points, normalize_vector, rotate_vector


def compute_theoretical_gaze(c_eye: np.ndarray, c_pupil: np.ndarray) -> np.ndarray:
    """Return unit vector from eye center to pupil center.

    Args:
        c_eye: np.ndarray (3,)
        c_pupil: np.ndarray (3,)
    """
    gaze_vec = vector_of_2_points(c_eye, c_pupil)
    gaze_vec = normalize_vector(gaze_vec)
    return gaze_vec


def apply_kappa_compensation(
    theoretical_gaze: np.ndarray,
    kappa_axis: np.ndarray,
    kappa_angle: float,
    eye_center: np.ndarray,
    pupil_center: np.ndarray
) -> np.ndarray:
    """
    应用kappa角补偿，修正理论视线向量
    
    Args:
        theoretical_gaze: 理论视线向量（已归一化）
        kappa_axis: kappa轴单位向量（眼球坐标系下的旋转轴）
        kappa_angle: kappa角（弧度）
        eye_center: 眼球中心坐标
        pupil_center: 瞳孔中心坐标
    
    Returns:
        补偿后的视线向量（已归一化）
    """
    # 如果没有kappa角，直接返回理论视线向量
    if np.abs(kappa_angle) < 1e-6:
        return theoretical_gaze
    
    # 将kappa轴从眼球坐标系转换到世界坐标系
    # 假设眼球坐标系：z轴指向眼球前方，x轴指向右，y轴指向上
    eye_to_pupil = normalize_vector(pupil_center - eye_center)
    
    # 构建眼球坐标系到世界坐标系的旋转矩阵
    # 使用眼球中心到瞳孔中心的方向作为z轴
    z_axis = eye_to_pupil
    
    # 构建正交的x轴和y轴
    # 选择世界坐标系的上方向作为参考
    world_up = np.array([0, 1, 0])
    if np.abs(np.dot(z_axis, world_up)) > 0.9:
        world_up = np.array([1, 0, 0])  # 如果z轴接近上方向，使用右方向
    
    x_axis = normalize_vector(np.cross(z_axis, world_up))
    y_axis = normalize_vector(np.cross(z_axis, x_axis))
    
    # 眼球坐标系到世界坐标系的旋转矩阵
    R_eye_to_world = np.column_stack([x_axis, y_axis, z_axis])
    
    # 将kappa轴从眼球坐标系转换到世界坐标系
    kappa_axis_world = R_eye_to_world @ kappa_axis
    
    # 绕kappa轴旋转理论视线向量
    compensated_gaze = rotate_vector(theoretical_gaze, kappa_axis_world, kappa_angle)
    
    # 归一化结果
    return normalize_vector(compensated_gaze)


def apply_kappa_compensation_simple(
    theoretical_gaze: np.ndarray,
    kappa_axis: np.ndarray,
    kappa_angle: float
) -> np.ndarray:
    """
    简化版kappa角补偿（假设kappa轴已在世界坐标系中）
    
    Args:
        theoretical_gaze: 理论视线向量（已归一化）
        kappa_axis: kappa轴单位向量（世界坐标系）
        kappa_angle: kappa角（弧度）
    
    Returns:
        补偿后的视线向量（已归一化）
    """
    if np.abs(kappa_angle) < 1e-6:
        return theoretical_gaze
    
    # 直接绕kappa轴旋转
    compensated_gaze = rotate_vector(theoretical_gaze, kappa_axis, kappa_angle)
    return normalize_vector(compensated_gaze)


def compute_gaze_with_kappa_compensation(
    eye_center: np.ndarray,
    pupil_center: np.ndarray,
    kappa_axis: np.ndarray,
    kappa_angle: float
) -> np.ndarray:
    """
    完整的视线计算流程：理论视线 + kappa补偿
    
    Args:
        eye_center: 眼球中心坐标
        pupil_center: 瞳孔中心坐标
        kappa_axis: kappa轴单位向量
        kappa_angle: kappa角（弧度）
    
    Returns:
        补偿后的视线向量
    """
    # 1. 计算理论视线向量
    theoretical_gaze = compute_theoretical_gaze(eye_center, pupil_center)
    
    # 2. 应用kappa补偿
    compensated_gaze = apply_kappa_compensation(
        theoretical_gaze, kappa_axis, kappa_angle, eye_center, pupil_center
    )
    
    return compensated_gaze



