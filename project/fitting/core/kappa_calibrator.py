"""Estimate Kappa parameters from calibration samples (placeholders).

Each sample should contain:
- c_eye: np.ndarray (3,)
- c_pupil: np.ndarray (3,)
- target_point: np.ndarray (3,)  # known fixation point on the screen plane in camera coords
"""

from typing import Dict, Iterable, Tuple, List
import numpy as np
from project.fitting.utils.geometry import normalize_vector, vector_of_2_points


def estimate_kappa(samples: Iterable[Dict]) -> Tuple[np.ndarray, float]:
    """
    从标定样本估计kappa轴和kappa角
    
    Args:
        samples: 标定样本列表，每个样本包含：
            - c_eye: 眼球中心坐标 (3,)
            - c_pupil: 瞳孔中心坐标 (3,)
            - target_point: 屏幕上的已知注视点 (3,)
    
    Returns:
        (kappa_axis, kappa_angle)
        - kappa_axis: kappa轴单位向量 (3,)
        - kappa_angle: kappa角（弧度）
    """
    samples_list = list(samples)
    if len(samples_list) < 3:
        raise ValueError("至少需要3个标定样本")
    
    # 收集所有样本的理论视线向量和实际注视方向
    theoretical_gazes = []
    actual_gazes = []
    eye_centers = []
    
    for sample in samples_list:
        c_eye = np.array(sample['c_eye'])
        c_pupil = np.array(sample['c_pupil'])
        target_point = np.array(sample['target_point'])
        
        # 计算理论视线向量（眼球中心到瞳孔中心）
        theoretical_gaze = normalize_vector(vector_of_2_points(c_eye, c_pupil))
        
        # 计算实际注视方向（眼球中心到目标点）
        actual_gaze = normalize_vector(vector_of_2_points(c_eye, target_point))
        
        theoretical_gazes.append(theoretical_gaze)
        actual_gazes.append(actual_gaze)
        eye_centers.append(c_eye)
    
    # 方法1：基于最小二乘的kappa轴估计
    kappa_axis, kappa_angle = _estimate_kappa_least_squares(
        theoretical_gazes, actual_gazes, eye_centers
    )
    
    # 方法2：基于几何约束的kappa轴估计（备选）
    if kappa_axis is None:
        kappa_axis, kappa_angle = _estimate_kappa_geometric(
            theoretical_gazes, actual_gazes, eye_centers
        )
    
    return kappa_axis, kappa_angle


def _estimate_kappa_least_squares(
    theoretical_gazes: List[np.ndarray],
    actual_gazes: List[np.ndarray],
    eye_centers: List[np.ndarray]
) -> Tuple[np.ndarray, float]:
    """
    基于最小二乘的kappa参数估计
    
    原理：寻找一个旋转轴和角度，使得理论视线向量旋转后与实际注视方向的差异最小
    """
    n_samples = len(theoretical_gazes)
    
    # 构建线性方程组：A * x = b
    # 其中 x = [kappa_axis_x, kappa_axis_y, kappa_axis_z, kappa_angle]
    A = np.zeros((3 * n_samples, 4))
    b = np.zeros(3 * n_samples)
    
    for i in range(n_samples):
        t_gaze = theoretical_gazes[i]
        a_gaze = actual_gazes[i]
        
        # 计算理论视线和实际注视方向的叉积（用于构建旋转矩阵）
        cross_product = np.cross(t_gaze, a_gaze)
        
        # 填充矩阵A和向量b
        row_start = 3 * i
        A[row_start:row_start + 3, :3] = np.eye(3)  # kappa轴分量
        A[row_start:row_start + 3, 3] = cross_product  # kappa角系数
        
        # 目标：理论视线 + 旋转 = 实际注视方向
        b[row_start:row_start + 3] = a_gaze - t_gaze
    
    try:
        # 求解最小二乘问题
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        
        kappa_axis = normalize_vector(x[:3])
        kappa_angle = np.clip(x[3], -np.pi/6, np.pi/6)  # 限制kappa角范围
        
        return kappa_axis, kappa_angle
        
    except np.linalg.LinAlgError:
        return None, 0.0


def _estimate_kappa_geometric(
    theoretical_gazes: List[np.ndarray],
    actual_gazes: List[np.ndarray],
    eye_centers: List[np.ndarray]
) -> Tuple[np.ndarray, float]:
    """
    基于几何约束的kappa参数估计
    
    原理：kappa轴应该垂直于理论视线和实际注视方向构成的平面
    """
    n_samples = len(theoretical_gazes)
    
    # 收集所有可能的kappa轴候选
    kappa_axis_candidates = []
    
    for i in range(n_samples):
        t_gaze = theoretical_gazes[i]
        a_gaze = actual_gazes[i]
        
        # 计算理论视线和实际注视方向的叉积
        cross_product = np.cross(t_gaze, a_gaze)
        if np.linalg.norm(cross_product) > 1e-6:
            kappa_axis_candidates.append(normalize_vector(cross_product))
    
    if not kappa_axis_candidates:
        return np.array([0, 0, 1]), 0.0
    
    # 使用聚类方法找到最一致的kappa轴
    kappa_axis = _cluster_kappa_axes(kappa_axis_candidates)
    
    # 估计kappa角（使用所有样本的平均值）
    kappa_angle = _estimate_kappa_angle(
        theoretical_gazes, actual_gazes, kappa_axis
    )
    
    return kappa_axis, kappa_angle


def _cluster_kappa_axes(kappa_axis_candidates: List[np.ndarray]) -> np.ndarray:
    """
    对kappa轴候选进行聚类，找到最一致的方向
    """
    if len(kappa_axis_candidates) == 1:
        return kappa_axis_candidates[0]
    
    # 计算所有候选轴之间的相似度矩阵
    n_candidates = len(kappa_axis_candidates)
    similarity_matrix = np.zeros((n_candidates, n_candidates))
    
    for i in range(n_candidates):
        for j in range(n_candidates):
            if i != j:
                # 使用点积作为相似度（考虑方向的正负）
                similarity = abs(np.dot(kappa_axis_candidates[i], kappa_axis_candidates[j]))
                similarity_matrix[i, j] = similarity
    
    # 找到与其他轴平均相似度最高的轴
    avg_similarities = np.mean(similarity_matrix, axis=1)
    best_idx = np.argmax(avg_similarities)
    
    return kappa_axis_candidates[best_idx]


def _estimate_kappa_angle(
    theoretical_gazes: List[np.ndarray],
    actual_gazes: List[np.ndarray],
    kappa_axis: np.ndarray
) -> float:
    """
    给定kappa轴，估计kappa角
    """
    angles = []
    
    for t_gaze, a_gaze in zip(theoretical_gazes, actual_gazes):
        # 计算理论视线绕kappa轴旋转到实际注视方向所需的角度
        # 使用罗德里格斯旋转公式的逆过程
        
        # 计算理论视线和实际注视方向在垂直于kappa轴的平面上的投影
        t_proj = t_gaze - np.dot(t_gaze, kappa_axis) * kappa_axis
        a_proj = a_gaze - np.dot(a_gaze, kappa_axis) * kappa_axis
        
        if np.linalg.norm(t_proj) > 1e-6 and np.linalg.norm(a_proj) > 1e-6:
            # 归一化投影向量
            t_proj_norm = normalize_vector(t_proj)
            a_proj_norm = normalize_vector(a_proj)
            
            # 计算旋转角度
            cos_angle = np.dot(t_proj_norm, a_proj_norm)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            
            # 确定旋转方向（使用叉积判断）
            cross_prod = np.cross(t_proj_norm, a_proj_norm)
            sign = np.sign(np.dot(cross_prod, kappa_axis))
            
            angle = sign * np.arccos(cos_angle)
            angles.append(angle)
    
    if angles:
        # 返回中位数角度（对异常值更鲁棒）
        return np.median(angles)
    else:
        return 0.0

