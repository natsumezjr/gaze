#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坐标转换器模块 (Coordinate Converter Module)

主要功能：
1. 像素坐标结合深度转三维坐标
2. 批量转换关键点
3. 坐标验证和质量评估

理论基础：
- 像素坐标：图像中的实际像素位置
- 三维坐标：相机参考系下的真实三维位置（米单位）

关键公式：
像素坐标转三维坐标：
X = (x_pixel - cx) × Z / fx
Y = (y_pixel - cy) × Z / fy
Z = depth_value
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Union
import logging
import cv2
from project.recg_fit_data.data_manager import CoordinatePoint



# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 默认参数
DEPTH_VALIDATION_THRESHOLD = 0.1
COORDINATE_QUALITY_THRESHOLD = 0.8

def pixel_to_3d(pixel_coords: Tuple[int, int], 
                depth_map_meters: np.ndarray, 
                camera_params: dict) -> CoordinatePoint:
    """
    像素坐标结合深度转三维坐标
    
    Args:
        pixel_coords: 像素坐标 (x_pixel, y_pixel)，支持浮点数
        depth_map_meters: 深度图，numpy数组，单位：米
        camera_params: 相机参数字典
    
    Returns:
        coord_3d: 三维坐标 [X, Y, Z]，单位：米，使用CoordinatePoint类型
    
    Raises:
        ValueError: 输入参数无效或深度值无效时抛出
    """
    # 输入验证
    if not isinstance(pixel_coords, tuple) or len(pixel_coords) != 2:
        raise ValueError("pixel_coords必须是包含2个元素的元组")
    
    if not isinstance(depth_map_meters, np.ndarray) or depth_map_meters.ndim != 2:
        raise ValueError("depth_map_meters必须是2维numpy数组")
    
    # 提取相机参数
    intrinsic_params = camera_params.get("intrinsic_params", {})
    fx = intrinsic_params.get("fx", 925.0)
    fy = intrinsic_params.get("fy", 925.0)
    cx = intrinsic_params.get("cx", 640.0)
    cy = intrinsic_params.get("cy", 360.0)
    
    x_pixel, y_pixel = pixel_coords
    height, width = depth_map_meters.shape
    
    # 检查像素坐标是否在图像范围内
    if not (0.0 <= x_pixel < width and 0.0 <= y_pixel < height):
        raise ValueError(f"像素坐标({x_pixel}, {y_pixel})超出图像范围({width}, {height})")
    
    # 获取深度值（使用插值）
    depth_meters = _get_depth_interpolated(depth_map_meters, (x_pixel, y_pixel))
    
    # 验证深度值
    if not np.isfinite(depth_meters) or depth_meters <= DEPTH_VALIDATION_THRESHOLD:
        raise ValueError(f"无效的深度值：{depth_meters}，必须大于{DEPTH_VALIDATION_THRESHOLD}米")
    
    # 转换公式：X = (x_pixel - cx) × Z / fx, Y = (y_pixel - cy) × Z / fy, Z = depth
    x_3d = (x_pixel - cx) * depth_meters / fx
    y_3d = (y_pixel - cy) * depth_meters / fy
    z_3d = depth_meters
    
    # 使用CoordinatePoint类型别名
    return np.array([x_3d, y_3d, z_3d], dtype=np.float32)

def _get_depth_interpolated(depth_map: np.ndarray, 
                           pixel_coords: Tuple[float, float]) -> float:
    """
    使用OpenCV remap进行高效插值获取深度值
    
    Args:
        depth_map: 深度图
        pixel_coords: 浮点像素坐标 (x, y)
    
    Returns:
        depth_value: 插值后的深度值
    """
    x, y = pixel_coords
    height, width = depth_map.shape
    
    # 边界处理
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    
    # 创建坐标映射数组
    map_x = np.array([[x]], dtype=np.float32)
    map_y = np.array([[y]], dtype=np.float32)
    
    # 使用OpenCV remap进行插值
    interpolated = cv2.remap(depth_map, map_x, map_y, 
                            cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=np.nan)
    
    return float(interpolated[0, 0])

def batch_convert_landmarks(landmarks: List[Union[Tuple[float, float, float], Tuple[float, float, float, float]]], 
                           depth_map_meters: np.ndarray, 
                           camera_params: dict) -> List[CoordinatePoint]:
    """
    批量转换关键点为三维坐标，支持4D数据（包含visibility）
    
    Args:
        landmarks: 关键点列表，每个元素为(x, y, z) 或 (x, y, z, visibility)
        depth_map_meters: 深度图，numpy数组
        camera_params: 相机参数字典
    
    Returns:
        coords_3d: 三维坐标列表，每个元素为[X, Y, Z] 或 [X, Y, Z, visibility]，使用CoordinatePoint类型
    """
    if not isinstance(landmarks, list):
        raise ValueError("landmarks必须是列表类型")
    
    if not isinstance(depth_map_meters, np.ndarray) or depth_map_meters.ndim != 2:
        raise ValueError("depth_map_meters必须是2维numpy数组")
    
    coords_3d = []
    success_count = 0
    
    for i, landmark in enumerate(landmarks):
        # 检查关键点格式：支持3D和4D
        if not isinstance(landmark, (tuple, list, np.ndarray)) or len(landmark) not in [3, 4]:
            logger.warning(f"跳过无效的关键点 {i}：{landmark}")
            continue
        
        # 转换为列表格式
        landmark_list = list(landmark)
        
        # 提取坐标部分（前3维）
        coord_part = landmark_list[:3]
        visibility = landmark_list[3] if len(landmark_list) == 4 else 1.0
        
        # 转换为3D坐标
        try:
            coord_3d = pixel_to_3d((coord_part[0], coord_part[1]), depth_map_meters, camera_params)
            
            # 如果有visibility信息，添加到3D坐标
            if len(landmark_list) == 4:
                # 使用CoordinatePoint类型别名
                coord_4d = np.append(coord_3d, visibility).astype(np.float32)
                coords_3d.append(coord_4d)
            else:
                coords_3d.append(coord_3d)
            success_count += 1
            
        except Exception as e:
            logger.warning(f"关键点 {i} 转换失败: {e}")
    
    logger.debug(f"批量转换完成：成功 {success_count}/{len(landmarks)} 个关键点")
    return coords_3d

def validate_3d_coordinates(coords_3d: List[np.ndarray]) -> bool:
    """
    验证三维坐标的有效性
    
    Args:
        coords_3d: 三维坐标列表
    
    Returns:
        is_valid: 坐标是否有效
    """
    if not isinstance(coords_3d, list) or len(coords_3d) == 0:
        return False
    
    valid_count = 0
    for coord in coords_3d:
        if not isinstance(coord, np.ndarray) or coord.shape[0] < 3:
            continue
        
        # 检查数值有效性
        if not np.all(np.isfinite(coord[:3])):
            continue
        
        x, y, z = coord[:3]
        
        # 检查坐标范围（根据实际应用调整）
        if not (-10.0 <= x <= 10.0 and -10.0 <= y <= 10.0 and 0.1 <= z <= 10.0):
            continue
        
        valid_count += 1
    
    # 计算有效坐标比例
    valid_ratio = valid_count / len(coords_3d)
    is_valid = valid_ratio >= COORDINATE_QUALITY_THRESHOLD
    
    logger.info(f"坐标验证结果：{valid_count}/{len(coords_3d)} 有效，比例：{valid_ratio:.2f}")
    
    return is_valid

def calculate_coordinate_quality(coords_3d: List[np.ndarray]) -> float:
    """
    计算坐标质量分数
    
    Args:
        coords_3d: 三维坐标列表
    
    Returns:
        quality_score: 质量分数，范围[0,1]
    """
    if not coords_3d:
        return 0.0
    
    valid_count = 0
    total_count = len(coords_3d)
    
    for coord in coords_3d:
        if (isinstance(coord, np.ndarray) and coord.shape[0] >= 3 and 
            np.all(np.isfinite(coord[:3]))):
            valid_count += 1
    
    return valid_count / total_count

# 测试函数
def test_coordinate_converter():
    """测试坐标转换器功能"""
    print("=== 坐标转换器测试 ===")
    
    # 创建测试数据
    test_camera_params = {
        "intrinsic_params": {
            "fx": 925.0,
            "fy": 925.0,
            "cx": 640.0,
            "cy": 360.0
        },
        "depth_scale": 0.001
    }
    
    # 创建模拟深度图
    depth_map = np.ones((720, 1280), dtype=np.float32) * 0.5  # 0.5米深度
    
    # 测试像素坐标转三维坐标
    print("\n1. 测试像素坐标转三维坐标")
    pixel_coords = (640, 360)  # 图像中心
    coord_3d = pixel_to_3d(pixel_coords, depth_map, test_camera_params)
    print(f"像素坐标 {pixel_coords} -> 三维坐标 {coord_3d}")
    
    # 测试批量转换
    print("\n2. 测试批量转换")
    test_landmarks = [
        np.array([640, 360, 0.1, 1.0]),  # 4D数据
        np.array([384, 504, 0.2, 0.8]),  # 4D数据
        np.array([1024, 144, 0.3, 0.9])  # 4D数据
    ]
    coords_3d = batch_convert_landmarks(test_landmarks, depth_map, test_camera_params)
    print(f"批量转换结果：{len(coords_3d)} 个有效坐标")
    
    # 测试坐标验证
    print("\n3. 测试坐标验证")
    is_valid = validate_3d_coordinates(coords_3d)
    print(f"坐标验证结果：{'通过' if is_valid else '失败'}")
    
    # 测试质量评估
    print("\n4. 测试质量评估")
    quality_score = calculate_coordinate_quality(coords_3d)
    print(f"质量分数：{quality_score:.2f}")
    
    print("\n=== 测试完成 ===")


__all__ = ['pixel_to_3d', 'batch_convert_landmarks', 'validate_3d_coordinates']

if __name__ == "__main__":
    test_coordinate_converter()