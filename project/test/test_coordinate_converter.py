#!/usr/bin/env python3
"""
测试坐标转换功能
演示如何使用coordinate_converter.py中的函数
"""

import cv2
import numpy as np
import sys
import os

# 添加项目路径到sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.coordinate_converter import (
    normalized_to_pixel, 
    pixel_to_3d, 
    batch_convert_landmarks, 
    validate_3d_coordinates
)

def test_normalized_to_pixel():
    """测试归一化坐标转像素坐标"""
    print("=== 测试 normalized_to_pixel 函数 ===")
    
    # 测试1: 正常情况
    norm_coords = (0.5, 0.3)
    image_shape = (720, 1280)
    result = normalized_to_pixel(norm_coords, image_shape)
    print(f"归一化坐标 {norm_coords} -> 像素坐标 {result}")
    
    # 测试2: 边界情况
    norm_coords = (0.0, 0.0)
    result = normalized_to_pixel(norm_coords, image_shape)
    print(f"归一化坐标 {norm_coords} -> 像素坐标 {result}")
    
    norm_coords = (1.0, 1.0)
    result = normalized_to_pixel(norm_coords, image_shape)
    print(f"归一化坐标 {norm_coords} -> 像素坐标 {result}")
    
    # 测试3: 错误输入
    try:
        result = normalized_to_pixel((1.5, 0.5), image_shape)
        print(f"超出范围测试: {result}")
    except ValueError as e:
        print(f"超出范围测试: 正确抛出异常 - {e}")

def test_pixel_to_3d():
    """测试像素坐标转三维坐标"""
    print("\n=== 测试 pixel_to_3d 函数 ===")
    
    # 创建模拟的深度图
    depth_map = np.ones((720, 1280), dtype=np.uint16) * 1000  # 1米深度
    
    # 创建相机参数
    camera_params = {
        "intrinsic_params": {
            "fx": 925.0,
            "fy": 925.0,
            "cx": 640.0,
            "cy": 360.0
        },
        "depth_scale": 0.001
    }
    
    # 测试1: 正常情况
    pixel_coords = (640, 360)  # 图像中心
    result = pixel_to_3d(pixel_coords, depth_map, camera_params)
    print(f"像素坐标 {pixel_coords} -> 三维坐标 {result}")
    
    # 测试2: 非中心点
    pixel_coords = (740, 460)  # 偏移100像素
    result = pixel_to_3d(pixel_coords, depth_map, camera_params)
    print(f"像素坐标 {pixel_coords} -> 三维坐标 {result}")
    
    # 测试3: 错误输入
    try:
        result = pixel_to_3d((2000, 1000), depth_map, camera_params)
        print(f"超出范围测试: {result}")
    except ValueError as e:
        print(f"超出范围测试: 正确抛出异常 - {e}")

def test_batch_convert_landmarks():
    """测试批量转换关键点"""
    print("\n=== 测试 batch_convert_landmarks 函数 ===")
    
    # 创建模拟的关键点列表
    landmarks = [
        (640, 360, 0.1),  # 图像中心
        (740, 460, 0.2),  # 偏移点
        (540, 260, 0.3),  # 另一个偏移点
        (2000, 1000, 0.4),  # 超出范围的点（应该被跳过）
    ]
    
    # 创建模拟的深度图
    depth_map = np.ones((720, 1280), dtype=np.uint16) * 1000
    
    # 创建相机参数
    camera_params = {
        "intrinsic_params": {
            "fx": 925.0,
            "fy": 925.0,
            "cx": 640.0,
            "cy": 360.0
        },
        "depth_scale": 0.001
    }
    
    # 批量转换
    coords_3d = batch_convert_landmarks(landmarks, depth_map, camera_params)
    print(f"输入关键点数量: {len(landmarks)}")
    print(f"成功转换数量: {len(coords_3d)}")
    
    for i, coord in enumerate(coords_3d):
        print(f"关键点 {i+1}: {coord}")

def test_validate_3d_coordinates():
    """测试三维坐标验证"""
    print("\n=== 测试 validate_3d_coordinates 函数 ===")
    
    # 测试1: 有效坐标
    valid_coords = [
        np.array([0.1, 0.2, 0.5]),
        np.array([-0.1, 0.3, 0.8]),
        np.array([0.05, -0.1, 1.2])
    ]
    result = validate_3d_coordinates(valid_coords)
    print(f"有效坐标测试: {result} (期望: True)")
    
    # 测试2: 无效坐标（超出范围）
    invalid_coords = [
        np.array([0.1, 0.2, 0.5]),
        np.array([200.0, 0.3, 0.8]),  # X超出范围
        np.array([0.05, -0.1, 1.2])
    ]
    result = validate_3d_coordinates(invalid_coords)
    print(f"无效坐标测试: {result} (期望: False)")
    
    # 测试3: 空列表
    result = validate_3d_coordinates([])
    print(f"空列表测试: {result} (期望: False)")
    
    # 测试4: 错误类型
    wrong_type_coords = [
        np.array([0.1, 0.2, 0.5]),
        [1, 2, 3],  # 不是numpy数组
        np.array([0.05, -0.1, 1.2])
    ]
    result = validate_3d_coordinates(wrong_type_coords)
    print(f"错误类型测试: {result} (期望: False)")

def test_integration():
    """测试集成功能"""
    print("\n=== 测试集成功能 ===")
    
    # 创建模拟数据
    landmarks = [
        (640, 360, 0.1),
        (740, 460, 0.2),
        (540, 260, 0.3)
    ]
    
    depth_map = np.ones((720, 1280), dtype=np.uint16) * 1000
    
    camera_params = {
        "intrinsic_params": {
            "fx": 925.0,
            "fy": 925.0,
            "cx": 640.0,
            "cy": 360.0
        },
        "depth_scale": 0.001
    }
    
    # 完整流程测试
    print("1. 批量转换关键点...")
    coords_3d = batch_convert_landmarks(landmarks, depth_map, camera_params)
    
    print("2. 验证三维坐标...")
    is_valid = validate_3d_coordinates(coords_3d)
    
    print(f"转换结果: {len(coords_3d)} 个有效坐标")
    print(f"验证结果: {is_valid}")
    
    if is_valid:
        print("3. 显示转换结果:")
        for i, coord in enumerate(coords_3d):
            print(f"  关键点 {i+1}: X={coord[0]:.3f}m, Y={coord[1]:.3f}m, Z={coord[2]:.3f}m")

def create_sample_depth_map():
    """创建示例深度图"""
    print("\n=== 创建示例深度图 ===")
    
    # 创建一个简单的深度图
    depth_map = np.ones((720, 1280), dtype=np.uint16) * 1000
    
    # 在中心区域设置不同的深度值
    center_y, center_x = 360, 640
    for y in range(center_y-100, center_y+100):
        for x in range(center_x-100, center_x+100):
            # 创建渐变深度效果
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            depth_value = int(800 + distance * 2)  # 800-1200mm范围
            depth_map[y, x] = depth_value
    
    # 保存深度图
    cv2.imwrite('sample_depth_map.png', depth_map)
    print("示例深度图已保存为 'sample_depth_map.png'")
    
    return depth_map

def main():
    """主测试函数"""
    print("坐标转换功能测试程序")
    print("=" * 50)
    
    # 运行各个测试
    test_normalized_to_pixel()
    test_pixel_to_3d()
    test_batch_convert_landmarks()
    test_validate_3d_coordinates()
    test_integration()
    
    # 创建示例深度图
    create_sample_depth_map()
    
    print("\n所有测试完成！")

if __name__ == "__main__":
    main() 