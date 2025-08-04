#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坐标转换器演示脚本

展示coordinate_converter.py模块的主要功能，包括：
1. 基本坐标转换
2. 批量处理
3. 眼睛坐标提取
4. 质量评估
"""

import numpy as np
import sys
from pathlib import Path

# 添加项目路径
project_path = Path(__file__).parent
sys.path.insert(0, str(project_path))

from core.coordinate_converter import CoordinateConverter


def demo_basic_conversion():
    """演示基本坐标转换功能"""
    print("=" * 50)
    print("演示1：基本坐标转换")
    print("=" * 50)
    
    # 相机参数
    camera_params = {
        "intrinsic_params": {
            "fx": 925.0,
            "fy": 925.0,
            "cx": 640.0,
            "cy": 360.0
        },
        "depth_scale": 0.001,
        "image_resolution": {
            "width": 1280,
            "height": 720
        }
    }
    
    # 创建转换器
    converter = CoordinateConverter(camera_params)
    
    # 创建测试深度图
    depth_map = np.ones((720, 1280), dtype=np.float32) * 0.5  # 0.5米深度
    
    # 测试归一化坐标转像素坐标
    print("\n1. 归一化坐标转像素坐标：")
    test_cases = [
        (0.5, 0.5),    # 图像中心
        (0.0, 0.0),    # 左上角
        (1.0, 1.0),    # 右下角
        (0.25, 0.75),  # 其他位置
    ]
    
    for norm_coords in test_cases:
        pixel_coords = converter.normalized_to_pixel(norm_coords, (720, 1280))
        print(f"   归一化坐标 {norm_coords} -> 像素坐标 {pixel_coords}")
    
    # 测试像素坐标转三维坐标
    print("\n2. 像素坐标转三维坐标：")
    test_pixels = [
        (640, 360),  # 图像中心
        (320, 180),  # 左上象限
        (960, 540),  # 右下象限
    ]
    
    for pixel_coords in test_pixels:
        coord_3d = converter.pixel_to_3d(pixel_coords, depth_map)
        print(f"   像素坐标 {pixel_coords} -> 三维坐标 [{coord_3d[0]:.3f}, {coord_3d[1]:.3f}, {coord_3d[2]:.3f}]")


def demo_batch_conversion():
    """演示批量转换功能"""
    print("\n" + "=" * 50)
    print("演示2：批量坐标转换")
    print("=" * 50)
    
    # 相机参数
    camera_params = {
        "intrinsic_params": {
            "fx": 925.0,
            "fy": 925.0,
            "cx": 640.0,
            "cy": 360.0
        },
        "depth_scale": 0.001
    }
    
    converter = CoordinateConverter(camera_params)
    depth_map = np.ones((720, 1280), dtype=np.float32) * 0.5
    
    # 创建测试关键点（模拟MediaPipe输出）
    print("\n创建测试关键点...")
    landmarks = []
    for i in range(20):  # 20个测试关键点
        x = 0.4 + 0.2 * np.sin(i * 0.3)  # 归一化x坐标
        y = 0.4 + 0.2 * np.cos(i * 0.3)  # 归一化y坐标
        z = 0.1 + 0.05 * np.sin(i * 0.5)  # 相对深度
        landmarks.append((x, y, z))
    
    print(f"生成了 {len(landmarks)} 个测试关键点")
    
    # 批量转换
    print("\n执行批量转换...")
    coords_3d = converter.batch_convert_landmarks(landmarks, depth_map)
    
    print(f"成功转换 {len(coords_3d)} 个关键点")
    
    # 显示前5个转换结果
    print("\n前5个转换结果：")
    for i, coord in enumerate(coords_3d[:5]):
        print(f"   关键点 {i+1}: X={coord[0]:.3f}m, Y={coord[1]:.3f}m, Z={coord[2]:.3f}m")


def demo_eye_coordinates():
    """演示眼睛坐标提取功能"""
    print("\n" + "=" * 50)
    print("演示3：眼睛坐标提取")
    print("=" * 50)
    
    # 相机参数
    camera_params = {
        "intrinsic_params": {
            "fx": 925.0,
            "fy": 925.0,
            "cx": 640.0,
            "cy": 360.0
        },
        "depth_scale": 0.001
    }
    
    converter = CoordinateConverter(camera_params)
    depth_map = np.ones((720, 1280), dtype=np.float32) * 0.5
    
    # 创建模拟的MediaPipe面部关键点（468个点）
    print("\n创建模拟面部关键点...")
    face_landmarks = []
    for i in range(478):  # MediaPipe面部关键点总数
        # 模拟面部关键点分布
        if i < 468:  # 面部关键点
            x = 0.5 + 0.15 * np.sin(i * 0.1)
            y = 0.5 + 0.15 * np.cos(i * 0.1)
        else:  # 瞳孔和虹膜关键点
            x = 0.5 + 0.05 * np.sin(i * 0.5)
            y = 0.5 + 0.05 * np.cos(i * 0.5)
        
        z = 0.1 + 0.02 * np.sin(i * 0.2)
        face_landmarks.append((x, y, z))
    
    print(f"生成了 {len(face_landmarks)} 个面部关键点")
    
    # 提取眼睛坐标
    print("\n提取眼睛坐标...")
    eye_coords = converter.extract_eye_coordinates(face_landmarks, depth_map)
    
    # 显示结果
    print("\n眼睛坐标提取结果：")
    if eye_coords['left_pupil'] is not None:
        left_pupil = eye_coords['left_pupil']
        print(f"   左眼瞳孔中心：X={left_pupil[0]:.3f}m, Y={left_pupil[1]:.3f}m, Z={left_pupil[2]:.3f}m")
    else:
        print("   左眼瞳孔中心：提取失败")
    
    if eye_coords['right_pupil'] is not None:
        right_pupil = eye_coords['right_pupil']
        print(f"   右眼瞳孔中心：X={right_pupil[0]:.3f}m, Y={right_pupil[1]:.3f}m, Z={right_pupil[2]:.3f}m")
    else:
        print("   右眼瞳孔中心：提取失败")
    
    print(f"   左眼虹膜边界点：{len(eye_coords['left_iris'])} 个")
    print(f"   右眼虹膜边界点：{len(eye_coords['right_iris'])} 个")
    
    # 显示虹膜边界点坐标
    if eye_coords['left_iris']:
        print("\n   左眼虹膜边界点坐标：")
        for i, point in enumerate(eye_coords['left_iris'][:3]):  # 显示前3个
            print(f"     点{i+1}: X={point[0]:.3f}m, Y={point[1]:.3f}m, Z={point[2]:.3f}m")


def demo_quality_assessment():
    """演示质量评估功能"""
    print("\n" + "=" * 50)
    print("演示4：坐标质量评估")
    print("=" * 50)
    
    # 相机参数
    camera_params = {
        "intrinsic_params": {
            "fx": 925.0,
            "fy": 925.0,
            "cx": 640.0,
            "cy": 360.0
        },
        "depth_scale": 0.001
    }
    
    converter = CoordinateConverter(camera_params)
    
    # 创建高质量坐标
    print("\n1. 高质量坐标测试：")
    high_quality_coords = [
        np.array([0.1, 0.2, 0.5]),
        np.array([-0.1, 0.3, 0.6]),
        np.array([0.0, -0.1, 0.4]),
        np.array([0.05, 0.15, 0.55]),
        np.array([-0.05, 0.25, 0.65])
    ]
    
    is_valid = converter.validate_3d_coordinates(high_quality_coords)
    quality_score = converter.calculate_coordinate_quality(high_quality_coords)
    
    print(f"   坐标验证结果：{'通过' if is_valid else '失败'}")
    print(f"   质量分数：{quality_score:.2f}")
    
    # 创建低质量坐标
    print("\n2. 低质量坐标测试：")
    low_quality_coords = [
        np.array([0.1, 0.2, 0.5]),      # 有效坐标
        np.array([np.nan, 0.3, 0.6]),   # 包含NaN
        np.array([0.0, -0.1, 0.4]),     # 有效坐标
        np.array([15.0, 0.15, 0.55]),   # 超出范围
        np.array([-0.05, 0.25, 0.65])   # 有效坐标
    ]
    
    is_valid = converter.validate_3d_coordinates(low_quality_coords)
    quality_score = converter.calculate_coordinate_quality(low_quality_coords)
    
    print(f"   坐标验证结果：{'通过' if is_valid else '失败'}")
    print(f"   质量分数：{quality_score:.2f}")
    
    # 显示统计信息
    print("\n3. 转换统计信息：")
    stats = converter.get_conversion_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")


def demo_error_handling():
    """演示错误处理功能"""
    print("\n" + "=" * 50)
    print("演示5：错误处理")
    print("=" * 50)
    
    # 相机参数
    camera_params = {
        "intrinsic_params": {
            "fx": 925.0,
            "fy": 925.0,
            "cx": 640.0,
            "cy": 360.0
        },
        "depth_scale": 0.001
    }
    
    converter = CoordinateConverter(camera_params)
    depth_map = np.ones((720, 1280), dtype=np.float32) * 0.5
    
    # 测试无效输入
    print("\n1. 测试无效归一化坐标：")
    try:
        pixel_coords = converter.normalized_to_pixel((-0.1, 0.5), (720, 1280))
        print("   错误：应该抛出异常但没有")
    except ValueError as e:
        print(f"   正确捕获异常：{e}")
    
    try:
        pixel_coords = converter.normalized_to_pixel((1.5, 0.5), (720, 1280))
        print("   错误：应该抛出异常但没有")
    except ValueError as e:
        print(f"   正确捕获异常：{e}")
    
    print("\n2. 测试无效像素坐标：")
    try:
        coord_3d = converter.pixel_to_3d((-1, 0), depth_map)
        print("   错误：应该抛出异常但没有")
    except ValueError as e:
        print(f"   正确捕获异常：{e}")
    
    try:
        coord_3d = converter.pixel_to_3d((1280, 720), depth_map)
        print("   错误：应该抛出异常但没有")
    except ValueError as e:
        print(f"   正确捕获异常：{e}")
    
    print("\n3. 测试无效深度值：")
    invalid_depth_map = np.ones((720, 1280), dtype=np.float32) * -1.0
    try:
        coord_3d = converter.pixel_to_3d((640, 360), invalid_depth_map)
        print("   错误：应该抛出异常但没有")
    except ValueError as e:
        print(f"   正确捕获异常：{e}")


def demo_performance():
    """演示性能测试"""
    print("\n" + "=" * 50)
    print("演示6：性能测试")
    print("=" * 50)
    
    # 相机参数
    camera_params = {
        "intrinsic_params": {
            "fx": 925.0,
            "fy": 925.0,
            "cx": 640.0,
            "cy": 360.0
        },
        "depth_scale": 0.001
    }
    
    converter = CoordinateConverter(camera_params)
    depth_map = np.ones((720, 1280), dtype=np.float32) * 0.5
    
    # 创建大量测试数据
    num_landmarks = 1000
    print(f"\n创建 {num_landmarks} 个测试关键点...")
    
    test_landmarks = []
    for i in range(num_landmarks):
        x = 0.5 + 0.1 * np.sin(i * 0.01)
        y = 0.5 + 0.1 * np.cos(i * 0.01)
        z = 0.1 + 0.05 * np.sin(i * 0.02)
        test_landmarks.append((x, y, z))
    
    # 性能测试
    import time
    
    print("执行批量转换性能测试...")
    start_time = time.time()
    coords_3d = converter.batch_convert_landmarks(test_landmarks, depth_map)
    end_time = time.time()
    
    processing_time = end_time - start_time
    throughput = len(coords_3d) / processing_time
    
    print(f"\n性能测试结果：")
    print(f"   处理关键点数：{num_landmarks}")
    print(f"   成功转换数：{len(coords_3d)}")
    print(f"   处理时间：{processing_time:.4f} 秒")
    print(f"   吞吐量：{throughput:.2f} 坐标/秒")
    print(f"   成功率：{len(coords_3d)/num_landmarks*100:.1f}%")


def main():
    """主函数"""
    print("坐标转换器演示程序")
    print("=" * 60)
    
    try:
        # 运行所有演示
        demo_basic_conversion()
        demo_batch_conversion()
        demo_eye_coordinates()
        demo_quality_assessment()
        demo_error_handling()
        demo_performance()
        
        print("\n" + "=" * 60)
        print("所有演示完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n演示过程中出现错误：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 