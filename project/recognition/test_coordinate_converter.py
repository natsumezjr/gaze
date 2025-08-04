#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坐标转换器测试文件

测试coordinate_converter.py模块的各项功能，包括：
1. 归一化坐标转像素坐标
2. 像素坐标转三维坐标
3. 批量转换功能
4. 眼睛坐标提取
5. 坐标验证和质量评估
"""

import sys
import os
import numpy as np
import unittest
from pathlib import Path

# 添加项目路径
project_path = Path(__file__).parent
sys.path.insert(0, str(project_path))

from core.coordinate_converter import CoordinateConverter


class TestCoordinateConverter(unittest.TestCase):
    """坐标转换器测试类"""
    
    def setUp(self):
        """测试前的初始化"""
        # 测试相机参数
        self.camera_params = {
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
        
        # 创建坐标转换器实例
        self.converter = CoordinateConverter(self.camera_params)
        
        # 创建测试深度图
        self.depth_map = np.ones((720, 1280), dtype=np.float32) * 0.5  # 0.5米深度
        
        # 创建测试关键点（模拟MediaPipe输出）
        self.test_landmarks = []
        for i in range(478):  # MediaPipe面部关键点总数
            x = 0.5 + 0.1 * np.sin(i * 0.1)  # 归一化x坐标
            y = 0.5 + 0.1 * np.cos(i * 0.1)  # 归一化y坐标
            z = 0.1 + 0.05 * np.sin(i * 0.2)  # 相对深度
            self.test_landmarks.append((x, y, z))
    
    def test_normalized_to_pixel(self):
        """测试归一化坐标转像素坐标"""
        print("\n=== 测试归一化坐标转像素坐标 ===")
        
        # 测试图像中心点
        norm_coords = (0.5, 0.5)
        image_shape = (720, 1280)
        pixel_coords = self.converter.normalized_to_pixel(norm_coords, image_shape)
        expected_coords = (640, 360)
        
        print(f"归一化坐标 {norm_coords} -> 像素坐标 {pixel_coords}")
        self.assertEqual(pixel_coords, expected_coords)
        
        # 测试边界点
        test_cases = [
            ((0.0, 0.0), (0, 0)),      # 左上角
            ((1.0, 1.0), (1279, 719)), # 右下角
            ((0.25, 0.75), (320, 540)) # 其他点
        ]
        
        for norm_coords, expected in test_cases:
            pixel_coords = self.converter.normalized_to_pixel(norm_coords, image_shape)
            print(f"归一化坐标 {norm_coords} -> 像素坐标 {pixel_coords}")
            self.assertEqual(pixel_coords, expected)
    
    def test_pixel_to_3d(self):
        """测试像素坐标转三维坐标"""
        print("\n=== 测试像素坐标转三维坐标 ===")
        
        # 测试图像中心点
        pixel_coords = (640, 360)
        coord_3d = self.converter.pixel_to_3d(pixel_coords, self.depth_map)
        
        print(f"像素坐标 {pixel_coords} -> 三维坐标 {coord_3d}")
        
        # 验证结果
        self.assertIsInstance(coord_3d, np.ndarray)
        self.assertEqual(coord_3d.shape, (3,))
        self.assertAlmostEqual(coord_3d[0], 0.0, places=6)  # X坐标应该接近0
        self.assertAlmostEqual(coord_3d[1], 0.0, places=6)  # Y坐标应该接近0
        self.assertAlmostEqual(coord_3d[2], 0.5, places=6)  # Z坐标应该是深度值
        
        # 测试其他点
        test_cases = [
            ((320, 180), (-0.173, -0.097, 0.5)),  # 左上象限
            ((960, 540), (0.173, 0.097, 0.5))     # 右下象限
        ]
        
        for pixel_coords, expected in test_cases:
            coord_3d = self.converter.pixel_to_3d(pixel_coords, self.depth_map)
            print(f"像素坐标 {pixel_coords} -> 三维坐标 {coord_3d}")
            
            # 验证X和Y坐标（允许一定误差）
            self.assertAlmostEqual(coord_3d[0], expected[0], places=3)
            self.assertAlmostEqual(coord_3d[1], expected[1], places=3)
            self.assertAlmostEqual(coord_3d[2], expected[2], places=3)
    
    def test_batch_convert_landmarks(self):
        """测试批量转换关键点"""
        print("\n=== 测试批量转换关键点 ===")
        
        # 选择部分关键点进行测试
        test_landmarks = self.test_landmarks[:10]  # 前10个关键点
        
        coords_3d = self.converter.batch_convert_landmarks(test_landmarks, self.depth_map)
        
        print(f"批量转换结果：{len(coords_3d)} 个有效坐标")
        
        # 验证结果
        self.assertIsInstance(coords_3d, list)
        self.assertGreater(len(coords_3d), 0)
        
        for coord in coords_3d:
            self.assertIsInstance(coord, np.ndarray)
            self.assertEqual(coord.shape, (3,))
            self.assertTrue(np.all(np.isfinite(coord)))
    
    def test_extract_eye_coordinates(self):
        """测试眼睛坐标提取"""
        print("\n=== 测试眼睛坐标提取 ===")
        
        eye_coords = self.converter.extract_eye_coordinates(self.test_landmarks, self.depth_map)
        
        print(f"眼睛坐标提取结果：")
        print(f"  左眼瞳孔：{'成功' if eye_coords['left_pupil'] is not None else '失败'}")
        print(f"  右眼瞳孔：{'成功' if eye_coords['right_pupil'] is not None else '失败'}")
        print(f"  左眼虹膜：{len(eye_coords['left_iris'])} 个点")
        print(f"  右眼虹膜：{len(eye_coords['right_iris'])} 个点")
        
        # 验证结果
        self.assertIsInstance(eye_coords, dict)
        self.assertIn('left_pupil', eye_coords)
        self.assertIn('right_pupil', eye_coords)
        self.assertIn('left_iris', eye_coords)
        self.assertIn('right_iris', eye_coords)
        
        # 验证瞳孔坐标
        if eye_coords['left_pupil'] is not None:
            self.assertIsInstance(eye_coords['left_pupil'], np.ndarray)
            self.assertEqual(eye_coords['left_pupil'].shape, (3,))
        
        if eye_coords['right_pupil'] is not None:
            self.assertIsInstance(eye_coords['right_pupil'], np.ndarray)
            self.assertEqual(eye_coords['right_pupil'].shape, (3,))
        
        # 验证虹膜坐标
        self.assertIsInstance(eye_coords['left_iris'], list)
        self.assertIsInstance(eye_coords['right_iris'], list)
        
        for iris_point in eye_coords['left_iris'] + eye_coords['right_iris']:
            self.assertIsInstance(iris_point, np.ndarray)
            self.assertEqual(iris_point.shape, (3,))
    
    def test_validate_3d_coordinates(self):
        """测试三维坐标验证"""
        print("\n=== 测试三维坐标验证 ===")
        
        # 创建有效坐标
        valid_coords = [
            np.array([0.1, 0.2, 0.5]),
            np.array([-0.1, 0.3, 0.6]),
            np.array([0.0, -0.1, 0.4])
        ]
        
        is_valid = self.converter.validate_3d_coordinates(valid_coords)
        print(f"有效坐标验证结果：{'通过' if is_valid else '失败'}")
        self.assertTrue(is_valid)
        
        # 创建无效坐标
        invalid_coords = [
            np.array([0.1, 0.2, 0.5]),
            np.array([np.nan, 0.3, 0.6]),  # 包含NaN
            np.array([0.0, -0.1, 0.4])
        ]
        
        is_valid = self.converter.validate_3d_coordinates(invalid_coords)
        print(f"无效坐标验证结果：{'通过' if is_valid else '失败'}")
        self.assertFalse(is_valid)
    
    def test_calculate_coordinate_quality(self):
        """测试坐标质量评估"""
        print("\n=== 测试坐标质量评估 ===")
        
        # 高质量坐标
        high_quality_coords = [
            np.array([0.1, 0.2, 0.5]),
            np.array([-0.1, 0.3, 0.6]),
            np.array([0.0, -0.1, 0.4])
        ]
        
        quality_score = self.converter.calculate_coordinate_quality(high_quality_coords)
        print(f"高质量坐标质量分数：{quality_score:.2f}")
        self.assertAlmostEqual(quality_score, 1.0, places=2)
        
        # 低质量坐标
        low_quality_coords = [
            np.array([0.1, 0.2, 0.5]),
            np.array([np.nan, 0.3, 0.6]),  # 无效坐标
            np.array([0.0, -0.1, 0.4])
        ]
        
        quality_score = self.converter.calculate_coordinate_quality(low_quality_coords)
        print(f"低质量坐标质量分数：{quality_score:.2f}")
        self.assertLess(quality_score, 1.0)
    
    def test_error_handling(self):
        """测试错误处理"""
        print("\n=== 测试错误处理 ===")
        
        # 测试无效的归一化坐标
        with self.assertRaises(ValueError):
            self.converter.normalized_to_pixel((-0.1, 0.5), (720, 1280))
        
        with self.assertRaises(ValueError):
            self.converter.normalized_to_pixel((1.5, 0.5), (720, 1280))
        
        # 测试无效的像素坐标
        with self.assertRaises(ValueError):
            self.converter.pixel_to_3d((-1, 0), self.depth_map)
        
        with self.assertRaises(ValueError):
            self.converter.pixel_to_3d((1280, 720), self.depth_map)
        
        # 测试无效的深度图
        invalid_depth_map = np.ones((720, 1280), dtype=np.float32) * -1.0  # 负深度值
        with self.assertRaises(ValueError):
            self.converter.pixel_to_3d((640, 360), invalid_depth_map)
        
        print("错误处理测试通过")
    
    def test_conversion_statistics(self):
        """测试转换统计信息"""
        print("\n=== 测试转换统计信息 ===")
        
        stats = self.converter.get_conversion_statistics()
        
        print(f"转换统计信息：{stats}")
        
        # 验证统计信息
        self.assertIsInstance(stats, dict)
        self.assertIn('camera_fx', stats)
        self.assertIn('camera_fy', stats)
        self.assertIn('camera_cx', stats)
        self.assertIn('camera_cy', stats)
        self.assertIn('depth_scale', stats)
        self.assertIn('depth_threshold', stats)
        self.assertIn('quality_threshold', stats)
        
        # 验证数值
        self.assertEqual(stats['camera_fx'], 925.0)
        self.assertEqual(stats['camera_fy'], 925.0)
        self.assertEqual(stats['camera_cx'], 640.0)
        self.assertEqual(stats['camera_cy'], 360.0)


def run_performance_test():
    """运行性能测试"""
    print("\n=== 性能测试 ===")
    
    # 创建大量测试数据
    num_landmarks = 1000
    test_landmarks = []
    for i in range(num_landmarks):
        x = 0.5 + 0.1 * np.sin(i * 0.01)
        y = 0.5 + 0.1 * np.cos(i * 0.01)
        z = 0.1 + 0.05 * np.sin(i * 0.02)
        test_landmarks.append((x, y, z))
    
    depth_map = np.ones((720, 1280), dtype=np.float32) * 0.5
    
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
    
    import time
    
    # 测试批量转换性能
    start_time = time.time()
    coords_3d = converter.batch_convert_landmarks(test_landmarks, depth_map)
    end_time = time.time()
    
    processing_time = end_time - start_time
    throughput = len(coords_3d) / processing_time
    
    print(f"处理 {len(test_landmarks)} 个关键点")
    print(f"成功转换 {len(coords_3d)} 个坐标")
    print(f"处理时间：{processing_time:.4f} 秒")
    print(f"吞吐量：{throughput:.2f} 坐标/秒")


if __name__ == "__main__":
    # 运行单元测试
    print("开始运行坐标转换器测试...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # 运行性能测试
    run_performance_test()
    
    print("\n所有测试完成！") 