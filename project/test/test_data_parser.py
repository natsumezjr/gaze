import unittest
import numpy as np
import cv2
import json
import tempfile
import os
from pathlib import Path

# 添加项目根目录到Python路径
import sys
sys.path.append(str(Path(__file__).parent.parent))

from recognition.utils.data_parser import (
    decode_base64_image,
    parse_rgbd_json,
    load_rgbd_from_file,
    encode_image_to_base64,
    validate_image_format
)

class TestDataParser(unittest.TestCase):
    
    def setUp(self):
        """设置测试环境"""
        # 创建一个测试图像
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 创建测试JSON数据
        self.test_json_data = {
            "frame_id": 1,
            "timestamp": "2024-01-15T10:30:00",
            "camera": {"resolution": [640, 480]},
            "rgb_data": encode_image_to_base64(self.test_image),
            "depth_data": None,
            "eye_centers": {"left": None, "right": None},
            "pupil_centers": {"left": None, "right": None},
            "iris_boundaries": {"left": [], "right": []}
        }
    
    def test_encode_decode_base64_image(self):
        """测试base64编码和解码"""
        # 编码
        b64_string = encode_image_to_base64(self.test_image)
        self.assertIsInstance(b64_string, str)
        self.assertTrue(len(b64_string) > 0)
        
        # 解码
        decoded_image = decode_base64_image(b64_string)
        self.assertIsInstance(decoded_image, np.ndarray)
        self.assertEqual(decoded_image.shape, self.test_image.shape)
        self.assertEqual(decoded_image.dtype, self.test_image.dtype)
        
        # 验证图像内容
        np.testing.assert_array_equal(decoded_image, self.test_image)
    
    def test_parse_rgbd_json_dict(self):
        """测试解析JSON字典"""
        rgb_image, depth_map, metadata = parse_rgbd_json(self.test_json_data)
        
        # 验证返回类型
        self.assertIsInstance(rgb_image, np.ndarray)
        self.assertIsNone(depth_map)
        self.assertIsInstance(metadata, dict)
        
        # 验证图像尺寸
        self.assertEqual(rgb_image.shape, self.test_image.shape)
        
        # 验证元数据
        self.assertEqual(metadata["frame_id"], 1)
        self.assertEqual(metadata["timestamp"], "2024-01-15T10:30:00")
        self.assertIn("camera", metadata)
    
    def test_parse_rgbd_json_string(self):
        """测试解析JSON字符串"""
        json_string = json.dumps(self.test_json_data)
        rgb_image, depth_map, metadata = parse_rgbd_json(json_string)
        
        # 验证返回类型
        self.assertIsInstance(rgb_image, np.ndarray)
        self.assertIsNone(depth_map)
        self.assertIsInstance(metadata, dict)
        
        # 验证图像尺寸
        self.assertEqual(rgb_image.shape, self.test_image.shape)
    
    def test_load_rgbd_from_file(self):
        """测试从文件加载数据"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.test_json_data, f)
            temp_file_path = f.name
        
        try:
            # 加载数据
            rgb_image, depth_map, metadata = load_rgbd_from_file(temp_file_path)
            
            # 验证返回类型
            self.assertIsInstance(rgb_image, np.ndarray)
            self.assertIsNone(depth_map)
            self.assertIsInstance(metadata, dict)
            
            # 验证图像尺寸
            self.assertEqual(rgb_image.shape, self.test_image.shape)
            
        finally:
            # 清理临时文件
            os.unlink(temp_file_path)
    
    def test_validate_image_format(self):
        """测试图像格式验证"""
        # 有效图像
        valid_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.assertTrue(validate_image_format(valid_image))
        
        # 无效图像：错误的维度
        invalid_image_1 = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        self.assertFalse(validate_image_format(invalid_image_1))
        
        # 无效图像：错误的通道数
        invalid_image_2 = np.random.randint(0, 255, (480, 640, 4), dtype=np.uint8)
        self.assertFalse(validate_image_format(invalid_image_2))
        
        # 无效图像：错误的数据类型
        invalid_image_3 = np.random.randint(0, 255, (480, 640, 3), dtype=np.float32)
        self.assertFalse(validate_image_format(invalid_image_3))
        
        # 无效图像：非numpy数组
        self.assertFalse(validate_image_format("not an image"))
    
    def test_error_handling(self):
        """测试错误处理"""
        # 测试无效的base64字符串
        with self.assertRaises(ValueError):
            decode_base64_image("invalid_base64")
        
        # 测试缺少必要字段的JSON
        invalid_json = {"frame_id": 1}  # 缺少rgb_data
        with self.assertRaises(ValueError):
            parse_rgbd_json(invalid_json)
        
        # 测试不存在的文件
        with self.assertRaises(FileNotFoundError):
            load_rgbd_from_file("nonexistent_file.json")
        
        # 测试无效的图像编码
        with self.assertRaises(ValueError):
            encode_image_to_base64("not an image")

if __name__ == '__main__':
    unittest.main() 