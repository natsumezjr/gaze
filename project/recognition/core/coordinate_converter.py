#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坐标转换器模块 (Coordinate Converter Module)

本模块负责将MediaPipe检测到的归一化坐标转换为像素坐标，
并结合深度信息转换为相机参考系下的三维坐标。

主要功能：
1. 归一化坐标转像素坐标
2. 像素坐标结合深度转三维坐标
3. 批量转换关键点
4. 坐标验证和质量评估

理论基础：
- 归一化坐标：MediaPipe输出的坐标范围在[0,1]，表示相对位置
- 像素坐标：图像中的实际像素位置
- 三维坐标：相机参考系下的真实三维位置（米单位）

关键公式：
1. 归一化坐标转像素坐标：
   x_pixel = x_norm × W, y_pixel = y_norm × H
2. 像素坐标转三维坐标：
   X = (x_pixel - cx) × Z / fx
   Y = (y_pixel - cy) × Z / fy
   Z = depth_value

作者：根据structure.md文档要求实现
日期：2024
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Union
import logging
from pathlib import Path
import cv2

# 导入项目配置
try:
    from ..config.constants import (
        LEFT_PUPIL_INDEX, RIGHT_PUPIL_INDEX,
        LEFT_IRIS_INDICES, RIGHT_IRIS_INDICES,
        STATUS_SUCCESS, STATUS_INVALID_DEPTH
    )
    from ..config.settings import (
        DEPTH_VALIDATION_THRESHOLD,
        COORDINATE_QUALITY_THRESHOLD
    )
except ImportError:
    try:
        # 尝试绝对导入
        from config.constants import (
            LEFT_PUPIL_INDEX, RIGHT_PUPIL_INDEX,
            LEFT_IRIS_INDICES, RIGHT_IRIS_INDICES,
            STATUS_SUCCESS, STATUS_INVALID_DEPTH
        )
        from config.settings import (
            DEPTH_VALIDATION_THRESHOLD,
            COORDINATE_QUALITY_THRESHOLD
        )
    except ImportError:
        # 如果导入失败，使用默认值
        LEFT_PUPIL_INDEX = 468
        RIGHT_PUPIL_INDEX = 473
        LEFT_IRIS_INDICES = [469, 470, 471, 472]
        RIGHT_IRIS_INDICES = [474, 475, 476, 477]
        STATUS_SUCCESS = 0
        STATUS_INVALID_DEPTH = 3
        DEPTH_VALIDATION_THRESHOLD = 0.1
        COORDINATE_QUALITY_THRESHOLD = 0.8

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoordinateConverter:
    """
    坐标转换器类
    
    提供完整的坐标转换功能，包括：
    - 归一化坐标到像素坐标的转换
    - 像素坐标结合深度信息到三维坐标的转换
    - 批量关键点转换
    - 坐标质量评估和验证
    """
    
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super(CoordinateConverter, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, camera_params: Optional[Dict] = None):
        """
        初始化坐标转换器
        
        Args:
            camera_params: 相机参数字典，包含内参等信息
                格式：
                {
                    "intrinsic_params": {
                        "fx": float,  # 焦距x（像素）
                        "fy": float,  # 焦距y（像素）
                        "cx": float,  # 主点x坐标（像素）
                        "cy": float   # 主点y坐标（像素）
                    },
                    "depth_scale": float,  # 深度值缩放因子
                    "image_resolution": {
                        "width": int,
                        "height": int
                    }
                }
        """
        self.camera_params = camera_params or self._get_default_camera_params()
        self._validate_camera_params()
        
        # 提取相机参数
        self.fx = self.camera_params["intrinsic_params"]["fx"]
        self.fy = self.camera_params["intrinsic_params"]["fy"]
        self.cx = self.camera_params["intrinsic_params"]["cx"]
        self.cy = self.camera_params["intrinsic_params"]["cy"]
        self.depth_scale = self.camera_params.get("depth_scale", 0.001)
        
        logger.info(f"坐标转换器初始化完成，相机参数：fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")
    
    def _get_default_camera_params(self) -> Dict:
        """获取默认相机参数"""
        return {
            "intrinsic_params": {
                "fx": 925.0,  # 默认焦距x
                "fy": 925.0,  # 默认焦距y
                "cx": 640.0,  # 默认主点x
                "cy": 360.0   # 默认主点y
            },
            "depth_scale": 0.001,
            "image_resolution": {
                "width": 1280,
                "height": 720
            }
        }
    
    def _validate_camera_params(self):
        """验证相机参数的有效性"""
        if not isinstance(self.camera_params, dict):
            raise ValueError("camera_params必须是字典类型")
        
        intrinsic_params = self.camera_params.get("intrinsic_params")
        if not intrinsic_params:
            raise ValueError("缺少intrinsic_params参数")
        
        required_keys = ["fx", "fy", "cx", "cy"]
        for key in required_keys:
            if key not in intrinsic_params:
                raise ValueError(f"缺少必需的相机内参：{key}")
            
            value = intrinsic_params[key]
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"相机内参{key}必须是正数，当前值：{value}")
    
    def normalized_to_pixel(self, norm_coords: Tuple[float, float], 
                           image_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        归一化坐标转像素坐标
        
        理论基础：
        归一化坐标是MediaPipe输出的相对坐标，范围在[0,1]之间。
        转换公式：x_pixel = x_norm × W, y_pixel = y_norm × H
        
        Args:
            norm_coords: 归一化坐标 (x_norm, y_norm)，范围[0,1]
            image_shape: 图像尺寸 (height, width)
        
        Returns:
            pixel_coords: 像素坐标 (x_pixel, y_pixel)
        
        Raises:
            ValueError: 输入参数无效时抛出
        
        示例：
            norm_coords = (0.5, 0.5)  # 图像中心
            image_shape = (720, 1280)  # 高度720，宽度1280
            pixel_coords = (640, 360)  # 像素坐标
        """
        # 输入验证
        if not isinstance(norm_coords, tuple) or len(norm_coords) != 2:
            raise ValueError("norm_coords必须是包含2个元素的元组")
        
        if not isinstance(image_shape, tuple) or len(image_shape) != 2:
            raise ValueError("image_shape必须是包含2个元素的元组")
        
        x_norm, y_norm = norm_coords
        height, width = image_shape
        
        # 验证归一化坐标范围
        if not (0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0):
            raise ValueError(f"归一化坐标必须在[0,1]范围内，当前值：({x_norm}, {y_norm})")
        
        # 验证图像尺寸
        if width <= 0 or height <= 0:
            raise ValueError(f"图像尺寸必须为正数，当前值：({width}, {height})")
        
        # 转换公式：x_pixel = x_norm × W, y_pixel = y_norm × H
        x_pixel = x_norm * width
        y_pixel = y_norm * height
        
        # 确保像素坐标在有效范围内（保留浮点精度）
        x_pixel = max(0.0, min(x_pixel, width - 1.0))
        y_pixel = max(0.0, min(y_pixel, height - 1.0))
        
        return (x_pixel, y_pixel)
    
    def pixel_to_3d(self, pixel_coords: Tuple[int, int], 
                   depth_map_meters: np.ndarray) -> np.ndarray:
        """
        像素坐标结合深度转三维坐标
        
        理论基础：
        根据相机内参和深度信息，将像素坐标转换为相机参考系下的三维坐标。
        转换公式：
        X = (x_pixel - cx) × Z / fx
        Y = (y_pixel - cy) × Z / fy
        Z = depth_value
        
        Args:
            pixel_coords: 像素坐标 (x_pixel, y_pixel)，支持浮点数
            depth_map_meters: 深度图，numpy数组，单位：米
        
        Returns:
            coord_3d: 三维坐标 [X, Y, Z]，单位：米
        
        Raises:
            ValueError: 输入参数无效或深度值无效时抛出
        
        示例：
            pixel_coords = (640.5, 360.3)  # 浮点像素坐标
            depth_map_meters = np.array([[0.5, 0.6], [0.7, 0.8]])  # 深度图（米）
            coord_3d = [0.0, 0.0, 0.5]  # 三维坐标（米）
        """
        # 输入验证
        if not isinstance(pixel_coords, tuple) or len(pixel_coords) != 2:
            raise ValueError("pixel_coords必须是包含2个元素的元组")
        
        if not isinstance(depth_map_meters, np.ndarray) or depth_map_meters.ndim != 2:
            raise ValueError("depth_map_meters必须是2维numpy数组")
        
        x_pixel, y_pixel = pixel_coords
        height, width = depth_map_meters.shape
        
        # 检查像素坐标是否在图像范围内（支持浮点坐标）
        if not (0.0 <= x_pixel < width and 0.0 <= y_pixel < height):
            raise ValueError(f"像素坐标({x_pixel}, {y_pixel})超出图像范围({width}, {height})")
        
        # 获取深度值（使用OpenCV插值）
        depth_meters = self._get_depth_interpolated(depth_map_meters, (x_pixel, y_pixel))
        
        # 验证深度值
        if not np.isfinite(depth_meters) or depth_meters <= DEPTH_VALIDATION_THRESHOLD:
            raise ValueError(f"无效的深度值：{depth_meters}，必须大于{DEPTH_VALIDATION_THRESHOLD}米")
        
        # 转换公式：X = (x_pixel - cx) × Z / fx, Y = (y_pixel - cy) × Z / fy, Z = depth
        x_3d = (x_pixel - self.cx) * depth_meters / self.fx
        y_3d = (y_pixel - self.cy) * depth_meters / self.fy
        z_3d = depth_meters
        
        return np.array([x_3d, y_3d, z_3d])
    
    def _get_depth_interpolated(self, depth_map: np.ndarray, 
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
    
    def convert_landmark_to_3d(self, landmark: Tuple[float, float, float], 
                              depth_map: np.ndarray) -> Optional[np.ndarray]:
        """
        转换单个关键点到三维坐标
        
        Args:
            landmark: 关键点 (x_norm, y_norm, z_relative)
            depth_map: 深度图
        
        Returns:
            coord_3d: 三维坐标，转换失败时返回None
        """
        try:
            # 归一化坐标转像素坐标（现在返回浮点数）
            norm_coords = (landmark[0], landmark[1])
            image_shape = (depth_map.shape[0], depth_map.shape[1])
            pixel_coords = self.normalized_to_pixel(norm_coords, image_shape)
            
            # 像素坐标转三维坐标（使用插值）
            coord_3d = self.pixel_to_3d(pixel_coords, depth_map)
            
            return coord_3d
            
        except (ValueError, IndexError) as e:
            logger.warning(f"关键点转换失败：{e}")
            return None
    
    def batch_convert_landmarks(self, landmarks: List[Tuple[float, float, float]], 
                               depth_map: np.ndarray) -> List[np.ndarray]:
        """
        批量转换关键点为三维坐标
        
        理论基础：
        对多个关键点进行批量转换，提高处理效率。
        每个关键点都经过归一化坐标→像素坐标→三维坐标的转换流程。
        
        Args:
            landmarks: 关键点列表，每个元素为(x_norm, y_norm, z_relative)
            depth_map: 深度图，numpy数组
        
        Returns:
            coords_3d: 三维坐标列表，每个元素为[X, Y, Z]
        
        示例：
            landmarks = [(0.5, 0.5, 0.1), (0.3, 0.7, 0.2)]  # 归一化坐标
            depth_map = np.array([[0.5, 0.6], [0.7, 0.8]])  # 深度图
            coords_3d = [[0.0, 0.0, 0.5], [-0.1, 0.1, 0.7]]  # 三维坐标
        """
        if not isinstance(landmarks, list):
            raise ValueError("landmarks必须是列表类型")
        
        if not isinstance(depth_map, np.ndarray) or depth_map.ndim != 2:
            raise ValueError("depth_map必须是2维numpy数组")
        
        coords_3d = []
        success_count = 0
        
        for i, landmark in enumerate(landmarks):
            if not isinstance(landmark, tuple) or len(landmark) != 3:
                logger.warning(f"跳过无效的关键点 {i}：{landmark}")
                continue
            
            coord_3d = self.convert_landmark_to_3d(landmark, depth_map)
            if coord_3d is not None:
                coords_3d.append(coord_3d)
                success_count += 1
            else:
                logger.warning(f"关键点 {i} 转换失败")
        
        logger.info(f"批量转换完成：成功 {success_count}/{len(landmarks)} 个关键点")
        return coords_3d
    
    def extract_eye_coordinates(self, landmarks: List[Tuple[float, float, float]], 
                               depth_map: np.ndarray) -> Dict[str, np.ndarray]:
        """
        提取眼睛相关关键点的三维坐标
        
        理论基础：
        根据MediaPipe的关键点索引，提取瞳孔中心和虹膜边界点的三维坐标。
        这些坐标用于后续的视线拟合和追踪。
        
        Args:
            landmarks: 468个面部关键点列表
            depth_map: 深度图
        
        Returns:
            eye_coords: 眼睛坐标字典，包含：
                - left_pupil: 左眼瞳孔中心
                - right_pupil: 右眼瞳孔中心
                - left_iris: 左眼虹膜边界点列表
                - right_iris: 右眼虹膜边界点列表
        """
        eye_coords = {
            "left_pupil": None,
            "right_pupil": None,
            "left_iris": [],
            "right_iris": []
        }
        
        try:
            # 提取左眼瞳孔中心
            left_pupil_landmark = landmarks[LEFT_PUPIL_INDEX]
            eye_coords["left_pupil"] = self.convert_landmark_to_3d(left_pupil_landmark, depth_map)
            
            # 提取右眼瞳孔中心
            right_pupil_landmark = landmarks[RIGHT_PUPIL_INDEX]
            eye_coords["right_pupil"] = self.convert_landmark_to_3d(right_pupil_landmark, depth_map)
            
            # 提取左眼虹膜边界点
            for idx in LEFT_IRIS_INDICES:
                iris_landmark = landmarks[idx]
                coord_3d = self.convert_landmark_to_3d(iris_landmark, depth_map)
                if coord_3d is not None:
                    eye_coords["left_iris"].append(coord_3d)
            
            # 提取右眼虹膜边界点
            for idx in RIGHT_IRIS_INDICES:
                iris_landmark = landmarks[idx]
                coord_3d = self.convert_landmark_to_3d(iris_landmark, depth_map)
                if coord_3d is not None:
                    eye_coords["right_iris"].append(coord_3d)
            
            logger.info(f"眼睛坐标提取完成：左眼瞳孔{'成功' if eye_coords['left_pupil'] is not None else '失败'}, "
                       f"右眼瞳孔{'成功' if eye_coords['right_pupil'] is not None else '失败'}, "
                       f"左眼虹膜{len(eye_coords['left_iris'])}个点, "
                       f"右眼虹膜{len(eye_coords['right_iris'])}个点")
            
        except (IndexError, ValueError) as e:
            logger.error(f"眼睛坐标提取失败：{e}")
        
        return eye_coords
    
    def validate_3d_coordinates(self, coords_3d: List[np.ndarray]) -> bool:
        """
        验证三维坐标的有效性
        
        理论基础：
        对转换后的三维坐标进行质量检查，确保坐标值在合理范围内。
        包括数值有效性、范围合理性等检查。
        
        Args:
            coords_3d: 三维坐标列表
        
        Returns:
            is_valid: 坐标是否有效
        
        示例：
            coords_3d = [[0.1, 0.2, 0.5], [-0.1, 0.3, 0.6]]  # 有效坐标
            is_valid = True
        """
        if not isinstance(coords_3d, list) or len(coords_3d) == 0:
            return False
        
        valid_count = 0
        for coord in coords_3d:
            if not isinstance(coord, np.ndarray) or coord.shape != (3,):
                continue
            
            # 检查数值有效性
            if not np.all(np.isfinite(coord)):
                continue
            
            x, y, z = coord
            
            # 检查坐标范围（根据实际应用调整）
            if not (-10.0 <= x <= 10.0 and -10.0 <= y <= 10.0 and 0.1 <= z <= 10.0):
                continue
            
            valid_count += 1
        
        # 计算有效坐标比例
        valid_ratio = valid_count / len(coords_3d)
        is_valid = valid_ratio >= COORDINATE_QUALITY_THRESHOLD
        
        logger.info(f"坐标验证结果：{valid_count}/{len(coords_3d)} 有效，比例：{valid_ratio:.2f}")
        
        return is_valid
    
    def calculate_coordinate_quality(self, coords_3d: List[np.ndarray]) -> float:
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
            if (isinstance(coord, np.ndarray) and coord.shape == (3,) and 
                np.all(np.isfinite(coord))):
                valid_count += 1
        
        return valid_count / total_count
    
    def get_conversion_statistics(self) -> Dict[str, Union[int, float]]:
        """
        获取转换统计信息
        
        Returns:
            stats: 统计信息字典
        """
        return {
            "camera_fx": self.fx,
            "camera_fy": self.fy,
            "camera_cx": self.cx,
            "camera_cy": self.cy,
            "depth_scale": self.depth_scale,
            "depth_threshold": DEPTH_VALIDATION_THRESHOLD,
            "quality_threshold": COORDINATE_QUALITY_THRESHOLD
        }


# 兼容性函数（保持向后兼容）
def normalized_to_pixel(norm_coords: Tuple[float, float], 
                       image_shape: Tuple[int, int]) -> Tuple[int, int]:
    """归一化坐标转像素坐标（兼容性函数）"""
    converter = CoordinateConverter()
    return converter.normalized_to_pixel(norm_coords, image_shape)


def pixel_to_3d(pixel_coords: Tuple[int, int], 
                depth_map_meters: np.ndarray, 
                camera_params: dict) -> np.ndarray:
    """像素坐标转三维坐标（兼容性函数）"""
    converter = CoordinateConverter(camera_params)
    return converter.pixel_to_3d(pixel_coords, depth_map_meters)


def batch_convert_landmarks(landmarks: List[Tuple[float, float, float]], 
                           depth_map_meters: np.ndarray, 
                           camera_params: dict) -> List[np.ndarray]:
    """批量转换关键点（兼容性函数）"""
    converter = CoordinateConverter(camera_params)
    return converter.batch_convert_landmarks(landmarks, depth_map_meters)


def validate_3d_coordinates(coords_3d: List[np.ndarray]) -> bool:
    """验证三维坐标（兼容性函数）"""
    converter = CoordinateConverter()
    return converter.validate_3d_coordinates(coords_3d)


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
    
    # 初始化转换器
    converter = CoordinateConverter(test_camera_params)
    
    # 测试归一化坐标转像素坐标
    print("\n1. 测试归一化坐标转像素坐标")
    norm_coords = (0.5, 0.5)  # 图像中心
    image_shape = (720, 1280)
    pixel_coords = converter.normalized_to_pixel(norm_coords, image_shape)
    print(f"归一化坐标 {norm_coords} -> 像素坐标 {pixel_coords}")
    
    # 测试像素坐标转三维坐标
    print("\n2. 测试像素坐标转三维坐标")
    coord_3d = converter.pixel_to_3d(pixel_coords, depth_map)
    print(f"像素坐标 {pixel_coords} -> 三维坐标 {coord_3d}")
    
    # 测试批量转换
    print("\n3. 测试批量转换")
    test_landmarks = [
        (0.5, 0.5, 0.1),  # 中心点
        (0.3, 0.7, 0.2),  # 其他点
        (0.8, 0.2, 0.3)
    ]
    coords_3d = converter.batch_convert_landmarks(test_landmarks, depth_map)
    print(f"批量转换结果：{len(coords_3d)} 个有效坐标")
    
    # 测试坐标验证
    print("\n4. 测试坐标验证")
    is_valid = converter.validate_3d_coordinates(coords_3d)
    print(f"坐标验证结果：{'通过' if is_valid else '失败'}")
    
    # 测试质量评估
    print("\n5. 测试质量评估")
    quality_score = converter.calculate_coordinate_quality(coords_3d)
    print(f"质量分数：{quality_score:.2f}")
    
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    test_coordinate_converter()
