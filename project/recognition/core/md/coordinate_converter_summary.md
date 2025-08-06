# 坐标转换器实现总结

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坐标转换器模块实现总结
"""

# 核心类定义
class CoordinateConverter:
    """坐标转换器类"""
    
    def __init__(self, camera_params=None):
        """初始化坐标转换器"""
        self.camera_params = camera_params or self._get_default_camera_params()
        self._validate_camera_params()
        
        # 提取相机参数
        self.fx = self.camera_params["intrinsic_params"]["fx"]
        self.fy = self.camera_params["intrinsic_params"]["fy"]
        self.cx = self.camera_params["intrinsic_params"]["cx"]
        self.cy = self.camera_params["intrinsic_params"]["cy"]
        self.depth_scale = self.camera_params.get("depth_scale", 0.001)
    
    def _get_default_camera_params(self):
        """获取默认相机参数"""
        return {
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
    
    def _validate_camera_params(self):
        """验证相机参数有效性"""
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
    
    def normalized_to_pixel(self, norm_coords, image_shape):
        """归一化坐标转像素坐标"""
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
        x_pixel = int(x_norm * width)
        y_pixel = int(y_norm * height)
        
        # 确保像素坐标在有效范围内
        x_pixel = max(0, min(x_pixel, width - 1))
        y_pixel = max(0, min(y_pixel, height - 1))
        
        return (x_pixel, y_pixel)
    
    def pixel_to_3d(self, pixel_coords, depth_map):
        """像素坐标结合深度转三维坐标"""
        if not isinstance(pixel_coords, tuple) or len(pixel_coords) != 2:
            raise ValueError("pixel_coords必须是包含2个元素的元组")
        
        if not isinstance(depth_map, np.ndarray) or depth_map.ndim != 2:
            raise ValueError("depth_map必须是2维numpy数组")
        
        x_pixel, y_pixel = pixel_coords
        height, width = depth_map.shape
        
        # 检查像素坐标是否在图像范围内
        if not (0 <= x_pixel < width and 0 <= y_pixel < height):
            raise ValueError(f"像素坐标({x_pixel}, {y_pixel})超出图像范围({width}, {height})")
        
        # 获取深度值
        depth_meters = depth_map[y_pixel, x_pixel]
        
        # 验证深度值
        if not np.isfinite(depth_meters) or depth_meters <= DEPTH_VALIDATION_THRESHOLD:
            raise ValueError(f"无效的深度值：{depth_meters}，必须大于{DEPTH_VALIDATION_THRESHOLD}米")
        
        # 转换公式：X = (x_pixel - cx) × Z / fx, Y = (y_pixel - cy) × Z / fy, Z = depth
        x_3d = (x_pixel - self.cx) * depth_meters / self.fx
        y_3d = (y_pixel - self.cy) * depth_meters / self.fy
        z_3d = depth_meters
        
        return np.array([x_3d, y_3d, z_3d])
    
    def convert_landmark_to_3d(self, landmark, depth_map):
        """转换单个关键点到三维坐标"""
        try:
            # 归一化坐标转像素坐标
            norm_coords = (landmark[0], landmark[1])
            image_shape = (depth_map.shape[0], depth_map.shape[1])
            pixel_coords = self.normalized_to_pixel(norm_coords, image_shape)
            
            # 像素坐标转三维坐标
            coord_3d = self.pixel_to_3d(pixel_coords, depth_map)
            
            return coord_3d
            
        except (ValueError, IndexError) as e:
            logger.warning(f"关键点转换失败：{e}")
            return None
    
    def batch_convert_landmarks(self, landmarks, depth_map):
        """批量转换关键点为三维坐标"""
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
    
    def extract_eye_coordinates(self, landmarks, depth_map):
        """提取眼睛相关关键点的三维坐标"""
        if len(landmarks) < 478:  # MediaPipe面部关键点总数
            raise ValueError(f"关键点数量不足，需要至少478个，当前：{len(landmarks)}")
        
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
    
    def validate_3d_coordinates(self, coords_3d):
        """验证三维坐标的有效性"""
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
    
    def calculate_coordinate_quality(self, coords_3d):
        """计算坐标质量分数"""
        if not coords_3d:
            return 0.0
        
        valid_count = 0
        total_count = len(coords_3d)
        
        for coord in coords_3d:
            if (isinstance(coord, np.ndarray) and coord.shape == (3,) and
                np.all(np.isfinite(coord))):
                valid_count += 1
        
        return valid_count / total_count
    
    def get_conversion_statistics(self):
        """获取转换统计信息"""
        return {
            "camera_fx": self.fx,
            "camera_fy": self.fy,
            "camera_cx": self.cx,
            "camera_cy": self.cy,
            "depth_scale": self.depth_scale,
            "depth_threshold": DEPTH_VALIDATION_THRESHOLD,
            "quality_threshold": COORDINATE_QUALITY_THRESHOLD
        }
```

```python
# 兼容性函数（保持向后兼容）
def normalized_to_pixel(norm_coords, image_shape):
    """归一化坐标转像素坐标（兼容性函数）"""
    converter = CoordinateConverter()
    return converter.normalized_to_pixel(norm_coords, image_shape)

def pixel_to_3d(pixel_coords, depth_map, camera_params=None):
    """像素坐标转三维坐标（兼容性函数）"""
    converter = CoordinateConverter(camera_params)
    return converter.pixel_to_3d(pixel_coords, depth_map)

def batch_convert_landmarks(landmarks, depth_map, camera_params=None):
    """批量转换关键点（兼容性函数）"""
    converter = CoordinateConverter(camera_params)
    return converter.batch_convert_landmarks(landmarks, depth_map)

def validate_3d_coordinates(coords_3d):
    """验证三维坐标（兼容性函数）"""
    converter = CoordinateConverter()
    return converter.validate_3d_coordinates(coords_3d)
```

```python
# 测试函数
def test_coordinate_converter():
    """测试坐标转换器功能"""
    import numpy as np
    
    # 初始化转换器
    converter = CoordinateConverter()
    
    # 测试数据
    depth_map = np.random.uniform(0.5, 2.0, (720, 1280))
    landmarks = [(0.5, 0.5, 0.1), (0.6, 0.5, 0.1)]
    
    # 测试批量转换
    coords_3d = converter.batch_convert_landmarks(landmarks, depth_map)
    
    # 测试验证
    is_valid = converter.validate_3d_coordinates(coords_3d)
    quality = converter.calculate_coordinate_quality(coords_3d)
    
    print(f"测试结果：{len(coords_3d)} 个坐标，有效性：{is_valid}，质量：{quality:.2f}")
    
    return len(coords_3d) > 0 and is_valid

if __name__ == "__main__":
    test_coordinate_converter()
``` 