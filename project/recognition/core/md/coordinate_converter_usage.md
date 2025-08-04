# 坐标转换器使用示例

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坐标转换器使用示例
"""

import numpy as np
from core.coordinate_converter import CoordinateConverter

# 初始化坐标转换器
camera_params = {
    "intrinsic_params": {
        "fx": 925.0,  # 焦距x（像素）
        "fy": 925.0,  # 焦距y（像素）
        "cx": 640.0,  # 主点x坐标（像素）
        "cy": 360.0   # 主点y坐标（像素）
    },
    "depth_scale": 0.001,
    "image_resolution": {
        "width": 1280,
        "height": 720
    }
}

converter = CoordinateConverter(camera_params)

# 示例1：归一化坐标转像素坐标
norm_coords = (0.5, 0.5)  # 图像中心
image_shape = (720, 1280)  # 高度720，宽度1280
pixel_coords = converter.normalized_to_pixel(norm_coords, image_shape)
print(f"归一化坐标 {norm_coords} -> 像素坐标 {pixel_coords}")

# 示例2：像素坐标转三维坐标
depth_map = np.random.uniform(0.5, 2.0, (720, 1280))  # 模拟深度图
coord_3d = converter.pixel_to_3d(pixel_coords, depth_map)
print(f"像素坐标 {pixel_coords} -> 三维坐标 {coord_3d}")

# 示例3：批量转换关键点
landmarks = [
    (0.5, 0.5, 0.1),  # 左眼瞳孔
    (0.6, 0.5, 0.1),  # 右眼瞳孔
    (0.52, 0.48, 0.1),  # 左眼虹膜点1
    (0.58, 0.52, 0.1)   # 右眼虹膜点1
]

coords_3d = converter.batch_convert_landmarks(landmarks, depth_map)
print(f"批量转换结果：{len(coords_3d)} 个有效坐标")

# 示例4：提取眼睛坐标
# 模拟468个面部关键点
face_landmarks = [(0.5 + i*0.001, 0.5 + i*0.001, 0.1) for i in range(468)]
eye_coords = converter.extract_eye_coordinates(face_landmarks, depth_map)
print(f"眼睛坐标提取完成：{eye_coords}")

# 示例5：坐标验证和质量评估
is_valid = converter.validate_3d_coordinates(coords_3d)
quality_score = converter.calculate_coordinate_quality(coords_3d)
print(f"坐标有效性：{is_valid}, 质量分数：{quality_score:.2f}")

# 示例6：获取转换统计信息
stats = converter.get_conversion_statistics()
print(f"转换统计信息：{stats}")
```

```python
# 相机标定参数获取示例
def get_camera_params_from_calibration():
    """从相机标定文件获取参数"""
    import json
    
    # 读取标定文件
    with open('camera_calibration.json', 'r') as f:
        calibration_data = json.load(f)
    
    # 提取内参
    intrinsic_matrix = calibration_data['intrinsic_matrix']
    fx = intrinsic_matrix[0][0]
    fy = intrinsic_matrix[1][1]
    cx = intrinsic_matrix[0][2]
    cy = intrinsic_matrix[1][2]
    
    return {
        "intrinsic_params": {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy
        },
        "depth_scale": 0.001,
        "image_resolution": {
            "width": 1280,
            "height": 720
        }
    }
```

```python
# 错误处理示例
def safe_coordinate_conversion(landmarks, depth_map):
    """安全的坐标转换，包含错误处理"""
    try:
        converter = CoordinateConverter()
        coords_3d = converter.batch_convert_landmarks(landmarks, depth_map)
        
        # 验证转换结果
        if not converter.validate_3d_coordinates(coords_3d):
            print("警告：部分坐标转换质量较低")
            
        return coords_3d
        
    except ValueError as e:
        print(f"坐标转换错误：{e}")
        return []
    except Exception as e:
        print(f"未知错误：{e}")
        return []
``` 