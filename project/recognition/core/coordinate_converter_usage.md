# 坐标转换器使用示例

## 函数列表

### 1. normalized_to_pixel(norm_coords, image_shape)
归一化坐标转像素坐标

### 2. pixel_to_3d(pixel_coords, depth_map, camera_params)
像素坐标结合深度转三维坐标

### 3. batch_convert_landmarks(landmarks, depth_map, camera_params)
批量转换关键点为三维坐标

### 4. validate_3d_coordinates(coords_3d)
验证三维坐标有效性

## 使用示例

### 基础坐标转换

```python
import cv2
import numpy as np
from core.coordinate_converter import (
    normalized_to_pixel, 
    pixel_to_3d, 
    batch_convert_landmarks, 
    validate_3d_coordinates
)

# 1. 归一化坐标转像素坐标
norm_coords = (0.5, 0.3)  # 归一化坐标
image_shape = (720, 1280)  # 图像尺寸 (高度, 宽度)
pixel_coords = normalized_to_pixel(norm_coords, image_shape)
print(f"归一化坐标 {norm_coords} -> 像素坐标 {pixel_coords}")

# 2. 像素坐标转三维坐标
depth_map = cv2.imread("depth_image.png", cv2.IMREAD_ANYDEPTH)  # 读取深度图
camera_params = {
    "intrinsic_params": {
        "fx": 925.0,  # 焦距x
        "fy": 925.0,  # 焦距y
        "cx": 640.0,  # 主点x坐标
        "cy": 360.0   # 主点y坐标
    },
    "depth_scale": 0.001  # 深度值缩放因子
}

coord_3d = pixel_to_3d(pixel_coords, depth_map, camera_params)
print(f"像素坐标 {pixel_coords} -> 三维坐标 {coord_3d}")
```

### 批量处理关键点

```python
# 从landmark_extractor获取的关键点
from core.landmark_extractor import extract_landmarks

# 读取RGB图像
rgb_image = cv2.imread("face_image.jpg")
landmarks = extract_landmarks(rgb_image)

if landmarks:
    # 批量转换为三维坐标
    coords_3d = batch_convert_landmarks(landmarks, depth_map, camera_params)
    
    # 验证三维坐标
    if validate_3d_coordinates(coords_3d):
        print(f"成功转换 {len(coords_3d)} 个关键点")
        
        # 显示前几个关键点的三维坐标
        for i, coord in enumerate(coords_3d[:5]):
            print(f"关键点 {i+1}: X={coord[0]:.3f}m, Y={coord[1]:.3f}m, Z={coord[2]:.3f}m")
    else:
        print("三维坐标验证失败")
```

### 实时处理示例

```python
import cv2
import numpy as np
from core.coordinate_converter import batch_convert_landmarks, validate_3d_coordinates

# 相机参数（需要根据实际相机标定结果调整）
camera_params = {
    "intrinsic_params": {
        "fx": 925.0,
        "fy": 925.0,
        "cx": 640.0,
        "cy": 360.0
    },
    "depth_scale": 0.001
}

# 打开RGB-D相机
cap_rgb = cv2.VideoCapture(0)  # RGB相机
cap_depth = cv2.VideoCapture(1)  # 深度相机（如果有的话）

while True:
    ret_rgb, rgb_frame = cap_rgb.read()
    ret_depth, depth_frame = cap_depth.read()
    
    if ret_rgb and ret_depth:
        # 提取关键点
        landmarks = extract_landmarks(rgb_frame)
        
        if landmarks:
            # 转换为三维坐标
            coords_3d = batch_convert_landmarks(landmarks, depth_frame, camera_params)
            
            if validate_3d_coordinates(coords_3d):
                # 在图像上显示关键点
                for i, (x, y, z) in enumerate(landmarks):
                    cv2.circle(rgb_frame, (int(x), int(y)), 2, (0, 255, 0), -1)
                
                # 显示三维坐标信息
                cv2.putText(rgb_frame, f"3D Points: {len(coords_3d)}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('RGB Frame', rgb_frame)
        cv2.imshow('Depth Frame', depth_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_rgb.release()
cap_depth.release()
cv2.destroyAllWindows()
```

### 相机标定参数示例

```python
# Intel RealSense D435i 相机参数示例
camera_params = {
    "camera_name": "Intel RealSense D435i",
    "camera_type": "RGB-D",
    
    "intrinsic_params": {
        "fx": 925.0,  # 焦距x（像素单位）
        "fy": 925.0,  # 焦距y（像素单位）
        "cx": 640.0,  # 主点x坐标（像素）
        "cy": 360.0   # 主点y坐标（像素）
    },
    
    "image_resolution": {
        "width": 1280,
        "height": 720
    },
    
    "depth_scale": 0.001,  # 深度值缩放因子（米/单位）
    "min_depth": 0.1,      # 最小有效深度（米）
    "max_depth": 10.0,     # 最大有效深度（米）
    
    "distortion_coeffs": {
        "k1": 0.0,  # 径向畸变系数1
        "k2": 0.0,  # 径向畸变系数2
        "p1": 0.0,  # 切向畸变系数1
        "p2": 0.0,  # 切向畸变系数2
        "k3": 0.0   # 径向畸变系数3
    }
}
```

## 运行测试

```bash
# 运行坐标转换测试
cd project/recognition
python test_coordinate_converter.py
```

## 注意事项

1. **深度图格式**: 确保深度图是16位无符号整数格式
2. **相机标定**: 使用前需要先进行相机标定获取准确的内参
3. **坐标范围**: 三维坐标验证的范围可以根据实际应用调整
4. **错误处理**: 所有函数都包含输入验证和错误处理 