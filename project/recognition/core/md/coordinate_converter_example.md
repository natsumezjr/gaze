# 坐标转换器示例代码

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坐标转换器完整示例
"""

import numpy as np
import cv2
from core.coordinate_converter import CoordinateConverter

# 相机参数配置
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

# 初始化转换器
converter = CoordinateConverter(camera_params)

# 模拟深度图数据
depth_map = np.random.uniform(0.5, 2.0, (720, 1280))

# 模拟MediaPipe关键点数据
landmarks = [
    (0.5, 0.5, 0.1),   # 左眼瞳孔
    (0.6, 0.5, 0.1),   # 右眼瞳孔
    (0.52, 0.48, 0.1), # 左眼虹膜点1
    (0.58, 0.52, 0.1), # 右眼虹膜点1
    (0.48, 0.52, 0.1), # 左眼虹膜点2
    (0.62, 0.48, 0.1)  # 右眼虹膜点2
]

# 执行坐标转换
coords_3d = converter.batch_convert_landmarks(landmarks, depth_map)

# 验证结果
is_valid = converter.validate_3d_coordinates(coords_3d)
quality_score = converter.calculate_coordinate_quality(coords_3d)

print(f"转换结果：{len(coords_3d)} 个有效坐标")
print(f"坐标有效性：{is_valid}")
print(f"质量分数：{quality_score:.2f}")

# 显示转换后的三维坐标
for i, coord in enumerate(coords_3d):
    print(f"关键点 {i+1}: X={coord[0]:.3f}m, Y={coord[1]:.3f}m, Z={coord[2]:.3f}m")
```

```python
# 实时处理示例
def real_time_coordinate_conversion():
    """实时坐标转换处理"""
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 模拟深度图（实际应用中需要从深度相机获取）
        depth_map = np.random.uniform(0.5, 2.0, (720, 1280))
        
        # 模拟关键点检测结果
        landmarks = [(0.5, 0.5, 0.1), (0.6, 0.5, 0.1)]
        
        # 坐标转换
        coords_3d = converter.batch_convert_landmarks(landmarks, depth_map)
        
        # 在图像上显示结果
        for i, (x_norm, y_norm, z) in enumerate(landmarks):
            x_pixel = int(x_norm * frame.shape[1])
            y_pixel = int(y_norm * frame.shape[0])
            cv2.circle(frame, (x_pixel, y_pixel), 5, (0, 255, 0), -1)
            
            if i < len(coords_3d):
                cv2.putText(frame, f"Z:{coords_3d[i][2]:.2f}m", 
                           (x_pixel+10, y_pixel), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow('Coordinate Conversion', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
```

```python
# 批量文件处理示例
def batch_process_landmarks(landmark_files, depth_files):
    """批量处理关键点文件"""
    results = []
    
    for landmark_file, depth_file in zip(landmark_files, depth_files):
        # 加载关键点数据
        landmarks = np.load(landmark_file)
        
        # 加载深度图
        depth_map = np.load(depth_file)
        
        # 坐标转换
        coords_3d = converter.batch_convert_landmarks(landmarks, depth_map)
        
        # 质量评估
        quality = converter.calculate_coordinate_quality(coords_3d)
        
        results.append({
            'file': landmark_file,
            'coords_3d': coords_3d,
            'quality': quality,
            'count': len(coords_3d)
        })
    
    return results
```

```python
# 眼睛坐标提取示例
def extract_eye_coordinates_example():
    """提取眼睛坐标示例"""
    # 模拟468个面部关键点
    face_landmarks = [(0.5 + i*0.001, 0.5 + i*0.001, 0.1) for i in range(468)]
    
    # 提取眼睛坐标
    eye_coords = converter.extract_eye_coordinates(face_landmarks, depth_map)
    
    print("眼睛坐标提取结果：")
    print(f"左眼瞳孔：{eye_coords['left_pupil']}")
    print(f"右眼瞳孔：{eye_coords['right_pupil']}")
    print(f"左眼虹膜点数：{len(eye_coords['left_iris'])}")
    print(f"右眼虹膜点数：{len(eye_coords['right_iris'])}")
    
    return eye_coords
```

```python
# 性能测试示例
def performance_test():
    """性能测试"""
    import time
    
    # 生成大量测试数据
    num_landmarks = 1000
    test_landmarks = [(np.random.random(), np.random.random(), 0.1) 
                     for _ in range(num_landmarks)]
    
    # 测试批量转换性能
    start_time = time.time()
    coords_3d = converter.batch_convert_landmarks(test_landmarks, depth_map)
    end_time = time.time()
    
    processing_time = end_time - start_time
    throughput = num_landmarks / processing_time
    
    print(f"处理 {num_landmarks} 个关键点耗时：{processing_time:.3f}秒")
    print(f"处理速度：{throughput:.1f} 关键点/秒")
    print(f"成功率：{len(coords_3d)/num_landmarks*100:.1f}%")
``` 