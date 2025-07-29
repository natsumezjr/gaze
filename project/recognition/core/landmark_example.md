# 人脸关键点提取函数使用示例

## 函数概述

`extract_landmarks(rgb_image)` 函数用于从RGB图像中提取468个人脸关键点。

## 参数说明

### 输入参数：`rgb_image`
- **类型**: `numpy.ndarray`
- **形状**: `(height, width, 3)` - 高度、宽度、3个颜色通道
- **数据类型**: `uint8` (0-255的整数)
- **颜色格式**: RGB或BGR格式都可以，函数会自动转换
- **来源**: 通常来自：
  - 摄像头捕获的图像
  - 图像文件读取
  - 视频帧

## 返回值说明

### 输出：关键点列表
- **类型**: `List[Tuple[float, float, float]]`
- **长度**: 468个关键点（如果检测到人脸）
- **格式**: 每个关键点是一个包含3个浮点数的元组 `(x, y, z)`
  - `x`: 像素坐标X（水平位置）
  - `y`: 像素坐标Y（垂直位置）
  - `z`: 相对深度值（值越小表示越近）

## 使用示例

### 示例1：从摄像头读取图像

```python
import cv2
from core.landmark_extractor import extract_landmarks

# 打开摄像头
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if ret:
    # 提取关键点
    landmarks = extract_landmarks(frame)
    
    if landmarks:
        print(f"检测到 {len(landmarks)} 个关键点")
        # landmarks[0] 是第一个关键点的坐标 (x, y, z)
        print(f"第一个关键点: {landmarks[0]}")
    else:
        print("未检测到人脸")
```

### 示例2：从图像文件读取

```python
import cv2
from core.landmark_extractor import extract_landmarks

# 读取图像文件
image_path = "path/to/your/image.jpg"
frame = cv2.imread(image_path)

if frame is not None:
    landmarks = extract_landmarks(frame)
    
    if landmarks:
        print(f"成功提取 {len(landmarks)} 个关键点")
        
        # 在图像上绘制关键点
        for x, y, z in landmarks:
            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
        
        # 显示结果
        cv2.imshow('Face Landmarks', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
```

### 示例3：获取特定部位的关键点

```python
from core.landmark_extractor import extract_landmarks, get_landmark_indices

# 获取关键点索引
indices = get_landmark_indices()

# 提取关键点
landmarks = extract_landmarks(image)

if landmarks:
    # 获取右眼关键点
    right_eye_indices = indices["right_eye_contour"]  # [0, 1, 2, ..., 9]
    right_eye_landmarks = [landmarks[i] for i in right_eye_indices]
    
    # 获取左眼关键点
    left_eye_indices = indices["left_eye_contour"]    # [10, 11, 12, ..., 19]
    left_eye_landmarks = [landmarks[i] for i in left_eye_indices]
    
    # 获取鼻子关键点
    nose_indices = indices["nose"]                    # [44, 45, 46, ..., 67]
    nose_landmarks = [landmarks[i] for i in nose_indices]
    
    print(f"右眼关键点数量: {len(right_eye_landmarks)}")
    print(f"左眼关键点数量: {len(left_eye_landmarks)}")
    print(f"鼻子关键点数量: {len(nose_landmarks)}")
```

## 关键点索引说明

MediaPipe Face Mesh的468个关键点按以下顺序排列：

| 部位 | 索引范围 | 关键点数量 |
|------|----------|------------|
| 右眼轮廓 | 0-9 | 10 |
| 左眼轮廓 | 10-19 | 10 |
| 右眼虹膜 | 20-31 | 12 |
| 左眼虹膜 | 32-43 | 12 |
| 鼻子 | 44-67 | 24 |
| 嘴巴外轮廓 | 68-83 | 16 |
| 嘴巴内轮廓 | 84-107 | 24 |
| 右眉毛 | 108-127 | 20 |
| 左眉毛 | 128-147 | 20 |
| 右脸颊 | 148-167 | 20 |
| 左脸颊 | 168-187 | 20 |
| 下巴 | 188-207 | 20 |
| 额头 | 208-227 | 20 |
| 右太阳穴 | 228-247 | 20 |
| 左太阳穴 | 248-267 | 20 |
| 右耳 | 268-287 | 20 |
| 左耳 | 288-307 | 20 |
| 右耳垂 | 308-327 | 20 |
| 左耳垂 | 328-347 | 20 |
| 右耳轮 | 348-367 | 20 |
| 左耳轮 | 368-387 | 20 |
| 右耳屏 | 388-407 | 20 |
| 左耳屏 | 408-427 | 20 |
| 右耳垂底部 | 428-447 | 20 |
| 左耳垂底部 | 448-467 | 20 |

## 常见问题

### Q: 函数返回空列表怎么办？
A: 可能的原因：
- 图像中没有检测到人脸
- 人脸角度过大或遮挡严重
- 图像质量太低
- 光线条件不好

### Q: 关键点坐标超出图像范围？
A: 这是正常现象，MediaPipe可能检测到图像边界外的人脸部分。

### Q: 如何提高检测精度？
A: 
- 确保人脸在图像中清晰可见
- 提供足够的光线
- 使用高质量的图像
- 调整 `min_detection_confidence` 参数

## 运行测试

运行测试脚本：
```bash
cd project/recognition
python test_landmark_extractor.py
```

或者直接测试图像文件：
```bash
python test_landmark_extractor.py path/to/your/image.jpg
``` 
# 人脸关键点提取完整使用示例

## 概述

本文档展示了如何使用 `landmark_extractor.py` 中的所有函数来提取和分析人脸关键点。

## 函数列表

1. `extract_landmarks(rgb_image)` - 提取468个人脸关键点
2. `validate_landmarks(landmarks)` - 验证关键点质量
3. `get_eye_landmarks(landmarks, eye_type)` - 提取指定眼睛的关键点
4. `get_pupil_landmarks(landmarks)` - 提取瞳孔中心关键点
5. `get_iris_landmarks(landmarks)` - 提取虹膜边界关键点
6. `get_landmark_indices()` - 获取关键点索引映射

## 完整使用示例

### 示例1：基础人脸关键点提取

```python
import cv2
import numpy as np
from core.landmark_extractor import extract_landmarks, validate_landmarks

# 读取图像
image = cv2.imread("face_image.jpg")

# 提取关键点
landmarks = extract_landmarks(image)

if landmarks:
    print(f"成功提取 {len(landmarks)} 个关键点")
    
    # 验证关键点质量
    if validate_landmarks(landmarks):
        print("关键点质量良好")
        
        # 显示前几个关键点的坐标
        for i in range(5):
            x, y, z = landmarks[i]
            print(f"关键点 {i}: ({x:.2f}, {y:.2f}, {z:.2f})")
    else:
        print("关键点质量较差")
else:
    print("未检测到人脸")
```

### 示例2：眼部关键点分析

```python
from core.landmark_extractor import get_eye_landmarks, get_pupil_landmarks, get_iris_landmarks

# 假设已经有了landmarks
if landmarks and validate_landmarks(landmarks):
    # 获取左眼关键点
    left_eye = get_eye_landmarks(landmarks, 'left')
    print(f"左眼关键点数量: {len(left_eye)}")
    
    # 获取右眼关键点
    right_eye = get_eye_landmarks(landmarks, 'right')
    print(f"右眼关键点数量: {len(right_eye)}")
    
    # 获取瞳孔中心
    pupil_centers = get_pupil_landmarks(landmarks)
    left_pupil = pupil_centers['left']
    right_pupil = pupil_centers['right']
    
    print(f"左瞳孔中心: {left_pupil}")
    print(f"右瞳孔中心: {right_pupil}")
    
    # 获取虹膜边界
    iris_boundaries = get_iris_landmarks(landmarks)
    left_iris = iris_boundaries['left']
    right_iris = iris_boundaries['right']
    
    print(f"左眼虹膜关键点数量: {len(left_iris)}")
    print(f"右眼虹膜关键点数量: {len(right_iris)}")
```

### 示例3：可视化关键点

```python
import cv2
import numpy as np
from core.landmark_extractor import (
    extract_landmarks, 
    get_eye_landmarks, 
    get_pupil_landmarks, 
    get_iris_landmarks
)

def visualize_landmarks(image, landmarks):
    """可视化所有关键点"""
    result_image = image.copy()
    
    # 绘制所有关键点（绿色小点）
    for x, y, z in landmarks:
        cv2.circle(result_image, (int(x), int(y)), 1, (0, 255, 0), -1)
    
    # 绘制左眼关键点（蓝色）
    left_eye = get_eye_landmarks(landmarks, 'left')
    for x, y, z in left_eye:
        cv2.circle(result_image, (int(x), int(y)), 3, (255, 0, 0), -1)
    
    # 绘制右眼关键点（红色）
    right_eye = get_eye_landmarks(landmarks, 'right')
    for x, y, z in right_eye:
        cv2.circle(result_image, (int(x), int(y)), 3, (0, 0, 255), -1)
    
    # 绘制瞳孔中心（黄色大点）
    pupil_centers = get_pupil_landmarks(landmarks)
    if pupil_centers['left']:
        x, y, z = pupil_centers['left']
        cv2.circle(result_image, (int(x), int(y)), 5, (0, 255, 255), -1)
    if pupil_centers['right']:
        x, y, z = pupil_centers['right']
        cv2.circle(result_image, (int(x), int(y)), 5, (0, 255, 255), -1)
    
    # 绘制虹膜边界（青色）
    iris_boundaries = get_iris_landmarks(landmarks)
    for x, y, z in iris_boundaries['left'] + iris_boundaries['right']:
        cv2.circle(result_image, (int(x), int(y)), 2, (255, 255, 0), -1)
    
    return result_image

# 使用示例
image = cv2.imread("face_image.jpg")
landmarks = extract_landmarks(image)

if landmarks:
    visualized_image = visualize_landmarks(image, landmarks)
    cv2.imshow('Face Landmarks Visualization', visualized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 保存结果
    cv2.imwrite('visualized_landmarks.jpg', visualized_image)
```

### 示例4：实时摄像头处理

```python
import cv2
import numpy as np
from core.landmark_extractor import extract_landmarks, validate_landmarks

def process_frame(frame):
    """处理单帧图像"""
    landmarks = extract_landmarks(frame)
    
    if landmarks and validate_landmarks(landmarks):
        # 在图像上绘制关键点
        for x, y, z in landmarks:
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
        
        # 添加文本信息
        cv2.putText(frame, f"Landmarks: {len(landmarks)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Quality: Good", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No face detected", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return frame

# 实时处理
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 处理帧
    processed_frame = process_frame(frame)
    
    # 显示结果
    cv2.imshow('Real-time Face Landmarks', processed_frame)
    
    # 按'q'退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 示例5：关键点数据分析

```python
import numpy as np
from core.landmark_extractor import extract_landmarks, get_landmark_indices

def analyze_landmarks(landmarks):
    """分析关键点数据"""
    if not landmarks:
        return None
    
    # 转换为numpy数组
    landmarks_array = np.array(landmarks)
    
    # 基本统计信息
    analysis = {
        'total_points': len(landmarks),
        'x_range': (np.min(landmarks_array[:, 0]), np.max(landmarks_array[:, 0])),
        'y_range': (np.min(landmarks_array[:, 1]), np.max(landmarks_array[:, 1])),
        'z_range': (np.min(landmarks_array[:, 2]), np.max(landmarks_array[:, 2])),
        'face_width': np.max(landmarks_array[:, 0]) - np.min(landmarks_array[:, 0]),
        'face_height': np.max(landmarks_array[:, 1]) - np.min(landmarks_array[:, 1])
    }
    
    # 获取关键点索引
    indices = get_landmark_indices()
    
    # 分析各个部位
    for part_name, part_indices in indices.items():
        if part_indices:
            part_points = landmarks_array[part_indices]
            analysis[part_name] = {
                'point_count': len(part_indices),
                'center': np.mean(part_points, axis=0).tolist(),
                'area': calculate_area(part_points)
            }
    
    return analysis

def calculate_area(points):
    """计算点集的面积（简化版本）"""
    if len(points) < 3:
        return 0
    
    # 使用凸包计算面积
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points[:, :2])  # 只使用x,y坐标
        return hull.volume  # 对于2D点，volume实际上是面积
    except:
        return 0

# 使用示例
image = cv2.imread("face_image.jpg")
landmarks = extract_landmarks(image)

if landmarks:
    analysis = analyze_landmarks(landmarks)
    print("关键点分析结果:")
    print(f"总关键点数: {analysis['total_points']}")
    print(f"人脸宽度: {analysis['face_width']:.2f}")
    print(f"人脸高度: {analysis['face_height']:.2f}")
    print(f"X坐标范围: {analysis['x_range']}")
    print(f"Y坐标范围: {analysis['y_range']}")
    print(f"Z坐标范围: {analysis['z_range']}")
    
    # 显示眼部信息
    if 'left_eye_contour' in analysis:
        left_eye = analysis['left_eye_contour']
        print(f"左眼中心: {left_eye['center']}")
        print(f"左眼面积: {left_eye['area']:.2f}")
```

## 错误处理

### 常见错误及解决方案

1. **未检测到人脸**
   ```python
   landmarks = extract_landmarks(image)
   if not landmarks:
       print("未检测到人脸，请检查图像质量和光线条件")
   ```

2. **关键点质量差**
   ```python
   if not validate_landmarks(landmarks):
       print("关键点质量较差，可能需要重新拍摄")
   ```

3. **参数错误**
   ```python
   try:
       eye_landmarks = get_eye_landmarks(landmarks, 'invalid')
   except ValueError as e:
       print(f"参数错误: {e}")
   ```

## 性能优化建议

1. **批量处理**: 如果需要处理多张图像，可以复用MediaPipe对象
2. **图像预处理**: 调整图像大小可以提高处理速度
3. **缓存结果**: 对于相同的图像，可以缓存关键点结果

## 运行测试

运行完整测试：
```bash
cd project/recognition
python test_all_functions.py
```

运行基础测试：
```bash
python test_landmark_extractor.py
``` 