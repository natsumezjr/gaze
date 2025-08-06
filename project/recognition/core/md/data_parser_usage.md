# 数据解析工具使用示例

## 概述

新的数据解析工具将JSON字符串解析逻辑从核心业务模块中分离出来，统一了数据接口。所有核心模块现在都接收 `np.ndarray` 格式的图像数据。

## 主要改进

### 1. 接口统一
- **之前**：各模块接收不同格式的数据（base64字符串、numpy数组等）
- **现在**：统一接收 `np.ndarray` 格式，形状为 `(H, W, 3)`，BGR格式

### 2. 职责分离
- **数据解析**：`utils/data_parser.py` 专门处理JSON字符串到numpy数组的转换
- **业务逻辑**：核心模块专注于算法处理

## 使用示例

### 示例1：从JSON文件加载数据

```python
from recognition.utils.data_parser import load_rgbd_from_file
from recognition.core.detector import FaceDetector
from recognition.config.settings import get_camera_params

# 加载数据
rgb_image, depth_map, metadata = load_rgbd_from_file("data/rgbd_input.json")

# 初始化检测器
camera_params = get_camera_params()
detector = FaceDetector(camera_params)

# 检测人脸（现在接收numpy数组）
success = detector.detect_face(rgb_image, depth_map)

if success:
    # 获取眼球中心
    eye_centers = detector.get_eye_centers()
    print(f"左眼球中心: {eye_centers['left']}")
    print(f"右眼球中心: {eye_centers['right']}")
```

### 示例2：直接处理numpy数组

```python
from recognition.core.landmark_extractor import extract_landmarks
import cv2

# 读取图像
image = cv2.imread("face.jpg")  # BGR格式

# 提取关键点（现在接收numpy数组）
landmarks = extract_landmarks(image)

if landmarks:
    print(f"检测到 {len(landmarks)} 个关键点")
    print(f"第一个关键点: {landmarks[0]}")
```

### 示例3：解析JSON字符串

```python
from recognition.utils.data_parser import parse_rgbd_json

# JSON数据（字符串格式）
json_string = '''
{
    "frame_id": 1,
    "timestamp": "2024-01-15T10:30:00",
    "rgb_data": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDA...",
    "depth_data": null
}
'''

# 解析数据
rgb_image, depth_map, metadata = parse_rgbd_json(json_string)

print(f"图像尺寸: {rgb_image.shape}")
print(f"帧ID: {metadata['frame_id']}")
```

## 接口对比

### 之前的接口

```python
# landmark_extractor.py
def extract_landmarks(bgr_image_b64: str) -> List[List[float]]:
    # 需要先解码base64字符串
    bgr_image = decode_base64_image(bgr_image_b64)
    # 处理逻辑...

# detector.py  
def detect_face(self, bgr_image_b64: str, depth_map: np.ndarray) -> bool:
    # 需要先解码base64字符串
    bgr_image = self._decode_base64_image(bgr_image_b64)
    # 处理逻辑...
```

### 现在的接口

```python
# landmark_extractor.py
def extract_landmarks(bgr_image: np.ndarray) -> List[List[float]]:
    # 直接处理numpy数组
    # 处理逻辑...

# detector.py
def detect_face(self, bgr_image: np.ndarray, depth_map: np.ndarray) -> bool:
    # 直接处理numpy数组
    # 处理逻辑...
```

## 优势

1. **解耦合**：数据解析逻辑与业务逻辑分离
2. **统一接口**：所有模块使用相同的数据格式
3. **易于测试**：可以直接传入numpy数组进行测试
4. **性能提升**：避免重复的base64解码操作
5. **类型安全**：明确的类型定义，减少运行时错误

## 迁移指南

### 对于现有代码

1. **数据加载**：使用 `load_rgbd_from_file()` 或 `parse_rgbd_json()`
2. **模块调用**：直接传入numpy数组，无需base64解码
3. **测试代码**：可以直接创建numpy数组进行测试

### 示例迁移

```python
# 旧代码
from recognition.core.landmark_extractor import extract_landmarks
landmarks = extract_landmarks(base64_string)

# 新代码
from recognition.utils.data_parser import decode_base64_image
from recognition.core.landmark_extractor import extract_landmarks

# 先解析数据
bgr_image = decode_base64_image(base64_string)
# 再处理业务逻辑
landmarks = extract_landmarks(bgr_image)
``` 