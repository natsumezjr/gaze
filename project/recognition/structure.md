# 识别模块（recognition）结构与功能说明

## 一、模块概述

识别模块的目标是：**从摄像头采集的RGB图像和深度图中，自动检测出人脸、瞳孔中心和虹膜边界点，并将这些点的二维归一化坐标转换为相机参考系下的三维坐标。**

### 主要功能
- 人脸检测与关键点提取
- 瞳孔中心和虹膜边界点识别
- 归一化坐标到像素坐标转换
- 像素坐标结合深度图转换为三维坐标
- 实时处理与质量控制

---

## 二、文件结构设计以及调用关系

```
recognition/
├── __init__.py                    # 模块初始化
├── core/
│   ├── __init__.py
│   ├── detector.py                # 人脸检测器核心类
│   ├── landmark_extractor.py      # 关键点提取器
│   └── coordinate_converter.py    # 坐标转换器
├── models/
│   ├── __init__.py
│   ├── face_detector.py          # 人脸检测模型
│   └── eye_detector.py           # 眼部检测模型
├── utils/
│   ├── __init__.py
│   ├── camera_calibration.py     # 相机标定工具
│   ├── depth_processor.py        # 深度图处理工具
│   └── validation.py             # 数据验证工具
├── config/
│   ├── __init__.py
│   ├── settings.py               # 配置文件
│   └── constants.py              # 常量定义
└── main.py                       # 主入口文件（包含大while循环）
```

调用关系
main.py (主循环)
    ↓ 调用
detector.py (核心检测器)
    ↓ 调用
├── landmark_extractor.py (关键点提取)
│   └── models/face_detector.py (人脸检测模型)
├── coordinate_converter.py (坐标转换)
└── utils/ (工具函数)
    ├── camera_calibration.py
    ├── depth_processor.py
    └── validation.py
---

## 三、各文件功能详细说明

### 3.1 核心文件（core/）

#### `detector.py` - 人脸检测器核心类
**功能**：整合人脸检测、关键点提取、坐标转换的完整流程

**主要方法**：
- `__init__(camera_params)`：初始化，接收相机内参
- `detect_face(rgb_image, depth_map)`：主检测方法
- `get_eye_centers()`：获取眼球中心三维坐标
- `get_pupil_centers()`：获取瞳孔中心三维坐标
- `get_iris_boundaries()`：获取虹膜边界点三维坐标
- `get_detection_confidence()`：获取检测置信度
- `get_detection_status()`：获取检测状态信息

**输入输出**：
- 输入：RGB图像(numpy数组)、深度图(numpy数组)、相机内参(dict)
- 输出：三维坐标字典，包含眼球中心、瞳孔中心、虹膜边界点

#### `landmark_extractor.py` - 关键点提取器
**功能**：专门处理MediaPipe人脸关键点的提取和验证

**主要方法**：
- `extract_landmarks(rgb_image)`：提取468个关键点
- `validate_landmarks(landmarks)`：验证关键点质量
- `get_eye_landmarks(landmarks, eye_type)`：提取指定眼睛的关键点
- `get_pupil_landmarks(landmarks)`：提取瞳孔中心关键点
- `get_iris_landmarks(landmarks)`：提取虹膜边界关键点

**关键点索引**：
- 左眼瞳孔中心：468
- 右眼瞳孔中心：473
- 左眼虹膜边界：469-472
- 右眼虹膜边界：474-477

#### `coordinate_converter.py` - 坐标转换器
**功能**：处理归一化坐标到像素坐标，再到三维坐标的转换

**主要方法**：
- `normalized_to_pixel(norm_coords, image_shape)`：归一化转像素
- `pixel_to_3d(pixel_coords, depth_map, camera_params)`：像素转三维
- `batch_convert_landmarks(landmarks, depth_map, camera_params)`：批量转换
- `validate_3d_coordinates(coords_3d)`：验证三维坐标有效性

**转换公式**：
```
x_pixel = x_norm × W
y_pixel = y_norm × H
X = (x_pixel - cx) × Z / fx
Y = (y_pixel - cy) × Z / fy
Z = D(x_pixel, y_pixel)
```

### 3.2 模型文件（models/）

#### `face_detector.py` - 人脸检测模型
**功能**：封装MediaPipe人脸检测功能

**主要方法**：
- `detect(rgb_image)`：检测人脸
- `get_face_mesh()`：获取人脸网格
- `is_face_detected()`：判断是否检测到人脸
- `get_face_confidence()`：获取人脸检测置信度
- `reset_detector()`：重置检测器状态

#### `eye_detector.py` - 眼部检测模型
**功能**：专门处理眼部区域的检测和验证

**主要方法**：
- `detect_eyes(face_landmarks)`：检测双眼
- `validate_eye_detection(eye_landmarks)`：验证眼部检测质量
- `get_eye_region(landmarks, eye_type)`：获取眼部区域
- `get_eye_confidence(eye_landmarks)`：获取眼部检测置信度
- `filter_eye_landmarks(landmarks, eye_type)`：滤波眼部关键点

### 3.3 工具文件（utils/）

#### `camera_calibration.py` - 相机标定工具
**功能**：处理相机内参的加载、验证和转换

**主要方法**：
- `load_camera_params(file_path)`：加载相机参数
- `validate_camera_params(params)`：验证参数有效性
- `get_intrinsic_matrix()`：获取内参矩阵
- `save_camera_params(params, file_path)`：保存相机参数
- `get_camera_info()`：获取相机信息

**相机参数格式**：
```python
camera_params = {
    'fx': 焦距x,
    'fy': 焦距y,
    'cx': 主点x,
    'cy': 主点y,
    'width': 图像宽度,
    'height': 图像高度
}
```

#### `depth_processor.py` - 深度图处理工具
**功能**：处理深度图数据，包括滤波、插值等

**主要方法**：
- `filter_depth_map(depth_map)`：深度图滤波
- `interpolate_depth(depth_map, pixel_coords)`：深度插值
- `validate_depth_value(depth_value)`：验证深度值有效性
- `get_depth_statistics(depth_map)`：获取深度统计信息
- `normalize_depth_map(depth_map)`：深度图归一化

#### `validation.py` - 数据验证工具
**功能**：验证输入数据的有效性和质量

**主要方法**：
- `validate_image(image)`：验证图像数据
- `validate_depth_map(depth_map)`：验证深度图
- `check_coordinate_quality(coords_3d)`：检查三维坐标质量
- `validate_camera_params(params)`：验证相机参数
- `get_validation_report()`：获取验证报告

### 3.4 配置文件（config/）

#### `settings.py` - 配置文件
**功能**：存储模块的配置参数

**主要内容**：
- 检测阈值设置
- 坐标转换参数
- 质量评估标准
- 错误处理策略
- 实时处理参数

**配置项示例**：
```python
DETECTION_CONFIDENCE_THRESHOLD = 0.7
DEPTH_VALIDATION_THRESHOLD = 0.1
COORDINATE_QUALITY_THRESHOLD = 0.8
MAX_PROCESSING_TIME = 0.1  # 秒
```

#### `constants.py` - 常量定义
**功能**：定义MediaPipe关键点索引等常量

**主要内容**：
- 瞳孔中心索引（468, 473）
- 虹膜边界点索引（469-472, 474-477）
- 其他关键点索引
- 错误代码定义
- 状态码定义

**常量示例**：
```python
# 关键点索引
LEFT_PUPIL_INDEX = 468
RIGHT_PUPIL_INDEX = 473
LEFT_IRIS_INDICES = [469, 470, 471, 472]
RIGHT_IRIS_INDICES = [474, 475, 476, 477]

# 状态码
STATUS_SUCCESS = 0
STATUS_NO_FACE_DETECTED = 1
STATUS_LOW_CONFIDENCE = 2
STATUS_INVALID_DEPTH = 3
```

### 3.5 主入口文件

#### `main.py` - 主入口文件
**功能**：包含大的while循环，处理实时数据流

**主要结构**：
```python
def main():
    # 初始化检测器
    detector = FaceDetector(camera_params)
    
    while True:
        try:
            # 获取RGB图像和深度图
            rgb_image, depth_map = get_frame_data()
            
            # 检测人脸和关键点
            result = detector.detect_face(rgb_image, depth_map)
            
            # 输出三维坐标
            eye_centers = detector.get_eye_centers()
            pupil_centers = detector.get_pupil_centers()
            
            # 处理结果...
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            handle_error(e)
```

---

## 四、接口设计规范

### 4.1 输入接口
- **RGB图像**：numpy数组格式，BGR或RGB通道，形状为(H, W, 3)
- **深度图**：numpy数组格式，与RGB图像对应，形状为(H, W)
- **相机内参**：包含fx, fy, cx, cy的字典或矩阵

### 4.2 输出接口
- **眼球中心坐标**：左右眼的三维坐标（相机参考系）
- **瞳孔中心坐标**：左右眼瞳孔的三维坐标（相机参考系）
- **置信度**：检测结果的置信度分数
- **状态信息**：检测是否成功、错误信息等

### 4.3 错误处理
- 人脸未检测到的情况
- 关键点质量不足的情况
- 深度图无效的情况
- 相机参数错误的情况
- 实时处理超时的情况

---

## 五、使用示例

### 5.1 基本使用
```python
from recognition.core.detector import FaceDetector
from recognition.utils.camera_calibration import load_camera_params

# 加载相机参数
camera_params = load_camera_params('camera_params.json')

# 初始化检测器
detector = FaceDetector(camera_params)

# 检测人脸
result = detector.detect_face(rgb_image, depth_map)

# 获取结果
eye_centers = detector.get_eye_centers()
pupil_centers = detector.get_pupil_centers()
```

### 5.2 实时处理
```python
from recognition.main import main

if __name__ == "__main__":
    main()
```

---

## 六、性能要求

### 6.1 实时性要求
- 单帧处理时间 < 100ms
- 支持30fps实时处理
- 内存使用优化

### 6.2 精度要求
- 瞳孔中心检测精度：±2像素
- 三维坐标精度：±5mm
- 检测置信度 > 0.7

### 6.3 鲁棒性要求
- 支持不同光照条件
- 支持部分遮挡情况
- 支持不同人脸角度
- 支持不同距离范围

---

## 七、测试规范

### 7.1 单元测试
- 每个模块独立测试
- 边界条件测试
- 错误处理测试

### 7.2 集成测试
- 端到端流程测试
- 实时性能测试
- 精度验证测试

### 7.3 测试数据
- 标准测试图像集
- 不同光照条件数据
- 不同距离范围数据
- 异常情况数据 