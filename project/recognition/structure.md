# 识别模块（recognition）结构与功能说明

> **数据路径说明：**
> 所有数据保存路径均由 `config/settings.py` 中的 `DATA_PATH` 变量统一配置，后续如需修改数据保存目录，只需修改该变量即可。

## 一、模块概述

识别模块的目标是：**从摄像头采集的RGB图像和深度图中，自动检测出人脸、瞳孔中心和虹膜边界点，并将这些点的二维归一化坐标转换为相机参考系下的三维坐标。**

### 主要功能
- 人脸检测与关键点提取（478个关键点）
- 瞳孔中心和虹膜边界点识别
- 归一化坐标到像素坐标转换
- 像素坐标结合深度图转换为三维坐标
- 实时处理与质量控制

---

## 二、文件结构设计以及调用关系
根据现有文件@coordinate_converter.py @detector.py @landmark_extractor.py @camera_calibration.py @data_manager.py 中的函数（直接阅读函数定义即可）以及文件结构，更新@core.md @structure.md @README.md并且根据@README.md 进行编译

我将根据现有文件中的函数定义来更新文档，然后进行编译。让我先分析现有文件中的函数定义。
现在我将根据现有文件中的函数定义来更新文档。首先更新core.md文件：
每个文件直接给我，不要写文件，我粘贴就好，然后执行编译

好的，我将根据现有文件中的函数定义为您提供更新后的文档内容。
1. core.md 更新内容
摄像头数据 → DataManager内存存储 → core模块处理 → 结果输出
2. structure.md 更新内容
recognition/
├── init.py # 模块初始化
├── core/
│ ├── init.py
│ ├── detector.py # 人脸检测器核心类
│ ├── landmark_extractor.py # 关键点提取器
│ └── coordinate_converter.py # 坐标转换器
├── utils/
│ ├── init.py
│ ├── data_manager.py # 数据管理器（单例模式）
│ ├── camera_calibration.py # 相机标定工具
├── config/
│ ├── init.py
│ ├── settings.py # 配置文件
│ └── constants.py # 常量定义
└── main.py # 主入口文件（包含大while循环）

调用关系
main.py (主循环)
    ↓ 调用
detector.py (核心检测器)
    ↓ 调用
├── landmark_extractor.py (关键点提取)
├── coordinate_converter.py (坐标转换)
└── utils/ (工具函数)
    ├── data_manager.py (数据管理)
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
- `detect_face(bgr_image, depth_map)`：主检测方法
- `get_eyes_contours()`：获取眼球轮廓三维坐标
- `get_pupil_center()`：获取瞳孔中心三维坐标
- `get_iris_boundaries()`：获取虹膜边界点三维坐标
- `get_detection_confidence()`：获取检测置信度
- `get_detection_status()`：获取检测状态信息

**输入输出**：
- 输入：BGR图像(numpy数组)、深度图(numpy数组)、相机内参(dict)
- 输出：三维坐标字典，包含眼球轮廓、瞳孔中心、虹膜边界点

#### `landmark_extractor.py` - 关键点提取器
**功能**：专门处理MediaPipe人脸关键点的提取和验证

**主要方法**：
- `extract_landmarks(bgr_image)`：提取478个关键点
- `validate_landmarks(landmarks)`：验证关键点质量
- `get_pupil_center_landmarks(landmarks)`：提取瞳孔中心关键点
- `get_eye_contours_landmarks(landmarks)`：提取眼球轮廓关键点
- `get_iris_boundaries_landmarks(landmarks)`：提取虹膜边界关键点

**关键点索引**：
- 右眼轮廓：33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
- 左眼轮廓：362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
- 右眼虹膜：468, 469, 470, 471, 472
- 左眼虹膜：473, 474, 475, 476, 477
- 瞳孔中心：468（右眼）、473（左眼）

#### `coordinate_converter.py` - 坐标转换器
**功能**：处理像素坐标到三维坐标的转换

**主要方法**：
- `pixel_to_3d(pixel_coords, depth_map_meters, camera_params)`：像素转三维
- `batch_convert_landmarks(landmarks, depth_map_meters, camera_params)`：批量转换
- `validate_3d_coordinates(coords_3d)`：验证三维坐标有效性

**转换公式**：

$$X = \frac{(x_{pixel} - c_x) \times Z}{f_x}$$

$$Y = \frac{(y_{pixel} - c_y) \times Z}{f_y}$$

$$Z = D(x_{pixel}, y_{pixel})$$

### 3.2 核心功能整合

**功能整合说明**：
- 原models层的功能已整合到core层中
- `landmark_extractor.py` 直接实现人脸检测和关键点提取
- `detector.py` 整合了所有检测流程
- 简化了调用关系，提高了代码效率

### 3.3 工具文件（utils/）

#### `data_manager.py` - 数据管理器（单例模式）
**功能**：统一管理BGR图像和深度图数据的内存存储

**主要方法**：
- `add_frame(frame_id, bgr_image, depth_map)`：添加帧数据到内存
- `get_image(frame_id)`：获取指定帧的BGR图像
- `get_depth(frame_id)`：获取指定帧的深度图
- `get_frame_data(frame_id)`：获取指定帧的完整数据
- `get_frame_count()`：获取当前帧数量
- `get_resolution()`：获取图像分辨率
- `clear_all()`：清空所有数据
- `remove_frame(frame_id)`：删除指定帧

**数据结构**：
```python
_data_dict = {
    frame_id: {
        "bgr_image": np.ndarray(shape=(H,W,3), dtype=np.uint8),
        "depth_map": np.ndarray(shape=(H,W), dtype=np.float32),
        "timestamp": str
    }
}
```

**特点**：
- 单例模式，全局唯一实例
- 内存存储，无需文件I/O
- 线程安全，支持并发访问
- 自动分辨率设置
- 数据格式验证

#### `camera_calibration.py` - 相机标定工具
**功能**：处理相机内参的加载、验证和转换

**主要方法**：
- `__init__(file_path, rgb_d)`：初始化相机标定器
- `get_cap()`：获取摄像头对象
- `load_camera_params()`：加载相机参数
- `get_image_resolution()`：获取图像分辨率
- `get_intrinsics()`：获取内参矩阵
- `get_depth_scale()`：获取深度缩放因子

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
            # 获取BGR图像和深度图
            bgr_image, depth_map = get_frame_data()
            
            # 检测人脸和关键点
            result = detector.detect_face(bgr_image, depth_map)
            
            # 输出三维坐标
            eye_contours = detector.get_eyes_contours()
            pupil_center = detector.get_pupil_center()
            
            # 处理结果...
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            handle_error(e)
```

---

## 四、接口设计规范

### 4.1 输入接口
- **BGR图像**：numpy数组格式，BGR通道，形状为(H, W, 3)
- **深度图**：numpy数组格式，与BGR图像对应，形状为(H, W)
- **相机内参**：包含fx, fy, cx, cy的字典或矩阵

### 4.2 输出接口
- **眼球轮廓坐标**：左右眼的三维坐标（相机参考系）
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
from recognition.utils.camera_calibration import CameraCalibrator

# 加载相机参数
calibrator = CameraCalibrator("config/camera_params.json", rgb_d=True)
camera_params = calibrator.load_camera_params()

# 初始化检测器
detector = FaceDetector(camera_params)

# 检测人脸
result = detector.detect_face(bgr_image, depth_map)

# 获取结果
eye_contours = detector.get_eyes_contours()
pupil_center = detector.get_pupil_center()
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