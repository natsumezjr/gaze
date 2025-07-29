# 识别模块（Recognition Module）

> **数据路径说明：**
> 所有数据保存路径均由 `config/settings.py` 中的 `DATA_PATH` 变量统一配置，后续如需修改数据保存目录，只需修改该变量即可。

## 概述

识别模块是眼动追踪系统的核心组件，负责从RGB-D摄像头采集的图像中检测人脸、提取瞳孔中心和虹膜边界点，并将这些点的二维归一化坐标转换为相机参考系下的三维坐标。

## 项目结构

```
recognition/
├── __init__.py                    # 模块初始化
├── structure.md                   # 详细结构说明文档
├── README.md                      # 本文件
├── main.py                        # 主入口文件（包含大while循环）
├── core/                          # 核心功能模块
│   ├── __init__.py
│   ├── detector.py                # 人脸检测器核心类
│   ├── landmark_extractor.py      # 关键点提取器
│   └── coordinate_converter.py    # 坐标转换器
├── utils/                         # 工具文件
│   ├── __init__.py
│   ├── camera_calibration.py     # 相机标定工具
│   ├── depth_processor.py        # 深度图处理工具
│   └── validation.py             # 数据验证工具
└── config/                        # 配置文件
    ├── __init__.py
    ├── settings.py               # 配置文件
    └── constants.py              # 常量定义
```

## 主要功能

### 输入
- RGB图像：numpy数组格式，BGR或RGB通道
- 深度图：numpy数组格式，与RGB图像对应
- 相机内参：包含fx, fy, cx, cy的字典或矩阵

### 输出
- 眼球中心坐标：左右眼的三维坐标（相机参考系）
- 瞳孔中心坐标：左右眼瞳孔的三维坐标（相机参考系）
- 置信度：检测结果的置信度分数
- 状态信息：检测是否成功、错误信息等

## 使用方法

### 基本使用
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
iris_boundaries = detector.get_iris_boundaries()
```

### 实时处理
```python
from recognition.main import main

if __name__ == "__main__":
    main()
```

## 性能要求

- **实时性**：单帧处理时间 < 100ms，支持30fps实时处理
- **精度**：瞳孔中心检测精度 ±2像素，三维坐标精度 ±5mm
- **鲁棒性**：支持不同光照条件、部分遮挡、不同人脸角度

## 依赖库

- OpenCV (cv2)
- MediaPipe
- NumPy
- SciPy

## 详细文档

更多详细信息请参考 `structure.md` 文件。 