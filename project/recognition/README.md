# 识别模块（Recognition Module）

> **数据路径说明：**
> 所有数据保存路径均由 `config/settings.py` 中的 `DATA_PATH` 变量统一配置，后续如需修改数据保存目录，只需修改该变量即可。

## 概述

识别模块是眼动追踪系统的核心组件，负责从RGB-D摄像头采集的图像中检测人脸、提取瞳孔中心和虹膜边界点，并将这些点的像素坐标转换为相机参考系下的三维坐标。

## 项目结构


```
recognition/
├── setup.py                       # 包安装配置文件
├── __init__.py                    # 模块初始化
├── structure.md                   # 详细结构说明文档
├── README.md                      # 本文件
├── main.py                        # 主入口文件（包含大while循环）
├── gaze_recognition.egg-info/     # 包安装信息（自动生成）
│   ├── PKG-INFO                   # 包信息
│   ├── SOURCES.txt                # 源文件列表
│   ├── dependency_links.txt       # 依赖链接
│   ├── requires.txt               # 依赖要求
│   └── top_level.txt             # 顶级包名
├── core/                          # 核心功能模块
│   ├── __init__.py
│   ├── detector.py                # 人脸检测器核心类
│   ├── landmark_extractor.py      # 关键点提取器
│   └── coordinate_converter.py    # 坐标转换器
├── utils/                         # 工具文件
│   ├── __init__.py
│   ├── data_manager.py            # 数据管理器（单例模式）
│   ├── camera_calibration.py     # 相机标定工具
└── config/                        # 配置文件
    ├── __init__.py
    ├── settings.py               # 配置文件
    └── constants.py              # 常量定义
```


## 安装和使用

### 1. 安装依赖
```bash
# 安装项目依赖
pip install -r requirements.txt

# 安装项目包（开发模式）
pip install -e .
```

### 2. 运行方式

#### 方式1：使用启动脚本（推荐）
```bash
# 运行主程序
python main.py
```

#### 方式2：直接运行模块
```bash
# 运行关键点提取器测试
python core/landmark_extractor.py

# 运行数据管理器测试
python utils/data_manager.py
```

#### 方式3：作为包导入
```python
# 导入数据管理器
from utils.data_manager import add_frame, get_image, get_depth

# 导入核心功能
from core.landmark_extractor import extract_landmarks, get_pupil_centers_landmarks
from core.detector import FaceDetector
```

## 主要功能

### 输入
- BGR图像：numpy数组格式，BGR通道
- 深度图：numpy数组格式，与BGR图像对应
- 相机内参：包含fx, fy, cx, cy的字典或矩阵

### 输出
- 478个面部关键点：包含眼睛、鼻子、嘴巴等面部特征
- 眼球轮廓坐标：左右眼的三维坐标（相机参考系）
- 瞳孔中心坐标：左右眼瞳孔的三维坐标（相机参考系）
- 虹膜边界点：左右眼虹膜的边界关键点
- 置信度：检测结果的置信度分数
- 状态信息：检测是否成功、错误信息等

## 使用方法

### 基本使用
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
pupil_centers = detector.get_pupil_centers()
iris_boundaries = detector.get_iris_boundaries()
```

### 实时处理
```python
from recognition.main import main

if __name__ == "__main__":
    main()
```

## 项目结构说明

### 包结构设计
本项目采用标准的Python包结构，具有以下优势：

1. **标准化**: 使用Python标准的包结构，符合最佳实践
2. **可维护性**: 清晰的模块划分，易于理解和维护
3. **可扩展性**: 新功能可以轻松添加到对应模块
4. **可安装性**: 可以作为包安装到任何Python环境
5. **开发友好**: 支持IDE的自动补全和错误检查

### gaze_recognition.egg-info 说明
`gaze_recognition.egg-info/` 目录是Python包安装时自动生成的元数据目录，包含：

- **PKG-INFO**: 包的版本、描述、作者等基本信息
- **SOURCES.txt**: 列出包中包含的所有源文件
- **dependency_links.txt**: 额外的依赖下载链接
- **requires.txt**: 包的依赖要求列表
- **top_level.txt**: 顶级包名列表

这个目录在运行 `pip install -e .` 时自动创建，用于Python包管理系统识别和安装包。**不要手动修改此目录中的文件**，它们会在重新安装包时自动更新。

### 模块说明
- **core/**: 核心算法模块，包含关键点提取、坐标转换等核心功能
- **utils/**: 工具模块，包含数据处理、文件操作等辅助功能
- **config/**: 配置模块，包含路径配置、常量定义等
- **启动脚本**: 解决Python模块导入问题，提供便捷的运行方式

### 导入问题解决方案
之前遇到的"无法解析导入"问题已通过以下方式解决：

1. **创建setup.py**: 将项目作为可安装的Python包
2. **完善__init__.py**: 定义包的导入接口
3. **创建启动脚本**: 设置正确的Python路径
4. **安装到开发环境**: 使用`pip install -e .`安装包

## 性能要求

- **实时性**：单帧处理时间 < 100ms，支持30fps实时处理
- **精度**：瞳孔中心检测精度 ±2像素，三维坐标精度 ±5mm
- **鲁棒性**：支持不同光照条件、部分遮挡、不同人脸角度

## 依赖库

- OpenCV (cv2) >= 4.5.0
- MediaPipe >= 0.10.0
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- Pillow >= 8.0.0
- matplotlib >= 3.3.0
- pandas >= 1.3.0

## 详细文档

更多详细信息请参考 `structure.md` 文件。