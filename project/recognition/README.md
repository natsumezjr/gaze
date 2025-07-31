# 识别模块（Recognition Module）

> **数据路径说明：**
> 所有数据保存路径均由 `config/settings.py` 中的 `DATA_PATH` 变量统一配置，后续如需修改数据保存目录，只需修改该变量即可。

## 概述

识别模块是眼动追踪系统的核心组件，负责从RGB-D摄像头采集的图像中检测人脸、提取瞳孔中心和虹膜边界点，并将这些点的二维归一化坐标转换为相机参考系下的三维坐标。

## 项目结构

```
recognition/
├── setup.py                       # 包安装配置文件
├── __init__.py                    # 模块初始化
├── structure.md                   # 详细结构说明文档
├── README.md                      # 本文件
├── main.py                        # 主入口文件（包含大while循环）
├── run_data_writer.py             # 数据写入器启动脚本
├── core/                          # 核心功能模块
│   ├── __init__.py
│   ├── detector.py                # 人脸检测器核心类
│   ├── landmark_extractor.py      # 关键点提取器
│   └── coordinate_converter.py    # 坐标转换器
├── utils/                         # 工具文件
│   ├── __init__.py
│   ├── data_writer.py             # 数据写入工具
│   ├── data_parser.py             # 数据解析工具
│   ├── camera_calibration.py     # 相机标定工具
│   ├── depth_processor.py        # 深度图处理工具
│   └── validation.py             # 数据验证工具
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
# 运行数据写入器
python run_data_writer.py

# 运行主程序
python main.py
```

#### 方式2：直接运行模块
```bash
# 运行数据写入器
python utils/data_writer.py

# 运行关键点提取器测试
python -c "from core.landmark_extractor import get_landmark_indices; print('测试成功')"
```

#### 方式3：作为包导入
```python
# 导入工具函数
from utils.data_writer import convert_to_json, clean_old_entries
from utils.data_parser import encode_image_to_base64

# 导入核心功能
from core.landmark_extractor import get_landmark_indices, validate_landmarks
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

## 项目结构说明

### 包结构设计
本项目采用标准的Python包结构，具有以下优势：

1. **标准化**: 使用Python标准的包结构，符合最佳实践
2. **可维护性**: 清晰的模块划分，易于理解和维护
3. **可扩展性**: 新功能可以轻松添加到对应模块
4. **可安装性**: 可以作为包安装到任何Python环境
5. **开发友好**: 支持IDE的自动补全和错误检查

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
- SciPy

## 详细文档

更多详细信息请参考 `structure.md` 文件。 