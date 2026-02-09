# 眼动追踪系统 (Gaze Tracking System)

基于计算机视觉的眼动追踪系统，用于实时检测和追踪用户注视点。

## 功能特性

- **实时识别**：使用 MediaPipe 进行人脸检测和眼部关键点提取
- **眼球拟合**：基于 RANSAC 和 Levenberg-Marquardt 算法的眼球几何建模
- **Kappa 角标定**：个体化的视线角度补偿
- **注视点计算**：精确计算视线与屏幕的交点
- **多线程处理**：高效的并发处理架构

## 系统要求

- Python >= 3.11
- Windows / Linux / macOS
- 摄像头设备

## 安装

### 使用 Poetry（推荐）

```bash
# 安装依赖
poetry install

# 激活虚拟环境（Poetry 2.x）
poetry env activate
# 按输出的命令执行，例如：
# & "path/to/venv/Scripts/activate.ps1"  (Windows PowerShell)
# source path/to/venv/bin/activate       (Linux/macOS)
```

### 使用 pip

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows CMD:
venv\Scripts\activate.bat
# Linux/macOS:
source venv/bin/activate

# 安装依赖
pip install -r project/requirements.txt
```

## 使用方法

### 使用 Poetry（推荐）

```bash
# 1. 安装所有依赖（包括项目本身的可编辑安装）
poetry install

# 2. 运行程序（无需激活虚拟环境）
poetry run python run.py

# 或先激活虚拟环境再运行
poetry env activate
# 执行输出的激活命令后：
python run.py
```

### 使用 pip

```bash
# 激活虚拟环境后，需要设置 PYTHONPATH
# Windows PowerShell:
$env:PYTHONPATH = "$PWD"
python run.py

# Linux/macOS:
export PYTHONPATH=$(pwd)
python run.py
```

## 项目结构

```
gaze/
├── project/              # 主项目目录（Python 包）
│   ├── core/            # 核心算法模块
│   │   ├── recognition/ # 识别模块
│   │   ├── fitting/     # 拟合模块
│   │   ├── track/       # 追踪模块
│   │   └── visualization/ # 可视化
│   ├── config/          # 配置文件
│   ├── data/            # 数据管理
│   ├── client/          # 客户端应用
│   ├── events/          # 事件处理
│   ├── utils/           # 工具函数
│   └── main.py          # 主程序入口
├── run.py               # 启动脚本（推荐使用）
├── pyproject.toml       # Poetry 配置
└── README.md            # 项目说明
```

## 依赖库

- opencv-python: 计算机视觉处理
- mediapipe: 人脸和关键点检测
- numpy, scipy: 数值计算
- pandas: 数据处理
- Pillow, matplotlib: 图像处理与可视化
- flask: Web 服务
- psutil: 系统监控
- pywin32: Windows 平台支持（仅 Windows）
- pytest, pytest-cov: 单元测试

## 开发

```bash
# 运行测试
poetry run pytest

# 代码格式化
poetry run black .

# 类型检查
poetry run mypy .
```

## 许可证

MIT License

## 作者

zeng-jingran

