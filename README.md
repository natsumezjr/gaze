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

# 激活虚拟环境
poetry shell
```

### 使用 pip

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 安装依赖
pip install -r project/requirements.txt
```

## 使用方法

```bash
# 进入项目目录
cd project

# 运行主程序
python main.py
```

## 项目结构

```
gaze-refactor/
├── project/              # 主项目目录
│   ├── core/            # 核心算法模块
│   │   ├── recognition/ # 识别模块
│   │   ├── fitting/     # 拟合模块
│   │   └── track/       # 追踪模块
│   ├── config/          # 配置文件
│   ├── data/            # 数据管理
│   ├── client/          # 客户端应用
│   └── utils/           # 工具函数
├── pyproject.toml       # Poetry 配置
└── README.md           # 项目说明
```

## 依赖库

- opencv-python: 计算机视觉处理
- mediapipe: 人脸和关键点检测
- numpy, scipy: 数值计算
- pandas: 数据处理
- pytest: 单元测试

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

