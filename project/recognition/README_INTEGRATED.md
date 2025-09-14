# 眼动追踪集成系统

## 概述

本系统将 `fitting_frontend` 的内容整合到 `recognition` 模块中，按照 `structure.md` 的流程组织，形成一个完整的眼动追踪系统。

## 系统架构

### 模块结构

```
project/recognition/
├── integrated_server.py          # 集成服务器主文件
├── run_integrated.py             # 统一启动脚本
├── config/
│   └── integrated_config.json    # 集成系统配置
├── frontend/                     # 前端文件
│   ├── templates/
│   │   └── calibration_integrated.html
│   └── static/
│       ├── css/
│       │   └── style.css
│       └── js/
│           └── script_integrated.js
├── core/                        # 识别模块核心
├── utils/                       # 识别模块工具
└── logs/                        # 日志文件
```

### 四大核心模块

1. **识别模块 (Recognition)**
   - 摄像头数据采集
   - 人脸检测与关键点提取
   - 瞳孔中心与虹膜边界点检测
   - 三维坐标转换

2. **拟合模块 (Fitting)**
   - 眼球形状拟合 (RANSAC)
   - Kappa角校准
   - 视线向量计算
   - 6点校准流程

3. **追踪模块 (Tracking)**
   - LSTM模型预测
   - 马尔可夫模型
   - 视线轨迹平滑
   - 快速眼动检测

4. **交互模块 (Interaction)**
   - ROI区域管理
   - 注视点交互
   - 优先级处理
   - 用户界面控制

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动集成系统

```bash
# 启动完整系统
python run_integrated.py --mode full

# 只启动识别模块
python run_integrated.py --mode recognition

# 只启动拟合模块
python run_integrated.py --mode fitting

# 调试模式
python run_integrated.py --mode full --debug --log-level DEBUG
```

### 3. 访问前端界面

打开浏览器访问: http://localhost:5000

## API 接口

### 识别模块 API

- `POST /api/recognition/camera/start` - 启动摄像头
- `POST /api/recognition/camera/stop` - 停止摄像头
- `GET /api/recognition/gaze/current` - 获取当前注视点

### 拟合模块 API

- `GET /api/fitting/calibration/status` - 获取校准状态
- `POST /api/fitting/calibration/start` - 开始校准
- `POST /api/fitting/calibration/collect` - 收集校准点
- `POST /api/fitting/calibration/complete` - 完成校准

### 追踪模块 API

- `GET /api/tracking/lstm/status` - 获取LSTM状态
- `POST /api/tracking/lstm/load` - 加载LSTM模型

### 交互模块 API

- `GET /api/interaction/roi/regions` - 获取ROI区域
- `POST /api/interaction/roi/add` - 添加ROI区域
- `POST /api/interaction/gaze/action` - 处理注视交互

### 系统状态 API

- `GET /api/system/status` - 获取系统整体状态
- `POST /api/system/reset` - 重置系统

## 配置说明

系统配置文件位于 `config/integrated_config.json`，包含以下配置项：

### 识别模块配置

```json
{
  "recognition": {
    "camera": {
      "rgb_d": true,
      "fps": 30,
      "resolution": {"width": 640, "height": 480}
    },
    "face_detection": {
      "max_faces": 1,
      "min_detection_confidence": 0.5
    }
  }
}
```

### 拟合模块配置

```json
{
  "fitting": {
    "calibration": {
      "total_points": 6,
      "point_layout": "hexagon"
    },
    "ransac": {
      "max_trials": 100,
      "threshold": 0.5
    }
  }
}
```

## 工作流程

### 1. 系统初始化

1. 启动摄像头
2. 初始化人脸检测器
3. 加载相机参数
4. 开始人脸检测循环

### 2. 粗拟合阶段

1. 检测人脸和瞳孔
2. 执行眼球形状拟合 (RANSAC)
3. 初始化校准会话
4. 等待用户开始校准

### 3. 校准阶段

1. 显示6个校准点
2. 收集每个点的注视数据
3. 进行Kappa角校准
4. 完成校准流程

### 4. 实时追踪

1. 持续获取注视点数据
2. 应用LSTM预测 (可选)
3. 处理ROI交互
4. 提供用户反馈

## 开发说明

### 添加新功能

1. 在相应的模块中添加API端点
2. 更新前端JavaScript代码
3. 修改配置文件
4. 更新文档

### 调试技巧

1. 使用 `--debug` 模式启动
2. 查看日志文件
3. 使用浏览器开发者工具
4. 检查API响应

### 性能优化

1. 调整摄像头帧率
2. 优化LSTM模型
3. 减少API调用频率
4. 使用缓存机制

## 故障排除

### 常见问题

1. **摄像头无法启动**
   - 检查摄像头权限
   - 确认摄像头未被其他程序占用
   - 检查OpenCV安装

2. **人脸检测失败**
   - 确保光线充足
   - 调整摄像头角度
   - 检查MediaPipe安装

3. **校准精度低**
   - 确保用户注视校准点
   - 检查瞳孔检测质量
   - 调整校准参数

4. **前端连接失败**
   - 检查后端服务器状态
   - 确认端口未被占用
   - 检查CORS设置

## 更新日志

### v1.0.0 (2024-01-XX)

- 整合 `fitting_frontend` 到 `recognition` 模块
- 实现四大核心模块的API接口
- 添加统一的配置管理
- 优化前端用户体验
- 完善错误处理和日志记录

## 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 发起 Pull Request

## 许可证

本项目采用 MIT 许可证。
