# 眼动追踪系统完整工作流程

## 系统概述

眼动追踪系统是一个基于计算机视觉的眼球运动检测和校准系统，主要包含以下核心模块：

- **识别模块** (`project/recognition/`): 人脸检测、关键点提取、瞳孔定位
- **拟合模块** (`project/fitting/`): 眼球形状拟合、Kappa校准
- **数据管理** (`project/recg_fit_data/`): 数据存储和状态管理

## 完整工作流程

### 阶段1: 系统初始化
```
1. 启动主程序 (main.py)
2. 配置日志系统
3. 初始化摄像头和相机参数
4. 加载人脸检测器
5. 设置全局状态变量
```

**关键文件**: `project/recognition/main.py`
**状态变量**:
- `calibration_started = False`
- `calibration_finished = False` 
- `coarse_fitting_done = False`
- `coarse_fitting_results = {...}`

### 阶段2: 实时视频处理循环
```
1. 读取摄像头帧
2. 生成固定深度图 (0.6米)
3. 添加帧数据到数据管理器
4. 检测人脸和关键点
```

**关键函数**: `face_detector.detect_face()`
**数据流**: 摄像头 → OpenCV → MediaPipe → 关键点数据

### 阶段3: 粗拟合（眼球形状拟合）
**触发条件**: 检测到人脸且 `coarse_fitting_done = False`

#### 3.1 眼球形状拟合
```
1. 检查数据管理器状态
2. 记录左眼/右眼瞳孔点数
3. 调用 fit_main() 进行眼球形状拟合
4. 生成 last_fitting 数据
```

**关键文件**: `project/fitting/main.py`
**核心函数**: `fit_all_eyes()`
**输出**: 眼球中心、半径、置信度等参数

#### 3.2 会话初始化
```
1. 调用 start_session("integrated-calibration")
2. 创建校准会话
3. 准备收集校准样本
```

**关键文件**: `project/fitting/app/api.py`
**核心函数**: `start_session()`

#### 3.3 状态更新
```
1. 更新 coarse_fitting_results
2. 发送结果给前端
3. 设置 coarse_fitting_done = True
```

### 阶段4: 校准点收集
**触发条件**: `coarse_fitting_done = True` 且 `calibration_started = False`

#### 4.1 初始化校准流程
```
1. 生成9点校准网格
2. 设置目标点坐标
3. 开始显示校准界面
```

**关键函数**: `nine_point_grid()`
**目标点**: 3×3网格，覆盖屏幕的10%-90%区域

#### 4.2 用户交互收集
**触发条件**: 用户按 's' 键

```
1. 获取当前目标点坐标
2. 从数据管理器获取瞳孔数据
3. 调用 push_sample() 收集样本
4. 更新当前点索引
```

**关键函数**: `push_sample()`
**数据收集**: 眼球中心、瞳孔中心、目标像素、时间戳

### 阶段5: 精细拟合（Kappa校准）
**触发条件**: 收集完所有9个校准点

#### 5.1 眼球形状重新拟合
```
1. 使用9个校准点的数据
2. 重新进行眼球形状拟合
3. 更新 last_fitting 数据
```

#### 5.2 Kappa校准
```
1. 提取校准样本数据
2. 构建kappa校准样本
3. 估计kappa参数
4. 评估拟合质量
```

**关键函数**: `run_kappa_calibration()`
**核心算法**: `estimate_kappa_pro()`

#### 5.3 视线计算和评估
```
1. 计算补偿后的视线向量
2. 评估拟合质量
3. 输出最终结果
```

**关键函数**: `compute_gaze()`

## 数据流图

```
摄像头 → 人脸检测 → 关键点提取 → 数据管理器
                                    ↓
眼球形状拟合 ← 数据管理器 ← 瞳孔定位
     ↓
会话管理器 ← last_fitting数据
     ↓
校准样本收集 ← 用户交互
     ↓
Kappa校准 ← 9个校准点
     ↓
最终视线计算
```

## 关键数据结构

### 1. 校准样本 (CalibrationSample)
```python
{
    "timestamp": float,
    "target_pixel": (int, int),
    "eye_center": [float, float, float],
    "pupil_center": [float, float, float], 
    "theoretical_gaze": [float, float, float],
    "quality": float,
    "frame_id": int
}
```

### 2. 拟合结果 (FittingResultLite)
```python
{
    "center": [float, float, float],
    "radius": float,
    "confidence": float,
    "converged": bool,
    "final_residual_norm": float,
    "strategy_used": str
}
```

### 3. 粗拟合结果 (coarse_fitting_results)
```python
{
    "kappa_angle": float,
    "fit_quality": dict,
    "status": str,  # "pending"/"success"/"failed"
    "message": str
}
```

## 状态转换图

```
初始化 → 检测人脸 → 粗拟合 → 收集校准点 → 精细拟合 → 完成
   ↓         ↓         ↓         ↓         ↓
  等待    人脸检测   眼球拟合   用户交互   Kappa校准
```

## 错误处理

### 常见错误情况
1. **摄像头初始化失败**: 检查摄像头连接
2. **人脸检测失败**: 调整光照和角度
3. **眼球拟合失败**: 检查关键点数据质量
4. **校准样本不足**: 确保收集到足够的校准点
5. **Kappa校准失败**: 检查数据质量和算法参数

### 日志输出
- **详细日志**: 每个步骤都有详细的日志记录
- **错误追踪**: 异常信息包含完整的堆栈跟踪
- **状态监控**: 实时显示系统状态和进度

## 前端接口

### 获取粗拟合结果
```python
results = get_coarse_fitting_results()
```

### 发送数据给前端
```python
send_to_frontend(data)
```

## 配置文件

### 相机参数 (`project/recognition/config/camera_params.json`)
```json
{
    "camera_name": "HD Camera",
    "intrinsic_params": {
        "fx": 1500.0, "fy": 1500.0,
        "cx": 960.0, "cy": 540.0
    },
    "image_resolution": {
        "width": 1920, "height": 1080
    }
}
```

## 运行命令

```bash
cd /Users/andrew/PycharmProjects/大创/gaze
python project/recognition/main.py
```

## 用户操作指南

1. **启动程序**: 运行主程序，等待摄像头初始化
2. **人脸检测**: 确保人脸在摄像头视野内
3. **粗拟合**: 系统自动进行眼球形状拟合
4. **校准收集**: 按 's' 键收集9个校准点
5. **精细拟合**: 系统自动进行Kappa校准
6. **完成**: 查看最终校准结果

## 技术特点

- **实时处理**: 基于OpenCV的实时视频处理
- **高精度**: 使用MediaPipe进行关键点检测
- **模块化**: 清晰的模块分离和接口设计
- **可扩展**: 支持多种拟合策略和校准方法
- **用户友好**: 直观的交互界面和状态提示

## 性能优化

- **GPU加速**: 支持CUDA加速（可配置）
- **内存管理**: 高效的数据存储和清理
- **日志控制**: 可配置的日志级别和输出
- **异常处理**: 完善的错误恢复机制
