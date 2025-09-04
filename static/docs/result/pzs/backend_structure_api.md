# 眼动追踪系统后端架构与接口文档

## 1. 系统架构概览

眼动追踪系统后端采用模块化设计，主要包含以下核心模块：

```
project/
├── fitting/                    # 眼球拟合与校准模块
│   ├── app/                   # 应用层接口
│   ├── core/                  # 核心算法
│   ├── config/                # 配置管理
│   └── utils/                 # 工具函数
├── recognition/               # 人脸识别与关键点检测模块
│   ├── core/                 # 核心检测算法
│   ├── config/               # 配置管理
│   └── utils/                # 工具函数
├── recg_fit_data/            # 数据管理模块
└── test/                     # 测试模块
```

## 2. 核心模块详解

### 2.1 拟合模块 (fitting/)

#### 2.1.1 应用层接口 (app/)

**主要文件：**
- `api.py` - 核心API接口
- `state.py` - 状态管理
- `calibration_ui.py` - 校准界面
- `run.py` - 运行入口

**核心API接口：**

```python
# 会话管理
def start_session(session_id: str) -> Dict
def push_sample(payload: Dict) -> Dict
def fit_kappa(session_id: Optional[str] = None, K: Optional[np.ndarray] = None) -> Dict
def compute_gaze(eye: str, target_pixel: Optional[Tuple[int, int]] = None) -> Dict
```

**数据结构：**

```python
@dataclass
class CalibrationSample:
    timestamp: float
    target_pixel: Tuple[int, int]
    eye_center: np.ndarray  # (3,), mm
    pupil_center: np.ndarray  # (3,), mm
    theoretical_gaze: np.ndarray  # (3,), unit vector
    quality: float
    frame_id: Optional[int] = None

@dataclass
class KappaModel:
    axis: np.ndarray  # (3,), unit vector
    angle_deg: float
    samples_count: int
    quality: float

@dataclass
class GazeVector:
    eye: Eye
    origin: np.ndarray  # (3,), mm
    direction: np.ndarray  # (3,), unit vector
    target_pixel: Optional[Tuple[int, int]] = None
```

#### 2.1.2 核心算法 (core/)

**主要文件：**
- `gaze_estimator.py` - 视线估计
- `kappa_calibrator_pro.py` - Kappa校准算法
- `fitting_strategy.py` - 拟合策略
- `eye_tracking_data_interface.py` - 数据接口

**核心功能：**
- 眼球几何拟合
- Kappa角补偿
- 视线向量计算
- 数据质量评估

### 2.2 识别模块 (recognition/)

#### 2.2.1 核心检测 (core/)

**主要文件：**
- `detector.py` - 人脸检测器
- `landmark_extractor.py` - 关键点提取
- `coordinate_converter.py` - 坐标转换

**核心功能：**
- 人脸检测与跟踪
- 眼部关键点提取
- 2D到3D坐标转换
- 深度信息处理

#### 2.2.2 配置管理 (config/)

**主要文件：**
- `settings.py` - 系统设置
- `camera_params.json` - 相机参数

### 2.3 数据管理模块 (recg_fit_data/)

**主要文件：**
- `data_manager.py` - 数据管理器

**核心功能：**
- 关键点坐标管理
- 数据格式转换
- 数据验证与清理
- 单位转换（米到毫米）

## 3. 主要接口规范

### 3.1 会话管理接口

#### 启动校准会话
```python
def start_session(session_id: str) -> Dict:
    """
    启动新的校准会话
    
    Args:
        session_id: 会话标识符
        
    Returns:
        Dict: 会话状态信息
        {
            "session_id": str,
            "stage": str,  # "pre_fitting" | "collecting" | "calibrated"
            "samples": int,
            "has_kappa": bool,
            "has_fitting": bool
        }
    """
```

#### 添加校准样本
```python
def push_sample(payload: Dict) -> Dict:
    """
    添加校准样本数据
    
    Args:
        payload: 样本数据
        {
            "timestamp": float,
            "target_pixel": [x, y],
            "eye": "left"|"right",
            "frame_id": Optional[int]
        }
        
    Returns:
        Dict: 操作结果
        {
            "ok": bool,
            "samples": int
        }
    """
```

### 3.2 Kappa校准接口

#### 执行Kappa校准
```python
def fit_kappa(session_id: Optional[str] = None, K: Optional[np.ndarray] = None) -> Dict:
    """
    使用当前会话样本估计Kappa模型
    
    Args:
        session_id: 会话ID（可选）
        K: 相机内参矩阵(3x3)（可选）
        
    Returns:
        Dict: Kappa模型信息
        {
            "kappa_model": {
                "axis": List[float],  # 旋转轴
                "angle_deg": float,   # 旋转角度（度）
                "samples_count": int, # 样本数量
                "quality": float      # 模型质量
            }
        }
    """
```

### 3.3 视线计算接口

#### 计算补偿后视线
```python
def compute_gaze(eye: str, target_pixel: Optional[Tuple[int, int]] = None) -> Dict:
    """
    计算补偿后的视线向量
    
    Args:
        eye: 眼睛类型 ("left" | "right")
        target_pixel: 目标像素坐标（可选）
        
    Returns:
        Dict: 视线向量信息
        {
            "gaze": {
                "eye": str,
                "origin": List[float],    # 视线起点
                "direction": List[float], # 视线方向（单位向量）
                "target_pixel": List[int] # 目标像素坐标
            }
        }
    """
```

### 3.4 数据管理接口

#### 获取关键点坐标
```python
def get_coordinate_point(eye: str, fitting_type: str) -> List[CoordinatePoint]:
    """
    获取指定眼睛和类型的坐标点
    
    Args:
        eye: 眼睛类型 ("left" | "right")
        fitting_type: 拟合类型 ("pupil" | "iris" | "inner_canthus" | 
                              "upper_eyelid" | "lower_eyelid" | "outer_canthus")
                              
    Returns:
        List[CoordinatePoint]: 坐标点列表
        CoordinatePoint: np.ndarray, shape=(4,), dtype=float32
        [x, y, z, visibility] 单位：毫米
    """
```

#### 设置关键点坐标
```python
def set_coordinate_point(eye: str, fitting_type: str, coordinate_point: List[CoordinatePoint]) -> None:
    """
    设置指定眼睛和类型的关键点坐标
    
    Args:
        eye: 眼睛类型
        fitting_type: 拟合类型
        coordinate_point: 坐标点列表
    """
```

## 4. 数据流架构

### 4.1 数据流向图

```
摄像头数据 → 人脸检测 → 关键点提取 → 坐标转换 → 数据管理器
    ↓
眼球拟合 ← 数据管理器 ← 关键点数据
    ↓
Kappa校准 ← 校准样本 ← 前端交互
    ↓
视线计算 → 补偿后视线 → 前端显示
```

### 4.2 关键数据格式

#### 关键点坐标格式
```python
CoordinatePoint = np.ndarray  # shape: (4,), dtype: float32
# [x, y, z, visibility]
# x, y, z: 3D坐标，单位：毫米
# visibility: 可见性，范围：0-1
```

#### 校准样本格式
```python
@dataclass
class CalibrationSample:
    timestamp: float                    # 时间戳
    target_pixel: Tuple[int, int]       # 目标像素坐标
    eye_center: np.ndarray             # 眼球中心 (3,) mm
    pupil_center: np.ndarray           # 瞳孔中心 (3,) mm
    theoretical_gaze: np.ndarray       # 理论视线 (3,) 单位向量
    quality: float                     # 质量分数 0-1
    frame_id: Optional[int]            # 帧ID
```

### 4.3 前端校准数据JSON格式

前端眼动追踪校准系统返回给后端的JSON数据格式规范：

#### 4.3.1 完整JSON结构
```json
{
  "metadata": {
    "description": "眼动追踪校准数据示例 - 9点校准",
    "created_at": "2024-12-01T10:30:00.000000",
    "data_format": "JSON",
    "coordinate_system": "camera_coordinates_mm",
    "sample_count": 9,
    "calibration_type": "9_point_grid",
    "hardware": "simulated_data"
  },
  "eyes": [
    [0.0, 0.0, 0.0],
    [0.05, 0.0, 0.0],
    [0.1, 0.0, 0.0],
    [0.0, 0.05, 0.0],
    [0.05, 0.05, 0.0],
    [0.1, 0.05, 0.0],
    [0.0, 0.1, 0.0],
    [0.05, 0.1, 0.0],
    [0.1, 0.1, 0.0]
  ],
  "pupils": [
    [0.0, 0.0, 0.1],
    [0.05, 0.0, 0.1],
    [0.1, 0.0, 0.1],
    [0.0, 0.05, 0.1],
    [0.05, 0.05, 0.1],
    [0.1, 0.05, 0.1],
    [0.0, 0.1, 0.1],
    [0.05, 0.1, 0.1],
    [0.1, 0.1, 0.1]
  ],
  "target_pixels": [
    [192, 108],
    [960, 108],
    [1728, 108],
    [192, 540],
    [960, 540],
    [1728, 540],
    [192, 972],
    [960, 972],
    [1728, 972]
  ],
  "camera_matrix": [
    [1000.0, 0.0, 960.0],
    [0.0, 1000.0, 540.0],
    [0.0, 0.0, 1.0]
  ],
  "screen_resolution": [1920, 1080],
  "units": {
    "coordinates": "millimeters",
    "pixels": "pixels",
    "angles": "degrees"
  },
  "calibration_info": {
    "method": "9_point_grid",
    "duration_seconds": 18,
    "samples_per_point": 20,
    "point_order": [
      "top_left", "top_center", "top_right",
      "middle_left", "center", "middle_right", 
      "bottom_left", "bottom_center", "bottom_right"
    ]
  }
}
```

#### 4.3.2 字段详细说明

**metadata（元数据）**
```typescript
interface Metadata {
  description: string;           // 数据描述
  created_at: string;           // 创建时间 (ISO 8601格式)
  data_format: "JSON";          // 数据格式
  coordinate_system: string;    // 坐标系统
  sample_count: number;         // 样本数量
  calibration_type: string;     // 校准类型
  hardware: string;             // 硬件信息
}
```

**eyes（眼球中心坐标）**
```typescript
type EyeCoordinates = number[][];  // [[x, y, z], ...]
// 每个子数组包含3个浮点数：[x, y, z]
// 单位：毫米
// 坐标系：相机坐标系
```

**pupils（瞳孔中心坐标）**
```typescript
type PupilCoordinates = number[][];  // [[x, y, z], ...]
// 每个子数组包含3个浮点数：[x, y, z]
// 单位：毫米
// 坐标系：相机坐标系
```

**target_pixels（目标像素坐标）**
```typescript
type TargetPixels = number[][];  // [[x, y], ...]
// 每个子数组包含2个整数：[x, y]
// 单位：像素
// 坐标系：屏幕像素坐标系
```

**camera_matrix（相机内参矩阵）**
```typescript
type CameraMatrix = number[][];  // 3x3矩阵
// [
//   [fx, 0, cx],
//   [0, fy, cy],
//   [0, 0, 1]
// ]
// fx, fy: 焦距（像素）
// cx, cy: 主点坐标（像素）
```

**screen_resolution（屏幕分辨率）**
```typescript
type ScreenResolution = [number, number];  // [width, height]
// 屏幕宽度和高度（像素）
```

**units（单位说明）**
```typescript
interface Units {
  coordinates: "millimeters";  // 坐标单位
  pixels: "pixels";           // 像素单位
  angles: "degrees";          // 角度单位
}
```

**calibration_info（校准信息）**
```typescript
interface CalibrationInfo {
  method: string;              // 校准方法
  duration_seconds: number;    // 校准持续时间（秒）
  samples_per_point: number;   // 每个点的样本数
  point_order: string[];       // 校准点顺序
}
```

#### 4.3.3 数据验证规则

**必需字段验证**
- `metadata`、`eyes`、`pupils`、`target_pixels` 为必需字段
- `camera_matrix`、`screen_resolution` 为必需字段
- `units`、`calibration_info` 为必需字段

**数据一致性验证**
- `eyes`、`pupils`、`target_pixels` 数组长度必须相等
- `eyes` 和 `pupils` 中每个坐标点必须包含3个数值
- `target_pixels` 中每个像素点必须包含2个数值
- `camera_matrix` 必须为3x3矩阵
- `screen_resolution` 必须包含2个正整数

**数值范围验证**
- 坐标值：合理的3D空间坐标范围
- 像素值：非负整数，不超过屏幕分辨率
- 相机内参：焦距和主点必须为正数
- 时间戳：有效的ISO 8601格式

#### 4.3.4 后端处理接口

**接收前端校准数据**
```python
def process_calibration_data(json_data: Dict) -> Dict:
    """
    处理前端返回的校准数据
    
    Args:
        json_data: 前端校准数据JSON
        
    Returns:
        Dict: 处理结果
        {
            "success": bool,
            "message": str,
            "processed_samples": int,
            "kappa_model": Optional[Dict]
        }
    """
    # 1. 验证JSON数据格式
    # 2. 提取eyes、pupils、target_pixels数据
    # 3. 构建校准样本
    # 4. 执行Kappa校准
    # 5. 返回校准结果
```

**数据转换示例**
```python
def convert_json_to_samples(json_data: Dict) -> List[CalibrationSample]:
    """
    将JSON数据转换为校准样本列表
    
    Args:
        json_data: 前端JSON数据
        
    Returns:
        List[CalibrationSample]: 校准样本列表
    """
    eyes = np.array(json_data["eyes"])
    pupils = np.array(json_data["pupils"])
    target_pixels = json_data["target_pixels"]
    
    samples = []
    for i in range(len(eyes)):
        sample = CalibrationSample(
            timestamp=time.time(),
            target_pixel=tuple(target_pixels[i]),
            eye_center=eyes[i],
            pupil_center=pupils[i],
            theoretical_gaze=compute_theoretical_gaze(eyes[i], pupils[i]),
            quality=1.0,
            frame_id=i
        )
        samples.append(sample)
    
    return samples
```

## 5. 配置管理

### 5.1 相机参数配置

```json
{
    "intrinsic_params": {
        "fx": 1000.0,    // 焦距X
        "fy": 1000.0,    // 焦距Y
        "cx": 960.0,     // 主点X
        "cy": 540.0      // 主点Y
    },
    "depth_scale": 0.001  // 深度缩放因子
}
```

### 5.2 拟合参数配置

```python
FITTING_CONFIG = {
    "max_iterations": 1000,
    "convergence_threshold": 1e-6,
    "initial_radius": 12.0,  # 初始眼球半径（毫米）
    "strategy": "adaptive"    # 拟合策略
}
```

## 6. 错误处理

### 6.1 常见错误类型

```python
class CalibrationError(Exception):
    """校准相关错误"""
    pass

class DataValidationError(Exception):
    """数据验证错误"""
    pass

class FittingError(Exception):
    """拟合错误"""
    pass
```

### 6.2 错误处理策略

- **数据验证**：输入数据格式和范围检查
- **异常捕获**：关键操作异常捕获和日志记录
- **降级处理**：关键功能失败时的备用方案
- **状态恢复**：错误后的状态重置和恢复

## 7. 性能优化

### 7.1 计算优化

- **向量化计算**：使用NumPy向量化操作
- **缓存机制**：频繁计算结果缓存
- **并行处理**：多眼并行拟合
- **增量更新**：Kappa模型增量更新

### 7.2 内存管理

- **数据复用**：避免不必要的数据复制
- **及时清理**：及时释放大对象内存
- **流式处理**：大数据流式处理

## 8. 测试策略

### 8.1 单元测试

- **算法测试**：核心算法正确性验证
- **接口测试**：API接口功能测试
- **数据测试**：数据格式和转换测试

### 8.2 集成测试

- **模块集成**：模块间接口集成测试
- **端到端测试**：完整流程测试
- **性能测试**：系统性能基准测试

## 9. 部署与维护

### 9.1 部署要求

- **Python版本**：Python 3.8+
- **依赖库**：NumPy, OpenCV, SciPy
- **硬件要求**：支持深度相机的计算设备
- **系统要求**：Windows/Linux/macOS

### 9.2 监控与日志

- **性能监控**：关键指标实时监控
- **错误日志**：详细错误信息记录
- **调试信息**：开发调试信息输出
- **用户行为**：用户操作行为记录

## 10. 扩展性设计

### 10.1 模块化架构

- **插件机制**：支持算法插件扩展
- **配置驱动**：通过配置文件控制行为
- **接口标准化**：统一的模块间接口

### 10.2 未来扩展

- **多相机支持**：支持多相机系统
- **实时处理**：实时数据处理优化
- **云端集成**：云端数据存储和分析
- **AI增强**：机器学习算法集成
