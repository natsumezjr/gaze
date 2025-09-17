# 眼动追踪系统重构方案

## 一、项目结构问题分析

### 1.1 当前问题

1. **模块职责混乱**：识别模块(`project/recognition/main.py`)集成了系统管理功能(`gaze_system`)，违反了单一职责原则
2. **算法实现分散**：`gaze_estimator.py`中的算法未被使用，注视点计算逻辑写在识别模块中且极其不精确
3. **配置管理不当**：屏幕、摄像机信息应统一配置在`settings.py`中，便于交点计算
4. **前端架构不匹配**：当前使用网页前端，但VR眼镜需要客户端应用

### 1.2 重构目标

- 明确模块职责分工
- 统一算法实现和配置管理
- 将网页前端改为客户端应用
- 保持功能完整性

## 二、重构后的项目结构

```
gaze_tracking_system/
├── main.py                          # 系统主入口，协调各模块
├── requirements.txt                 # 依赖管理
├── config/                          # 全局配置
│   ├── __init__.py
│   ├── settings.py                  # 统一配置管理
│   ├── camera_params.json          # 相机参数
│   └── screen_config.json          # 屏幕配置
├── core/                           # 核心算法模块
│   ├── __init__.py
│   ├── recognition/                 # 识别模块
│   │   ├── __init__.py
│   │   ├── detector.py             # 人脸检测
│   │   ├── landmark_extractor.py   # 关键点提取
│   │   └── coordinate_converter.py # 坐标转换
│   ├── fitting/                     # 拟合模块
│   │   ├── __init__.py
│   │   ├── sphere_fitter.py        # 眼球球面拟合
│   │   ├── kappa_calibrator.py     # Kappa角标定
│   │   └── gaze_estimator.py       # 视线估计
│   └── intersection/                # 交点计算模块
│       ├── __init__.py
│       ├── screen_intersection.py  # 屏幕交点计算
│       └── geometry_utils.py       # 几何工具
├── data/                           # 数据管理
│   ├── __init__.py
│   ├── data_manager.py             # 数据管理器
│   └── session_manager.py          # 会话管理
├── client/                         # 客户端应用
│   ├── __init__.py
│   ├── main_window.py              # 主窗口
│   ├── calibration_ui.py           # 校准界面
│   ├── gaze_visualizer.py          # 注视点可视化
│   └── utils/                      # 客户端工具
│       ├── __init__.py
│       ├── ui_components.py        # UI组件
│       └── event_handlers.py       # 事件处理
└── utils/                          # 通用工具
    ├── __init__.py
    ├── logging_config.py           # 日志配置
    └── system_utils.py             # 系统工具
```

## 三、各模块职责分工

### 3.1 主入口模块 (`main.py`)
- **职责**：系统启动、模块协调、全局状态管理
- **功能**：
  - 初始化各模块
  - 管理系统生命周期
  - 协调模块间通信
  - 处理系统级异常

### 3.2 配置模块 (`config/`)
- **职责**：统一配置管理
- **功能**：
  - 屏幕物理参数（尺寸、分辨率、3D坐标）
  - 相机参数（内参、外参、畸变系数）
  - 系统参数（算法阈值、拟合参数等）
  - 环境配置（日志级别、调试模式等）

### 3.3 核心算法模块 (`core/`)

#### 3.3.1 识别模块 (`core/recognition/`)
- **职责**：从图像中提取眼部关键点
- **功能**：
  - 人脸检测和跟踪
  - 眼部关键点提取（瞳孔、虹膜、眼角等）
  - 3D坐标转换
  - 数据质量评估

#### 3.3.2 拟合模块 (`core/fitting/`)
- **职责**：眼球几何建模和Kappa角标定
- **功能**：
  - 眼球球面拟合（RANSAC算法）
  - Kappa角估计和补偿
  - 理论视线计算
  - 拟合质量评估

#### 3.3.3 交点计算模块 (`core/intersection/`)
- **职责**：视线与屏幕的交点计算
- **功能**：
  - 视线向量与屏幕平面求交
  - 像素坐标转换
  - 多屏幕支持
  - 交点有效性验证

### 3.4 数据管理模块 (`data/`)
- **职责**：数据存储和会话管理
- **功能**：
  - 关键点数据存储
  - 校准样本管理
  - 拟合结果缓存
  - 会话状态维护

### 3.5 客户端模块 (`client/`)
- **职责**：用户界面和交互
- **功能**：
  - 主窗口管理
  - 校准流程引导
  - 实时注视点显示
  - 系统状态监控
  - VR眼镜适配

## 四、拟合模块算法详解

### 4.1 眼球球面拟合算法 (`sphere_fitter.py`)

#### 4.1.1 算法原理
基于RANSAC + Levenberg-Marquardt的鲁棒球面拟合算法，用于从眼部关键点估计眼球几何参数。该算法结合了RANSAC的鲁棒性和LM算法的精确性。

#### 4.1.2 数学公式

**球面方程**：
$$(x - c_x)^2 + (y - c_y)^2 + (z - c_z)^2 = r^2$$

其中：
- $C = (c_x, c_y, c_z)$ 为球心坐标
- $r$ 为球体半径

**残差定义**：
系统定义了多种残差类型来约束拟合过程：

1. **表面残差**：
$$f_{surface,i} = \sqrt{\alpha} \cdot (||p_i - c|| - r - bias_{surface,i}) \cdot w_{huber,i}$$

2. **中心残差**：
$$f_{center,i} = \sqrt{1-\alpha} \cdot (||p_i - c|| - bias_{center,i}) \cdot w_{huber,i}$$

3. **深度残差**（Softplus约束）：
$$f_{depth} = \sqrt{\lambda_{depth}} \cdot \sigma \cdot \ln(1 + \exp(u/\sigma))$$

其中 $u = \max(0, \max_z + \epsilon - c_z)$

4. **眼球外约束残差**：
$$f_{outside,i} = \sqrt{\lambda_{outside}} \cdot \sigma_{outside} \cdot \ln(1 + \exp(u_i/\sigma_{outside}))$$

其中 $u_i = \max(0, r - ||p_i - c||)$

**Huber权重函数**：
$$w_{huber,i} = \min(1, \frac{\kappa}{|f_{raw,i}|})$$

**目标函数**：
$$\Phi(\theta) = \frac{1}{2} \sum_{i} f_i^2$$

#### 4.1.3 算法步骤

**RANSAC阶段**：
1. **随机采样**：从虹膜边界点中随机选择4个点
2. **初始拟合**：使用最小二乘法拟合球面参数
3. **内点检测**：计算所有点到拟合球面的距离，判断内点
4. **迭代优化**：重复上述步骤，选择内点最多的模型
5. **置信度评估**：基于几何约束和正态分布计算置信度

**Levenberg-Marquardt优化阶段**：
1. **残差计算**：计算所有残差项及其雅可比矩阵
2. **正规方程求解**：$(J^T J + \lambda I) \Delta = -J^T r$
3. **步长控制**：裁剪步长范数，防止发散
4. **收敛判断**：检查参数变化和残差变化
5. **阻尼调整**：根据目标函数变化调整阻尼因子

#### 4.1.4 关键参数
```python
# RANSAC参数
RANSAC_PARAMS = {
    "max_iterations": 20,           # 最大迭代次数
    "min_fitting_points": 3,        # 最小拟合点数
    "min_constraint_points": 3,     # 最小约束点数
    "base_optimizer": "lm"          # 基础优化器
}

# Levenberg-Marquardt参数
LM_PARAMS = {
    "max_iterations": 30,           # 最大迭代次数
    "lambda_init": 0.01,            # 初始阻尼因子
    "lambda_factor": 10.0,          # 阻尼因子调整倍数
    "delta_convergence_threshold": 1.0,  # 参数收敛阈值
    "residual_convergence_threshold": 0.01,  # 残差收敛阈值
    "max_step_size": 4.0,           # 最大步长
    "alpha": 0.1,                   # 表面残差权重
    "huber_kappa": 1.5,             # Huber阈值
    "depth_epsilon": 5.0,           # 深度容差(mm)
    "depth_sigma": 100.0,           # 深度平滑参数
    "depth_lambda": 0.01,           # 深度约束权重
    "outside_eyeball_sigma": 1.0,   # 眼球外约束平滑参数
    "outside_eyeball_lambda": 0.1   # 眼球外约束权重
}

# 解剖学约束
ANATOMICAL_CONSTRAINTS = {
    "TO_SURFACE": {                 # 表面偏差(mm)
        "pupil": (-4.5, -2.5),
        "iris": (-1.8, -0.4),
        "inner_canthus": (4.0, 8.0),
        "upper_eyelid": (0.0, 2.0),
        "lower_eyelid": (0.2, 1.6),
        "outer_canthus": (2.5, 6.7)
    },
    "TO_CENTER": {                  # 中心偏差(mm)
        "pupil": (7.95, 9.95),
        "iris": (10.40, 12.04),
        "inner_canthus": (16.0, 20.0),
        "upper_eyelid": (11.3, 13.3),
        "lower_eyelid": (12.0, 14.0),
        "outer_canthus": (14.9, 18.9)
    },
    "EYEBALL_RADIUS": (11.0, 13.0), # 眼球半径范围(mm)
    "EYEBALL_RADIUS_DEFAULT": 12.0   # 默认半径(mm)
}
```

### 4.2 Kappa角标定算法 (`kappa_calibrator.py`)

#### 4.2.1 算法原理
Kappa角是光轴与视轴之间的微小角度差异，通过校准样本估计个体化的Kappa参数。使用小角度线性化将非线性旋转问题转化为线性最小二乘问题。

#### 4.2.2 数学公式

**旋转关系**：
$$\vec{v}_{visual} = R(\vec{\alpha}) \cdot \vec{v}_{optical}$$

其中：
- $\vec{v}_{optical}$ 为光轴向量（眼球中心到瞳孔中心）
- $\vec{v}_{visual}$ 为视轴向量（指向注视目标）
- $R(\vec{\alpha})$ 为旋转矩阵，$\vec{\alpha} = [\alpha_x, \alpha_y, \alpha_z]$ 为Kappa角

**小角度线性化**：
$$R(\vec{\alpha}) \vec{v} \approx \vec{v} + \vec{\alpha} \times \vec{v}$$

**Rodrigues旋转公式**：
$$R(\vec{\alpha}) = I + \sin(|\vec{\alpha}|) \frac{[\vec{\alpha}]_\times}{|\vec{\alpha}|} + (1-\cos(|\vec{\alpha}|)) \frac{[\vec{\alpha}]_\times^2}{|\vec{\alpha}|^2}$$

其中 $[\vec{\alpha}]_\times$ 为反对称矩阵：
$$[\vec{\alpha}]_\times = \begin{bmatrix}
0 & -\alpha_z & \alpha_y \\
\alpha_z & 0 & -\alpha_x \\
-\alpha_y & \alpha_x & 0
\end{bmatrix}$$

**线性化方程**：
$$\vec{\alpha} \times \vec{v} = -[\vec{v}]_\times \vec{\alpha}$$

因此：
$$-\hat{v} \vec{\alpha} = \vec{d} - \vec{v}$$

其中 $\hat{v} = [\vec{v}]_\times$ 为反对称矩阵。

**加权最小二乘**：
$$\min_{\vec{\alpha}} ||W^{1/2} (M\vec{\alpha} - b)||^2$$

其中：
- $M = [-\hat{v}_1; -\hat{v}_2; \ldots; -\hat{v}_N]$ 为 $3N \times 3$ 矩阵
- $b = [\vec{d}_1 - \vec{v}_1; \vec{d}_2 - \vec{v}_2; \ldots; \vec{d}_N - \vec{v}_N]$ 为 $3N$ 向量
- $W$ 为权重矩阵

**Huber鲁棒权重**：
$$w_i = \begin{cases}
1 & \text{if } |r_i| \leq \delta \\
\frac{\delta}{|r_i|} & \text{if } |r_i| > \delta
\end{cases}$$

其中 $r_i$ 为角度残差，$\delta$ 为Huber阈值。

#### 4.2.3 算法步骤

1. **样本收集**：收集多个已知注视点的校准样本
2. **数据预处理**：验证样本有效性，计算光轴和视轴向量
3. **线性化**：构建线性方程组 $M\vec{\alpha} = b$
4. **初始求解**：使用加权最小二乘法求解Kappa角
5. **鲁棒优化**：使用Huber权重进行迭代重加权
6. **质量评估**：计算角度误差和像素误差

#### 4.2.4 关键参数
```python
KAPPA_CALIBRATION_PARAMS = {
    "lock_roll": True,              # 锁定滚转角（αz=0）
    "robust_iters": 3,              # 鲁棒迭代次数
    "huber_delta_deg": 5.0,         # Huber阈值(度)
    "min_samples": 9,               # 最小样本数
    "max_angular_error": 2.0,       # 最大角度误差(度)
    "max_pixel_error": 50.0,        # 最大像素误差(像素)
    "weight_threshold": 1e-12       # 权重阈值
}

# 角度误差计算
def angle_error_deg(kappa, V, D):
    """计算角度误差（度）"""
    R = rodrigues(kappa)
    V_rot = (R @ V.T).T
    dots = np.clip(np.sum(V_rot * D, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(dots))
```

### 4.3 屏幕交点计算算法 (`screen_intersection.py`)

#### 4.3.1 算法原理
计算视线向量与屏幕平面的交点，并转换为像素坐标。使用射线与平面求交的几何方法，结合坐标变换实现精确的像素映射。

#### 4.3.2 数学公式

**射线方程**：
$$\vec{P}(t) = \vec{O} + t \cdot \vec{D}$$

其中：
- $\vec{O}$ 为射线起点（眼球中心）
- $\vec{D}$ 为射线方向（视线向量）
- $t$ 为参数

**平面方程**：
$$\vec{N} \cdot (\vec{P} - \vec{P_0}) = 0$$

其中：
- $\vec{N}$ 为平面法向量
- $\vec{P_0}$ 为平面上一点

**交点参数**：
$$t = \frac{\vec{N} \cdot (\vec{P_0} - \vec{O})}{\vec{N} \cdot \vec{D}}$$

**交点坐标**：
$$\vec{P}_{intersect} = \vec{O} + t \cdot \vec{D}$$

**平面坐标系转换**：
$$\vec{p}_{2D} = \begin{bmatrix}
\vec{x}_{axis} \cdot (\vec{P}_{intersect} - \vec{P_0}) \\
\vec{y}_{axis} \cdot (\vec{P}_{intersect} - \vec{P_0})
\end{bmatrix}$$

**像素坐标映射**：
$$u = \frac{x - x_{min}}{x_{max} - x_{min}} \cdot W_{screen}$$
$$v = \frac{y - y_{min}}{y_{max} - y_{min}} \cdot H_{screen}$$

其中：
- $(x_{min}, y_{min})$ 和 $(x_{max}, y_{max})$ 为屏幕在平面坐标系中的边界
- $W_{screen}$ 和 $H_{screen}$ 为屏幕像素分辨率

#### 4.3.3 算法步骤

1. **平面定义**：定义屏幕平面（法向量、中心点、坐标轴）
2. **射线求交**：计算视线与屏幕平面的交点
3. **坐标转换**：将3D交点转换为屏幕2D坐标
4. **像素映射**：将物理坐标映射到像素坐标
5. **边界检查**：验证交点是否在屏幕范围内
6. **有效性验证**：检查交点是否在合理范围内

#### 4.3.4 关键参数
```python
SCREEN_CONFIG = {
    "physical_size": {              # 物理尺寸(m)
        "width": 0.6,
        "height": 0.34
    },
    "resolution": {                 # 分辨率(像素)
        "width": 1920,
        "height": 1080
    },
    "position": {                   # 屏幕位置(mm)
        "center": [0, 0, 0],        # 屏幕中心
        "normal": [0, 0, 1],        # 法向量
        "x_axis": [1, 0, 0],        # X轴方向
        "y_axis": [0, 1, 0]         # Y轴方向
    },
    "corners": {                    # 屏幕四角坐标(mm)
        "top_left": [-300, -170, 0],
        "top_right": [300, -170, 0],
        "bottom_left": [-300, 170, 0],
        "bottom_right": [300, 170, 0]
    },
    "camera_offset": {              # 相机偏移(mm)
        "x": 0.0,
        "y": 5.0
    }
}

# 几何计算参数
GEOMETRY_PARAMS = {
    "intersection_tolerance": 1e-8,  # 交点计算容差
    "parallel_threshold": 1e-6,      # 平行判断阈值
    "boundary_margin": 0.01,         # 边界容差(m)
    "max_distance": 10.0             # 最大视线距离(m)
}
```

## 五、客户端应用设计

### 5.1 技术选型
- **框架**：PyQt6/PySide6（跨平台GUI框架）
- **优势**：
  - 原生桌面应用，性能优异
  - 支持VR眼镜显示
  - 跨平台兼容性好
  - 丰富的UI组件

### 5.2 界面设计
- **主窗口**：系统状态显示、校准控制
- **校准界面**：9点校准流程、实时反馈
- **可视化界面**：注视点显示、轨迹记录
- **设置界面**：参数配置、系统调试

### 5.3 VR适配
- **全屏模式**：支持VR眼镜全屏显示
- **低延迟**：优化渲染性能，减少延迟
- **高刷新率**：支持高刷新率显示
- **眼动追踪**：集成眼动追踪数据

## 六、实施计划

### 6.1 第一阶段：模块重构
1. 创建新的项目结构
2. 重构配置管理模块
3. 分离识别和拟合模块
4. 实现统一的算法接口

### 6.2 第二阶段：算法优化
1. 完善眼球拟合算法
2. 优化Kappa角标定
3. 实现精确的屏幕交点计算
4. 添加算法质量评估

### 6.3 第三阶段：客户端开发
1. 设计客户端界面
2. 实现校准流程
3. 添加实时可视化
4. VR眼镜适配

### 6.4 第四阶段：集成测试
1. 模块集成测试
2. 端到端功能测试
3. 性能优化
4. 用户验收测试

## 七、预期效果

### 7.1 架构改进
- 模块职责清晰，易于维护
- 算法实现统一，便于优化
- 配置管理集中，便于部署

### 7.2 性能提升
- 注视点计算精度提高
- 系统响应速度优化
- VR眼镜适配性能提升

### 7.3 可维护性
- 代码结构清晰
- 接口设计合理
- 文档完善

### 7.4 可扩展性
- 支持多屏幕配置
- 支持不同VR设备
- 支持算法参数调优

---

*本重构方案基于对现有代码的深入分析，旨在解决当前架构问题，提升系统性能和可维护性，为VR眼镜应用做好准备。*
