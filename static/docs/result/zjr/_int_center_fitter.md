# 眼球中心拟合算法实现文档

## 一、算法概述

### 1.1 核心思想
基于现有`geometry.py`中的几何计算函数，实现分层拟合的眼球中心定位算法。该算法采用RANSAC采样 + Levenberg-Marquardt优化的策略，确保所有关键点都在眼球外部，符合解剖学关系。

### 1.2 算法流程
1. **数据预处理**：置信度过滤 → 数据质量检查 → 权重计算
2. **初值估计**：2D圆拟合XY → 解剖先验Z/R → 初始参数
3. **RANSAC几何判定**：采样点集 → 几何判定 → 内点统计
4. **LM精化优化**：构造雅可比矩阵 → 求解LM方程 → 参数更新 → 收敛检测

## 二、核心算法实现

### 2.1 几何计算基础函数

#### 2.1.1 向量操作
```python
def vector_of_2_points(point1: np.ndarray, point2: np.ndarray) -> np.ndarray:
    """计算两点间向量"""
    return point2 - point1

def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """向量归一化"""
    return vector / np.linalg.norm(vector)

def angle_between_vectors(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """计算两向量夹角（弧度）"""
    return np.arccos(np.dot(vector1, vector2) / 
                    (np.linalg.norm(vector1) * np.linalg.norm(vector2)))
```

#### 2.1.2 屏幕投影计算
```python
def intersect_pixel_on_screen(vector: np.ndarray, rgb_d=False) -> np.ndarray:
    """计算向量与屏幕的交点（像素坐标）"""
    screen = Plane(rgb_d)
    point_2d = screen.intersection_on_plane(vector)
    if point_2d is None:
        return np.array([np.nan, np.nan])
    
    # 转换到屏幕像素坐标
    top_left_2d = screen._point_3d_to_2d(screen.top_left)
    point_2d = point_2d - top_left_2d
    
    x, y = point_2d
    screen_width, screen_height = screen.width_m, screen.height_m
    w, h = SCREEN_WITH_RGBD["resolution_px"]
    
    return np.array([w * x / screen_width, h * y / screen_height])
```

#### 2.1.3 向量旋转
```python
def rotate_vector(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """使用罗德里格斯旋转公式绕轴旋转向量"""
    axis = normalize_vector(axis)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    v_parallel = np.dot(vector, axis) * axis
    v_perpendicular = vector - v_parallel
    
    rotated_vector = (v_parallel + 
                     v_perpendicular * cos_angle + 
                     np.cross(axis, vector) * sin_angle)
    
    return rotated_vector
```

### 2.2 权重计算系统

#### 2.2.1 两层权重架构
```python
def _calculate_weights(self, points: List[np.ndarray], structure_types: List[str], center_only: bool = False) -> WeightResult:
    """
    两层权重计算：权重 = 解剖学权重 × 几何权重
    
    解剖学权重（固定值）：
    - 瞳孔中心：w_anat = 0.991
    - 虹膜边界：w_anat = 0.839  
    - 眼睛轮廓：w_anat = 0.607
    
    几何权重（动态计算）：
    - w_geom = exp(-d²/(2σ²))
    - d为距离偏差，σ为初始化标准差
    """
```

#### 2.2.2 残差驱动权重调整
```python
def _calculate_final_weight(self, w_anat: float, w_geom: float) -> float:
    """
    残差驱动权重调整策略
    
    权重策略：
    1. 残差范数 > 截止阈值×10：只使用解剖学权重
    2. 残差范数 > 截止阈值：解剖学权重 × min(几何权重, 0.8)
    3. 残差范数 ≤ 截止阈值：解剖学权重 × 几何权重
    """
```

### 2.3 采样策略

#### 2.3.1 RANSAC采样
```python
def _ransac_sampling(self, points: List[np.ndarray], structure_types: List[str], 
                     confidence_scores: List[float]) -> SamplingResult:
    """
    分层采样策略：
    1. 瞳孔中心：随机选1个
    2. 虹膜边界：随机选3个
    3. 眼睛轮廓：随机选4-10个
    4. 补充点：随机选择其他点
    """
```

#### 2.3.2 置信度过滤
```python
def _filter_low_confidence_points(self, points: List[np.ndarray], 
                                 structure_types: List[str], 
                                 confidence_scores: List[float]) -> Tuple[List, List, List]:
    """过滤置信度低于阈值的点"""
    threshold = self._fitting_config.confidence_threshold
    filtered_indices = [i for i, conf in enumerate(confidence_scores) if conf >= threshold]
    
    filtered_points = [points[i] for i in filtered_indices]
    filtered_types = [structure_types[i] for i in filtered_indices]
    filtered_confidences = [confidence_scores[i] for i in filtered_indices]
    
    return filtered_points, filtered_types, filtered_confidences
```

### 2.4 初值估计

#### 2.4.1 2D圆拟合
```python
def _estimate_initial_params(self, points: List[np.ndarray], structure_types: List[str], center_only: bool = False) -> np.ndarray:
    """
    最精简的眼球中心初始估计
    
    策略：
    1. 使用虹膜边界点进行2D圆拟合
    2. 瞳孔Z坐标 + 1.2mm解剖偏移作为球心Z
    3. 使用解剖学先验半径12mm
    """
    # 提取虹膜边界点
    iris_points = [p for p, t in zip(points, structure_types) if t == "iris_boundary"]
    
    if iris_points:
        iris_array = np.array(iris_points)
        # 使用质心作为初始中心
        center = np.mean(iris_array, axis=0)
    else:
        points_array = np.array(points)
        center = np.mean(points_array, axis=0)
    
    self._current_fit_center = center
    
    if center_only:
        self._estimated_initial_params = center
    else:
        radius = EYEBALL_RADIUS  # 12mm
        self._current_fit_radius = radius
        self._estimated_initial_params = np.array([center[0], center[1], center[2], radius])
    
    self._initial_params_estimated = True
    return self._estimated_initial_params
```

### 2.5 Levenberg-Marquardt优化

#### 2.5.1 核心算法
```python
def _levenberg_marquardt_optimization(self, initial_params: np.ndarray, points: List[np.ndarray], 
                                    structure_types: List[str], center_only: bool = False) -> Tuple[np.ndarray, bool]:
    """
    LM优化算法实现
    
    数学公式：
    - 目标函数：Φ(θ) = ½ * f(θ)ᵀ * W * f(θ)
    - LM更新方程：(Jᵀ * W * J + λ * I) * Δ = -Jᵀ * W * f(θ)
    - 参数更新：θ_new = θ_old - Δ
    
    收敛判据：
    - 参数收敛：||Δ|| < param_tolerance
    - 残差收敛：|residual_norm_new - residual_norm_old| < residual_tolerance
    """
```

#### 2.5.2 雅可比矩阵计算
```python
def _compute_jacobian(self, params: np.ndarray, points: List[np.ndarray], center_only: bool = False) -> np.ndarray:
    """
    计算雅可比矩阵 J = ∂f/∂θ
    
    对于球体拟合：
    - f_i(θ) = ||p_i - c|| - r
    - ∂f_i/∂c_x = -(x_i - c_x)/d_i
    - ∂f_i/∂c_y = -(y_i - c_y)/d_i  
    - ∂f_i/∂c_z = -(z_i - c_z)/d_i
    - ∂f_i/∂r = -1
    
    其中 d_i = ||p_i - c||
    """
```

#### 2.5.3 残差计算
```python
def _compute_residuals(self, params: np.ndarray, points: List[np.ndarray], center_only: bool = False) -> np.ndarray:
    """
    计算残差向量 f(θ) = [f₁, f₂, ..., fₙ]ᵀ
    
    其中 f_i = ||p_i - c|| - r
    """
```

### 2.6 内点判定

#### 2.6.1 几何判定
```python
def _calculate_inlier_ratio(self, points: List[np.ndarray], center: np.ndarray, 
                           radius: float, threshold: float) -> Tuple[int, float]:
    """
    计算内点比例
    
    内点判定条件：|d_i - r| ≤ threshold
    其中 d_i = ||p_i - c|| 是点到球心的距离
    """
    inlier_count = 0
    total_points = len(points)
    
    for point in points:
        distance = np.linalg.norm(point - center)
        if abs(distance - radius) <= threshold:
            inlier_count += 1
    
    inlier_ratio = inlier_count / total_points if total_points > 0 else 0.0
    return inlier_count, inlier_ratio
```

## 三、配置参数

### 3.1 拟合配置
```python
@dataclass
class FittingConfig:
    # 优化参数
    max_iterations: int = 50
    param_tolerance: float = 1e-6        # 参数收敛阈值
    residual_tolerance: float = 1e-8     # 残差收敛阈值
    
    # 残差驱动权重调整参数
    residual_cutoff_threshold: float = 0.01      # 残差截止阈值
    residual_protection_factor: float = 5.0      # 残差保护因子
    geometric_weight_cap: float = 0.8            # 几何权重上限
    
    # 采样参数
    min_iris_points: int = 3             # 最小虹膜点数
    min_contour_points: int = 4          # 最小轮廓点数
    max_contour_points: int = 10         # 最大轮廓点数
    
    # RANSAC参数
    ransac_threshold: float = 0.002     # 内点判定阈值（2mm）
    max_trials: int = 100               # 最大RANSAC试验次数
    min_inlier_ratio: float = 0.4       # 最小内点比例
```

### 3.2 权重配置
```python
# 解剖学权重参数
ANATOMICAL_WEIGHT_PARAMS = {
    "pupil_center": 0.991,      # 瞳孔中心权重
    "iris_boundary": 0.839,     # 虹膜边界权重
    "eye_contour": 0.607        # 眼睛轮廓权重
}

# 几何权重参数
GEOMETRIC_WEIGHT_PARAMS = {
    "pupil_center": {"initial_sigma": 0.2},      # 瞳孔中心标准差
    "iris_boundary": {"initial_sigma": 0.85},    # 虹膜边界标准差
    "eye_contour": {"initial_sigma": 1.5}        # 眼睛轮廓标准差
}

# 全局权重配置
GLOBAL_WEIGHT_CONFIG = {
    "min_weight": 0.1,          # 最小权重保护
    "normalization": True        # 权重归一化
}
```

## 四、算法流程

### 4.1 主拟合流程
```python
def center_fitter(self, key_coordinates: SingleEyeKeyCoordinates, 
                 trials_times: int, config: Optional[FittingConfig] = None) -> Tuple[EllipsoidParams, np.ndarray, FittingResult]:
    """
    主拟合接口
    
    流程：
    1. 更新配置
    2. 判断是否超过最大试验次数
    3. 执行拟合（中心拟合或完整拟合）
    4. 返回结果
    """
```

### 4.2 完整拟合流程
```python
def _fit_ellipsoid(self, key_coordinates: SingleEyeKeyCoordinates, center_only: bool = False) -> Tuple[EllipsoidParams, np.ndarray, FittingResult]:
    """
    椭球体拟合主流程
    
    步骤：
    1. 数据预处理和权重计算
    2. RANSAC采样和几何判定
    3. Levenberg-Marquardt精化优化
    4. 结果验证和输出
    """
```

### 4.3 单次RANSAC迭代
```python
def _run_single_ransac_iteration(self, points: List[np.ndarray], structure_types: List[str]) -> Tuple[np.ndarray, int, float]:
    """
    单次RANSAC迭代
    
    步骤：
    1. 随机采样点集
    2. 估计初始参数
    3. LM优化
    4. 计算内点比例
    5. 返回最佳参数
    """
```

## 五、性能优化

### 5.1 数值稳定性
- **步长限制**：限制参数更新步长，防止数值不稳定
- **阻尼调整**：自适应LM阻尼因子调整
- **奇异矩阵处理**：使用伪逆求解奇异矩阵问题

### 5.2 收敛控制
- **双重收敛判据**：参数收敛 + 残差收敛
- **最大迭代限制**：防止无限循环
- **残差回退**：残差增加时自动回退

### 5.3 权重保护
- **权重下限**：防止权重过低导致数值不稳定
- **权重归一化**：保持权重系统平衡
- **动态调整**：根据拟合质量动态调整权重策略

## 六、质量评估

### 6.1 拟合质量指标
```python
def _calculate_quality_metrics(self) -> QualityMetrics:
    """
    计算拟合质量指标
    
    指标包括：
    1. 内点比例：ρ = inlier_count / total_points
    2. 参数稳定性：参数变化的平滑程度
    3. 收敛速度：单位迭代次数的残差改善程度
    4. 最终残差：拟合精度
    """
```

### 6.2 质量等级
- **优秀**：内点比例 ≥ 90%，残差 < 0.001
- **良好**：内点比例 ≥ 80%，残差 < 0.005  
- **一般**：内点比例 ≥ 70%，残差 < 0.01
- **较差**：内点比例 < 70% 或残差 ≥ 0.01

## 七、错误处理

### 7.1 异常情况处理
- **点数不足**：虹膜点数 < 3时发出警告
- **置信度过低**：所有点置信度都低于阈值时返回默认值
- **收敛失败**：达到最大迭代次数仍未收敛时返回当前最佳结果

### 7.2 默认值策略
```python
def _get_default_params(self) -> EllipsoidParams:
    """返回默认参数"""
    return EllipsoidParams(
        center=np.array([0.0, 0.0, 0.6]),  # 默认Z坐标0.6m
        axes=np.array([EYEBALL_RADIUS, EYEBALL_RADIUS, EYEBALL_RADIUS]),
        rotation=np.eye(3)
    )
```

## 八、待办优化项目

### 8.1 核心问题修复：约束拟合算法（最高优先级）

#### 8.1.1 问题描述
当前算法存在严重错误：**眼球中心位置错误，所有关键点都在眼球内部**，这完全违背解剖学原理。

**正确的解剖关系**：
- 瞳孔中心：在眼球前方，距离眼球表面约0.1-0.3mm
- 虹膜边界：围绕眼球，距离眼球表面约0.5-1.2mm  
- 眼睛轮廓：在眼球外部，距离眼球表面约1-2mm
- 半径：应该在 10-14mm范围内

#### 8.1.2 解决方案：约束拟合算法

**核心思想**：强制约束所有关键点都在拟合球体外部，通过几何约束确保眼球位置正确。

**数学约束**：对于每个关键点 $p_i$，必须满足 $||p_i - c|| \geq r$

**算法流程**：
1. **约束初始估计**：瞳孔Z + 8mm偏移作为球心Z，虹膜质心作为XY
2. **约束目标函数**：点在球内时添加大惩罚项
3. **约束雅可比矩阵**：包含约束惩罚的导数项
4. **约束验证**：每次迭代后验证所有点都在球外

#### 8.1.3 关键接口设计

```python
def _compute_constrained_residuals(self, params: np.ndarray, points: List[np.ndarray], 
                                  structure_types: List[str], center_only: bool = False) -> np.ndarray:
    """
    计算带约束的残差向量
    约束条件：所有关键点必须在眼球外部
    惩罚函数：如果点在球内，添加大的惩罚项
    """

def _compute_constrained_jacobian(self, params: np.ndarray, points: List[np.ndarray], 
                                 structure_types: List[str], center_only: bool = False) -> np.ndarray:
    """
    计算带约束的雅可比矩阵
    包含约束惩罚的导数项
    """

def _estimate_constrained_initial_params(self, points: List[np.ndarray], 
                                       structure_types: List[str], center_only: bool = False) -> np.ndarray:
    """
    约束初始参数估计
    策略：瞳孔Z + 8mm偏移，虹膜质心XY，确保所有点在球外
    """

def _levenberg_marquardt_constrained_optimization(self, initial_params: np.ndarray, 
                                                points: List[np.ndarray], 
                                                structure_types: List[str], 
                                                center_only: bool = False) -> Tuple[np.ndarray, bool]:
    """
    约束LM优化算法
    确保每次迭代后所有点都在球外
    """
```

#### 8.1.4 实施步骤
1. **第1步**：实现约束残差和雅可比矩阵计算
2. **第2步**：修改初始参数估计，确保解剖学正确
3. **第3步**：集成到LM优化流程中
4. **第4步**：添加约束验证和调试输出


### 8.2 测试验证重点

#### 8.2.1 约束验证测试
1. **解剖关系验证**：确保所有关键点都在眼球外部
2. **距离合理性**：验证点到眼球表面的距离符合解剖学范围
3. **收敛稳定性**：约束条件下的算法收敛性测试

#### 8.2.2 性能测试
1. **精度提升**：对比约束前后的拟合精度
2. **鲁棒性增强**：异常值和噪声情况下的表现
3. **计算效率**：约束算法的计算复杂度分析