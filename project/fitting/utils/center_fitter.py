from typing import Dict, List, Optional, Tuple, NamedTuple
import numpy as np
import logging
from project.fitting.config import (
    EllipsoidParams,
    BaseFittingConfig,
    FITTING_STRATEGIES,
    DEFAULT_FITTING_STRATEGY,
    ANATOMICAL_CONSTRAINTS,
    MEDIAPIPE_QUALITY_THRESHOLDS,
    FittingStatus,
    FittingResult,
    ConvergenceStep,
    SingleEyeKeyCoordinates
)


class SampleData(NamedTuple):
    """采样数据结构 - 统一管理点、类型和visibility"""
    points: np.ndarray
    types: List[str]
    visibility: np.ndarray
    method: str


class CenterFitter:
    """
    眼球中心拟合器 - 单例类
    支持约束拟合、策略选择和4D坐标输入（包含visibility）
    """
    
    # 统一的点类型常量
    POINT_TYPES = {
        "pupil_center": "pupil_center",      # 瞳孔中心
        "iris_boundaries": "iris_boundaries", # 虹膜边界（复数）
        "eye_contours": "eye_contours"       # 眼睛轮廓（复数）
    }
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            logging.debug("🆕 创建新的CenterFitter单例实例")
        else:
            logging.debug("🔄 返回已存在的CenterFitter单例实例")
        return cls._instance
    
    def __init__(self, sphere: bool = True, strategy: Optional[BaseFittingConfig] = None):
        """
        初始化CenterFitter
        
        Args:
            sphere: 是否使用球体拟合（默认True）
            strategy: 拟合策略配置（默认使用DEFAULT_FITTING_STRATEGY）
        """
        # 单例初始化检查
        if hasattr(self, '_initialized'):
            # 如果已经初始化，但传入了新的strategy，则更新
            if strategy is not None:
                self._strategy = strategy
                logging.debug(f"🔄 更新策略配置: {strategy.strategy_name}")
            return
        self._initialized = True
        
        # 模型类型
        self._sphere = sphere
        
        # 策略配置
        self._strategy = strategy or DEFAULT_FITTING_STRATEGY
        
        # 初始化椭球参数
        self._ellipsoid_params = EllipsoidParams()
        if sphere:
            self._r = ANATOMICAL_CONSTRAINTS["EYEBALL_RADIUS_DEFAULT"]
            self._ellipsoid_params.axes = np.array([self._r, self._r, self._r])
            self._ellipsoid_params.rotation = np.eye(3)
            self._ellipsoid_params.center = np.array([0.0, 0.0, 0.0])
        
        # 生理偏差中值 - 用于残差计算
        self._physiological_bias_median = ANATOMICAL_CONSTRAINTS["PHYSIOLOGICAL_BIAS_MEDIAN"]
        
        # 数据预处理结果（全局维护）
        self._coords_3d: Dict[str, np.ndarray] = {}
        self._visibility_scores: Dict[str, np.ndarray] = {}
        self._total_points: int = 0
        self._avg_visibility: float = 0.0
        
        # 采样结果（统一管理）
        self._sample_data: Optional[SampleData] = None
        
        # 状态管理
        self._current_weights = None
        self._weights_calculated = False
        self._current_fit_center = None
        self._current_fit_radius = None
        
        # 拟合状态
        self._fitting_status = FittingStatus(
            is_running=False,
            current_iteration=0,
            max_iterations=self._strategy.lm_config.max_iterations,
            current_residual=float('inf'),
            converged=False
        )
        self._convergence_history: List[ConvergenceStep] = []
        
        # 策略函数映射表
        self._strategy_functions = {
            "constrained_lm": self._constrained_lm_fitting,
            "constrained_ransac_lm": self._constrained_ransac_lm_fitting,
            "ellipsoid_fallback": self._fit_ellipsoid,
            "fail": self._handle_failure
        }
        
        logging.debug("CenterFitter单例初始化完成")
    
    # ==================== 主要接口函数 ====================
    
    def center_fitter(self, key_coordinates: SingleEyeKeyCoordinates, 
                     trials_times: int, 
                     strategy: Optional[str] = None) -> Tuple[EllipsoidParams, np.ndarray, FittingResult]:
        """
        主拟合接口 - 保持对外接口不变，支持4D坐标输入
        
        Args:
            key_coordinates: 关键点坐标，支持3D或4D格式
                           3D: {"pupil_center": np.array([[x,y,z], ...]), ...}
                           4D: {"pupil_center": np.array([[x,y,z,v], ...]), ...} (v为visibility)
            trials_times: 实验次数
            strategy: 拟合策略名称，None表示自动选择
            
        Returns:
            Tuple[EllipsoidParams, np.ndarray, FittingResult]: 拟合结果
        """
        logging.info(f"🔍 开始眼球拟合: trials_times={trials_times}, strategy={strategy}")
        
        # 调试信息：检查对象状态
        logging.debug(f"🔍 对象状态检查: _initialized={hasattr(self, '_initialized')}, _strategy_functions={hasattr(self, '_strategy_functions')}")
        if hasattr(self, '_strategy_functions'):
            logging.debug(f"🔍 策略函数映射表: {list(self._strategy_functions.keys())}")
        
        # 数据预处理和全局维护
        self._preprocess_data(key_coordinates)
        
        # 自动选择策略（如果未指定）
        if strategy is None:
            strategy = self._assess_data_quality()
            logging.info(f"📊 自动选择拟合策略: {strategy}")
        
        # 直接调用对应的策略函数
        if not hasattr(self, '_strategy_functions'):
            logging.error(f"❌ 缺少策略函数映射表，重新初始化")
            self._strategy_functions = {
                "constrained_lm": self._constrained_lm_fitting,
                "constrained_ransac_lm": self._constrained_ransac_lm_fitting,
                "ellipsoid_fallback": self._fit_ellipsoid,
                "fail": self._handle_failure
            }
        
        if strategy in self._strategy_functions:
            logging.info(f"🚀 执行拟合策略: {strategy}")
            return self._strategy_functions[strategy](trials_times)
        else:
            logging.error(f"❌ 未知拟合策略: {strategy}")
            return self._handle_failure(f"未知策略: {strategy}")
    
    # ==================== 数据预处理和全局维护 ====================
    
    def _preprocess_data(self, key_coordinates: SingleEyeKeyCoordinates) -> None:
        """
        数据预处理：提取3D坐标和visibility，进行全局维护
        支持四维向量：[x, y, z, visibility]，其中visibility是MediaPipe的可见性置信度
        
        Args:
            key_coordinates: 关键点坐标（3D或4D）
        """
        # 清空之前的数据
        self._coords_3d.clear()
        self._visibility_scores.clear()
        
        total_visibility = 0.0
        total_points = 0
        
        for key, points in key_coordinates.items():
            if points is None or len(points) == 0:
                continue
            
            try:
                # 统一处理：将各种格式转换为标准格式
                points_array = np.array(points)
                processed_points, processed_visibility = self._process_numpy_points(points_array)

                if processed_points is not None:
                    self._coords_3d[key] = processed_points
                    self._visibility_scores[key] = processed_visibility
                    total_visibility += np.sum(processed_visibility)
                    total_points += len(processed_points)
                    
            except Exception as e:
                logging.warning(f"处理{key}失败: {e}")
                continue
        
        # 全局维护
        self._total_points = total_points
        self._avg_visibility = total_visibility / total_points if total_points > 0 else 0.0
        
        logging.info(f"📋 数据预处理完成: 总点数={self._total_points}, 平均visibility={self._avg_visibility:.3f}")
        # 输出各类型点的统计信息
        for key, coords in self._coords_3d.items():
            if coords is not None and len(coords) > 0:
                avg_vis = np.mean(self._visibility_scores[key]) if key in self._visibility_scores else 0.0
                logging.info(f"  📍 {key}: {len(coords)}点, 平均visibility={avg_vis:.3f}")
    
    def _process_numpy_points(self, points: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        处理numpy数组格式的点，统一转换为3D坐标和visibility
        
        Args:
            points: 输入点数组
            
        Returns:
            (3D坐标数组, visibility数组) 或 (None, None) 如果处理失败
        """
        if len(points.shape) == 1:
            # 单个点
            if points.shape[0] == 4:
                # 4D点：[x, y, z, visibility]
                coords_3d = points[:3].reshape(1, 3)
                visibility = np.array([points[3]])
            elif points.shape[0] == 3:
                # 3D点：[x, y, z]，补充visibility=1.0
                coords_3d = points.reshape(1, 3)
                visibility = np.array([1.0])
            else:
                return None, None
                
        elif len(points.shape) == 2:
            # 多个点
            if points.shape[1] == 4:
                # 4D点：[[x, y, z, visibility], ...]
                coords_3d = points[:, :3]
                visibility = points[:, 3]
            elif points.shape[1] == 3:
                # 3D点：[[x, y, z], ...]，补充visibility=1.0
                coords_3d = points
                visibility = np.ones(len(points))
            else:
                return None, None
        else:
            return None, None
        
        return coords_3d, visibility
    
    # ==================== 策略选择接口 ====================
    
    def _assess_data_quality(self) -> str:
        """
        基于预处理后的数据评估质量，自动选择拟合策略
        
        Returns:
            str: 策略名称 ("constrained_lm", "constrained_ransac_lm", "fail")
        """
        # 基于MEDIAPIPE_QUALITY_THRESHOLDS判断
        if (self._total_points >= MEDIAPIPE_QUALITY_THRESHOLDS["HIGH"]["min_points"] and 
            self._avg_visibility >= MEDIAPIPE_QUALITY_THRESHOLDS["HIGH"]["confidence"]):
            return "constrained_lm"
        elif (self._total_points >= MEDIAPIPE_QUALITY_THRESHOLDS["MEDIUM"]["min_points"] and 
              self._avg_visibility >= MEDIAPIPE_QUALITY_THRESHOLDS["MEDIUM"]["confidence"]):
            return "constrained_ransac_lm"
        else:
            return "fail"
    
    # ==================== 约束拟合接口 ====================
    
    def _constrained_lm_fitting(self, trials_times: int) -> Tuple[EllipsoidParams, np.ndarray, FittingResult]:
        """
        约束LM拟合 - 高质量数据使用全部采样+直接LM优化
        
        Args:
            trials_times: 实验次数
            
        Returns:
            Tuple[EllipsoidParams, np.ndarray, FittingResult]: 拟合结果
        """
        # 使用全部采样
        self._full_sampling()
        logging.info(f"📊 使用全部采样策略: {len(self._sample_data.types) if self._sample_data else 0}个点")
        
        # 约束LM优化
        logging.info("🔄 开始约束LM优化...")
        return self._constrained_optimization()
    
    def _constrained_ransac_lm_fitting(self, trials_times: int) -> Tuple[EllipsoidParams, np.ndarray, FittingResult]:
        """
        约束RANSAC+LM拟合 - 中质量数据使用RANSAC采样+约束LM优化
        
        Args:
            trials_times: 实验次数
            
        Returns:
            Tuple[EllipsoidParams, np.ndarray, FittingResult]: 拟合结果
        """
        class RANSACOptimizer:
            """RANSAC优化器内部类"""
            
            def __init__(self, outer_instance):
                self.outer = outer_instance
                self.best_center = None
                self.best_radius = None
                self.best_inliers = None
                self.best_inlier_count = 0
                self._config = outer_instance._strategy.ransac_config
                
            def optimize(self) -> Tuple[Optional[np.ndarray], Optional[float], Optional[np.ndarray]]:
                """执行RANSAC优化"""
                logging.info(f"🔄 RANSAC优化开始: 最大迭代次数={self._config.max_iterations}")
                
                for iteration in range(self._config.max_iterations):
                    # 1. 随机采样
                    sampled_points, point_types = self._random_sampling()
                    
                    # 2. 单次RANSAC拟合（使用约束优化）
                    center, radius, inliers = self._single_ransac_fit(sampled_points, point_types)
                    
                    # 3. 更新最佳结果
                    if inliers is not None and len(inliers) > self.best_inlier_count:
                        self.best_center = center
                        self.best_radius = radius
                        self.best_inliers = inliers
                        self.best_inlier_count = len(inliers)
                        
                        # 检查是否达到足够好的结果
                        inlier_ratio = self.best_inlier_count / len(sampled_points)
                        
                        # 使用配置的内点比例阈值
                        if inlier_ratio >= self._config.min_inlier_ratio:
                            logging.info(f"✅ RANSAC提前收敛: 内点比例={inlier_ratio:.1%}")
                            break
                        
                        # 使用新的内点比例阈值进行快速收敛判断
                        if inlier_ratio >= self._config.inlier_ratio_threshold:
                            logging.info(f"🚀 RANSAC快速收敛: 内点比例={inlier_ratio:.1%} >= {self._config.inlier_ratio_threshold:.1%}")
                            break
                
                logging.info(f"🎯 RANSAC优化完成: 最佳内点数={self.best_inlier_count}, 总采样点数={len(sampled_points) if 'sampled_points' in locals() else 0}")
                return self.best_center, self.best_radius, self.best_inliers
            
            def _random_sampling(self) -> Tuple[np.ndarray, List[str]]:
                """随机采样策略"""
                # 采样配置：类型 -> (目标数量, 最小要求数量)
                sampling_config = {
                    "pupil_center": (1, 1),
                    "iris_boundaries": (self._config.min_iris_points, self._config.min_iris_points),
                    "eye_contours": (np.random.randint(self._config.min_contour_points, self._config.max_contour_points + 1), self._config.min_contour_points)
                }
                
                # 统一采样并合并结果
                all_samples = [
                    self._sample_point_type(point_type, target, required)
                    for point_type, (target, required) in sampling_config.items()
                ]
                
                # 展平结果
                sampled_points = [p for points, _, _ in all_samples for p in points]
                point_types = [t for _, types, _ in all_samples for t in types]
                sampled_visibility = [v for _, _, visibility in all_samples for v in visibility]
                
                if sampled_points:
                    self.outer._update_sample_data(
                        points=np.vstack(sampled_points),
                        types=point_types,
                        visibility=np.array(sampled_visibility),
                        method="ransac_temporary"
                    )
                    return self.outer._sample_data.points, self.outer._sample_data.types
                
                return np.array([]), []
            
            def _sample_point_type(self, point_type: str, target_count: int, required_count: int) -> Tuple[List[np.ndarray], List[str], List[float]]:
                """采样指定类型的点"""
                # 检查是否有足够的点
                if point_type in self.outer._coords_3d and len(self.outer._coords_3d[point_type]) >= target_count:
                    # 直接采样
                    indices = np.random.choice(len(self.outer._coords_3d[point_type]), target_count, replace=False)
                    points = [self.outer._coords_3d[point_type][indices]]
                    types = [point_type] * target_count
                    visibility = (self.outer._visibility_scores.get(point_type, np.ones(len(self.outer._coords_3d[point_type])))[indices]).tolist()
                    return points, types, visibility
                
                # 点数不足，从其他点补充
                return self._supplement_sampling(required_count, point_type)
            
            def _supplement_sampling(self, required_count: int, target_type: str) -> Tuple[List[np.ndarray], List[str], List[float]]:
                """补充采样：当某种类型点数不足时，从其他点中随机补充"""
                # 收集所有可用点
                all_points = []
                all_visibility = []
                
                for key, points in self.outer._coords_3d.items():
                    if points is not None and len(points) > 0:
                        all_points.extend(points)
                        all_visibility.extend(self.outer._visibility_scores.get(key, [1.0] * len(points)))
                
                if len(all_points) >= required_count:
                    # 随机选择补充点
                    indices = np.random.choice(len(all_points), required_count, replace=False)
                    return (
                        [all_points[i] for i in indices],
                        [target_type] * required_count,
                        [all_visibility[i] for i in indices]
                    )
                
                return [], [], []
            
            def _single_ransac_fit(self, sampled_points: np.ndarray, point_types: List[str]) -> Tuple[np.ndarray, float, np.ndarray]:
                """单次RANSAC拟合 - 使用约束优化函数"""
                if len(sampled_points) < self._config.min_samples:
                    return None, None, None
                
                # 临时更新采样结果用于约束优化
                original_sample_data = self.outer._sample_data
                
                # 更新为当前RANSAC采样结果
                self.outer._update_sample_data(
                    points=sampled_points,
                    types=point_types,
                    visibility=np.ones(len(sampled_points)), # 临时使用全1
                    method="ransac_temporary"
                )
                
                try:
                    # 使用约束优化函数进行拟合
                    ellipsoid_params, center, result = self.outer._constrained_optimization()
                    
                    if center is not None:
                        # 使用新的内点计算函数
                        inlier_indices = self._calculate_inliers(
                            sampled_points, 
                            center, 
                            result.ellipsoid_params.axes[0],  # 使用拟合的半径
                            self._config.threshold
                        )
                        
                        return center, result.ellipsoid_params.axes[0], inlier_indices
                    else:
                        return None, None, None
                        
                finally:
                    # 恢复原始采样结果
                    self.outer._sample_data = original_sample_data
            
            def _calculate_inliers(self, points: np.ndarray, center: np.ndarray, radius: float, threshold: float) -> np.ndarray:
                """计算内点索引"""
                # 1. 计算每个点到球心的距离
                distances = np.linalg.norm(points - center, axis=1)
                
                # 2. 计算残差（点到球面的距离）
                residuals = np.abs(distances - radius)
                
                # 3. 找出残差小于阈值的点索引
                inlier_indices = np.where(residuals < threshold)[0]
                
                return inlier_indices
        
        # 创建RANSAC优化器并执行
        logging.info("🎯 开始RANSAC优化...")
        ransac_optimizer = RANSACOptimizer(self)
        best_center, best_radius, best_inliers = ransac_optimizer.optimize()
        
        # 使用最佳RANSAC结果进行约束LM优化
        if best_center is not None:
            logging.info(f"✅ RANSAC优化成功: 最佳内点数={len(best_inliers) if best_inliers is not None else 0}")
            # 更新采样结果为最佳内点
            self._update_sampling_result(best_inliers)
            logging.info("🔄 使用RANSAC内点进行约束LM优化...")
            return self._constrained_optimization()
        else:
            logging.error("❌ RANSAC优化失败")
            return self._handle_failure("RANSAC拟合失败")
    

    # ==================== 采样策略接口 ====================
    
    def _full_sampling(self) -> None:
        """全部采样策略 - 使用所有可用点，结果存储到全局变量"""
        all_points = []
        point_types = []
        sampled_points_visibility = []
        
        for key, points in self._coords_3d.items():
            if points is not None and len(points) > 0:
                all_points.append(points)
                point_types.extend([key] * len(points))
                sampled_points_visibility.extend(self._visibility_scores[key])
        
        if all_points:
            self._update_sample_data(
                points=np.vstack(all_points),
                types=point_types,
                visibility=np.array(sampled_points_visibility),
                method="full"
            )
        else:
            self._update_sample_data(
                points=np.array([]),
                types=[],
                visibility=np.array([]),
                method="full"
            )
        
        logging.info(f"📊 全部采样完成: 采样点数={self._get_sample_count()}")
    
    def _update_sampling_result(self, inlier_indices: np.ndarray) -> None:
        """更新采样结果为最佳内点"""
        if len(inlier_indices) > 0 and self._has_sample_data():
            # 使用新的辅助方法过滤采样数据
            self._filter_sample_data(inlier_indices)
            logging.debug(f"采样结果已更新为RANSAC内点: 内点数={len(inlier_indices)}")
        else:
            logging.warning("无法更新采样结果：内点索引为空或采样点为空")
    
    # ==================== 约束优化接口 ====================
    
    def _constrained_optimization(self) -> Tuple[EllipsoidParams, np.ndarray, FittingResult]:
        """
        约束优化 - 按照文档公式实现，去除先验权重
        
        目标函数：Φ(θ) = F(θ) + (1/2)λ_depth * h(θ)²
        其中：
        - F(θ) = (1/2)∑w_i(||p_i - c|| - (r + δ_i))² （由weight_manager处理）
        - h(θ) = max{0, ε + z_max - c_z} （深度约束）
        
        Returns:
            Tuple[EllipsoidParams, np.ndarray, FittingResult]: 优化结果
        """
        
        # 调试信息：显示约束优化开始
        logging.debug(f"🔍 约束优化开始:")
        logging.debug(f"   目标函数: Φ(θ) = F(θ) + (1/2)λ_depth * h(θ)²")
        logging.debug(f"   数据项: F(θ) = (1/2)∑w_i(||p_i - c|| - (r + δ_i))²")
        logging.debug(f"   约束项: h(θ) = max{{0, ε + z_max - c_z}}")
        
        class ConstrainedLMOptimizer:
            """约束LM优化器内部类"""
            
            def __init__(self, outer_instance, center: np.ndarray, radius: float):
                self.outer = outer_instance
                self.center = center.copy()
                self.radius = radius
                self.params = np.concatenate([center, [radius]])  # θ = [c_x, c_y, c_z, r]
                
                # 调试信息：显示初始化参数
                logging.debug(f"🔍 约束LM优化器初始化:")
                logging.debug(f"   初始球心: {self.center}")
                logging.debug(f"   初始半径: {self.radius:.6f}")
                logging.debug(f"   参数向量 θ: {self.params.reshape(-1, 1)}")  # 列向量形式
                
                # 优化状态维护
                self.iteration_count = 0
                self.is_converged = False
                self._current_residual_norm = float('inf')
                self.inlier_ratio = 1.0
                self.quality_score = 1.0
                self.warnings = []
                
                # 获取配置
                self._config = outer_instance._strategy.lm_config
                self._constraint = outer_instance._strategy.constraint_config
                
                # 获取样本长度
                self._sample_length = outer_instance._get_sample_count()
                
                # 调试信息：显示配置信息
                logging.debug(f"   样本数量: {self._sample_length}")
                logging.debug(f"   LM配置: {self._config}")
                logging.debug(f"   约束配置: {self._constraint}")
                
                # 初始化权重管理器（简化版本）
                from project.fitting.utils.weight_manager import DEFAULT_WEIGHT_MANAGER
                self._weight_manager = DEFAULT_WEIGHT_MANAGER
                
            def optimize(self) -> Tuple[np.ndarray, float]:
                """执行约束LM优化"""
                logging.info(f"🔄 约束LM优化开始: 最大迭代次数={self._config.max_iterations}")
                
                # 调试信息：显示优化开始状态
                logging.debug(f"🔍 优化开始状态:")
                logging.debug(f"   当前参数: {self.params.reshape(-1, 1)}")  # 列向量形式
                logging.debug(f"   当前球心: {self.center}")
                logging.debug(f"   当前半径: {self.radius:.6f}")
                logging.debug(f"   最大迭代次数: {self._config.max_iterations}")
                
                def show_delta(delta: np.ndarray, iteration: int) -> None:
                    """显示delta"""
                    def analyze_delta(delta: np.ndarray) -> str:
                        delta_x = delta[0]
                        delta_y = delta[1]
                        delta_z = delta[2]
                        delta_r = delta[3]
                        
                        info = "向"  # ✅ 初始化info变量
                        
                        if delta_x > 0:
                            info += "左"
                        elif delta_x < 0:
                            info += "右"
                        info += f"{abs(delta_x*1000):.6f}mm，向"
                        
                        if delta_y > 0:
                            info += "下"
                        elif delta_y < 0:
                            info += "上"
                        info += f"{abs(delta_y*1000):.6f}mm，向"
                        
                        if delta_z > 0:
                            info += "后"
                        elif delta_z < 0:
                            info += "前"
                        info += f"{abs(delta_z*1000):.6f}mm，半径变"
                        
                        if delta_r > 0:
                            info += "大"
                        elif delta_r < 0:
                            info += "小"
                        info += f"{abs(delta_r*1000):.6f}mm"
                        
                        return info
                    
                    logging.info("-----------------------------------------------------------------------------------")
                    logging.info(f"🔍 迭代第{iteration}次，球心{analyze_delta(delta)}")
                    
                    # 显示变化前后的对比
                    if hasattr(self.outer, '_last_params') and self.outer._last_params is not None:
                        last_center = self.outer._last_params[:3]
                        last_radius = self.outer._last_params[3]

                        logging.info(f"📊 变化前: 中心({self.center[0]:.3f}, {self.center[1]:.3f}, {self.center[2]:.3f}), 半径{self.radius*1000:.2f}mm")
                        logging.info(f"📊 变化后: 中心({(last_center[0] + delta[0]):.3f}, {(last_center[1] + delta[1]):.3f}, {(last_center[2] + delta[2]):.3f}), 半径{(last_radius + delta[3])*1000:.2f}mm")
                        
                        # 计算变化量
                        center_change = np.linalg.norm(delta[:2]) * 1000  # 毫米
                        radius_change = abs(delta[3]) * 1000  # 毫米
                        logging.info(f"📈 变化量: 中心移动{center_change:.2f}mm, 半径变化{radius_change:.2f}mm")
                    else:
                        logging.info(f"📊 当前状态: 中心({self.center[0]:.3f}, {self.center[1]:.3f}, {self.center[2]:.3f}), 半径{self.radius*1000:.2f}mm")
                    
                    # 获取瞳孔中心数据
                    pupil_center = self.outer._coords_3d.get("pupil_center")  # ✅ 使用正确的键名
                    eyeball_center = self.center
                    
                    # 计算眼球中心到瞳孔中心的方向
                    if pupil_center is not None and len(pupil_center) > 0:
                        pupil_center_avg = np.mean(pupil_center, axis=0) if len(pupil_center.shape) > 1 else pupil_center
                        direction = eyeball_center - pupil_center_avg
                        
                        # 分析方向，就像分析delta一样
                        def analyze_direction(direction: np.ndarray) -> str:
                            """分析方向向量，返回中文描述"""
                            dx = direction[0]
                            dy = direction[1]
                            dz = direction[2]
                            
                            info = "眼球中心在瞳孔中心的"
                            
                            if dx > 0:
                                info += "右"
                            elif dx < 0:
                                info += "左"
                            info += f"{abs(dx*1000):.2f}mm，"
                            
                            if dy > 0:
                                info += "下"
                            elif dy < 0:
                                info += "上"
                            info += f"{abs(dy*1000):.2f}mm，"
                            
                            if dz > 0:
                                info += "后"
                            elif dz < 0:
                                info += "前"
                            info += f"{abs(dz*1000):.2f}mm"
                            
                            return info
                        
                        direction_info = analyze_direction(direction)
                        logging.info(f"📍 眼球中心: {eyeball_center}")
                        logging.info(f"📏 眼球半径: {self.radius*1000:.2f}mm")
                        logging.info(f"👁️ 瞳孔中心: {pupil_center_avg}")
                        logging.info(f"📐 {direction_info}")
                        logging.info(f"📏 距离: {np.linalg.norm(direction)*1000:.2f}mm")
                    else:
                        logging.info("⚠️ 瞳孔中心数据不可用")
                    
                    logging.info("-----------------------------------------------------------------------------------")
                
                # 将show_delta函数绑定到self，使其可以在类内部访问
                self.show_delta = show_delta
                
                for iteration in range(self._config.max_iterations):
                    self.iteration_count = iteration
                    
                    # 调试信息：显示迭代开始
                    logging.debug(f"🔍 迭代 {iteration + 1}/{self._config.max_iterations}:")
                    
                    # 2.1 计算残差
                    data_residuals = self._calculate_fitting_residual()      # f(θ)
                    constraint_residuals = self._calculate_constraint_penalty()  # g(θ)
                    radius_residuals = self._calculate_radius_residual()  # r(θ)
                    
                    # 调试信息：显示残差计算结果
                    logging.debug(f"   数据残差 f(θ): {data_residuals.reshape(-1, 1)}")  # 列向量形式
                    logging.debug(f"   约束残差 g(θ): {constraint_residuals.reshape(-1, 1)}")  # 列向量形式
                    logging.debug(f"   半径残差 r(θ): {radius_residuals.reshape(-1, 1)}")  # 列向量形式
                    # 2.2 更新权重管理器状态
                    self._update_weight_manager_state(data_residuals)
                    
                    # 2.3 求解LM更新方程
                    delta = self._solve_lm_equation(data_residuals, constraint_residuals, radius_residuals)
                    
                    if delta is None:
                        # 求解失败，增加阻尼因子
                        if not hasattr(self, '_lambda_lm'):
                            self._lambda_lm = self._config.lambda_lm
                        self._lambda_lm *= self._config.lambda_factor
                        # 限制阻尼因子范围
                        self._lambda_lm = np.clip(
                            self._lambda_lm,
                            self._config.min_lambda,
                            self._config.max_lambda
                        )
                        logging.warning(f"⚠️ LM迭代{iteration}求解失败，增加阻尼因子到{self._lambda_lm:.2e}")
                        
                        # 调试信息：显示阻尼因子调整
                        logging.debug(f"   阻尼因子调整: λ_LM = {self._lambda_lm:.2e}")
                        continue
                    
                    # 2.4 参数更新
                    params_new = self.params + delta
                    
                    # 调试信息：显示参数更新
                    logging.debug(f"   参数更新:")
                    logging.debug(f"     旧参数: {self.params.reshape(-1, 1)}")  # 列向量形式
                    logging.debug(f"     更新向量: {delta.reshape(-1, 1)}")  # 列向量形式
                    logging.debug(f"     新参数: {params_new.reshape(-1, 1)}")  # 列向量形式
                    
                    show_delta(delta,iteration)
                    
                    # 2.4 检查参数有效性且更新参数
                    valid = self._check_parameters_valid(params_new, data_residuals, constraint_residuals, radius_residuals)
                    

                    logging.debug(f"🔍 参数有效: {valid}")

                    
                    if valid:
                        self.params = params_new
                        self.center = self.params[:3]
                        self.radius = self.params[3]
                        
                        # 调试信息：显示参数更新成功
                        logging.debug(f"   参数更新成功:")
                        logging.debug(f"     新球心: {self.center}")
                        logging.debug(f"     新半径: {self.radius:.6f}")
                        
                        # 2.6 检查收敛性
                        if self._check_config_convergence():
                            self.is_converged = True
                            logging.info(f"✅ LM优化收敛: 迭代{iteration}次")
                            break
                    else:
                        # 参数无效，增加阻尼因子
                        if not hasattr(self, '_lambda_lm'):
                            self._lambda_lm = self._config.lambda_lm
                        self._lambda_lm *= self._config.lambda_factor
                        # 限制阻尼因子范围
                        self._lambda_lm = np.clip(
                            self._lambda_lm,
                            self._config.min_lambda,
                            self._config.max_lambda
                        )
                        self.warnings.append(f"迭代{iteration}: 参数无效，阻尼因子={self._lambda_lm:.2e}")
                        
                        # 调试信息：显示参数无效时的阻尼因子调整
                        logging.debug(f"   参数无效，阻尼因子调整: λ_LM = {self._lambda_lm:.2e}")
                
                # 调试信息：显示优化结束状态
                logging.debug(f"🔍 优化结束状态:")
                logging.debug(f"   最终参数: {self.params.reshape(-1, 1)}")  # 列向量形式
                logging.debug(f"   最终球心: {self.center}")
                logging.debug(f"   最终半径: {self.radius:.6f}")
                logging.debug(f"   是否收敛: {self.is_converged}")
                logging.debug(f"   迭代次数: {self.iteration_count}")
                
                # 更新最终状态
                self._update_final_state()
                if not self.is_converged:
                    logging.warning(f"⚠️ LM优化未收敛: 达到最大迭代次数{self._config.max_iterations}")
                    self._check_config_convergence()
                return self.center, self.radius
            
            def _calculate_fitting_residual(self) -> np.ndarray:
                """
                计算拟合残差向量 f(θ) - 点到球面的误差，含个体生理偏差修正
                
                残差定义：f_i(θ) = ||p_i - c|| - (r + δ_i)
                其中：
                - p_i: 第i个采样点坐标
                - c: 球心坐标
                - r: 球体半径
                - δ_i: 第i个点对应的生理偏差中值
                
                Returns:
                    np.ndarray: 残差向量 [f_1, f_2, ..., f_N]
                """
                if self._sample_length == 0:
                    logging.warning("⚠️ 采样点数量为0，无法计算残差")
                    return np.array([])

                # 1. 点到球心的欧氏距离
                points = self.outer._get_sample_points()
                distances = np.linalg.norm(points - self.center, axis=1)
                logging.debug(f"🔍 计算点到球心距离: 形状={distances.shape}")

                # 2. 生理偏差（点类型 → 个体修正值）
                point_types = self.outer._get_sample_types()
                physiological_bias = self.outer._get_physiological_bias_vector(point_types)
                logging.debug(f"获取生理偏差: 点类型={point_types}, 偏差值={physiological_bias}")

                # 3. 残差：距离 - (半径 + 生理偏差)
                # 注意：这里使用 (半径 + 生理偏差) 作为理论距离
                theoretical_distances = self.radius + physiological_bias
                residuals = distances - theoretical_distances

                # 保存当前残差范数
                self._current_residual_norm = np.linalg.norm(residuals)

                # 详细的调试信息
                logging.info("-----------------------------------------------------------------------------------")
                logging.debug(f"🔍 拟合残差计算详情:")
                logging.debug(f"   球心坐标: {self.center}")
                logging.debug(f"   球体半径: {self.radius:.6f}")
                logging.debug(f"   采样点数: {self._sample_length}")
                logging.debug(f"   点类型分布: {dict(zip(*np.unique(point_types, return_counts=True)))}")
                logging.debug(f"   理论距离向量: {theoretical_distances}")
                logging.debug(f"   实际距离向量: {distances}")
                logging.debug(f"   生理偏差向量: {physiological_bias}")
                logging.debug(f"   残差向量 f(θ): \n{residuals.reshape(-1, 1)}")
                logging.debug(f"   平均残差范数: {(self._current_residual_norm / self._sample_length):.6f}")
                logging.debug(f"   残差范数={self._current_residual_norm:.6f}")

                return residuals


            def _calculate_constraint_penalty(self) -> np.ndarray:
                """
                计算深度约束惩罚 h(θ) - 使用softplus软先验
                
                深度约束：h(θ) = τ log(1 + exp((ε + z_max - c_z)/τ))
                其中：
                - ε: 深度偏移量，避免数值抖动
                - z_max: 所有采样点的最大Z坐标
                - c_z: 球心的Z坐标
                - τ: softplus平滑参数，控制约束的软硬程度
                
                当 z_max > c_z - ε 时，约束被违反，产生惩罚
                使用softplus函数确保约束的连续性和可微性
                
                Returns:
                    np.ndarray: 深度约束惩罚值 [h(θ)]
                """
                if self._sample_length == 0:
                    logging.warning("⚠️ 采样点数量为0，无法计算深度约束")
                    return np.array([])

                # 获取所有采样点的Z坐标
                points = self.outer._get_sample_points()
                z_coords = points[:, 2]
                max_z = np.max(z_coords)
                
                # 获取约束参数
                depth_offset = self._constraint.depth_offset_epsilon
                tau = self._constraint.softplus_tau
                
                # 计算深度约束违反程度
                depth_violation_raw = depth_offset + max_z - self.center[2]
                
                # 使用softplus函数计算软约束惩罚
                # softplus(x) = τ log(1 + exp(x/τ))，当x>0时平滑增长
                u = depth_violation_raw / tau
                depth_violation = tau * np.log1p(np.exp(u))
                
                # 详细的调试信息
                logging.info("-----------------------------------------------------------------------------------")
                logging.debug(f"🔍 深度约束计算详情 (softplus):")
                logging.debug(f"   采样点Z坐标范围: [{np.min(z_coords):.6f}, {max_z:.6f}]")
                logging.debug(f"   球心Z坐标: {self.center[2]:.6f}")
                logging.debug(f"   深度偏移ε: {depth_offset:.6f}")
                logging.debug(f"   softplus参数τ: {tau:.6f}")
                logging.debug(f"   原始违反值: {depth_violation_raw:.6f}")
                logging.debug(f"   softplus输入 u: {u:.6f}")
                logging.debug(f"   约束惩罚 h(θ): {depth_violation:.6f}")
                
                # 约束状态分析
                if depth_violation_raw > 0:
                    logging.debug(f"   🔴 深度约束被违反: 球心过浅")
                else:
                    logging.debug(f"   🟢 深度约束满足: 球心深度足够")
                
                return np.array([depth_violation])


            def _calculate_radius_residual(self) -> np.ndarray:
                """
                计算半径残差项 r(θ)
                
                半径残差：r(θ) = r - r_prior
                
                
                Returns:
                    np.ndarray: 半径残差值 [r(θ)]
                """
                if self._sample_length == 0:
                    logging.warning("⚠️ 采样点数量为0，无法计算半径惩罚")
                    return np.array([])
                
                # 获取半径先验值
                radius_prior = self._constraint.radius_prior
                
                # 计算半径惩罚
                radius_residual = self.radius - radius_prior
                
                # 详细的调试信息
                logging.info("-----------------------------------------------------------------------------------")
                logging.debug(f"🔍 半径惩罚计算详情:")
                logging.debug(f"   当前半径: {self.radius:.6f}")
                logging.debug(f"   半径先验值: {radius_prior:.6f}")
                logging.debug(f"   半径残差 r(θ): {radius_residual:.6f}")
                
                return np.array([radius_residual])

            def _update_weight_manager_state(self, data_residuals: np.ndarray) -> None:
                """
                更新权重管理器状态 - 控制几何权重启用
                
                功能：
                1. 监控参数和残差的变化趋势
                2. 根据变化情况决定是否启用几何权重
                3. 计算并更新几何权重矩阵
                4. 维护优化历史状态
                
                Args:
                    data_residuals: 当前迭代的数据残差向量
                """
                # 计算参数和残差的变化量
                if hasattr(self.outer, '_last_params') and hasattr(self.outer, '_last_residuals'):
                    param_change = np.linalg.norm(self.params - self.outer._last_params)
                    residual_change = np.abs(self._current_residual_norm - np.linalg.norm(self.outer._last_residuals))
                    logging.debug(f"🔍 状态变化计算: 参数变化={param_change:.6f}, 残差变化={residual_change:.6f}")
                else:
                    param_change = 0.0
                    residual_change = 0.0
                    logging.debug("🔍 首次迭代，无历史状态可比较")

                # 详细的调试信息
                logging.info("-----------------------------------------------------------------------------------")
                logging.debug(f"🔍 权重管理器状态更新详情:")
                logging.debug(f"   当前迭代: {self.iteration_count}")
                logging.debug(f"   参数变化: {param_change:.6f} (阈值: {self._config.param_tolerance:.6f})")
                logging.debug(f"   残差变化: {residual_change:.6f} (阈值: {self._config.residual_tolerance:.6f})")
                logging.debug(f"   几何权重因子: {self._config.geometric_weight_factor}")

                # 判断是否启用几何权重
                # 当参数和残差变化都小于阈值时，启用几何权重以提高精度
                self._weight_manager.should_enable_geometric_weights(
                    param_change=param_change,
                    residual_change=residual_change,
                    param_tolerance=self._config.param_tolerance,
                    residual_tolerance=self._config.residual_tolerance,
                    geometric_weight_factor=self._config.geometric_weight_factor
                )

                # 保存当前状态到历史记录
                self.outer._last_params = self.params.copy()
                self.outer._last_residuals = data_residuals.copy()
                logging.debug(f"状态已保存: 参数={self.params}, 残差范数={self._current_residual_norm:.6f}")

                # 获取点类型和visibility信息
                point_types = self.outer._get_sample_types()
                if self.outer._has_sample_data():
                    visibility = self.outer._get_sample_visibility()
                    logging.debug(f"🔍 获取visibility信息: 形状={visibility.shape}, 范围=[{np.min(visibility):.3f}, {np.max(visibility):.3f}]")
                else:
                    logging.warning("⚠️ 未找到visibility信息，使用默认全1权重")
                    visibility = np.ones(len(data_residuals))

                # 设置权重管理器的置信度
                self._weight_manager.set_image_confidence(visibility)

                # 计算权重矩阵
                logging.debug(f"🔍 开始计算权重矩阵:")
                logging.debug(f"   点类型: {point_types}")
                logging.debug(f"   残差向量: 形状={data_residuals.shape}, 范数={np.linalg.norm(data_residuals):.6f}")
                logging.debug(f"   Visibility: 形状={visibility.shape}, 均值={np.mean(visibility):.3f}")
                
                weight_matrix = self._weight_manager.calculate_weight_matrix(
                    point_types=point_types,
                    residuals=data_residuals,
                    visibility=visibility
                )

                # 从权重矩阵中提取几何权重向量
                if weight_matrix.size > 0:
                    self._current_geometric_weights = np.diag(weight_matrix)
                    logging.debug(f"🔍 权重矩阵计算成功: 形状={weight_matrix.shape}")
                else:
                    self._current_geometric_weights = np.ones(len(data_residuals))
                    logging.warning("⚠️ 权重矩阵为空，使用默认权重")

                # 输出权重统计信息
                weights = self._current_geometric_weights
                logging.debug(f"🔍 几何权重统计:")
                logging.debug(f"   权重向量: {weights.reshape(-1, 1)}")
                logging.debug(f"   权重范围: [{np.min(weights):.6f}, {np.max(weights):.6f}]")
                logging.debug(f"   权重均值: {np.mean(weights):.6f}")
                logging.debug(f"   权重标准差: {np.std(weights):.6f}")
                
                # 权重分布分析
                high_weight_count = np.sum(weights > 0.8)
                low_weight_count = np.sum(weights < 0.2)
                logging.debug(f"   高权重点(>0.8): {high_weight_count}/{len(weights)}")
                logging.debug(f"   低权重点(<0.2): {low_weight_count}/{len(weights)}")

            def _solve_lm_equation(self, data_residuals: np.ndarray, constraint_residuals: np.ndarray, radius_residuals: np.ndarray) -> Optional[np.ndarray]:
                """
                求解LM更新方程：(J̃^T J̃ + λ_LM I)Δ = -J̃^T r̃
                
                目标函数：Φ(θ) = F(θ) + (1/2)λ_depth * h(θ)² + (1/2)λ_radius * r(θ)²
                其中：
                - F(θ): 数据项，加权残差平方和
                - h(θ): 深度约束项
                - r(θ): 半径先验项
                
                Args:
                    data_residuals: 数据残差向量 f(θ)
                    constraint_residuals: 约束残差向量 [h(θ)]
                    
                Returns:
                    Optional[np.ndarray]: 参数更新向量 Δθ，失败时返回None
                """
                
                def calculate_jacobian_f() -> np.ndarray:
                    """
                    计算数据残差雅可比矩阵 J_f
                    
                    J_f[i,j] = ∂f_i/∂θ_j，其中：
                    - f_i = ||p_i - c|| - (r + δ_i)
                    - θ = [c_x, c_y, c_z, r]
                    
                    Returns:
                        np.ndarray: 形状为 (N, 4) 的雅可比矩阵
                    """
                    if self._sample_length == 0:
                        logging.warning("⚠️ 采样点数量为0，无法计算雅可比矩阵")
                        return np.array([]).reshape(0, 4)
                    
                    N = self._sample_length
                    J_f = np.zeros((N, 4))
                    points = self.outer._get_sample_points()
                    
                    logging.debug(f"🔍 计算数据残差雅可比矩阵 J_f: 形状=({N}, 4)")
                    
                    for i in range(N):
                        diff = points[i] - self.center
                        dist = np.linalg.norm(diff)
                        
                        if dist > 1e-8:  # 避免除零
                            # ∂f_i/∂c = (p_i - c) / ||p_i - c||
                            J_f[i, 0:3] = diff / dist
                            # ∂f_i/∂r = -1
                            J_f[i, 3] = -1.0
                        else:
                            logging.warning(f"⚠️ 第{i}个点与球心距离过近: {dist:.2e}")
                    
                    logging.debug(f"🔍 J_f矩阵计算完成: 条件数={np.linalg.cond(J_f):.2e}")
                    return J_f

                def calculate_jacobian_h() -> np.ndarray:
                    """
                    计算深度约束雅可比矩阵 J_h
                    
                    J_h[0,j] = ∂h/∂θ_j，其中：
                    - h = τ log(1 + exp((ε + z_max - c_z)/τ))
                    - ∂h/∂c_z = -σ，σ = 1/(1 + exp(-u))
                    
                    Returns:
                        np.ndarray: 形状为 (1, 4) 的雅可比矩阵
                    """
                    if self._sample_length == 0:
                        logging.warning("⚠️ 采样点数量为0，无法计算深度约束雅可比")
                        return np.array([]).reshape(0, 4)
                    
                    # 获取深度约束相关参数
                    max_z = np.max(self.outer._get_sample_points()[:, 2])
                    depth_offset = self._constraint.depth_offset_epsilon
                    tau = self._constraint.softplus_tau
                    
                    # 计算softplus函数的导数
                    u = (depth_offset + max_z - self.center[2]) / tau
                    sigma = 1.0 / (1.0 + np.exp(-u))  # logistic函数
                    
                    J_h = np.zeros((1, 4))
                    J_h[0, 2] = -sigma  # ∂h/∂c_z = -σ
                    
                    logging.debug(f"🔍 深度约束雅可比计算: u={u:.6f}, σ={sigma:.6f}, J_h={J_h}")
                    return J_h

                def calculate_jacobian_r() -> np.ndarray:
                    """
                    计算半径先验雅可比矩阵 J_r
                    
                    J_r[0,j] = ∂r/∂θ_j，其中：
                    - r = r_current - r_prior
                    - ∂r/∂r = 1
                    
                    Returns:
                        np.ndarray: 形状为 (1, 4) 的雅可比矩阵
                    """
                    J_r = np.zeros((1, 4))
                    J_r[0, 3] = 1.0  # ∂r/∂r = 1
                    
                    logging.debug(f"🔍 半径先验雅可比: J_r={J_r}")
                    return J_r

                try:
                    logging.info("-----------------------------------------------------------------------------------")
                    logging.debug(f"🔍 开始求解LM更新方程:")
                    logging.debug(f"   数据残差: 形状={data_residuals.shape}, 范数={np.linalg.norm(data_residuals):.6f}")
                    logging.debug(f"   约束残差: 形状={constraint_residuals.shape}, 范数={np.linalg.norm(constraint_residuals):.6f}")
                    
                    # 1. 计算各雅可比矩阵
                    J_f = calculate_jacobian_f()
                    J_h = calculate_jacobian_h()
                    J_r = calculate_jacobian_r()
                    
                    if J_f.size == 0:
                        logging.error("❌ 数据雅可比矩阵为空，无法求解")
                        return None
                    
                    # 2. 构建权重矩阵
                    if hasattr(self, '_current_geometric_weights') and self._current_geometric_weights is not None:
                        W = np.diag(self._current_geometric_weights)
                        logging.debug(f"🔍 使用几何权重矩阵: 形状={W.shape}")
                    else:
                        W = np.eye(self._sample_length)
                        logging.warning(f"⚠️ 没有维护几何权重，使用单位权重矩阵: 形状={W.shape}")

                    # 3. 计算Hessian矩阵：H = J_f^T W J_f + λ_depth J_h^T J_h + λ_radius J_r^T J_r
                    H_data = J_f.T @ W @ J_f
                    H_h = self._constraint.lambda_depth * (J_h.T @ J_h)
                    H_r = self._constraint.lambda_radius * (J_r.T @ J_r)
                    H_total = H_data + H_h + H_r
                    
                    logging.debug(f"🔍 Hessian矩阵计算:")
                    logging.debug(f"   H_data = J_f^T W J_f: 形状={H_data.shape}, 条件数={np.linalg.cond(H_data):.2e}")
                    logging.debug(f"   H_h = λ_depth J_h^T J_h: 形状={H_h.shape}, λ_depth={self._constraint.lambda_depth}")
                    logging.debug(f"   H_r = λ_radius J_r^T J_r: 形状={H_r.shape}, λ_radius={self._constraint.lambda_radius}")
                    logging.debug(f"   H_total: 形状={H_total.shape}, 条件数={np.linalg.cond(H_total):.2e}")

                    # 4. 计算梯度：g = -(J_f^T W f + λ_depth J_h^T h + λ_radius J_r^T r)
                    g_data = J_f.T @ W @ data_residuals
                    g_h = self._constraint.lambda_depth * J_h.T @ constraint_residuals
                    g_r = self._constraint.lambda_radius * J_r.T @ radius_residuals
                    g_total = -(g_data + g_h + g_r)
                    
                    logging.debug(f"🔍 梯度计算:")
                    logging.debug(f"   g_data: 形状={g_data.shape}, 范数={np.linalg.norm(g_data):.6f}")
                    logging.debug(f"   g_h: 形状={g_h.shape}, 范数={np.linalg.norm(g_h):.6f}")
                    logging.debug(f"   g_r: 形状={g_r.shape}, 范数={np.linalg.norm(g_r):.6f}")
                    logging.debug(f"   g_total: 形状={g_total.shape}, 范数={np.linalg.norm(g_total):.6f}")

                    # 5. 添加LM阻尼项：(H_total + λ_LM I)Δ = g_total
                    if not hasattr(self, '_lambda_lm'):
                        self._lambda_lm = self._config.lambda_lm
                        logging.debug(f"🔍 初始化LM阻尼因子: λ_LM = {self._lambda_lm}")
                    
                    H_lm = H_total + self._lambda_lm * np.eye(4)
                    logging.debug(f"🔍 LM阻尼矩阵: H_lm = H_total + {self._lambda_lm} * I")
                    logging.debug(f"   H_lm条件数: {np.linalg.cond(H_lm):.2e}")

                    # 6. 求解线性方程组
                    delta = np.linalg.solve(H_lm, g_total)
                    logging.debug(f"🔍 线性方程组求解成功:")
                    logging.debug(f"   参数更新向量 Δθ: {delta.reshape(-1, 1)}")
                    logging.debug(f"   更新向量范数: {np.linalg.norm(delta):.6f}")
                    logging.debug(f"   各分量更新: Δc_x={delta[0]:.6f}, Δc_y={delta[1]:.6f}, Δc_z={delta[2]:.6f}, Δr={delta[3]:.6f}")
                    
                    return delta

                except np.linalg.LinAlgError as e:
                    logging.error(f"❌ LM方程求解失败 - 线性代数错误: {e}")
                    logging.error(f"   可能原因: Hessian矩阵奇异或病态")
                    return None
                except Exception as e:
                    logging.error(f"❌ LM方程求解异常: {e}")
                    return None
            
            def _check_parameters_valid(self, params: np.ndarray, data_residuals: np.ndarray, constraint_residuals: np.ndarray, radius_residuals: np.ndarray) -> bool:
                """
                检查参数有效性 - 重新设计基于新的理论
                
                新的理论：减去残差之后各点都在球上，所有点一视同仁
                空间分布完美：最终残差的平均范式 < 误差容忍度
                
                Args:
                    params: 参数向量 [cx, cy, cz, r]
                    
                Returns:
                    Tuple[bool, bool]: (半径有效, 空间分布完美)
                """
                center = params[:3]
                radius = params[3]
                
                # 调试信息：显示参数检查输入
                logging.info("-----------------------------------------------------------------------------------")
                logging.debug(f"🔍 参数有效性检查:")
                logging.debug(f"   参数向量: {params.reshape(-1, 1)}")  # 列向量形式
                logging.debug(f"   球心: {center}")
                logging.debug(f"   半径: {radius:.6f}")
                                
                # 1. 检查残差函数值是否变小
                # 使用辅助函数维护完整的优化目标函数值
                if hasattr(self, '_current_geometric_weights') and self._current_geometric_weights is not None:
                    weights = self._current_geometric_weights
                else:
                    logging.warning("⚠️ 没有维护几何权重，使用单位权重矩阵")
                    weights = np.ones(self._sample_length)
                # 使用辅助函数检查函数值改善
                current_function_value, is_improved = self._maintain_function_value(
                    residuals=data_residuals,
                    weights=weights,
                    constraint_residuals=constraint_residuals,
                    radius_residuals=radius_residuals
                )
                
                return is_improved
                
                
            
            def _check_single_point_constraint(self, point: np.ndarray, center: np.ndarray, 
                                             radius: float, point_type: str) -> bool:
                """检查单点约束"""
                # 计算点到球心的距离
                distance_to_center = np.linalg.norm(point - center)
                
                # 计算点到眼球表面的距离
                surface_distance = abs(distance_to_center - radius)
                
                # 调试信息：显示单点约束检查
                logging.info("-----------------------------------------------------------------------------------")
                logging.debug(f"🔍 单点约束检查:")
                logging.debug(f"   点坐标: {point}")
                logging.debug(f"   球心: {center}")
                logging.debug(f"   半径: {radius:.6f}")
                logging.debug(f"   点类型: {point_type}")
                logging.debug(f"   到球心距离: {distance_to_center:.6f}")
                logging.debug(f"   到表面距离: {surface_distance:.6f}")
                
                # 根据点类型获取正常范围
                if point_type == "pupil_center":
                    min_dist = ANATOMICAL_CONSTRAINTS["PUPIL_TO_SURFACE_MIN"]
                    max_dist = ANATOMICAL_CONSTRAINTS["PUPIL_TO_SURFACE_MAX"]
                elif point_type == "iris_boundaries":
                    min_dist = ANATOMICAL_CONSTRAINTS["IRIS_TO_SURFACE_MIN"]
                    max_dist = ANATOMICAL_CONSTRAINTS["IRIS_TO_SURFACE_MAX"]
                elif point_type == "eye_contours":
                    min_dist = ANATOMICAL_CONSTRAINTS["CONTOUR_TO_SURFACE_MIN"]
                    max_dist = ANATOMICAL_CONSTRAINTS["CONTOUR_TO_SURFACE_MAX"]
                else:
                    logging.debug(f"❌ 未知点类型: {point_type}")
                    return False  # 未知类型，默认不通过
                
                # 检查距离是否在正常范围内
                constraint_satisfied = min_dist <= surface_distance <= max_dist
                
                logging.debug(f"   约束范围: [{min_dist:.6f}, {max_dist:.6f}]")
                logging.debug(f"   约束满足: {constraint_satisfied}")
                
                return constraint_satisfied
            
            def _check_config_convergence(self) -> bool:
                """检查LM算法收敛性"""
                # 调试信息：显示收敛性检查
                logging.info("-----------------------------------------------------------------------------------")
                logging.debug(f"🔍 LM收敛性检查:")
                logging.debug(f"   当前迭代次数: {self.iteration_count}")
                logging.debug(f"   最大迭代次数: {self._config.max_iterations}")
                
                # 1. 参数收敛检查：||Δθ|| < param_tolerance，||Δr|| < residual_tolerance
                if hasattr(self.outer, '_last_params') and hasattr(self.outer, '_last_residuals'):
                    param_change = np.linalg.norm(self.params - self.outer._last_params)
                    residual_change = self._current_residual_norm - np.linalg.norm(self.outer._last_residuals)
                    
                    logging.debug(f"   参数变化: {param_change:.6f}")
                    logging.debug(f"   参数容忍度: {self._config.param_tolerance}")
                    logging.debug(f"   残差变化: {residual_change:.6f}")
                    logging.debug(f"   残差容忍度: {self._config.residual_tolerance}")
                    
                    if param_change < self._config.param_tolerance and residual_change < self._config.residual_tolerance:
                        logging.info(f"✅ LM收敛：参数变化{param_change:.2e} < {self._config.param_tolerance}，平均残差变化{residual_change:.2e} < {self._config.residual_tolerance}")
                        return True
                    # 2. 最大迭代次数检查
                    if self.iteration_count >= self._config.max_iterations - 1:
                        logging.warning(f"⚠️ LM达到最大迭代次数: {self._config.max_iterations}")
                        logging.warning(f"⚠️ LM优化未收敛: 参数变化{param_change:.2e} 参数变化阈值{self._config.param_tolerance}，残差变化{residual_change:.2e} 残差变化阈值{self._config.residual_tolerance}")
                        return False
                else:
                    logging.debug(f"   缺少历史参数或残差信息")
                
                return False
            
            
            def _maintain_function_value(self, residuals: np.ndarray, weights: np.ndarray, 
                                constraint_residuals: np.ndarray, radius_residuals: np.ndarray) -> Tuple[float, bool]:
                """
                维护完整的优化目标函数值并检查是否改善
                
                函数表达式：Φ(θ) = F(θ) + (1/2)λ_depth * h(θ)² + (1/2)λ_r * (r - r₀)²
                其中：
                - F(θ) = (1/2)∑w_i * f_i² (数据项)
                - h(θ) = τ log(1 + exp((ε + z_max - c_z)/τ)) (深度约束)
                - r(θ) = r - r₀ (半径先验)
                
                Args:
                    residuals: 数据残差向量 f(θ)
                    weights: 权重向量
                    constraint_residuals: 约束残差向量 [h(θ)]
                    radius: 当前半径值
                    
                Returns:
                    Tuple[float, bool]: (当前函数值, 是否改善)
                """
                # 1. 数据项：F(θ) = (1/2)∑w_i * f_i²
                data_term = 0.5 * np.sum(weights * (residuals ** 2))
                
                # 2. 深度约束项：(1/2)λ_depth * h(θ)²
                depth_term = 0.5 * self._constraint.lambda_depth * (constraint_residuals[0] ** 2)
                
                # 3. 半径先验项：(1/2)λ_r * (r - r₀)²
                radius_term = 0.5 * self._constraint.lambda_radius * (radius_residuals[0] ** 2)
                
                # 总函数值
                current_value = data_term + depth_term + radius_term
                
                # 检查是否有历史函数值
                if hasattr(self, '_last_function_value'):
                    improvement = self._last_function_value - current_value
                    is_improved = improvement >= 0
                    
                    # 更新历史值
                    self._last_function_value = current_value
                    
                    # 详细的调试信息
                    logging.debug(f"🔍 函数值计算详情:")
                    logging.debug(f"   数据项 F(θ): {data_term:.6f}")
                    logging.debug(f"   深度约束项: {depth_term:.6f}")
                    logging.debug(f"   半径先验项: {radius_term:.6f}")
                    logging.debug(f"   总函数值 Φ(θ): {current_value:.6f}")
                    logging.debug(f"   函数值改善: {improvement:.6f}")
                    
                    return current_value, is_improved
                else:
                    # 首次计算，初始化
                    self._last_function_value = current_value
                    logging.debug(f"🔍 初始化函数值 Φ(θ): {current_value:.6f}")
                    return current_value, True  # 首次计算认为"改善"

            
            def _update_final_state(self) -> None:
                """更新最终状态"""
                # 调试信息：显示最终状态更新
                logging.info("-----------------------------------------------------------------------------------")
                logging.debug(f"🔍 更新最终状态:")
                
                # 计算最终残差
                final_data_residuals = self._calculate_fitting_residual()
                self._current_residual_norm = np.linalg.norm(final_data_residuals)
                
                logging.debug(f"   最终残差向量: {final_data_residuals.reshape(-1, 1)}")  # 列向量形式
                logging.debug(f"   最终残差范数: {self._current_residual_norm:.6f}")
                
                # 计算内点比例（简化版本）
                if self._sample_length > 0:
                    self.inlier_ratio = 1.0  # 约束优化中所有点都是内点
                    logging.debug(f"   内点比例: {self.inlier_ratio:.6f}")
                
                # 计算质量分数
                self.quality_score = max(0.0, 1.0 - self._current_residual_norm / 0.01)  # 归一化到[0,1]
                logging.debug(f"   质量分数: {self.quality_score:.6f}")
                
                # 保存到外部实例
                self.outer._convergence_iterations = self.iteration_count
                self.outer._is_converged = self.is_converged
                self.outer._current_residual_norm = self._current_residual_norm
                self.outer._inlier_ratio = self.inlier_ratio
                self.outer._quality_score = self.quality_score
                self.outer._warnings = self.warnings.copy()
                
                logging.debug(f"   收敛迭代次数: {self.outer._convergence_iterations}")
                logging.debug(f"   是否收敛: {self.outer._is_converged}")
                logging.debug(f"   警告信息: {self.outer._warnings}")
        
        if not self._has_sample_data():
            return self._handle_failure("没有可用的采样点")
        
        # 1. 参数初始化
        logging.info("🔧 开始参数初始化...")
        center, radius = self._initialize_parameters()
        logging.info(f"📍 初始中心点: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
        logging.info(f"📏 初始半径: {radius:.3f}m")
        
        # 2. 创建优化器并执行优化
        logging.info("⚙️ 创建约束LM优化器...")
        optimizer = ConstrainedLMOptimizer(self, center, radius)
        center, radius = optimizer.optimize()
        
        # 3. 构建结果
        logging.info(f"✅ 优化完成: 最终中心点=({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}), 半径={radius:.3f}m")
        result = self._build_fitting_result(center, radius)
        return self._ellipsoid_params, center, result
    
    # ==================== 参数初始化接口 ====================
    
    def _initialize_parameters(self) -> Tuple[np.ndarray, float]:
        """智能初始化：基于三个虹膜点拟合外心，通过勾股定理计算半径和深度"""
        # 调试信息：显示参数初始化开始
        logging.debug(f"🔍 参数初始化开始:")
        
        # 1. 获取虹膜边界点
        iris_points = self._coords_3d.get("iris_boundaries", np.array([]))
        logging.info(f"🔍 虹膜边界点数量: {len(iris_points)}")
        
        # 调试信息：显示虹膜点数据
        if len(iris_points) > 0:
            logging.debug(f"   虹膜点数据:")
            logging.debug(f"     形状: {iris_points.shape}")
            logging.debug(f"     前3个点: {iris_points[:3]}")
            logging.debug(f"     所有点: {iris_points}")
        
        if len(iris_points) < 3:
            logging.warning("⚠️ 虹膜点不足3个，使用备选初始化")
            return self._fallback_initialization()
        
        # 2. 随机选择3个虹膜点进行拟合
        if len(iris_points) > 3:
            # 随机选择3个点，提高鲁棒性
            indices = np.random.choice(len(iris_points), 3, replace=False)
            selected_points = iris_points[indices]
            logging.debug(f"   随机选择3个点，索引: {indices}")
        else:
            selected_points = iris_points
            logging.debug(f"   使用所有3个点")
        
        # 调试信息：显示选中的点
        logging.debug(f"   选中的点:")
        logging.debug(f"     形状: {selected_points.shape}")
        logging.debug(f"     坐标: {selected_points}")
        
        # 3. 拟合外心（三个点确定的圆的圆心）
        logging.info("🔧 开始2D圆拟合...")
        center_2d, radius_2d = self._fit_circle_2d(selected_points)
        
        if center_2d is None:
            logging.warning("⚠️ 2D圆拟合失败，使用备选初始化")
            return self._fallback_initialization()
        
        # 调试信息：显示2D圆拟合结果
        logging.debug(f"   2D圆拟合结果:")
        logging.debug(f"     圆心: {center_2d}")
        logging.debug(f"     半径: {radius_2d:.6f}")
        
        # 4. 计算3D球心：使用2D圆心的x,y坐标，z坐标基于解剖学约束
        # 虹膜边界到眼球表面的距离范围：0.5-1.2mm
        iris_to_surface_min = ANATOMICAL_CONSTRAINTS["IRIS_TO_SURFACE_MIN"]
        iris_to_surface_max = ANATOMICAL_CONSTRAINTS["IRIS_TO_SURFACE_MAX"]
        
        # 调试信息：显示解剖学约束
        logging.debug(f"   解剖学约束:")
        logging.debug(f"     虹膜到表面最小距离: {iris_to_surface_min:.6f}")
        logging.debug(f"     虹膜到表面最大距离: {iris_to_surface_max:.6f}")
        
        # 5. 球半径设置为默认值
        radius_3d = ANATOMICAL_CONSTRAINTS["EYEBALL_RADIUS_DEFAULT"]
        # 选择中间值作为中值距离加半径作为d
        
        d = (iris_to_surface_min + iris_to_surface_max) / 2 + radius_3d
        
        # 调试信息：显示半径计算
        logging.debug(f"   半径计算:")
        logging.debug(f"     默认半径: {radius_3d:.6f}")
        logging.debug(f"     中值距离: {(iris_to_surface_min + iris_to_surface_max) / 2:.6f}")
        logging.debug(f"     中值距离+半径 d: {d:.6f}")
        
        # 6. 通过勾股定理计算深度偏移z
        # 勾股定理：d² = r² + z²
        # 其中：d是中值距离+半径（斜边），r是2D圆的半径（直角边），z是深度偏移（直角边）
        # 所以：z = sqrt(d² - r²)
        if d > radius_2d:
            logging.info(f"🔍 中值距离+半径: {d:.3f}m，2D半径: {radius_2d:.3f}m，使用虹膜深度偏移")
            z = np.sqrt(d**2 - radius_2d**2)
            logging.debug(f"   勾股定理计算深度偏移: z = sqrt({d:.6f}² - {radius_2d:.6f}²) = {z:.6f}")
        else:
            # 如果中值距离小于2D半径，说明约束不合理，使用默认偏移
            logging.warning(f"中值距离{d:.6f}小于2D半径{radius_2d:.6f}，使用默认偏移")
            z = iris_to_surface_min
            logging.debug(f"   使用默认深度偏移: {z:.6f}")
        
        # 球心z坐标：虹膜点的平均z坐标 + 深度偏移（远离摄像机）
        avg_z = np.mean(selected_points[:, 2])
        center_3d = np.array([center_2d[0], center_2d[1], avg_z + z])
        
        # 调试信息：显示3D球心计算
        logging.debug(f"   3D球心计算:")
        logging.debug(f"     虹膜点平均Z坐标: {avg_z:.6f}")
        logging.debug(f"     深度偏移: {z:.6f}")
        logging.debug(f"     3D球心: {center_3d}")
        
        # 7. 验证半径是否在合理范围内
        min_radius = ANATOMICAL_CONSTRAINTS["EYEBALL_RADIUS_MIN"]
        max_radius = ANATOMICAL_CONSTRAINTS["EYEBALL_RADIUS_MAX"]
        
        if radius_3d < min_radius or radius_3d > max_radius:
            logging.warning(f"⚠️ 计算半径{radius_3d:.6f}超出范围[{min_radius:.6f}, {max_radius:.6f}]，使用默认值")
            radius_3d = ANATOMICAL_CONSTRAINTS["EYEBALL_RADIUS_DEFAULT"]
            logging.debug(f"   使用默认半径: {radius_3d:.6f}")
        
        # 调试信息：显示最终参数
        logging.debug(f"   最终初始化参数:")
        logging.debug(f"     球心: {center_3d}")
        logging.debug(f"     半径: {radius_3d:.6f}")
        
        return center_3d, radius_3d
    
    def _fit_circle_2d(self, points: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """基于三个点拟合2D圆的圆心和半径"""
        if len(points) != 3:
            return None, None
        
        # 调试信息：显示2D圆拟合开始
        logging.debug(f"🔍 2D圆拟合开始:")
        logging.debug(f"   输入点: {points}")
        logging.debug(f"   点数量: {len(points)}")
        
        try:
            # 提取x,y坐标
            x_coords = points[:, 0]
            y_coords = points[:, 1]
            
            # 调试信息：显示坐标提取
            logging.debug(f"   坐标提取:")
            logging.debug(f"     X坐标: {x_coords}")
            logging.debug(f"     Y坐标: {y_coords}")
            
            # 构建线性方程组：Ax = b
            # 对于点(xi, yi)，有：(xi² + yi²) = 2x0*xi + 2y0*yi + (r² - x0² - y0²)
            # 其中(x0, y0)是圆心，r是半径
            
            # 构建系数矩阵A
            A = np.array([
                [2*x_coords[0], 2*y_coords[0], 1],
                [2*x_coords[1], 2*y_coords[1], 1],
                [2*x_coords[2], 2*y_coords[2], 1]
            ])
            
            # 构建常数向量b
            b = np.array([
                x_coords[0]**2 + y_coords[0]**2,
                x_coords[1]**2 + y_coords[1]**2,
                x_coords[2]**2 + y_coords[2]**2
            ])
            
            # 调试信息：显示线性方程组
            logging.debug(f"   线性方程组:")
            logging.debug(f"     系数矩阵 A:\n{A}")
            logging.debug(f"     常数向量 b: {b.reshape(-1, 1)}")  # 列向量形式
            
            # 求解线性方程组
            solution = np.linalg.solve(A, b)
            
            # 调试信息：显示求解结果
            logging.debug(f"   线性方程组求解结果:")
            logging.debug(f"     解向量: {solution.reshape(-1, 1)}")  # 列向量形式
            
            # 提取圆心坐标和半径
            center_x = solution[0]
            center_y = solution[1]
            center_2d = np.array([center_x, center_y])
            
            # 计算半径：r = sqrt(x0² + y0² + c)，其中c = r² - x0² - y0²
            c = solution[2]
            radius_2d = np.sqrt(center_x**2 + center_y**2 + c)
            
            # 调试信息：显示圆心和半径计算
            logging.debug(f"   圆心和半径计算:")
            logging.debug(f"     圆心X坐标: {center_x:.6f}")
            logging.debug(f"     圆心Y坐标: {center_y:.6f}")
            logging.debug(f"     圆心: {center_2d}")
            logging.debug(f"     常数c: {c:.6f}")
            logging.debug(f"     半径: {radius_2d:.6f}")
            
            # 验证结果的有效性
            if np.isnan(radius_2d) or radius_2d <= 0:
                logging.warning("2D圆拟合结果无效")
                return None, None
            
            # 调试信息：显示最终结果
            logging.debug(f"   2D圆拟合成功:")
            logging.debug(f"     圆心: {center_2d}")
            logging.debug(f"     半径: {radius_2d:.6f}")
            
            return center_2d, radius_2d
            
        except np.linalg.LinAlgError:
            logging.warning("2D圆拟合线性方程组求解失败")
            return None, None
        except Exception as e:
            logging.error(f"2D圆拟合异常: {e}")
            return None, None
    
    def _fallback_initialization(self) -> Tuple[np.ndarray, float]:
        """备选初始化：基于瞳孔中心，考虑解剖学约束"""
        pupil_points = self._coords_3d.get("pupil_center", np.array([]))
        
        if len(pupil_points) > 0:
            # 1. 计算瞳孔中心
            pupil_center = np.mean(pupil_points, axis=0)
            logging.info(f"🔍 备选初始化: 瞳孔中心=({pupil_center[0]:.3f}, {pupil_center[1]:.3f}, {pupil_center[2]:.3f})")
            
            # 2. 基于解剖学约束计算初始球心
            # 瞳孔到眼球表面的距离范围：0.1-0.3mm
            pupil_to_surface_min = ANATOMICAL_CONSTRAINTS["PUPIL_TO_SURFACE_MIN"]
            pupil_to_surface_max = ANATOMICAL_CONSTRAINTS["PUPIL_TO_SURFACE_MAX"]
            
            # 初始半径设置为默认值，选择中间值作为初始偏移
            initial_radius = ANATOMICAL_CONSTRAINTS["EYEBALL_RADIUS_DEFAULT"]
            initial_offset = (pupil_to_surface_min + pupil_to_surface_max) / 2 + initial_radius
            
            # 3. 计算初始球心：瞳孔中心 + 偏移向量
            # 偏移方向：从瞳孔中心指向眼球内部（Z轴正方向，远离摄像机）
            offset_vector = np.array([0.0, 0.0, initial_offset])
            initial_center = pupil_center + offset_vector
            
            logging.info(f"✅ 备选初始化完成: 初始球心=({initial_center[0]:.3f}, {initial_center[1]:.3f}, {initial_center[2]:.3f}), 偏移={initial_offset:.3f}m")
            
            return initial_center, initial_radius
        else:
            # 所有点都不足，使用默认值
            logging.warning("⚠️ 关键点不足，使用默认初始化")
            return np.array([0.0, 0.0, 0.0]), ANATOMICAL_CONSTRAINTS["EYEBALL_RADIUS_DEFAULT"]
    
    # ==================== 结果构建接口 ====================
    
    def _build_fitting_result(self, center: np.ndarray, radius: float) -> FittingResult:
        """构建拟合结果"""
        # 更新椭球参数
        self._ellipsoid_params.center = center
        self._ellipsoid_params.axes = np.array([radius, radius, radius])
        
        # 计算内点数量（约束优化中所有点都是内点）
        inlier_count = self._get_sample_count()
        total_points = self._get_sample_count()
        inlier_ratio = 1.0 if total_points > 0 else 0.0
        
        # 计算最终残差
        if hasattr(self, '_current_residual_norm'):
            final_residual = self._current_residual_norm
        else:
            final_residual = 0.0
        
        # 获取收敛信息
        convergence_iterations = getattr(self, '_convergence_iterations', 0)
        converged = getattr(self, '_is_converged', False)
        quality_score = getattr(self, '_quality_score', 0.0)
        warnings = getattr(self, '_warnings', [])
        
        # 构建结果对象
        result = FittingResult(
            ellipsoid_params=self._ellipsoid_params,
            center=center,
            inlier_count=inlier_count,
            total_points=total_points,
            inlier_ratio=inlier_ratio,
            final_residual=final_residual,
            convergence_iterations=convergence_iterations,
            converged=converged,
            quality_score=quality_score,
            warnings=warnings
        )
        
        return result
    
    # ==================== 椭球拟合接口（保持原有功能） ====================
    
    def _fit_ellipsoid(self, key_coordinates: SingleEyeKeyCoordinates, 
                       center_only: bool = False) -> Tuple[EllipsoidParams, np.ndarray, FittingResult]:
        """
        椭球拟合接口 - 保持原有功能不变，作为约束拟合失败时的回退方案
        
        Args:
            key_coordinates: 关键点坐标
            center_only: 是否只拟合中心
            
        Returns:
            Tuple[EllipsoidParams, np.ndarray, FittingResult]: 拟合结果
        """
        # 这里应该调用原有的椭球拟合逻辑
        # 为了保持向后兼容，暂时返回默认结果
        logging.warning("椭球拟合功能暂未实现，返回默认结果")
        
        # 使用默认参数
        center = np.array([0.0, 0.0, 0.0])
        radius = ANATOMICAL_CONSTRAINTS["EYEBALL_RADIUS_DEFAULT"]
        
        # 构建结果
        result = self._build_fitting_result(center, radius)
        return self._ellipsoid_params, center, result
    
    # ==================== 失败处理接口 ====================
    
    def _handle_failure(self, reason: str) -> Tuple[EllipsoidParams, np.ndarray, FittingResult]:
        """
        处理拟合失败情况
        
        Args:
            reason: 失败原因
            
        Returns:
            Tuple[EllipsoidParams, np.ndarray, FittingResult]: 失败结果
        """
        logging.error(f"拟合失败: {reason}")
        
        # 返回失败结果
        result = FittingResult(
            ellipsoid_params=self._ellipsoid_params,
            center=np.array([0.0, 0.0, 0.0]),
            inlier_count=0,
            total_points=0,
            inlier_ratio=0.0,
            final_residual=float('inf'),
            convergence_iterations=0,
            converged=False,
            quality_score=0.0,
            warnings=[reason]
        )
        
        return self._ellipsoid_params, np.array([0.0, 0.0, 0.0]), result

    # ==================== 辅助方法 ====================
    
    def _get_sample_data(self) -> Optional[SampleData]:
        """获取当前采样数据"""
        return self._sample_data
    
    def _has_sample_data(self) -> bool:
        """检查是否有采样数据"""
        return self._sample_data is not None and len(self._sample_data.points) > 0
    
    def _get_sample_points(self) -> np.ndarray:
        """获取采样点"""
        return self._sample_data.points if self._has_sample_data() else np.array([])
    
    def _get_sample_types(self) -> List[str]:
        """获取采样点类型"""
        return self._sample_data.types if self._has_sample_data() else []
    
    def _get_sample_visibility(self) -> np.ndarray:
        """获取采样点visibility"""
        return self._sample_data.visibility if self._has_sample_data() else np.array([])
    
    def _get_sample_count(self) -> int:
        """获取采样点数量"""
        return len(self._sample_data.points) if self._has_sample_data() else 0
    
    def _update_sample_data(self, points: np.ndarray, types: List[str], 
                           visibility: np.ndarray, method: str) -> None:
        """更新采样数据"""
        self._sample_data = SampleData(
            points=points,
            types=types,
            visibility=visibility,
            method=method
        )
        logging.debug(f"📊 采样数据已更新: {method}, 点数={len(points)}")
    
    def _filter_sample_data(self, indices: np.ndarray) -> None:
        """根据索引过滤采样数据"""
        if not self._has_sample_data() or len(indices) == 0:
            return
        
        valid_indices = [i for i in indices if 0 <= i < len(self._sample_data.points)]
        if valid_indices:
            self._sample_data = SampleData(
                points=self._sample_data.points[valid_indices],
                types=[self._sample_data.types[i] for i in valid_indices],
                visibility=self._sample_data.visibility[valid_indices],
                method="filtered"
            )
            logging.debug(f"🔍 采样数据已过滤: 保留{len(valid_indices)}个点")
    
    def _get_point_type_count(self, point_type: str) -> int:
        """获取指定类型的点数"""
        if not self._has_sample_data():
            return 0
        return sum(1 for t in self._sample_data.types if t == point_type)
    
    def _get_points_by_type(self, point_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """根据类型获取点和对应的visibility"""
        if not self._has_sample_data():
            return np.array([]), np.array([])
        
        indices = [i for i, t in enumerate(self._sample_data.types) if t == point_type]
        if not indices:
            return np.array([]), np.array([])
        
        return self._sample_data.points[indices], self._sample_data.visibility[indices]
    
    def _get_physiological_bias_median(self, point_type: str) -> float:
        """
        获取指定点类型的生理偏差中值
        
        生理偏差中值是基于解剖学约束范围计算得出的，用于残差计算中的个体修正：
        - 瞳孔中心: 0.2mm (0.1-0.3mm范围的中值)
        - 虹膜边界: 0.85mm (0.5-1.2mm范围的中值)  
        - 眼睛轮廓: 1.5mm (1.0-2.0mm范围的中值)
        
        Args:
            point_type: 点类型字符串，支持 "pupil_center", "iris_boundaries", "eye_contours"
            
        Returns:
            float: 对应的生理偏差中值，如果类型不存在则返回0.0
            
        Example:
            >>> bias = fitter._get_physiological_bias_median("pupil_center")
            >>> print(f"瞳孔中心生理偏差: {bias:.6f}")  # 0.000200
        """
        bias = self._physiological_bias_median.get(point_type, 0.0)
        
        if bias == 0.0 and point_type not in self._physiological_bias_median:
            logging.warning(f"⚠️ 未知点类型 '{point_type}'，使用默认偏差0.0")
            
        return bias
    
    def _get_physiological_bias_vector(self, point_types: List[str]) -> np.ndarray:
        """
        根据点类型列表获取对应的生理偏差中值向量
        
        这个函数将点类型列表映射为对应的生理偏差向量，用于批量残差计算。
        使用列表推导式和字典映射，避免了if-else语句，提高了代码的可维护性。
        
        Args:
            point_types: 点类型列表，每个元素应为有效的点类型字符串
            
        Returns:
            np.ndarray: 对应的生理偏差中值向量，形状与point_types相同
            
        Example:
            >>> types = ["pupil_center", "iris_boundaries", "pupil_center"]
            >>> bias_vector = fitter._get_physiological_bias_vector(types)
            >>> print(f"偏差向量: {bias_vector}")  # [0.0002, 0.00085, 0.0002]
        """
        if not point_types:
            logging.warning("⚠️ 点类型列表为空，返回空偏差向量")
            return np.array([])
            
        # 使用列表推导式和字典映射，避免if-else语句
        bias_vector = np.array([self._get_physiological_bias_median(pt) for pt in point_types])
        
        # 调试信息
        logging.debug(f"🔍 生理偏差向量计算:")
        logging.debug(f"   点类型列表: {point_types}")
        logging.debug(f"   偏差向量: {bias_vector}")
        logging.debug(f"   向量形状: {bias_vector.shape}")
        logging.debug(f"   偏差统计: 均值={np.mean(bias_vector):.6f}, 标准差={np.std(bias_vector):.6f}")
        
        # 验证偏差向量的合理性
        if np.any(bias_vector < 0):
            logging.warning("⚠️ 发现负的生理偏差值，这可能表示配置错误")
        
        return bias_vector


# ==================== 模块级接口函数（保持向后兼容） ====================

def center_fitter(key_coordinates: SingleEyeKeyCoordinates, 
                 trials_times: int, 
                 config: Optional[BaseFittingConfig] = None) -> Tuple[EllipsoidParams, np.ndarray]:
    """
    模块级接口函数 - 保持向后兼容
    
    Args:
        key_coordinates: 关键点坐标
        trials_times: 实验次数
        config: 拟合配置（可选）
        
    Returns:
        Tuple[EllipsoidParams, np.ndarray]: 拟合结果（椭球参数和中心点）
    """
    # 创建CenterFitter单例实例
    fitter = CenterFitter(strategy=config)
    
    # 调用实例方法，获取三个返回值
    ellipsoid_params, center, fitting_result = fitter.center_fitter(key_coordinates, trials_times)
    
    # 记录详细的拟合结果信息到日志
    logging.info(f"拟合完成 - {fitting_result}")
    
    # 只返回前两个值，保持向后兼容
    return ellipsoid_params, center

# 导出列表
__all__ = ["center_fitter", "CenterFitter"]
