from typing import Dict, List, Optional, Tuple
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
from project.fitting.config.models import EYEBALL_RADIUS

class CenterFitter:
    """
    眼球中心拟合器 - 单例类
    支持约束拟合、策略选择和4D坐标输入（包含visibility）
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
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
        
        # 数据预处理结果（全局维护）
        self._coords_3d: Dict[str, np.ndarray] = {}
        self._visibility_scores: Dict[str, np.ndarray] = {}
        self._total_points: int = 0
        self._avg_visibility: float = 0.0
        
        # 采样结果（全局维护）
        self._sampled_points: np.ndarray = np.array([])
        self._sampled_point_types: List[str] = []
        self._sampling_method: str = ""
        
        # 状态管理
        self._current_weights = None
        self._weights_calculated = False
        self._current_fit_center = None
        self._current_fit_radius = None
        
        # 拟合状态
        self._fitting_status = FittingStatus(
            is_running=False,
            current_iteration=0,
            max_iterations=self._strategy.max_lm_iterations,
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
        
        # 数据预处理和全局维护
        self._preprocess_data(key_coordinates)
        
        # 自动选择策略（如果未指定）
        if strategy is None:
            strategy = self._assess_data_quality()
            logging.info(f"📊 自动选择拟合策略: {strategy}")
        
        # 直接调用对应的策略函数
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
                if isinstance(points, np.ndarray):
                    # numpy数组：直接处理
                    processed_points, processed_visibility = self._process_numpy_points(points)
                elif isinstance(points, list) and len(points) > 0 and isinstance(points[0], np.ndarray):
                    # 点列表：转换为numpy数组后处理
                    points_array = np.array(points)
                    processed_points, processed_visibility = self._process_numpy_points(points_array)
                else:
                    logging.warning(f"跳过不支持的数据类型: {key}, 类型: {type(points)}")
                    continue
                
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
        if (self._total_points >= MEDIAPIPE_QUALITY_THRESHOLDS["HIGH_QUALITY_MIN_POINTS"] and 
            self._avg_visibility >= MEDIAPIPE_QUALITY_THRESHOLDS["HIGH_QUALITY_CONFIDENCE"]):
            return "constrained_lm"
        elif (self._total_points >= MEDIAPIPE_QUALITY_THRESHOLDS["MEDIUM_QUALITY_MIN_POINTS"] and 
              self._avg_visibility >= MEDIAPIPE_QUALITY_THRESHOLDS["MEDIUM_QUALITY_CONFIDENCE"]):
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
        logging.info(f"📊 使用全部采样策略: {len(self._sampled_point_types)}个点")
        
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
                
            def optimize(self) -> Tuple[Optional[np.ndarray], Optional[float], Optional[np.ndarray]]:
                """执行RANSAC优化"""
                logging.info(f"🔄 RANSAC优化开始: 最大迭代次数={self.outer._strategy.max_ransac_iterations}")
                
                for iteration in range(self.outer._strategy.max_ransac_iterations):
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
                        if self.best_inlier_count >= len(sampled_points) * 0.7:  # 70%内点比例
                            logging.info(f"✅ RANSAC提前收敛: 内点比例={self.best_inlier_count/len(sampled_points):.1%}")
                            break
                
                logging.info(f"🎯 RANSAC优化完成: 最佳内点数={self.best_inlier_count}, 总采样点数={len(sampled_points) if 'sampled_points' in locals() else 0}")
                return self.best_center, self.best_radius, self.best_inliers
            
            def _random_sampling(self) -> Tuple[np.ndarray, List[str]]:
                """随机采样策略"""
                sampled_points = []
                point_types = []
                
                # 1. 瞳孔中心：固定采样1个点
                if "pupil_center" in self.outer._coords_3d and len(self.outer._coords_3d["pupil_center"]) > 0:
                    pupil_points = self.outer._coords_3d["pupil_center"]
                    indices = np.random.choice(len(pupil_points), 1, replace=False)
                    sampled_points.append(pupil_points[indices])
                    point_types.extend(["pupil_center"] * 1)
                else:
                    # 如果瞳孔点不足，从其他点中随机补充
                    supplement_points, supplement_types = self._supplement_sampling(1, "pupil_center")
                    sampled_points.extend(supplement_points)
                    point_types.extend(supplement_types)
                
                # 2. 虹膜边界：固定采样3个点
                if "iris_boundary" in self.outer._coords_3d and len(self.outer._coords_3d["iris_boundary"]) >= 3:
                    iris_points = self.outer._coords_3d["iris_boundary"]
                    indices = np.random.choice(len(iris_points), 3, replace=False)
                    sampled_points.append(iris_points[indices])
                    point_types.extend(["iris_boundary"] * 3)
                else:
                    # 如果虹膜点不足，从其他点中随机补充
                    supplement_points, supplement_types = self._supplement_sampling(3, "iris_boundary")
                    sampled_points.extend(supplement_points)
                    point_types.extend(supplement_types)
                
                # 3. 眼睛轮廓：随机采样4-10个点
                contour_samples = np.random.randint(4, 11)  # 4到10之间的随机数
                if "eye_contour" in self.outer._coords_3d and len(self.outer._coords_3d["eye_contour"]) >= contour_samples:
                    contour_points = self.outer._coords_3d["eye_contour"]
                    indices = np.random.choice(len(contour_points), contour_samples, replace=False)
                    sampled_points.append(contour_points[indices])
                    point_types.extend(["eye_contour"] * contour_samples)
                else:
                    # 如果轮廓点不足，从其他点中随机补充
                    supplement_points, supplement_types = self._supplement_sampling(contour_samples, "eye_contour")
                    sampled_points.extend(supplement_points)
                    point_types.extend(supplement_types)
                
                if sampled_points:
                    return np.vstack(sampled_points), point_types
                else:
                    return np.array([]), []
            
            def _supplement_sampling(self, required_count: int, target_type: str) -> Tuple[List[np.ndarray], List[str]]:
                """补充采样：当某种类型点数不足时，从其他点中随机补充"""
                supplement_points = []
                supplement_types = []
                
                # 收集所有可用的点
                all_available_points = []
                all_available_types = []
                
                for key, points in self.outer._coords_3d.items():
                    if points is not None and len(points) > 0:
                        all_available_points.extend(points)
                        all_available_types.extend([key] * len(points))
                
                if len(all_available_points) >= required_count:
                    # 随机选择补充点
                    indices = np.random.choice(len(all_available_points), required_count, replace=False)
                    for idx in indices:
                        supplement_points.append(all_available_points[idx])
                        supplement_types.append(target_type)
                
                return supplement_points, supplement_types
            
            def _single_ransac_fit(self, sampled_points: np.ndarray, point_types: List[str]) -> Tuple[np.ndarray, float, np.ndarray]:
                """单次RANSAC拟合 - 使用约束优化函数"""
                if len(sampled_points) < self.outer._strategy.min_samples:
                    return None, None, None
                
                # 临时更新采样结果用于约束优化
                original_points = self.outer._sampled_points.copy() if len(self.outer._sampled_points) > 0 else np.array([])
                original_types = self.outer._sampled_point_types.copy()
                original_method = self.outer._sampling_method
                
                # 更新为当前RANSAC采样结果
                self.outer._sampled_points = sampled_points
                self.outer._sampled_point_types = point_types
                self.outer._sampling_method = "ransac_temporary"
                
                try:
                    # 使用约束优化函数进行拟合
                    ellipsoid_params, center, result = self.outer._constrained_optimization()
                    
                    if center is not None:
                        # 使用新的内点计算函数
                        inlier_indices = self._calculate_inliers(
                            sampled_points, 
                            center, 
                            result.ellipsoid_params.axes[0],  # 使用拟合的半径
                            self.outer._strategy.ransac_threshold
                        )
                        
                        return center, result.ellipsoid_params.axes[0], inlier_indices
                    else:
                        return None, None, None
                        
                finally:
                    # 恢复原始采样结果
                    self.outer._sampled_points = original_points
                    self.outer._sampled_point_types = original_types
                    self.outer._sampling_method = original_method
            
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
        
        for key, points in self._coords_3d.items():
            if points is not None and len(points) > 0:
                all_points.append(points)
                point_types.extend([key] * len(points))
        
        if all_points:
            self._sampled_points = np.vstack(all_points)
            self._sampled_point_types = point_types
            self._sampling_method = "full"
        else:
            self._sampled_points = np.array([])
            self._sampled_point_types = []
            self._sampling_method = "full"
        
        logging.info(f"📊 全部采样完成: 采样点数={len(self._sampled_point_types)}")
    
    def _update_sampling_result(self, inlier_indices: np.ndarray) -> None:
        """更新采样结果为最佳内点"""
        if len(inlier_indices) > 0 and len(self._sampled_points) > 0:
            # 确保索引在有效范围内
            valid_indices = [i for i in inlier_indices if 0 <= i < len(self._sampled_points)]
            
            if valid_indices:
                # 使用numpy的索引操作
                self._sampled_points = self._sampled_points[valid_indices]
                self._sampled_point_types = [self._sampled_point_types[i] for i in valid_indices]
                self._sampling_method = "ransac_inliers"
                logging.debug(f"采样结果已更新为RANSAC内点: 内点数={len(valid_indices)}")
            else:
                logging.warning("内点索引无效，保持原有采样结果")
        else:
            logging.warning("无法更新采样结果：内点索引为空或采样点为空")
    
    # ==================== 约束优化接口 ====================
    
    def _constrained_optimization(self) -> Tuple[EllipsoidParams, np.ndarray, FittingResult]:
        """
        约束优化 - 实现完整的约束LM算法
        
        目标函数：Φ(θ) = (1/2) * ||f(θ)||² + ||g(θ)||²
        其中：f(θ)是拟合残差向量，g(θ)是约束惩罚向量
        
        Returns:
            Tuple[EllipsoidParams, np.ndarray, FittingResult]: 优化结果
        """
        
        class ConstrainedLMOptimizer:
            """约束LM优化器内部类"""
            
            def __init__(self, outer_instance, center: np.ndarray, radius: float):
                self.outer = outer_instance
                self.center = center.copy()
                self.radius = radius
                self.params = np.concatenate([center, [radius]])  # θ = [c_x, c_y, c_z, r]
                
                # 优化状态维护
                self.iteration_count = 0
                self.is_converged = False
                self.current_residual = float('inf')
                self.inlier_ratio = 1.0
                self.quality_score = 1.0
                self.warnings = []
                
                # 初始化权重管理器（简化版本）
                from project.fitting.utils.weight_manager import DEFAULT_WEIGHT_MANAGER
                self._weight_manager = DEFAULT_WEIGHT_MANAGER
                
            
            def optimize(self) -> Tuple[np.ndarray, float]:
                """执行约束LM优化"""
                logging.info(f"🔄 约束LM优化开始: 最大迭代次数={self.outer._strategy.max_lm_iterations}")
                
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
                
                # 约束LM优化循环
                for iteration in range(self.outer._strategy.max_lm_iterations):
                    self.iteration_count = iteration
                    
                    # 2.1 计算残差向量
                    data_residuals = self._calculate_fitting_residual()      # f(θ)
                    constraint_residuals = self._calculate_constraint_penalty()  # g(θ)
                    
                    # 2.2 更新权重管理器状态
                    self._update_weight_manager_state(data_residuals)
                    
                    # 2.3 求解LM更新方程
                    delta = self._solve_lm_equation(data_residuals, constraint_residuals)
                    
                    if delta is None:
                        # 求解失败，增加阻尼因子
                        if not hasattr(self.outer._strategy, 'lambda_lm'):
                            self.outer._strategy.lambda_lm = 0.01
                        self.outer._strategy.lambda_lm *= 10.0
                        logging.warning(f"⚠️ LM迭代{iteration}求解失败，增加阻尼因子")
                        continue
                    
                    # 2.4 参数更新
                    params_new = self.params + delta
                    
                    show_delta(delta,iteration)
                    
                    # 2.4 检查参数有效性且更新参数
                    update_enable, converge_enable = self._check_parameters_valid(params_new)
                    if update_enable:
                        self.params = params_new
                        self.center = self.params[:3]
                        self.radius = self.params[3]
                        
                        # 2.6 检查收敛性
                        if self._check_convergence() and converge_enable:
                            self.is_converged = True
                            logging.info(f"✅ LM优化收敛: 迭代{iteration}次")
                            break
                    else:
                        # 参数无效，增加阻尼因子
                        if not hasattr(self.outer._strategy, 'lambda_lm'):
                            self.outer._strategy.lambda_lm = 0.01
                        self.outer._strategy.lambda_lm *= 10.0
                        self.warnings.append(f"迭代{iteration}: 参数无效")
                
                # 更新最终状态
                self._update_final_state()
                if not self.is_converged:
                    logging.warning(f"⚠️ LM优化未收敛: 达到最大迭代次数{self.outer._strategy.max_lm_iterations}")
                    logging.warning(f"⚠️ 其中最后一次半径有效性为{update_enable}，几何空间分布良好性为{converge_enable}")
                    self._check_convergence()
                return self.center, self.radius
            
            def _calculate_fitting_residual(self) -> np.ndarray:
                """计算拟合残差向量 - 每个点的残差f(θ)"""
                if len(self.outer._sampled_points) == 0:
                    return np.array([])
                
                # 1. 计算每个点到球心的距离
                distances = np.linalg.norm(self.outer._sampled_points - self.center, axis=1)
                
                # 2. 计算每个点的残差：点到球面的距离
                # f(θ) = [distance_1 - radius, distance_2 - radius, ..., distance_N - radius]
                residuals = distances - self.radius
                
                return residuals
            
            def _calculate_constraint_penalty(self) -> np.ndarray:
                """计算约束违反惩罚向量 - 包含点约束和深度约束"""
                if len(self.outer._sampled_points) == 0:
                    return np.array([])
                
                N = len(self.outer._sampled_points)
                constraint_dim = 1  # 深度约束维度
                constraint_residuals = np.zeros(N + constraint_dim)
                
                # 1. 点约束残差：g_i(θ) = √λ₁ * softplus_τ(r - d_i)
                for i in range(N):
                    distance = np.linalg.norm(self.outer._sampled_points[i] - self.center)
                    violation = max(0, self.radius - distance)
                    
                    if violation > 0:
                        softplus_value = self.outer._strategy.softplus_tau * np.log1p(np.exp(violation / self.outer._strategy.softplus_tau))
                        constraint_residuals[i] = np.sqrt(self.outer._strategy.constraint_penalty_factor_1) * softplus_value
                
                # 2. 深度约束残差：g_z(θ) = √λ₂ * softplus_τ(ε + z̄_max - c_z)
                max_z = np.max(self.outer._sampled_points[:, 2])
                depth_offset = 0.0002  # 200微米深度偏移
                depth_violation = max(0, depth_offset + max_z - self.center[2])
                
                if depth_violation > 0:
                    softplus_value = self.outer._strategy.softplus_tau * np.log1p(np.exp(depth_violation / self.outer._strategy.softplus_tau))
                    constraint_residuals[N] = np.sqrt(self.outer._strategy.constraint_penalty_factor_2) * softplus_value
                
                return constraint_residuals
            
            def _update_weight_manager_state(self, data_residuals: np.ndarray) -> None:
                """更新权重管理器状态"""
                if hasattr(self.outer, '_last_params') and hasattr(self.outer, '_last_residuals'):
                    param_change = np.linalg.norm(self.params - self.outer._last_params)
                    residual_change = np.sum(data_residuals - self.outer._last_residuals) / len(self.outer._sampled_points)
                else:
                    param_change = 0.0
                    residual_change = 0.0
                                        
                self._weight_manager.should_enable_geometric_weights(
                    param_change=param_change,
                    residual_change=residual_change,
                    param_tolerance=self.outer._strategy.param_tolerance,
                    residual_tolerance=self.outer._strategy.residual_tolerance,
                    geometric_weight_factor=10.0  # 默认值
                )
                
                # 保存当前状态
                self.outer._last_params = self.params.copy()
                self.outer._last_residuals = data_residuals.copy()
            
            def _solve_lm_equation(self, data_residuals: np.ndarray, constraint_residuals: np.ndarray) -> Optional[np.ndarray]:
                """求解LM更新方程：(J̃^T J̃ + λ_LM I)Δ = -J̃^T r̃"""
                
                def calculate_jacobian_f() -> np.ndarray:
                    """计算数据残差雅可比矩阵 J_f"""
                    if len(self.outer._sampled_points) == 0:
                        return np.array([]).reshape(0, 4)
                    
                    N = len(self.outer._sampled_points)
                    J_f = np.zeros((N, 4))
                    
                    for i in range(N):
                        point = self.outer._sampled_points[i]
                        distance = np.linalg.norm(point - self.center)
                        
                        if distance > 1e-8:  # 避免除零
                            J_f[i, 0] = (point[0] - self.center[0]) / distance
                            J_f[i, 1] = (point[1] - self.center[1]) / distance  
                            J_f[i, 2] = (point[2] - self.center[2]) / distance
                            J_f[i, 3] = -1.0
                    
                    return J_f
                
                def calculate_jacobian_g() -> np.ndarray:
                    """计算约束残差雅可比矩阵 J_g"""
                    if len(self.outer._sampled_points) == 0:
                        return np.array([]).reshape(0, 4)
                    
                    N = len(self.outer._sampled_points)
                    constraint_dim = 1  # 深度约束维度
                    J_g = np.zeros((N + constraint_dim, 4))
                    
                    # 1. 点约束雅可比
                    for i in range(N):
                        point = self.outer._sampled_points[i]
                        distance = np.linalg.norm(point - self.center)
                        violation = max(0, self.radius - distance)
                        
                        if violation > 0:
                            exp_term = np.exp(violation / self.outer._strategy.softplus_tau)
                            softplus_derivative = exp_term / (1 + exp_term)
                            
                            if distance > 1e-8:
                                J_g[i, 0] = np.sqrt(self.outer._strategy.constraint_penalty_factor_1) * softplus_derivative * (point[0] - self.center[0]) / distance
                                J_g[i, 1] = np.sqrt(self.outer._strategy.constraint_penalty_factor_1) * softplus_derivative * (point[1] - self.center[1]) / distance
                                J_g[i, 2] = np.sqrt(self.outer._strategy.constraint_penalty_factor_1) * softplus_derivative * (point[2] - self.center[2]) / distance
                                J_g[i, 3] = np.sqrt(self.outer._strategy.constraint_penalty_factor_1) * softplus_derivative
                    
                    # 2. 深度约束雅可比
                    max_z = np.max(self.outer._sampled_points[:, 2])
                    depth_offset = 0.0002  # 200微米深度偏移
                    depth_violation = max(0, depth_offset + max_z - self.center[2])
                    
                    if depth_violation > 0:
                        exp_term = np.exp(depth_violation / self.outer._strategy.softplus_tau)
                        softplus_derivative = exp_term / (1 + exp_term)
                        J_g[N, 2] = -np.sqrt(self.outer._strategy.constraint_penalty_factor_2) * softplus_derivative
                    
                    return J_g
                
                def calculate_weight_matrix() -> np.ndarray:
                    """计算权重矩阵 W"""
                    N = len(self.outer._sampled_points)
                    if N == 0:
                        return np.array([]).reshape(0, 0)
                    
                    distances = np.array([
                        np.linalg.norm(self.outer._sampled_points[i] - self.center) 
                        for i in range(N)
                    ])
                    
                    residuals = np.array([
                        abs(distances[i] - self.radius) 
                        for i in range(N)
                    ])
                    
                    weight_matrix = self._weight_manager.calculate_weight_matrix(
                        point_types=self.outer._sampled_point_types,
                        distances=distances,
                        residuals=residuals
                    )
                    
                    return weight_matrix
                
                try:
                    # 1. 计算雅可比矩阵
                    J_f = calculate_jacobian_f()
                    J_g = calculate_jacobian_g()
                    
                    # 2. 计算权重矩阵
                    weight_matrix = calculate_weight_matrix()
                    
                    # 3. 计算Hessian矩阵：J_f^T W J_f + J_g^T J_g
                    H_data = J_f.T @ weight_matrix @ J_f
                    H_constraint = J_g.T @ J_g
                    H_total = H_data + H_constraint
                    
                    # 4. 计算梯度：-(J_f^T W f + J_g^T g)
                    g_data = J_f.T @ weight_matrix @ data_residuals
                    g_constraint = J_g.T @ constraint_residuals
                    g_total = -(g_data + g_constraint)
                    
                    # 5. 添加LM阻尼项：(H_total + λ_LM I)Δ = g_total
                    if not hasattr(self.outer._strategy, 'lambda_lm'):
                        self.outer._strategy.lambda_lm = 0.01
                    H_lm = H_total + self.outer._strategy.lambda_lm * np.eye(H_total.shape[0])
                    
                    # 6. 求解线性方程组
                    delta = np.linalg.solve(H_lm, g_total)
                    
                    return delta
                    
                except np.linalg.LinAlgError:
                    logging.warning("LM方程求解失败：线性代数错误")
                    return None
                except Exception as e:
                    logging.error(f"LM方程求解异常: {e}")
                    return None
            
            def _check_parameters_valid(self, params: np.ndarray) -> Tuple[bool, bool]:
                """检查参数有效性"""
                center = params[:3]
                radius = params[3]
                
                # 检查半径是否在合理范围内
                min_radius = ANATOMICAL_CONSTRAINTS["EYEBALL_RADIUS_MIN"]
                max_radius = ANATOMICAL_CONSTRAINTS["EYEBALL_RADIUS_MAX"]
                
                if radius < min_radius or radius > max_radius:
                    return False, False
                
                # 检查所有点是否满足约束
                for i in range(len(self.outer._sampled_points)):
                    if not self._check_single_point_constraint(
                        self.outer._sampled_points[i], center, radius, 
                        self.outer._sampled_point_types[i]
                    ):
                        return True, False
                
                return True, True
            
            def _check_single_point_constraint(self, point: np.ndarray, center: np.ndarray, 
                                             radius: float, point_type: str) -> bool:
                """检查单点约束"""
                # 计算点到球心的距离
                distance_to_center = np.linalg.norm(point - center)
                
                # 计算点到眼球表面的距离
                surface_distance = abs(distance_to_center - radius)
                
                # 根据点类型获取正常范围
                if point_type == "pupil_center":
                    min_dist = ANATOMICAL_CONSTRAINTS["PUPIL_TO_SURFACE_MIN"]
                    max_dist = ANATOMICAL_CONSTRAINTS["PUPIL_TO_SURFACE_MAX"]
                elif point_type == "iris_boundaries":
                    min_dist = ANATOMICAL_CONSTRAINTS["IRIS_TO_SURFACE_MIN"]
                    max_dist = ANATOMICAL_CONSTRAINTS["IRIS_TO_SURFACE_MAX"]
                elif point_type == "eye_contour":
                    min_dist = ANATOMICAL_CONSTRAINTS["CONTOUR_TO_SURFACE_MIN"]
                    max_dist = ANATOMICAL_CONSTRAINTS["CONTOUR_TO_SURFACE_MAX"]
                else:
                    return False  # 未知类型，默认不通过
                
                # 检查距离是否在正常范围内
                return min_dist <= surface_distance <= max_dist
            
            def _check_convergence(self) -> bool:
                """检查LM算法收敛性"""
                # 1. 参数收敛检查：||Δθ|| < param_tolerance，||Δr|| < residual_tolerance
                if hasattr(self.outer, '_last_params') and hasattr(self.outer, '_last_residuals'):
                    param_change = np.linalg.norm(self.params - self.outer._last_params)
                    residual_change = np.sum(self.current_residual - self.outer._last_residuals) / len(self.outer._sampled_points)
                    if param_change < self.outer._strategy.param_tolerance and residual_change < self.outer._strategy.residual_tolerance:
                        logging.info(f"✅ LM收敛：参数变化{param_change:.2e} < {self.outer._strategy.param_tolerance}，平均残差变化{residual_change:.2e} < {self.outer._strategy.residual_tolerance}")
                        return True
                    # 2. 最大迭代次数检查
                    if self.iteration_count >= self.outer._strategy.max_lm_iterations - 1:
                        logging.warning(f"⚠️ LM达到最大迭代次数: {self.outer._strategy.max_lm_iterations}")
                        logging.warning(f"⚠️ LM优化未收敛: 参数变化{param_change:.2e} 参数变化阈值{self.outer._strategy.param_tolerance}，残差变化{residual_change:.2e} 残差变化阈值{self.outer._strategy.residual_tolerance}")
                        return False
                return False
            
            def _update_final_state(self) -> None:
                """更新最终状态"""
                # 计算最终残差
                final_data_residuals = self._calculate_fitting_residual()
                self.current_residual = np.linalg.norm(final_data_residuals)
                
                # 计算内点比例（简化版本）
                if len(self.outer._sampled_points) > 0:
                    self.inlier_ratio = 1.0  # 约束优化中所有点都是内点
                
                # 计算质量分数
                self.quality_score = max(0.0, 1.0 - self.current_residual / 0.01)  # 归一化到[0,1]
                
                # 保存到外部实例
                self.outer._convergence_iterations = self.iteration_count
                self.outer._is_converged = self.is_converged
                self.outer._current_residual = self.current_residual
                self.outer._inlier_ratio = self.inlier_ratio
                self.outer._quality_score = self.quality_score
                self.outer._warnings = self.warnings.copy()
        
        if len(self._sampled_points) == 0:
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
        # 1. 获取虹膜边界点
        iris_points = self._coords_3d.get("iris_boundaries", np.array([]))
        logging.info(f"🔍 虹膜边界点数量: {len(iris_points)}")
        
        if len(iris_points) < 3:
            logging.warning("⚠️ 虹膜点不足3个，使用备选初始化")
            return self._fallback_initialization()
        
        # 2. 随机选择3个虹膜点进行拟合
        if len(iris_points) > 3:
            # 随机选择3个点，提高鲁棒性
            indices = np.random.choice(len(iris_points), 3, replace=False)
            selected_points = iris_points[indices]
        else:
            selected_points = iris_points
        
        # 3. 拟合外心（三个点确定的圆的圆心）
        logging.info("🔧 开始2D圆拟合...")
        center_2d, radius_2d = self._fit_circle_2d(selected_points)
        
        if center_2d is None:
            logging.warning("⚠️ 2D圆拟合失败，使用备选初始化")
            return self._fallback_initialization()
        
        # 4. 计算3D球心：使用2D圆心的x,y坐标，z坐标基于解剖学约束
        # 虹膜边界到眼球表面的距离范围：0.5-1.2mm
        iris_to_surface_min = ANATOMICAL_CONSTRAINTS["IRIS_TO_SURFACE_MIN"]
        iris_to_surface_max = ANATOMICAL_CONSTRAINTS["IRIS_TO_SURFACE_MAX"]
        
        # 5. 球半径设置为默认值
        radius_3d = EYEBALL_RADIUS
        # 选择中间值作为中值距离加半径作为d
        
        d = (iris_to_surface_min + iris_to_surface_max) / 2 + radius_3d
        
        # 6. 通过勾股定理计算深度偏移z
        # 勾股定理：d² = r² + z²
        # 其中：d是中值距离+半径（斜边），r是2D圆的半径（直角边），z是深度偏移（直角边）
        # 所以：z = sqrt(d² - r²)
        if d > radius_2d:
            logging.info(f"🔍 中值距离+半径: {d:.3f}m，2D半径: {radius_2d:.3f}m，使用虹膜深度偏移")
            z = np.sqrt(d**2 - radius_2d**2)
        else:
            # 如果中值距离小于2D半径，说明约束不合理，使用默认偏移
            logging.warning(f"中值距离{d:.6f}小于2D半径{radius_2d:.6f}，使用默认偏移")
            z = iris_to_surface_min
        
        # 球心z坐标：虹膜点的平均z坐标 + 深度偏移（远离摄像机）
        avg_z = np.mean(selected_points[:, 2])
        center_3d = np.array([center_2d[0], center_2d[1], avg_z + z])
        
        
        
        # 7. 验证半径是否在合理范围内
        min_radius = ANATOMICAL_CONSTRAINTS["EYEBALL_RADIUS_MIN"]
        max_radius = ANATOMICAL_CONSTRAINTS["EYEBALL_RADIUS_MAX"]
        
        if radius_3d < min_radius or radius_3d > max_radius:
            logging.warning(f"⚠️ 计算半径{radius_3d:.6f}超出范围[{min_radius:.6f}, {max_radius:.6f}]，使用默认值")
            radius_3d = ANATOMICAL_CONSTRAINTS["EYEBALL_RADIUS_DEFAULT"]
        
        logging.info(f"✅ 智能初始化完成: 2D圆心=({center_2d[0]:.3f}, {center_2d[1]:.3f}), 3D球心=({center_3d[0]:.3f}, {center_3d[1]:.3f}, {center_3d[2]:.3f})")
        logging.info(f"📏 2D半径={radius_2d:.3f}m, 3D半径={radius_3d:.3f}m, 深度偏移={z:.3f}m")
        
        return center_3d, radius_3d
    
    def _fit_circle_2d(self, points: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """基于三个点拟合2D圆的圆心和半径"""
        if len(points) != 3:
            return None, None
        
        try:
            # 提取x,y坐标
            x_coords = points[:, 0]
            y_coords = points[:, 1]
            
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
            
            # 求解线性方程组
            solution = np.linalg.solve(A, b)
            
            # 提取圆心坐标和半径
            center_x = solution[0]
            center_y = solution[1]
            center_2d = np.array([center_x, center_y])
            
            # 计算半径：r = sqrt(x0² + y0² + c)，其中c = r² - x0² - y0²
            c = solution[2]
            radius_2d = np.sqrt(center_x**2 + center_y**2 + c)
            
            # 验证结果的有效性
            if np.isnan(radius_2d) or radius_2d <= 0:
                logging.warning("2D圆拟合结果无效")
                return None, None
            
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
            initial_radius = EYEBALL_RADIUS
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
        inlier_count = len(self._sampled_points)
        total_points = len(self._sampled_points)
        inlier_ratio = 1.0 if total_points > 0 else 0.0
        
        # 计算最终残差
        if hasattr(self, '_current_residual'):
            final_residual = self._current_residual
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
