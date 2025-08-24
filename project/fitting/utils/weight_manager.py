"""
权重管理模块
统一管理解剖学权重、几何权重和权重融合策略
"""
from typing import Dict, Optional, Tuple
import numpy as np
from project.fitting.config.initial_weight import (
    ANATOMICAL_WEIGHT_PARAMS, 
    GEOMETRIC_WEIGHT_PARAMS, 
    GLOBAL_WEIGHT_CONFIG
)
from project.fitting.config.fitting_strategy import WeightConfig
import logging
class WeightManager:
    """
    权重管理器
    统一管理解剖学权重、几何权重和权重融合策略
    """
    
    def __init__(self, config: WeightConfig):
        """
        初始化权重管理器
        
        Args:
            config: 权重配置参数
        """
        self.config = config
        self._geometric_params = GEOMETRIC_WEIGHT_PARAMS
        self._global_config = GLOBAL_WEIGHT_CONFIG
        
        # 权重状态管理
        self._current_weights: Optional[np.ndarray] = None
        self._geometric_weights: Optional[np.ndarray] = None
        self._image_confidence_weights: Optional[np.ndarray] = None
        self._point_types: Optional[list] = None
        self._last_update_iteration: int = -1
        self._geometric_weight_enabled: bool = False
        
        # 图像置信度权重（visibility）
        self._image_confidence: Optional[np.ndarray] = None
        
    
    
    def _calculate_geometric_weight(self, point_type: str, residual: float) -> float:
        """
        基于高斯公式计算几何权重 - 按照fitting_strategy.md中的公式实现
        
        权重公式：w_i = exp(-f_i(θ)²/(2σ²))
        其中 f_i(θ) 是拟合残差，σ 是配置的高斯标准差
        
        Args:
            point_type: 点类型字符串 ("pupil_center", "iris_boundary", "eye_contour")
            residual: 拟合残差
            
        Returns:
            float: 几何权重值
        """
            
        if point_type in self._geometric_params:
            params = self._geometric_params[point_type]
            sigma = params.initial_sigma              # 标准差
            min_weight = params.min_weight            # 最小权重
            
            # 高斯公式：w_i = exp(-f_i(θ)²/(2σ²))
            weight = np.exp(-(residual ** 2) / (2 * sigma ** 2))
            
            # 应用最小权重限制
            return max(min_weight, weight)
        else:
            # 默认几何权重计算
            default_sigma = 0.0015
            weight = np.exp(-(residual ** 2) / (2 * default_sigma ** 2))
            return max(0.2, weight)
    
    def _gaussian_weight_fusion(self, image_weight: float, geometric_weight: float) -> float:
        """
        高斯权重融合策略 - 按照新的权重策略实现
        
        使用高斯核函数融合图像置信度权重和几何权重：
        w_fused = K(w_img, w_geom) * w_img + (1-K(w_img, w_geom)) * w_geom
        
        其中：K(w_img, w_geom) = exp(-(w_img - w_geom)²/(2σ²))
        
        Args:
            image_weight: 图像置信度权重（visibility）
            geometric_weight: 几何权重
            
        Returns:
            float: 融合后的权重
        """
        # 计算权重差异
        weight_diff = image_weight - geometric_weight
        
        # 高斯核函数：K(w_img, w_geom) = exp(-(w_img - w_geom)²/(2σ²))
        kernel = np.exp(-(weight_diff ** 2) / (2 * self.config.gaussian_fusion_sigma ** 2))
        
        # 融合权重：w_fused = K * w_img + (1-K) * w_geom
        fused_weight = kernel * image_weight + (1 - kernel) * geometric_weight
        
        return fused_weight
    
    def set_image_confidence(self, image_confidence: np.ndarray) -> None:
        """
        设置图像置信度权重（visibility）
        
        Args:
            image_confidence: 图像置信度数组，形状为(N,)
        """
        self._image_confidence = image_confidence.copy()
        logging.debug(f"🔧 设置图像置信度权重: {len(image_confidence)}个点")
    
    def get_image_confidence(self) -> Optional[np.ndarray]:
        """
        获取图像置信度权重
        
        Returns:
            np.ndarray: 图像置信度数组，如果未设置则返回None
        """
        return self._image_confidence.copy() if self._image_confidence is not None else None
    
    def _calculate_final_weight(self, point_type: str, residual: float, visibility: float) -> float:
        """
        计算最终权重 - 新的权重策略：图像置信度 + 几何权重的高斯融合
        
        Args:
            point_type: 点类型字符串
            residual: 拟合残差
            visibility: 图像置信度权重（visibility）
            
        Returns:
            float: 最终权重值
        """
        # 1. 直接使用传入的visibility权重
        image_weight = visibility
        return max(self.config.min_weight, image_weight)
        # 2. 如果未启用几何权重，直接返回图像置信度权重
        if not self._geometric_weight_enabled:
            logging.warning("⚠️ 权重管理器：没有启用几何权重，使用默认全1权重")
            return max(self.config.min_weight, image_weight)
        
        logging.info("🔍 权重管理器：启用几何权重，计算几何权重")
        # 3. 启用几何权重后，计算几何权重
        geometric_weight = self._calculate_geometric_weight(point_type, residual)
        
        # 4. 使用高斯融合策略融合图像置信度和几何权重
        if self.config.weight_fusion_enabled:
            final_weight = self._gaussian_weight_fusion(image_weight, geometric_weight)
        else:
            # 简单乘法策略
            final_weight = image_weight * geometric_weight
        
        # 5. 应用最小权重限制
        final_weight = max(self.config.min_weight, final_weight)
        
        logging.debug(f"🔍 类型={point_type}, 残差={residual:.6f}, 图像置信度={image_weight:.6f}, 几何权重={geometric_weight:.6f}, 最终权重={final_weight:.6f}")
        return final_weight
    
    def calculate_weight_matrix(self, point_types: list, residuals: np.ndarray, visibility: np.ndarray) -> np.ndarray:
        """
        批量计算权重矩阵 - 新的权重策略：图像置信度 + 几何权重
        
        Args:
            point_types: 点类型列表
            residuals: 残差数组 (N,)
            visibility: 图像置信度数组 (N,)
        Returns:
            np.ndarray: 权重矩阵，形状为(N, N)
        """
        N = len(point_types)
        if N == 0:
            return np.array([]).reshape(0, 0)
        
        # 直接使用传入的visibility，不再依赖内部存储
        if visibility is None or len(visibility) != N:
            logging.warning("⚠️ 权重管理器：没有维护visibility信息，使用默认全1权重")
            visibility = np.ones(N)  # 默认权重
        
        # 计算每个点的权重
        weights = np.zeros(N)
        for i in range(N):
            weights[i] = self._calculate_final_weight(
                point_types[i], 
                residuals[i], 
                visibility[i]
            )
        
        # 权重归一化
        if self.config.normalization and np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        
        # 构建对角权重矩阵
        weight_matrix = np.diag(weights)
        
        # 缓存权重信息
        self._current_weights = weights.copy()
        self._geometric_weights = np.array([
            self._calculate_geometric_weight(point_types[i], residuals[i]) 
            for i in range(N)
        ])
        self._image_confidence_weights = visibility.copy()
        self._point_types = point_types.copy()
        
        return weight_matrix
    
    def calculate_extended_weight_matrix(self, point_types: list, residuals: np.ndarray, 
                                       visibility: np.ndarray, constraint_violations: np.ndarray, lambda_depth: float) -> np.ndarray:
        """
        计算扩展权重矩阵 - 包含数据权重和约束权重
        
        用于目标函数 Φ(θ) = F(θ) + (1/2)λ_depth * h(θ)²
        
        Args:
            point_types: 点类型列表
            residuals: 残差数组 (N,)
            visibility: 图像置信度数组 (N,)
            constraint_violations: 约束违反值数组 (M,)
            lambda_depth: 深度约束权重系数
            
        Returns:
            np.ndarray: 扩展权重矩阵，形状为(N+M, N+M)
        """
        N = len(point_types)
        M = len(constraint_violations)
        total_dim = N + M
        
        if total_dim == 0:
            return np.array([]).reshape(0, 0)
        
        # 直接使用传入的visibility，不再依赖内部存储
        if visibility is None or len(visibility) != N:
            visibility = np.ones(N)  # 默认权重
        
        # 计算数据权重
        data_weights = np.zeros(N)
        for i in range(N):
            data_weights[i] = self._calculate_final_weight(
                point_types[i], 
                residuals[i], 
                visibility[i]
            )
        
        # 计算约束权重
        constraint_weights = self.calculate_constraint_weights(constraint_violations, lambda_depth)
        
        # 构建扩展权重矩阵
        # [W_data    0    ]
        # [0     W_constraint]
        weight_matrix = np.zeros((total_dim, total_dim))
        weight_matrix[:N, :N] = np.diag(data_weights)
        weight_matrix[N:, N:] = np.diag(constraint_weights)
        
        # 缓存权重信息
        self._current_weights = np.concatenate([data_weights, constraint_weights])
        self._geometric_weights = np.array([
            self._calculate_geometric_weight(point_types[i], residuals[i]) 
            for i in range(N)
        ])
        self._image_confidence_weights = visibility.copy()
        self._point_types = point_types.copy()
        
        logging.debug(f"🔧 扩展权重矩阵: 数据维度={N}, 约束维度={M}, 总维度={total_dim}")
        
        return weight_matrix
    
    def adaptive_weight_adjustment(self, current_iteration: int, max_iterations: int,
                                 damping_factor: float, convergence_ratio: float) -> float:
        """
        自适应权重调整 - 根据优化进度和阻尼因子调整权重策略
        
        在LM优化过程中，根据迭代进度和收敛情况动态调整权重
        
        Args:
            current_iteration: 当前迭代次数
            max_iterations: 最大迭代次数
            damping_factor: 当前LM阻尼因子
            convergence_ratio: 收敛比例 (0-1)
            
        Returns:
            float: 权重调整因子
        """
        # 1. 基于迭代进度的权重调整
        iteration_progress = current_iteration / max_iterations
        if iteration_progress < 0.3:
            # 早期阶段：保持解剖学权重，避免过早收敛
            weight_factor = 0.5
        elif iteration_progress < 0.7:
            # 中期阶段：逐步引入几何权重
            weight_factor = 0.5 + 0.5 * (iteration_progress - 0.3) / 0.4
        else:
            # 后期阶段：主要使用几何权重，精细调整
            weight_factor = 1.0
        
        # 2. 基于阻尼因子的权重调整
        # 阻尼因子越大，说明优化困难，需要更保守的权重策略
        damping_adjustment = 1.0 / (1.0 + np.log10(1 + damping_factor))
        weight_factor *= damping_adjustment
        
        # 3. 基于收敛比例的权重调整
        # 收敛比例越高，越可以激进地使用几何权重
        convergence_adjustment = 0.5 + 0.5 * convergence_ratio
        weight_factor *= convergence_adjustment
        
        # 限制权重因子范围
        weight_factor = np.clip(weight_factor, 0.1, 1.0)
        
        logging.debug(f"🔄 自适应权重调整: 迭代{current_iteration}/{max_iterations}, "
                     f"阻尼因子={damping_factor:.2e}, 收敛比例={convergence_ratio:.2f}, "
                     f"权重因子={weight_factor:.3f}")
        
        return weight_factor
    
    def should_enable_geometric_weights(self, param_change: float, residual_change: float, 
                                       param_tolerance: float, residual_tolerance: float,
                                       geometric_weight_factor: float) -> bool:
        """
        判断是否应该启用几何权重 - 按照fitting_strategy.md中的策略实现
        
        几何权重启用条件：
        当残差下降到一定阈值时启用几何权重，允许个体差异被逐步吸收
        
        Args:
            param_change: 参数变化量
            residual_change: 残差变化量
            param_tolerance: 参数收敛阈值
            residual_tolerance: 残差收敛阈值
            geometric_weight_factor: 几何权重启用因子
            
        Returns:
            bool: 是否启用几何权重
        """
        # 按照fitting_strategy.md的策略：
        # 当残差变化在合理范围内时启用几何权重
        residual_condition = (
            residual_tolerance < abs(residual_change) < 
            residual_tolerance * geometric_weight_factor
        )
        
        # 参数变化也需要在合理范围内
        param_condition = (
            param_tolerance < param_change < 
            param_tolerance * geometric_weight_factor
        )
        
        # 两个条件都满足时才启用几何权重
        self._geometric_weight_enabled = param_condition and residual_condition
        
        if self._geometric_weight_enabled:
            logging.debug(f"✅ 启用几何权重: 参数变化={param_change:.2e}, 残差变化={residual_change:.2e}")
        else:
            logging.debug(f"⏸️ 保持图像置信度权重: 参数变化={param_change:.2e}, 残差变化={residual_change:.2e}")
        
        return self._geometric_weight_enabled
    
    def get_weight_statistics(self) -> Dict[str, any]:
        """
        获取权重统计信息
        
        Returns:
            Dict: 权重统计信息
        """
        if self._current_weights is None:
            return {}
        
        return {
            "total_points": len(self._current_weights),
            "mean_weight": np.mean(self._current_weights),
            "std_weight": np.std(self._current_weights),
            "min_weight": np.min(self._current_weights),
            "max_weight": np.max(self._current_weights),
            "image_confidence_weights": self._image_confidence_weights.tolist() if self._image_confidence_weights is not None else [],
            "geometric_weights": self._geometric_weights.tolist() if self._geometric_weights is not None else [],
            "point_types": self._point_types
        }
    
    def reset_weights(self) -> None:
        """重置权重状态"""
        self._current_weights = None
        self._geometric_weights = None
        self._image_confidence_weights = None
        self._point_types = None
        self._last_update_iteration = -1
    
    def update_config(self, new_config: WeightConfig) -> None:
        """
        更新权重配置
        
        Args:
            new_config: 新的权重配置
        """
        self.config = new_config
        self.reset_weights()
    
    def update_weights_for_damping(self, current_residuals: np.ndarray, 
                                  damping_factor: float, 
                                  base_sigma: float) -> np.ndarray:
        """
        为阻尼因子优化更新权重 - 支持LM算法的阻尼调整
        
        当LM算法增加阻尼因子时，需要调整权重以保持数值稳定性
        
        Args:
            current_residuals: 当前残差向量
            damping_factor: 当前LM阻尼因子
            base_sigma: 基础高斯标准差
            
        Returns:
            np.ndarray: 调整后的权重向量
        """
        # 根据阻尼因子调整sigma
        # 阻尼因子越大，权重分布越均匀（避免过拟合）
        adjusted_sigma = base_sigma * (1 + np.log10(1 + damping_factor))
        
        # 计算调整后的几何权重
        adjusted_weights = np.exp(-current_residuals**2 / (2 * adjusted_sigma**2))
        
        # 应用最小权重限制
        adjusted_weights = np.clip(adjusted_weights, self.config.min_weight, 1.0)
        
        # 记录权重调整信息
        logging.debug(f"🔄 阻尼因子权重调整: λ={damping_factor:.2e}, σ={adjusted_sigma:.6f}")
        
        return adjusted_weights
    
    def calculate_constraint_weights(self, constraint_violations: np.ndarray, 
                                   lambda_depth: float) -> np.ndarray:
        """
        计算约束权重 - 支持深度约束的权重计算
        
        按照目标函数 Φ(θ) = F(θ) + (1/2)λ_depth * h(θ)²
        其中 h(θ) 是深度约束违反值
        
        Args:
            constraint_violations: 约束违反值向量
            lambda_depth: 深度约束权重系数
            
        Returns:
            np.ndarray: 约束权重向量
        """
        # 约束权重：w_constraint = √(λ_depth * h(θ))
        # 当约束违反时，权重增加以加强惩罚
        constraint_weights = np.sqrt(lambda_depth * np.abs(constraint_violations))
        
        # 应用权重上限，避免数值不稳定
        max_weight = 100.0  # 最大约束权重
        constraint_weights = np.clip(constraint_weights, 0.0, max_weight)
        
        logging.debug(f"🔒 约束权重计算: λ_depth={lambda_depth}, 最大权重={np.max(constraint_weights):.2f}")
        
        return constraint_weights

# 默认权重配置
DEFAULT_WEIGHT_CONFIG = WeightConfig()

# 默认权重管理器
DEFAULT_WEIGHT_MANAGER = WeightManager(DEFAULT_WEIGHT_CONFIG)