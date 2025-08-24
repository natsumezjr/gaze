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
        self._anatomical_params = ANATOMICAL_WEIGHT_PARAMS
        self._geometric_params = GEOMETRIC_WEIGHT_PARAMS
        self._global_config = GLOBAL_WEIGHT_CONFIG
        
        # 权重状态管理
        self._current_weights: Optional[np.ndarray] = None
        self._anatomical_weights: Optional[np.ndarray] = None
        self._geometric_weights: Optional[np.ndarray] = None
        self._point_types: Optional[list] = None
        self._last_update_iteration: int = -1
        self._geometric_weight_enabled: bool = False
        
    
    def _calculate_anatomical_weight(self, point_type: str, distance: float) -> float:
        """
        基于高斯公式计算解剖学权重
        
        权重公式：w = exp(-(d - d_median)²/(2σ²))
        
        Args:
            point_type: 点类型字符串 ("pupil_center", "iris_boundary", "eye_contour")
            distance: 当前点到眼球表面的距离
            
        Returns:
            float: 解剖学权重值
        """
            
        if point_type in self._anatomical_params:
            params = self._anatomical_params[point_type]
            median_distance = params.median_distance  # 中值距离
            sigma = params.sigma                      # 标准差
            
            # 高斯公式：w = exp(-(d - d_median)²/(2σ²))
            weight = np.exp(-((distance - median_distance) ** 2) / (2 * sigma ** 2))
            return weight
        else:
            # 默认权重
            return 1.0
    
    def _calculate_geometric_weight(self, point_type: str, residual: float) -> float:
        """
        基于高斯公式计算几何权重
        
        权重公式：w = exp(-residual²/(2σ²))
        
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
            
            # 高斯公式：w = exp(-residual²/(2σ²))
            weight = np.exp(-(residual ** 2) / (2 * sigma ** 2))
            
            # 应用最小权重限制
            return max(min_weight, weight)
        else:
            # 默认几何权重计算
            default_sigma = 0.0015
            weight = np.exp(-(residual ** 2) / (2 * default_sigma ** 2))
            return max(0.2, weight)
    
    def _gaussian_weight_fusion(self, anatomical_weight: float, geometric_weight: float) -> float:
        """
        高斯权重融合策略
        
        使用高斯核函数融合解剖学权重和几何权重：
        w_fused = K(w_anat, w_geom) * w_anat + (1-K(w_anat, w_geom)) * w_geom
        
        其中：K(w_anat, w_geom) = exp(-(w_anat - w_geom)²/(2σ²))
        
        Args:
            anatomical_weight: 解剖学权重
            geometric_weight: 几何权重
            
        Returns:
            float: 融合后的权重
        """
        # 计算权重差异
        weight_diff = anatomical_weight - geometric_weight
        
        # 高斯核函数：K(w_anat, w_geom) = exp(-(w_anat - w_geom)²/(2σ²))
        kernel = np.exp(-(weight_diff ** 2) / (2 * self.config.gaussian_fusion_sigma ** 2))
        
        # 融合权重：w_fused = K * w_anat + (1-K) * w_geom
        fused_weight = kernel * anatomical_weight + (1 - kernel) * geometric_weight
        
        return fused_weight
    
    def _calculate_final_weight(self, point_type: str, distance: float, residual: float) -> float:
        """
        计算最终权重，集成解剖学权重和几何权重
        
        Args:
            point_type: 点类型字符串
            distance: 点到眼球表面的距离
            residual: 拟合残差
            
        Returns:
            float: 最终权重值
        """
        # 1. 计算解剖学权重
        anatomical_weight = self._calculate_anatomical_weight(point_type, distance)
        
        # 2. 计算几何权重
        geometric_weight = self._calculate_geometric_weight(point_type, residual)
        
        # 3. 判断现在是否需要启用几何权重
        if not self._geometric_weight_enabled:
            return anatomical_weight
        
        # 4. 权重融合
        if self.config.weight_fusion_enabled:
            final_weight = self._gaussian_weight_fusion(anatomical_weight, geometric_weight)
        else:
            # 简单乘法策略
            final_weight = anatomical_weight * geometric_weight
        
        # 4. 应用最小权重限制
        final_weight = max(self.config.min_weight, final_weight)
        
        return final_weight
    
    def calculate_weight_matrix(self, point_types: list, distances: np.ndarray, 
                               residuals: np.ndarray) -> np.ndarray:
        """
        批量计算权重矩阵
        
        Args:
            point_types: 点类型列表
            distances: 距离数组 (N,)
            residuals: 残差数组 (N,)
        Returns:
            np.ndarray: 权重矩阵，形状为(N, N)
        """
        N = len(point_types)
        if N == 0:
            return np.array([]).reshape(0, 0)
        
        # 计算每个点的权重
        weights = np.zeros(N)
        for i in range(N):
            weights[i] = self._calculate_final_weight(
                point_types[i], 
                distances[i], 
                residuals[i]
            )
        
        # 权重归一化
        if self.config.normalization and np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        
        # 构建对角权重矩阵
        weight_matrix = np.diag(weights)
        
        # 缓存权重信息
        self._current_weights = weights.copy()
        self._anatomical_weights = np.array([
            self._calculate_anatomical_weight(point_types[i], distances[i]) 
            for i in range(N)
        ])
        self._geometric_weights = np.array([
            self._calculate_geometric_weight(point_types[i], residuals[i]) 
            for i in range(N)
        ])
        self._point_types = point_types.copy()
        
        return weight_matrix
    
    def should_enable_geometric_weights(self, param_change: float, residual_change: float, 
                                       param_tolerance: float, residual_tolerance: float,
                                       geometric_weight_factor: float) -> bool:
        """
        判断是否应该启用几何权重
        
        几何权重启用条件：
        1. param_tolerance < ||Δθ|| < param_tolerance × geometric_weight_factor
        2. residual_tolerance < |Δr| < residual_tolerance × geometric_weight_factor
        
        Args:
            param_change: 参数变化量
            residual_change: 残差变化量
            param_tolerance: 参数收敛阈值
            residual_tolerance: 残差收敛阈值
            geometric_weight_factor: 几何权重启用因子
            
        Returns:
            bool: 是否启用几何权重
        """
        # 检查参数收敛条件
        param_condition = (
            param_tolerance < param_change < 
            param_tolerance * geometric_weight_factor
        )
        
        # 检查残差收敛条件
        residual_condition = (
            residual_tolerance < residual_change < 
            residual_tolerance * geometric_weight_factor
        )
        
        # 两个条件都满足时才启用几何权重
        self._geometric_weight_enabled = param_condition and residual_condition
        
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
            "anatomical_weights": self._anatomical_weights.tolist() if self._anatomical_weights is not None else [],
            "geometric_weights": self._geometric_weights.tolist() if self._geometric_weights is not None else [],
            "point_types": self._point_types
        }
    
    def reset_weights(self) -> None:
        """重置权重状态"""
        self._current_weights = None
        self._anatomical_weights = None
        self._geometric_weights = None
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

# 默认权重配置
DEFAULT_WEIGHT_CONFIG = WeightConfig()

# 默认权重管理器
DEFAULT_WEIGHT_MANAGER = WeightManager(DEFAULT_WEIGHT_CONFIG)