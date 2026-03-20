# 眼球球面拟合模块
"""
拟合策略配置模块 - 重构版本
分层设计：
  1. 核心数据类（无依赖）
  2. 数据管理类
  3. 采样策略类
  4. 残差计算类
  5. 优化器类
  6. 策略类（纯策略，无状态）
  7. 主控制器类
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Dict, Union, Callable
from project.data.data_manager import RecgFitDataManager
from project.data.data_models import EYE_TYPE, Point3DWithVisibility, FittingType, Point3D
import numpy as np
import logging
from project.config.logging_config import setup_logging
logger = setup_logging(__name__)

# ==================== 解剖学约束参数 ====================
ANATOMICAL_CONSTRAINTS = {
    "TO_SURFACE": {
        "pupil": (-4.5, -2.5),
        "iris": (-1.8, -0.4),
        "inner_canthus": (4.0, 8.0),
        "upper_eyelid": (0.0, 2.0),
        "lower_eyelid": (0.2, 1.6),
        "outer_canthus": (2.5, 6.7),
    },
    "TO_CENTER": {
        "pupil": (7.95, 9.95),
        "iris": (10.40, 12.04),
        "inner_canthus": (16.0, 20.0),
        "upper_eyelid": (11.3, 13.3),
        "lower_eyelid": (12.0, 14.0),
        "outer_canthus": (14.9, 18.9),
    },
    "EYEBALL_RADIUS": (11.0, 13.0),
    "EYEBALL_RADIUS_DEFAULT": 12.0,
}

FITTING_ENABLE = ["pupil", "iris"]
OUTSIDE_EYEBALL = ["inner_canthus", "upper_eyelid", "lower_eyelid", "outer_canthus"]
RESIDUAL_TYPE = ["TO_SURFACE", "TO_CENTER"]

# ==================== 核心数据类（无依赖）====================
@dataclass
class FittingParameters:
    """拟合参数 - 纯数据类，无依赖"""
    center: np.ndarray  # 球心坐标 [x, y, z]
    radius: float       # 半径
    sphere_fitting: bool = True  # 是否使用球体拟合
    
    def copy(self) -> 'FittingParameters':
        """创建参数的深拷贝"""
        return FittingParameters(
            center=self.center.copy(),
            radius=self.radius,
            sphere_fitting=self.sphere_fitting
        )
    
    def __len__(self):
        return len(self.center) + len([self.radius])
    
    def to_theta(self) -> np.ndarray:
        """转换为参数向量"""
        return np.concatenate([self.center, [self.radius]])
    
    @classmethod
    def from_theta(cls, theta: np.ndarray, sphere_fitting: bool = True) -> 'FittingParameters':
        """从参数向量创建"""
        return cls(center=theta[:3], radius=theta[3], sphere_fitting=sphere_fitting)
    
    def __str__(self):
        return f"球心：{self.center}, 半径：{self.radius}"

@dataclass
class FittingResult:
    """拟合结果 - 纯数据类，无依赖"""
    parameters: FittingParameters
    confidence: float
    converged: bool
    iteration_count: int
    final_residual_norm: float
    final_delta_norm: float
    strategy_used: str
    
    def get_parameters(self) -> FittingParameters:
        return self.parameters
    
    def __str__(self):
        return f"拟合结果：\n 球心：{self.parameters.center}\n 半径：{self.parameters.radius}\n 置信度：{self.confidence}\n 收敛：{self.converged}\n 迭代次数：{self.iteration_count}\n 最终残差范数：{self.final_residual_norm}\n 最终delta范数：{self.final_delta_norm}\n 使用策略：{self.strategy_used}"
    
    def __repr__(self):
        return self.__str__()

# ==================== 数据管理类 ====================
class DataProvider:
    """数据提供者 - 只负责数据获取，不处理业务逻辑"""
    
    def __init__(self, data_manager: RecgFitDataManager):
        self.data_manager = data_manager
    
    def get_fitting_points(self, eye: str) -> List[Tuple[np.ndarray, str]]:
        """获取拟合点及其类型"""
        points_with_types = []
        for fitting_type in FITTING_ENABLE:
            try:
                points = self.data_manager.get_coordinate_point(eye, fitting_type)
                if points:
                    for point in points:
                        # 验证数据类型和格式
                        if self._is_valid_coordinate_point(point):
                            # 从Point3DWithVisibility对象提取3D坐标
                            coord_3d = point.to_point3d().to_ndarray()
                            points_with_types.append((coord_3d, fitting_type))
                        else:
                            logger.warning(f"跳过无效的拟合点: {point}, 类型: {type(point)}")
            except Exception as e:
                logger.error(f"获取{fitting_type}拟合点时出错: {e}")
        return points_with_types
    
    def get_constraint_points(self, eye: str) -> List[Tuple[np.ndarray, str]]:
        """获取约束点及其类型"""
        points_with_types = []
        for fitting_type in OUTSIDE_EYEBALL:
            try:
                points = self.data_manager.get_coordinate_point(eye, fitting_type)
                if points:
                    for point in points:
                        # 验证数据类型和格式
                        if self._is_valid_coordinate_point(point):
                            # 从Point3DWithVisibility对象提取3D坐标
                            coord_3d = point.to_point3d().to_ndarray()
                            points_with_types.append((coord_3d, fitting_type))
                        else:
                            logger.warning(f"跳过无效的约束点: {point}, 类型: {type(point)}")
            except Exception as e:
                logger.error(f"获取{fitting_type}约束点时出错: {e}")
        return points_with_types
    
    def get_data_quality(self, eye: str) -> Dict[str, float]:
        """获取数据质量指标"""
        try:
            fitting_points = self.get_fitting_points(eye)
            constraint_points = self.get_constraint_points(eye)
            
            total_points = len(fitting_points) + len(constraint_points)
            fitting_count = len(fitting_points)
            constraint_count = len(constraint_points)
            
            logger.debug(f"{eye}眼数据统计:")
            logger.debug(f"  拟合点数量: {fitting_count}")
            logger.debug(f"  约束点数量: {constraint_count}")
            logger.debug(f"  总点数: {total_points}")
            
            # 计算平均可见性 - 直接从数据管理器获取原始数据
            visibility_values = []
            for fitting_type in FITTING_ENABLE + OUTSIDE_EYEBALL:
                points = self.data_manager.get_coordinate_point(eye, fitting_type)
                if points:
                    for point in points:
                        if isinstance(point, Point3DWithVisibility):
                            visibility_values.append(float(point.visibility))
            
            mean_visibility = np.mean(visibility_values) if visibility_values else 0.0
            logger.debug(f"  平均可见性: {mean_visibility:.3f}")
            logger.debug(f"  可见性样本数: {len(visibility_values)}")
            
            quality = {
                "point_count": total_points,
                "fitting_point_count": fitting_count,
                "constraint_point_count": constraint_count,
                "visibility": mean_visibility
            }
            
            logger.debug(f"{eye}眼数据质量: {quality}")
            return quality
            
        except Exception as e:
            logger.error(f"计算{eye}眼数据质量时出错: {e}")
            return {
                "point_count": 0,
                "fitting_point_count": 0,
                "constraint_point_count": 0,
                "visibility": 0.0
            }

    def _is_valid_coordinate_point(self, point) -> bool:
        """验证坐标点是否有效"""
        try:
            # 检查是否为Point3DWithVisibility对象
            if not isinstance(point, Point3DWithVisibility):
                return False
            
            # 检查坐标是否有效
            coord_3d = point.to_point3d()
            coord_array = coord_3d.to_ndarray()
            
            # 检查是否包含NaN或无穷大值
            if np.any(np.isnan(coord_array)) or np.any(np.isinf(coord_array)):
                return False
            
            # 检查可见性是否有效
            if not (0.0 <= point.visibility <= 1.0):
                return False
            
            return True
        except Exception:
            return False

# ==================== 采样策略类 ====================
class SamplingStrategy(ABC):
    """采样策略抽象基类"""
    
    @abstractmethod
    def sample(self, fitting_points: List[Tuple[np.ndarray, str]], 
               constraint_points: List[Tuple[np.ndarray, str]]) -> Tuple[List[Tuple[np.ndarray, str]], List[Tuple[np.ndarray, str]]]:
        """采样方法"""
        pass

class FullSamplingStrategy(SamplingStrategy):
    """全采样策略"""
    
    def sample(self, fitting_points: List[Tuple[np.ndarray, str]], 
               constraint_points: List[Tuple[np.ndarray, str]]) -> Tuple[List[Tuple[np.ndarray, str]], List[Tuple[np.ndarray, str]]]:
        return fitting_points, constraint_points

class RandomSamplingStrategy(SamplingStrategy):
    """随机采样策略"""
    
    def __init__(self, min_fitting_points: int = 4, min_constraint_points: int = 4):
        self.min_fitting_points = min_fitting_points
        self.min_constraint_points = min_constraint_points
    
    def sample(self, fitting_points: List[Tuple[np.ndarray, str]], 
               constraint_points: List[Tuple[np.ndarray, str]]) -> Tuple[List[Tuple[np.ndarray, str]], List[Tuple[np.ndarray, str]]]:
        import random
        
        # 拟合点采样
        if len(fitting_points) >= self.min_fitting_points:
            fitting_sample_num = np.random.randint(self.min_fitting_points, len(fitting_points) + 1)
            logger.debug(f"拟合点随机采样数量：{fitting_sample_num},最小采样数量：{self.min_fitting_points}")
            sampled_fitting = random.sample(fitting_points, fitting_sample_num)
        else:
            sampled_fitting = fitting_points
        
        # 约束点采样
        if len(constraint_points) >= self.min_constraint_points:
            constraint_sample_num = np.random.randint(self.min_constraint_points, len(constraint_points) + 1)
            logger.debug(f"约束点随机采样数量：{constraint_sample_num},最小采样数量：{self.min_constraint_points}")
            sampled_constraint = random.sample(constraint_points, constraint_sample_num)
        else:
            sampled_constraint = constraint_points
        
        return sampled_fitting, sampled_constraint

class FailSamplingStrategy(SamplingStrategy):
    """失败采样策略"""
    
    def sample(self, fitting_points: List[Tuple[np.ndarray, str]], 
               constraint_points: List[Tuple[np.ndarray, str]]) -> Tuple[List[Tuple[np.ndarray, str]], List[Tuple[np.ndarray, str]]]:
        logger.warning("采样失败，返回空列表")
        return [], []

# ==================== 残差计算类 ====================
class ResidualCalculator:
    """残差计算器 - 纯函数式设计"""
    
    def __init__(self, anatomical_constraints: Dict = ANATOMICAL_CONSTRAINTS):
        self.constraints = anatomical_constraints
        self.bias_medians = {"TO_SURFACE": {}, "TO_CENTER": {}}
        self.bias_medians_computed = False
        self._precompute_bias_medians()
        
    
    def _precompute_bias_medians(self):
        """预计算生理偏差中值"""
        if self.bias_medians_computed:
            return
        for key in self.constraints:
            if key in RESIDUAL_TYPE:
                residual_type = key
                for fitting_type in self.constraints[residual_type]:
                    self.bias_medians[residual_type][fitting_type] = (
                        (self.constraints[residual_type][fitting_type][0] + self.constraints[residual_type][fitting_type][1]) / 2
                    )
        self.bias_medians_computed = True
    
    def calculate_residuals(self, parameters: FittingParameters, 
                          points: List[Tuple[np.ndarray, str]]) -> Tuple[np.ndarray, np.ndarray]:
        """计算残差"""
        if not points:
            return np.array([]), np.array([])
        
        # 数据已经在DataProvider中验证过，直接使用
        points_array = np.array([p[0] for p in points])
        point_types = [p[1] for p in points]
        
        # 计算距离
        distances = np.linalg.norm(points_array - parameters.center, axis=1)
        
        # 获取偏差
        surface_bias = np.array([self.bias_medians.get("TO_SURFACE", {}).get(pt, 0.0) for pt in point_types])
        center_bias = np.array([self.bias_medians.get("TO_CENTER", {}).get(pt, 0.0) for pt in point_types])
        
        # 计算残差
        surface_residual = distances - (parameters.radius + surface_bias)
        center_residual = distances - center_bias
        
        return surface_residual, center_residual


# ==================== 目标函数 ====================
def target_function(all_residuals: np.ndarray, alpha: float = 0.5, huber_kappa: float = 0.6) -> float:
    """目标函数：Φ(θ) = (1/2) * Σ r_i^2，其中 r_i 是各个残差项"""
    
    # 简化为平方和形式，因为残差已经在各自的函数中进行了适当的缩放
    total_value = 0.5 * np.sum(all_residuals**2)
    
    return total_value 

# ==================== 置信度计算类 ====================
class ConfidenceCalculator:
    """置信度计算器 - 基于几何约束和正态分布"""
    
    def __init__(self, anatomical_constraints: Dict = ANATOMICAL_CONSTRAINTS, fitting_only: bool = True):
        self.constraints = anatomical_constraints
        self.fitting_only = fitting_only
        self.residual_calculator = ResidualCalculator(anatomical_constraints)
        self.sigma_dict = {}
        self.sigma_dict_computed = False  
        self._precompute_sigma_dict()
        
    
    def _precompute_sigma_dict(self):
        """预计算所有类型点的sigma字典"""
        if self.sigma_dict_computed:
            return
        
        for residual_type in RESIDUAL_TYPE:
            self.sigma_dict[residual_type] = {}
            
            constraints_dict = self.constraints.get(residual_type, {})
            
            for point_type in constraints_dict:
                # 获取生理偏差中值
                mu = (constraints_dict[point_type][0] + constraints_dict[point_type][1]) / 2
                constraints = constraints_dict[point_type]
                
                # 计算最大偏移
                max_offset = (constraints[1] - constraints[0]) / 2
                
                # 解方程计算sigma平方
                sigma_squared = self._solve_sigma_squared(max_offset, 0.9)  # 使用0.9作为优秀阈值
                sigma = np.sqrt(sigma_squared)
                
                # 存储sigma和mu
                self.sigma_dict[residual_type][point_type] = {
                    "sigma": sigma,
                    "mu": mu
                }
                #logger.debug(f"预计算sigma字典，残差类型：\n{residual_type}\n 点类型：\n{point_type}\n sigma：{sigma} mu：{mu} 最大偏差：{max_offset}")
                #logger.debug(f"最大偏差置信度：{self.normal_distribution_vector(max_offset, 0, sigma)}")
        self.sigma_dict_computed = True
    
    def _solve_sigma_squared(self, max_offset: float, confidence: float) -> float:
        """
        解方程计算sigma平方
        公式：σ² = max_offset² / (-2 * ln(confidence))
        """
        if confidence <= 0 or confidence >= 1:
            raise ValueError("置信度必须在(0,1)范围内")
        
        sigma_squared = max_offset ** 2 / (-2 * np.log(confidence))
        return sigma_squared
    
    def normal_distribution_vector(self, d_vector: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        """
        正态分布函数（向量化版本)
        Args:
            d_vector: 残差向量
            mu: 均值
            sigma: 标准差
        Returns:
            正态分布概率密度归一化向量
        """
        return np.exp(-0.5 * ((d_vector - mu) / sigma) ** 2)
    
    def calculate_confidence_vector(self, residual_vector: np.ndarray, point_types: List[str], 
                                  residual_type: str) -> np.ndarray:
        """
        计算置信度向量（向量化版本）
        Args:
            residual_vector: 残差向量
            point_types: 点类型列表
            residual_type: 残差类型（"TO_SURFACE" 或 "TO_CENTER"）
        Returns:
            置信度向量
        """
        confidence_vector = np.zeros_like(residual_vector)
        
        for i, point_type in enumerate(point_types):
            if point_type in self.sigma_dict[residual_type]:
                sigma_info = self.sigma_dict[residual_type][point_type]
                sigma = sigma_info["sigma"]
                
                confidence_vector[i] = self.normal_distribution_vector(
                    residual_vector[i], 0, sigma
                )
                #logger.debug(f"计算置信度向量，残差类型：\n{residual_type}\n 点类型：\n{point_type}\n 置信度：\n{confidence_vector[i]}")
            else:
                # 如果点类型不在预计算字典中，使用默认值
                confidence_vector[i] = 0.0
        
        return confidence_vector
    
    def gaussian_fusion_vector(self, confidence1_vector: np.ndarray, confidence2_vector: np.ndarray, 
                              sigma: float = 0.5) -> np.ndarray:
        """
        高斯融合两个权重向量
        Args:
            confidence1_vector: 第一个置信度向量
            confidence2_vector: 第二个置信度向量
            sigma: 高斯融合参数sigma
        Returns:
            融合后的置信度向量
        """
        # 高斯融合公式：exp(-((c1-c2)^2)/(2*sigma^2)) * (c1 + c2) / 2
        weight = np.exp(-((confidence1_vector - confidence2_vector) ** 2) / (2 * sigma ** 2))
        fused_confidence = weight * (confidence1_vector + confidence2_vector) / 2
        
        return fused_confidence
    
    def calculate_geometry_confidence(self, parameters: FittingParameters, 
                                    fitting_points: List[Tuple[np.ndarray, str]],
                                    constraint_points: List[Tuple[np.ndarray, str]]) -> float:
        """
        计算几何置信度均值
        Args:
            parameters: 拟合参数
            fitting_points: 拟合点列表
            constraint_points: 约束点列表
        Returns:
            几何置信度均值
        """
        # 合并所有点
        if self.fitting_only:
            all_points = fitting_points
        else:
            all_points = fitting_points + constraint_points
        
        if not all_points:
            return 0.0
        
        # 计算残差
        surface_residual, center_residual = self.residual_calculator.calculate_residuals(
            parameters, all_points
        )
        
        # 获取点类型
        all_point_types = [p[1] for p in all_points]
        
        # 向量化计算置信度
        surface_confidence = self.calculate_confidence_vector(surface_residual, all_point_types, "TO_SURFACE")
        center_confidence = self.calculate_confidence_vector(center_residual, all_point_types, "TO_CENTER")
        
        # 高斯融合
        fused_confidence = self.gaussian_fusion_vector(surface_confidence, center_confidence)
        
        # 计算均值
        if len(fused_confidence) > 0:
            return np.mean(fused_confidence)
        else:
            return 0.0
    
    def calculate_confidence_by_type(self, parameters: FittingParameters, 
                                   points: List[Tuple[np.ndarray, str]], 
                                   residual_type: str) -> float:
        """
        计算特定残差类型的置信度
        Args:
            parameters: 拟合参数
            points: 点列表
            residual_type: 残差类型
        Returns:
            置信度
        """
        if not points:
            return 0.0
        
        # 计算残差
        surface_residual, center_residual = self.residual_calculator.calculate_residuals(parameters, points)
        
        # 选择对应的残差
        residual_vector = surface_residual if residual_type == "TO_SURFACE" else center_residual
        
        # 获取点类型
        point_types = [p[1] for p in points]
        
        # 计算置信度向量
        confidence_vector = self.calculate_confidence_vector(residual_vector, point_types, residual_type)
        
        # 返回均值
        return np.mean(confidence_vector) if len(confidence_vector) > 0 else 0.0
    
    def calculate_weighted_confidence(self, parameters: FittingParameters, 
                                    fitting_points: List[Tuple[np.ndarray, str]],
                                    constraint_points: List[Tuple[np.ndarray, str]],
                                    fitting_weight: float = 0.9, constraint_weight: float = 0.1) -> float:
        """
        计算加权置信度
        Args:
            parameters: 拟合参数
            fitting_points: 拟合点列表
            constraint_points: 约束点列表
            fitting_weight: 拟合点权重
            constraint_weight: 约束点权重
        Returns:
            加权置信度
        """
        # 计算拟合点置信度
        fitting_confidence = self.calculate_geometry_confidence(parameters, fitting_points, [])
        
        # 计算约束点置信度
        constraint_confidence = self.calculate_geometry_confidence(parameters, [], constraint_points)
        
        # 加权平均
        weighted_confidence = (fitting_weight * fitting_confidence + 
                             constraint_weight * constraint_confidence)
        
        return weighted_confidence
    
    def get_confidence_breakdown(self, parameters: FittingParameters, 
                               fitting_points: List[Tuple[np.ndarray, str]],
                               constraint_points: List[Tuple[np.ndarray, str]]) -> Dict[str, float]:
        """
        获取置信度详细分解
        Args:
            parameters: 拟合参数
            fitting_points: 拟合点列表
            constraint_points: 约束点列表
        Returns:
            置信度分解字典
        """
        # 计算各种置信度
        surface_confidence = self.calculate_confidence_by_type(parameters, fitting_points, "TO_SURFACE")
        center_confidence = self.calculate_confidence_by_type(parameters, fitting_points, "TO_CENTER")
        fitting_confidence = self.calculate_geometry_confidence(parameters, fitting_points, [])
        constraint_confidence = self.calculate_geometry_confidence(parameters, [], constraint_points)
        overall_confidence = self.calculate_geometry_confidence(parameters, fitting_points, constraint_points)
        weighted_confidence = self.calculate_weighted_confidence(parameters, fitting_points, constraint_points)
        
        return {
            "surface_confidence": surface_confidence,
            "center_confidence": center_confidence,
            "fitting_confidence": fitting_confidence,
            "constraint_confidence": constraint_confidence,
            "overall_confidence": overall_confidence,
            "weighted_confidence": weighted_confidence
        }

# ==================== 优化器类 ====================
class Optimizer(ABC):
    """优化器抽象基类"""
    
    @abstractmethod
    def optimize(self, initial_params: FittingParameters, 
                fitting_points: List[Tuple[np.ndarray, str]],
                constraint_points: List[Tuple[np.ndarray, str]]) -> FittingResult:
        """优化方法"""
        pass


class LevenbergMarquardtOptimizer(Optimizer):
    """Levenberg-Marquardt优化器（重构版本）"""

    def __init__(self, max_iterations: int = 30, lambda_init: float = 0.01, 
                 lambda_factor: float = 10.0, delta_convergence_threshold: float = 0.1,
                 max_lambda: float = 1e8, min_lambda: float = 1e-8, alpha: float = 0.1,
                 residual_convergence_threshold: float = 0.1,
                 max_step_size: float = 4,
                 huber_kappa: float = 1.5,
                 depth_epsilon: float = 0.5,
                 depth_sigma: float = 10000.0,
                 depth_lambda: float = 1e-8,
                 outside_eyeball_sigma: float = 10000.0,
                 outside_eyeball_lambda: float = 1e-8,
                 ):
        # 原有参数保持不变
        self.max_iterations = max_iterations
        self.lambda_init = lambda_init
        self.lambda_factor = lambda_factor
        self.max_lambda = max_lambda
        self.min_lambda = min_lambda
        self.delta_convergence_threshold = delta_convergence_threshold
        self.residual_convergence_threshold = residual_convergence_threshold
        self.max_step_size = max_step_size
        self.last_target_value = float('inf')

        # 残差权重相关
        self.alpha = alpha
        self.huber_kappa = huber_kappa
        self.depth_epsilon = depth_epsilon
        self.depth_sigma = depth_sigma
        self.depth_lambda = depth_lambda
        self.outside_eyeball_sigma = outside_eyeball_sigma
        self.outside_eyeball_lambda = outside_eyeball_lambda

        # 保持原有计算器
        self.residual_calculator = ResidualCalculator()
        self.confidence_calculator = ConfidenceCalculator()

        # 新增：残差项注册表
        self.residual_terms: List[Callable] = []
        self.register_default_residuals()

    # ================== 残差注册 ==================
    def register_residual(self, func: Callable):
        """注册残差计算函数: func(params, fitting_points, constraint_points) -> (residual, jacobian)"""
        self.residual_terms.append(func)

    def register_default_residuals(self):
        """注册默认残差项（表面、中心、深度、眼球外）"""
        self.register_residual(self._surface_residual)
        self.register_residual(self._center_residual)
        self.register_residual(self._depth_residual)
        self.register_residual(self._outside_eyeball_residual)


    # ================== 残差项实现 ==================
    def _surface_residual(self, params, fitting_points, constraint_points):
        """
        表面残差 - 使用原有的ResidualCalculator逻辑
        
        数学公式：
        对于每个拟合点 p_i，表面残差定义为：
        f_surface,i = √α * (||p_i - c|| - r - bias_surface,i) * w_huber,i
        
        其中：
        - p_i: 第i个拟合点的3D坐标
        - c: 球心坐标 [x, y, z]
        - r: 球体半径
        - bias_surface,i: 第i个点的表面生理偏差（从ANATOMICAL_CONSTRAINTS获取）
        - α: 表面残差权重参数
        - w_huber,i: Huber权重，w_huber,i = min(1, κ/|f_raw,i|)，其中κ是Huber阈值
        
        雅可比矩阵：
        ∂f_surface,i/∂c = -√α * w_huber,i * (p_i - c) / ||p_i - c||
        ∂f_surface,i/∂r = -√α * w_huber,i
        """
        if not fitting_points:
            return np.zeros(0), np.zeros((0, 4))
        
        # 使用原有的残差计算器
        surface_residual, _ = self.residual_calculator.calculate_residuals(params, fitting_points)
        
        # 计算雅可比矩阵
        N = len(fitting_points)
        jacobians = []
        for i, (point, _) in enumerate(fitting_points):
            diff = point - params.center
            dist = np.linalg.norm(diff)
            J = np.zeros(4)
            if dist > 1e-8:
                J[:3] = -np.sqrt(self.alpha) * (diff / dist)
                J[3] = -np.sqrt(self.alpha)
            jacobians.append(J)
        
        # 应用Huber权重
        weighted_residuals = []
        for i, f in enumerate(surface_residual):
            if abs(f) > self.huber_kappa:
                weight = self.huber_kappa / abs(f)
            else:
                weight = 1.0
            weighted_residuals.append(np.sqrt(self.alpha) * f * weight)
            jacobians[i] *= weight
        
        return np.array(weighted_residuals), np.array(jacobians)

    def _surface_residual(self, params, fitting_points, constraint_points):
        """
        表面残差 - 使用 ResidualCalculator 计算原始残差，然后应用 Huber 权重和 √α 缩放
        
        数学公式：
        f_surface,i = √α * (||p_i - c|| - r - bias_surface,i) * w_huber,i
        
        其中：
        - bias_surface,i: 从 ANATOMICAL_CONSTRAINTS["TO_SURFACE"] 获取的中值偏差
        - w_huber,i: Huber权重，w_huber,i = min(1, κ/|f_raw,i|)
        """
        if not fitting_points:
            return np.zeros(0), np.zeros((0, 4))
        
        # 使用 ResidualCalculator 计算原始残差
        surface_residual, _ = self.residual_calculator.calculate_residuals(params, fitting_points)
        
        residuals = []
        jacobians = []
        for i, (point, _) in enumerate(fitting_points):
            diff = point - params.center
            dist = np.linalg.norm(diff)
            
            # 获取原始残差
            f_raw = surface_residual[i]
            
            # Huber weight（IRLS 风格）
            if abs(f_raw) > self.huber_kappa:
                w = self.huber_kappa / (abs(f_raw) + 1e-12)
            else:
                w = 1.0
            
            # 最终残差与 jacobian
            r = np.sqrt(self.alpha) * f_raw * w
            residuals.append(r)
            
            J = np.zeros(4)
            if dist > 1e-12:
                # ∂dist/∂c = -(p - c)/dist  => ∂f/∂c = -(p-c)/dist
                J[:3] = -np.sqrt(self.alpha) * w * (diff / dist)
                J[3]  = -np.sqrt(self.alpha) * w
            jacobians.append(J)
        logger.debug(f"表面残差: {residuals}")
        return np.array(residuals), np.vstack(jacobians)

    def _center_residual(self, params, fitting_points, constraint_points):
        """
        中心残差 - 使用 ResidualCalculator 计算原始残差，然后应用 Huber 权重和 √(1-α) 缩放
        
        数学公式：
        f_center,i = √(1-α) * (||p_i - c|| - bias_center,i) * w_huber,i
        
        其中：
        - bias_center,i: 从 ANATOMICAL_CONSTRAINTS["TO_CENTER"] 获取的中值偏差
        - w_huber,i: Huber权重，w_huber,i = min(1, κ/|f_raw,i|)
        """
        if not fitting_points:
            return np.zeros(0), np.zeros((0, 4))
        
        # 使用 ResidualCalculator 计算原始残差
        _, center_residual = self.residual_calculator.calculate_residuals(params, fitting_points)
        
        residuals = []
        jacobians = []
        for i, (point, _) in enumerate(fitting_points):
            diff = point - params.center
            dist = np.linalg.norm(diff)
            
            # 获取原始残差
            f_raw = center_residual[i]
            
            # Huber weight（IRLS 风格）
            if abs(f_raw) > self.huber_kappa:
                w = self.huber_kappa / (abs(f_raw) + 1e-12)
            else:
                w = 1.0
            
            # 最终残差与 jacobian
            r = np.sqrt(1.0 - self.alpha) * f_raw * w
            residuals.append(r)
            
            J = np.zeros(4)
            if dist > 1e-12:
                J[:3] = -np.sqrt(1.0 - self.alpha) * w * (diff / dist)
                # J[3] = 0（中心残差不依赖于半径）
            jacobians.append(J)
        
        logger.debug(f"中心残差: {residuals}")
        return np.array(residuals), np.vstack(jacobians)

    def _depth_residual(self, params, fitting_points, constraint_points):
        """
        深度残差（Softplus），并正确缩放 √depth_lambda
        
        数学公式：
        f_depth = √λ_depth * σ * ln(1 + exp(u/σ))
        
        其中：
        - u = max(0, max_z + ε - c_z)
        - max_z: 所有拟合点中的最大z坐标（最小深度）
        - c_z: 球心的z坐标
        - ε: 深度容差参数（depth_epsilon）
        - σ: 深度平滑参数（depth_sigma）
        - λ_depth: 深度约束权重参数
        """
        if not fitting_points:
            return np.zeros(0), np.zeros((0, 4))

        max_z = max(p[0][2] for p in fitting_points)
        u = max(0.0, max_z + self.depth_epsilon - params.center[2])
        exp_term = np.exp(u / self.depth_sigma)
        f_raw = self.depth_sigma * np.log1p(exp_term)   # sigma * ln(1+exp(u/sigma))
        # scale by sqrt(depth_lambda)
        r = np.sqrt(self.depth_lambda) * f_raw

        # df/du = exp(u/σ)/(1+exp(u/σ)) = sigmoid(u/σ)
        df_du = exp_term / (1.0 + exp_term)
        # du/dcz = -1 (when u>0), so df/dcz = -df_du
        J = np.zeros((1, 4))
        J[0, 2] = np.sqrt(self.depth_lambda) * (- df_du)

        logger.debug(f"深度残差: {r}")
        return np.array([r]), J

    def _outside_eyeball_residual(self, params, fitting_points, constraint_points):
        """
        眼球外约束残差 - 使用与深度约束相同的控制方式
        
        数学公式：
        对于每个约束点 p_i，眼球外约束残差定义为：
        f_outside,i = √λ_outside * σ_outside * ln(1 + exp(u_i/σ_outside))
        
        其中：
        - u_i = max(0, r - ||p_i - c||)
        - p_i: 第i个约束点的3D坐标
        - c: 球心坐标 [x, y, z]
        - r: 球体半径
        - λ_outside: 眼球外约束权重参数（outside_eyeball_lambda）
        - σ_outside: 眼球外约束平滑参数（outside_eyeball_sigma）
        
        物理意义：
        - 当 r > ||p_i - c|| 时，u_i > 0，表示约束点位于球体内部，残差增加（惩罚）
        - 当 r ≤ ||p_i - c|| 时，u_i = 0，表示约束点位于球体外部或表面，残差为0（无惩罚）
        
        雅可比矩阵：
        ∂f_outside,i/∂c = √λ_outside * (-exp(u_i/σ_outside)/(1+exp(u_i/σ_outside))) * (p_i - c) / ||p_i - c||
        ∂f_outside,i/∂r = √λ_outside * (exp(u_i/σ_outside)/(1+exp(u_i/σ_outside)))
        """
        if not constraint_points:
            return np.zeros(0), np.zeros((0, 4))

        residuals, jacobians = [], []
        
        lambda_outside = self.outside_eyeball_lambda
        
        for point, _ in constraint_points:
            diff = point - params.center
            dist = np.linalg.norm(diff)

            # u = max(0, r - d) - 确保约束点在球体外部
            u = max(0.0, params.radius - dist)

            # 使用与深度约束相同的Softplus形式
            exp_term = np.exp(u / self.outside_eyeball_sigma)
            f_raw = self.outside_eyeball_sigma * np.log1p(exp_term)  # sigma * ln(1+exp(u/sigma))
            # scale by sqrt(outside_eyeball_lambda)
            lambda_outside /= len(constraint_points)
            r = np.sqrt(lambda_outside) * f_raw
            residuals.append(r)

            # 雅可比
            J = np.zeros(4)
            if u > 1e-8 and dist > 1e-8:
                # df/du = exp(u/σ)/(1+exp(u/σ)) = sigmoid(u/σ)
                df_du = exp_term / (1.0 + exp_term)
                # ∂u/∂c = -∂dist/∂c = (p-c)/dist, ∂u/∂r = 1
                J[:3] = np.sqrt(lambda_outside) * (-df_du) * (diff / dist)  # ∂f/∂c
                J[3] = np.sqrt(lambda_outside) * df_du                      # ∂f/∂r
            jacobians.append(J)
        logger.debug(f"眼球外残差: {residuals}")
        return np.array(residuals), np.array(jacobians)
    # ================== 工具函数 ==================
    def _softplus(self, x, beta=1.0):
        """
        Softplus函数
        
        数学公式：
        Softplus(x, β) = ln(1 + exp(β * x)) / β
        
        性质：
        - 当 x > 0 时，Softplus(x) ≈ x
        - 当 x < 0 时，Softplus(x) ≈ 0
        - 在 x = 0 处平滑过渡
        - β 控制过渡的陡峭程度
        """
        return np.log(1 + np.exp(beta * x)) / beta

    def _sigmoid(self, x):
        """
        Sigmoid函数
        
        数学公式：
        Sigmoid(x) = 1 / (1 + exp(-x))
        
        性质：
        - 输出范围：[0, 1]
        - 在 x = 0 处值为 0.5
        - 是 Softplus 函数的导数
        """
        return 1 / (1 + np.exp(-x))

    # ================== 重构残差拼接 ==================
    def _compute_all_residuals_and_jacobians(self, params, fitting_points, constraint_points):
        all_residuals, all_jacobians = [], []
        dim_params = len(params)
        for func in self.residual_terms:
            r, J = func(params, fitting_points, constraint_points)
            if r.size > 0:
                all_residuals.append(r)
                all_jacobians.append(J)
        if not all_residuals:
            return np.zeros(0), np.zeros((0, dim_params))
        return np.concatenate(all_residuals), np.vstack(all_jacobians)

    def optimize(self, initial_params: FittingParameters, 
                fitting_points: List[Tuple[np.ndarray, str]],
                constraint_points: List[Tuple[np.ndarray, str]]) -> FittingResult:
        """Levenberg-Marquardt优化"""
        if not fitting_points:
            logger.warning("没有拟合点，无法进行优化")
            return FittingResult(
                parameters=initial_params,
                confidence=0.0,
                converged=False,
                iteration_count=0,
                final_residual_norm=float('inf'),
                final_delta_norm=float('inf'),
                strategy_used="lm"
            )
        
        current_params = initial_params.copy()
        lambda_lm = self.lambda_init
        lamda_factor = self.lambda_factor
        max_lambda = self.max_lambda
        min_lambda = self.min_lambda
        delta_convergence_threshold = self.delta_convergence_threshold
        residual_convergence_threshold = self.residual_convergence_threshold
        self.last_target_value = float('inf')
        
        for iteration in range(self.max_iterations):
            logger.debug(f"------------------------------------------------------------------------------------------------------")
            
            logger.debug(f"第{iteration}次LM优化，计算残差：\n")
            # 使用重构的残差计算方法
            old_residual, J = self._compute_all_residuals_and_jacobians(
                current_params, fitting_points, constraint_points)
            
            
            # 求解正规方程
            delta = self._solve_normal_equations(J, old_residual, lambda_lm)
            
            logger.debug(f"第{iteration}次LM优化，未裁剪前计算的步长：{np.linalg.norm(delta)}")
            delta = self._clip_step_size(delta)
            logger.debug(f"第{iteration}次LM优化，裁剪后计算的步长：{np.linalg.norm(delta)}")
            
            # 更新参数（确保 float64，避免 center 为 int 时 += delta 报 UFuncOutputCastingError）
            new_params = current_params.copy()
            new_params.center = np.asarray(current_params.center, dtype=np.float64) + delta[:3]
            new_params.radius = float(current_params.radius) + float(delta[3])
            
            logger.debug(f"第{iteration}次LM优化初始参数: {current_params}")
            logger.debug(f"第{iteration}次LM优化准备更新参数: {new_params}")
            
            # 计算新目标函数值
            new_residual, _ = self._compute_all_residuals_and_jacobians(
                new_params, fitting_points, constraint_points)
            new_target = target_function(new_residual, self.alpha, self.huber_kappa)
            
            logger.debug(f"第{iteration}次LM优化，计算新目标函数值: {new_target}")
            
            logger.debug(f"目前的 置信度 {self.confidence_calculator.get_confidence_breakdown(current_params, fitting_points, constraint_points)}")
            
            # 检查更新是否有效
            if self._check_update_enable(new_target):
                current_params = new_params
                
                lambda_lm = max(lambda_lm/lamda_factor, min_lambda)
                
                # 检查收敛
                if self._check_convergence(delta, delta_convergence_threshold, old_residual, new_residual, residual_convergence_threshold):
                    logger.debug(f"LM优化收敛，迭代次数: {iteration + 1}")
                    logger.debug(f"LM优化收敛，最终参数: {current_params}")
                    logger.debug("------------------------------------------------------------------------------------------------------")
                    return FittingResult(
                        parameters=current_params,
                        confidence=self.confidence_calculator.calculate_weighted_confidence(current_params, fitting_points, constraint_points),
                        converged=True,
                        iteration_count=iteration + 1,
                        final_residual_norm=np.linalg.norm(new_residual),
                        final_delta_norm=np.linalg.norm(delta),
                        strategy_used="lm"
                    )
            else:
                lambda_lm = min(lambda_lm*lamda_factor, max_lambda)
                
            logger.debug(f"第{iteration}次LM优化，lambda_lm: {lambda_lm}")
        
        # 未收敛
        return FittingResult(
            parameters=current_params,
            confidence=self.confidence_calculator.calculate_geometry_confidence(current_params, fitting_points, constraint_points),
            converged=False,
            iteration_count=self.max_iterations,
            final_residual_norm=np.linalg.norm(old_residual),
            final_delta_norm=np.linalg.norm(delta),
            strategy_used="lm"
        )
        
    def _clip_step_size(self, delta: np.ndarray) -> np.ndarray:
        """裁剪步长范数"""
        delta_norm = np.linalg.norm(delta)
        if delta_norm > self.max_step_size:
            return delta * (self.max_step_size / delta_norm)
        return delta
        
    def _check_convergence(self, delta: np.ndarray, delta_convergence_threshold: float, old_residual: np.ndarray, new_residual: np.ndarray, residual_convergence_threshold: float) -> bool:
        """检查收敛"""
        return np.linalg.norm(delta) < delta_convergence_threshold and np.linalg.norm(new_residual - old_residual) < residual_convergence_threshold
    
    def _check_update_enable(self, target_value: float) -> bool:
        """检查更新是否有效"""
        # 检查目标值是否下降
        
        improved = False
        if target_value <= self.last_target_value:
            improved = True
            self.last_target_value = target_value
        logger.debug(f"检查更新是否有效: {target_value} <= {self.last_target_value} = {improved}")
        return improved
    
    def _solve_normal_equations(self, J: np.ndarray, residual: np.ndarray, lambda_lm: float) -> np.ndarray:
        """求解正规方程"""
        # 检查维度匹配
        if len(residual) != J.shape[0]:
            logger.error(f"维度不匹配: J.shape[0]={J.shape[0]}, residual长度={len(residual)}")
            return np.zeros(4)
        
        # 计算Hessian矩阵和梯度
        H = J.T @ J
        g = J.T @ residual
        
        # 添加阻尼项
        H_lm = H + lambda_lm * np.diagflat(np.diag(H))
        
        try:
            # 求解线性方程组
            delta = np.linalg.solve(H_lm, -g)
        except np.linalg.LinAlgError:
            logger.warning("线性方程组求解失败，使用最小二乘法求解")
            delta = np.linalg.lstsq(H_lm, -g, rcond=None)[0]
        finally:
            logger.debug(f"求解线性方程组结果: {delta}")
            return delta 
        
class RANSACOptimizer(Optimizer):
    """RANSAC优化器"""
    
    def __init__(self, max_iterations: int = 20, surface_threshold: float = None, 
                 center_threshold: float = None, base_optimizer: Optimizer = None, base_sampling_strategy: SamplingStrategy = None):
        self.max_iterations = max_iterations
        
        # 设置默认阈值（当点类型不在约束中时使用）
        if surface_threshold is None:
            self.surface_threshold = 0.5
        else:
            self.surface_threshold = surface_threshold
            
        if center_threshold is None:
            self.center_threshold = 1.0
        else:
            self.center_threshold = center_threshold
            
        # 打印各点类型的阈值信息
        logger.debug("RANSAC各点类型阈值设置：")
        for point_type in FITTING_ENABLE + OUTSIDE_EYEBALL:
            if point_type in ANATOMICAL_CONSTRAINTS["TO_SURFACE"]:
                min_val, max_val = ANATOMICAL_CONSTRAINTS["TO_SURFACE"][point_type]
                surface_threshold = (max_val - min_val) / 2
                logger.debug(f"  {point_type} 表面阈值: ({min_val}, {max_val}) -> {surface_threshold:.4f}")
            
            if point_type in ANATOMICAL_CONSTRAINTS["TO_CENTER"]:
                min_val, max_val = ANATOMICAL_CONSTRAINTS["TO_CENTER"][point_type]
                center_threshold = (max_val - min_val) / 2
                logger.debug(f"  {point_type} 中心阈值: ({min_val}, {max_val}) -> {center_threshold:.4f}")
            
        self.base_sampling_strategy = base_sampling_strategy or RandomSamplingStrategy(min_fitting_points=3, min_constraint_points=3)
        self.base_optimizer = base_optimizer or LevenbergMarquardtOptimizer()
        self.residual_calculator = ResidualCalculator()
        self.min_sample_count = 4
    
    def optimize(self, initial_params: FittingParameters, 
                fitting_points: List[Tuple[np.ndarray, str]],
                constraint_points: List[Tuple[np.ndarray, str]]) -> FittingResult:
        
        best_result = None
        best_confidence = 0.0
        best_iteration = 0
        
        # 用于维护所有迭代的距离信息
        # all_iterations_distances = []
        
        for iteration in range(self.max_iterations):
            logger.debug(f"======================================================================================================")
            # 随机采样

            sampled_fitting_points, sampled_constraint_points = self.base_sampling_strategy.sample(fitting_points, constraint_points)

            logger.debug(f"RANSAC优化器，第{iteration}次迭代，采样点数：{len(sampled_fitting_points)}")
            
            # 使用基础优化器
            result = self.base_optimizer.optimize(initial_params, sampled_fitting_points, sampled_constraint_points)
            
            # 计算并打印所有点到中心和表面的距离
            # point_distances = self._print_all_points_distances(result.parameters, iteration)
            # all_iterations_distances.append(point_distances)
            
            # 使用confidence作为更新标准
            logger.debug(f"RANSAC优化器，第{iteration}次迭代，confidence：{result.confidence:.4f}")    
            
            if result.confidence > best_confidence and result.converged:
                best_confidence = result.confidence
                best_result = result
                best_iteration = iteration
                logger.debug(f"RANSAC优化更新，第{best_iteration}次迭代，confidence：{best_confidence:.4f}")
            
            logger.debug(f"======================================================================================================")
        
        # 打印所有迭代的平均距离
        # self._print_average_distances(all_iterations_distances)
        
        if best_result is None:
            # 如果没有找到好的结果，返回初始参数
            return FittingResult(
                parameters=initial_params,
                confidence=0.0,
                converged=False,
                iteration_count=0,
                final_residual_norm=float('inf'),
                final_delta_norm=float('inf'),
                strategy_used="ransac"
            )
        
        best_result.strategy_used = "ransac"
        best_result.iteration_count = best_iteration
        return best_result
    

    def _print_average_distances(self, all_iterations_distances: List[List[Tuple[str, float, float]]]):
        """打印所有迭代中每个点的平均距离统计"""
        if not all_iterations_distances:
            logger.debug("RANSAC优化完成：无距离数据")
            return
        
        # 统计每个点类型在所有迭代中的距离
        point_type_stats = {}
        
        # 遍历所有迭代的距离数据
        for iteration, point_distances in enumerate(all_iterations_distances):
            for point_type, center_dist, surface_dist in point_distances:
                if point_type not in point_type_stats:
                    point_type_stats[point_type] = {
                        'center_distances': [],
                        'surface_distances': [],
                        'count': 0
                    }
                point_type_stats[point_type]['center_distances'].append(center_dist)
                point_type_stats[point_type]['surface_distances'].append(surface_dist)
                point_type_stats[point_type]['count'] += 1
        
        # 计算每个点类型的平均距离、最大距离、最小距离
        logger.debug("RANSAC优化完成 - 每个点类型在所有迭代中的距离统计：")
        for point_type, stats in point_type_stats.items():
            center_distances = stats['center_distances']
            surface_distances = stats['surface_distances']
            
            avg_center = np.mean(center_distances)
            avg_surface = np.mean(surface_distances)
            min_center = np.min(center_distances)
            max_center = np.max(center_distances)
            min_surface = np.min(surface_distances)
            max_surface = np.max(surface_distances)
            
            logger.debug(f"  {point_type}: 平均到中心距离={avg_center:.4f} (范围: {min_center:.4f}-{max_center:.4f}), "
                         f"平均到表面距离={avg_surface:.4f} (范围: {min_surface:.4f}-{max_surface:.4f}) (出现{stats['count']}次)")
        
        # 计算所有点的整体平均距离、最大距离、最小距离
        all_center_distances = []
        all_surface_distances = []
        for stats in point_type_stats.values():
            all_center_distances.extend(stats['center_distances'])
            all_surface_distances.extend(stats['surface_distances'])
        
        if all_center_distances:
            overall_avg_center = np.mean(all_center_distances)
            overall_avg_surface = np.mean(all_surface_distances)
            overall_min_center = np.min(all_center_distances)
            overall_max_center = np.max(all_center_distances)
            overall_min_surface = np.min(all_surface_distances)
            overall_max_surface = np.max(all_surface_distances)
            
            logger.debug(f"RANSAC整体平均距离: 到中心={overall_avg_center:.4f} (范围: {overall_min_center:.4f}-{overall_max_center:.4f}), "
                         f"到表面={overall_avg_surface:.4f} (范围: {overall_min_surface:.4f}-{overall_max_surface:.4f})")   
            

class FailOptimizer(Optimizer):
    """失败优化器"""
    
    def optimize(self, initial_params: FittingParameters, 
                fitting_points: List[Tuple[np.ndarray, str]],
                constraint_points: List[Tuple[np.ndarray, str]]) -> FittingResult:
        logger.warning("优化失败")
        return FittingResult(
            parameters=FittingParameters(center=np.zeros(3, dtype=np.float64), radius=0.0),
            confidence=0.0,
            converged=False,
            iteration_count=0,
            final_residual_norm=float('inf'),
            final_delta_norm=float('inf'),
            strategy_used="fail"
        )

# ==================== 策略类（纯策略，无状态）====================
@dataclass
class FittingStrategy:
    """拟合策略 - 纯配置类，无状态"""
    name: str
    sampling_strategy: SamplingStrategy
    optimizer: Optimizer
    confidence: float
    min_visibility: float
    min_points: int
    sphere_fitting: bool = True

class StrategySelector:
    """策略选择器 - 根据数据质量选择策略"""
    
    def __init__(self, strategies: List[FittingStrategy]):
        self.strategies = strategies
    
    def select_strategy(self, data_quality: Dict[str, float]) -> FittingStrategy:
        """根据数据质量选择最佳策略"""
        logger.debug(f"数据质量: {data_quality}")
        logger.debug(f"可用策略数量: {len(self.strategies)}")
        
        # 按可见性降序排列策略
        sorted_strategies = sorted(self.strategies, key=lambda s: s.min_visibility, reverse=True)
        
        for i, strategy in enumerate(sorted_strategies):
            logger.debug(f"策略 {i+1}: {strategy.name}")
            logger.debug(f"  最小点数: {strategy.min_points}, 最小可见性: {strategy.min_visibility}")
            logger.debug(f"  当前点数: {data_quality['point_count']}, 当前可见性: {data_quality['visibility']:.3f}")
            
            if (data_quality["point_count"] >= strategy.min_points and 
                data_quality["visibility"] >= strategy.min_visibility):
                logger.debug(f" 当前点数: {data_quality['point_count']}, 当前可见性: {data_quality['visibility']:.3f}")
                logger.debug(f"选择策略: {strategy.name}")
                return strategy
            else:
                logger.debug(f"  不满足条件，跳过")
        
        # 如果没有找到合适的策略，返回置信度最高的策略
        best_strategy = max(self.strategies, key=lambda s: s.confidence)
        logger.warning(f"没有找到满足条件的策略，使用置信度最高的策略: {best_strategy.name}")
        return best_strategy

# ==================== 主控制器类 ====================
class FittingController:
    """拟合控制器 - 协调各个组件"""
    
    def __init__(self, data_provider: DataProvider, strategy_selector: StrategySelector):
        self.data_provider = data_provider
        self.strategy_selector = strategy_selector
    
    def fit_eye(self, eye: str) -> FittingResult:
        """拟合单只眼睛"""
        # 1. 获取数据
        fitting_points = self.data_provider.get_fitting_points(eye)
        constraint_points = self.data_provider.get_constraint_points(eye)
        data_quality = self.data_provider.get_data_quality(eye)
        
        # 2. 选择策略
        strategy = self.strategy_selector.select_strategy(data_quality)
        
        # 3. 采样
        sampled_fitting, sampled_constraint = strategy.sampling_strategy.sample(
            fitting_points, constraint_points
        )
        
        # 4. 初始化参数
        initial_params = self._initialize_parameters(sampled_fitting)
        
        # 5. 优化
        result = strategy.optimizer.optimize(initial_params, sampled_fitting, sampled_constraint)
        
        return result
    
    def _fit_circle_to_points(self, points: List[np.ndarray]) -> Tuple[np.ndarray, float]:
        """三点确定外接圆"""
        if len(points) < 3:
            raise ValueError("至少需要3个点")
        
        # 取前三个点
        p1, p2, p3 = points[:3]
        
        # 计算三条边的中点
        mid1 = (p1 + p2) / 2
        mid2 = (p2 + p3) / 2
        
        # 计算边的方向向量
        v1 = p2 - p1
        v2 = p3 - p2
        
        # 计算垂直向量（法向量）
        normal = np.cross(v1, v2)
        if np.linalg.norm(normal) < 1e-6:
            raise ValueError("三点共线，无法确定外接圆")
        
        # 计算垂直平分线的方向
        perp1 = np.cross(normal, v1)
        perp2 = np.cross(normal, v2)
        
        # 归一化
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = perp2 / np.linalg.norm(perp2)
        
        # 圆心是两条垂直平分线的交点
        # 简化：使用两条垂直平分线的中点作为圆心
        center = (mid1 + mid2) / 2
        
        # 半径是圆心到任意点的距离
        radius = np.linalg.norm(center - p1)
        
        return center, radius

    def _initialize_parameters(self, fitting_points: List[Tuple[np.ndarray, str]]) -> FittingParameters:
        """初始化参数 - 自动选择最佳策略"""
        if not fitting_points:
            return FittingParameters(center=np.zeros(3, dtype=np.float64), radius=float(ANATOMICAL_CONSTRAINTS["EYEBALL_RADIUS_DEFAULT"]))
        
        # 按类型分组
        iris_points = [point[0] for point in fitting_points if point[1] == "iris"]
        pupil_points = [point[0] for point in fitting_points if point[1] == "pupil"]
        
        # 策略1：优先使用虹膜点方案
        if len(iris_points) >= 3:
            try:
                # 随机取3个虹膜点拟合外接圆
                import random
                selected_iris = random.sample(iris_points, 3)
                circle_center, circle_radius = self._fit_circle_to_points(selected_iris)
                logger.debug(f"虹膜点拟合外接圆，圆心：{circle_center}，半径：{circle_radius}")
                
                # 计算delta z - 验证数学有效性
                iris_to_center_median = (ANATOMICAL_CONSTRAINTS["TO_CENTER"]["iris"][0] + 
                                       ANATOMICAL_CONSTRAINTS["TO_CENTER"]["iris"][1]) / 2  # 斜边
                
                # 确保三角形数学有效：hypotenuse > leg
                if iris_to_center_median**2 < circle_radius**2:
                    raise ValueError(f"Invalid triangle: iris_to_center_median({iris_to_center_median}) < circle_radius({circle_radius})")
                
                delta_z = np.sqrt(iris_to_center_median**2 - circle_radius**2)  # 直角边
                
                # 圆心坐标加上delta z作为眼球中心
                eyeball_center = circle_center.copy()
                eyeball_center[2] += delta_z
                
                return FittingParameters(center=np.asarray(eyeball_center, dtype=np.float64), radius=float(circle_radius))
            except Exception as e:
                logger.warning(f"虹膜点拟合失败: {e}")
        
        # 策略2：使用瞳孔点方案
        if pupil_points:
            pupil_point = pupil_points[0]  # 取第一个瞳孔点
            pupil_to_center_median = (ANATOMICAL_CONSTRAINTS["TO_CENTER"]["pupil"][0] + 
                                    ANATOMICAL_CONSTRAINTS["TO_CENTER"]["pupil"][1]) / 2
            
            # 瞳孔点加上到中心的初始生理偏移中值
            eyeball_center = pupil_point.copy()
            eyeball_center[2] += pupil_to_center_median
            
            return FittingParameters(center=np.asarray(eyeball_center, dtype=np.float64), radius=float(ANATOMICAL_CONSTRAINTS["EYEBALL_RADIUS_DEFAULT"]))
        
        # 策略3：失败情况
        logger.error("无法找到有效的瞳孔点或虹膜点进行初始化")
        return FittingParameters(center=np.zeros(3, dtype=np.float64), radius=float(ANATOMICAL_CONSTRAINTS["EYEBALL_RADIUS_DEFAULT"]))
    
    def fit_all_eyes(self) -> Dict[str, FittingResult]:
        """拟合所有眼睛"""
        results = {}
        for eye in EYE_TYPE:
            results[eye] = self.fit_eye(eye)
        return results

# ==================== 预定义策略 ====================
def create_default_strategies() -> List[FittingStrategy]:
    """创建默认策略"""
    return [
        # 高质量数据策略
        FittingStrategy(
            name="high_quality",
            sampling_strategy=FullSamplingStrategy(),
            optimizer=LevenbergMarquardtOptimizer(),
            confidence=0.7,
            min_visibility=0.9,
            min_points=5
        ),
        # 中质量数据策略
        FittingStrategy(
            name="medium_quality",
            sampling_strategy=RandomSamplingStrategy(min_fitting_points=3, min_constraint_points=3),
            optimizer=RANSACOptimizer(base_sampling_strategy=RandomSamplingStrategy(min_fitting_points=3, min_constraint_points=3)),
            confidence=0.8,
            min_visibility=0.7,
            min_points=3
        ),
        # 低质量数据策略
        FittingStrategy(
            name="low_quality",
            sampling_strategy=RandomSamplingStrategy(min_fitting_points=2, min_constraint_points=1),
            optimizer=RANSACOptimizer(max_iterations=30, base_sampling_strategy=RandomSamplingStrategy(min_fitting_points=2, min_constraint_points=1)),
            confidence=0.9,
            min_visibility=0.5,
            min_points=2,
            sphere_fitting=False
        ),
        # 失败策略
        FittingStrategy(
            name="fail",
            sampling_strategy=FailSamplingStrategy(),
            optimizer=FailOptimizer(),
            confidence=0.0,
            min_visibility=0.0,
            min_points=0
        )
    ]

# ==================== 便捷函数 ====================
def create_fitting_controller(data_manager: RecgFitDataManager) -> FittingController:
    """创建拟合控制器"""
    data_provider = DataProvider(data_manager)
    strategies = create_default_strategies()
    strategy_selector = StrategySelector(strategies)
    return FittingController(data_provider, strategy_selector)

def fit_eye(eye: str, data_manager: RecgFitDataManager, params_only: bool = False) -> FittingResult:
    """简单的拟合函数"""
    controller = create_fitting_controller(data_manager)
    result = controller.fit_eye(eye)
    if params_only:
        return result.parameters
    return result

def fit_all_eyes(data_manager: RecgFitDataManager, params_only: bool = False) -> Dict[str, Union[FittingResult, FittingParameters]]:
    """
    简单的拟合所有眼睛函数

    Args:
        data_manager: 数据管理器
        params_only: 是否只返回参数，默认False返回完整结果
        
    Returns:
        Dict[str, Union[FittingResult, FittingParameters]]: 
            - 当params_only=False时: {"left": FittingResult, "right": FittingResult}
            - 当params_only=True时: {"left": FittingParameters, "right": FittingParameters}
            
        FittingResult包含:
            - parameters: FittingParameters (球心坐标和半径)
            - confidence: float (置信度，0.0-1.0)
            - converged: bool (是否收敛)
            - iteration_count: int (迭代次数)
            - final_residual_norm: float (最终残差范数)
            - strategy_used: str (使用的策略名称)
            
        FittingParameters包含:
            - center: np.ndarray (球心坐标 [x, y, z])
            - radius: float (球体半径)
            - sphere_fitting: bool (是否使用球体拟合)
    """
    controller = create_fitting_controller(data_manager)
    results = controller.fit_all_eyes()
    if params_only:
        return {eye: result.parameters for eye, result in results.items()}
    return results