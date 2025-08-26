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
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Dict, Optional
from project.recg_fit_data.data_manager import EYE_TYPE, RECG_FIT_DATA_MANAGER, RecgFitDataManager
import numpy as np
import logging

# ==================== 解剖学约束参数 ====================
ANATOMICAL_CONSTRAINTS = {
    "TO_SURFACE": {
        "pupil": (-4.05, -3.5),
        "iris": (-1.2, -0.6),
        "inner_canthus": (0.0, 0.5),
        "upper_eyelid": (1.0, 2.0),
        "lower_eyelid": (0.5, 1.5),
        "outer_canthus": (3.5, 5.0),
    },
    "TO_CENTER": {
        "pupil": (7.95, 8.95),
        "iris": (10.3, 11.8),
        "inner_canthus": (12.0, 13.0),
        "upper_eyelid": (13.5, 14.5),
        "lower_eyelid": (13.0, 14.0),
        "outer_canthus": (15.0, 17.5),
    },
    "EYEBALL_RADIUS": (8.0, 20.0),
    "EYEBALL_RADIUS_DEFAULT": 12.0,
}

FITTING_ENABLE = ["pupil", "iris", "inner_canthus"]
OUTSIDE_EYEBALL = ["upper_eyelid", "lower_eyelid", "outer_canthus"]
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
    strategy_used: str
    
    def __str__(self):
        return f"拟合结果：\n 球心：{self.parameters.center}\n 半径：{self.parameters.radius}\n 置信度：{self.confidence}\n 收敛：{self.converged}\n 迭代次数：{self.iteration_count}\n 最终残差范数：{self.final_residual_norm}\n 使用策略：{self.strategy_used}"

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
                            # 确保是3D坐标点
                            coord_3d = point[:3] if len(point) >= 3 else point
                            points_with_types.append((coord_3d, fitting_type))
                        else:
                            logging.warning(f"跳过无效的拟合点: {point}, 类型: {type(point)}")
            except Exception as e:
                logging.error(f"获取{fitting_type}拟合点时出错: {e}")
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
                            # 确保是3D坐标点
                            coord_3d = point[:3] if len(point) >= 3 else point
                            points_with_types.append((coord_3d, fitting_type))
                        else:
                            logging.warning(f"跳过无效的约束点: {point}, 类型: {type(point)}")
            except Exception as e:
                logging.error(f"获取{fitting_type}约束点时出错: {e}")
        return points_with_types
    
    def get_data_quality(self, eye: str) -> Dict[str, float]:
        """获取数据质量指标"""
        try:
            fitting_points = self.get_fitting_points(eye)
            constraint_points = self.get_constraint_points(eye)
            
            total_points = len(fitting_points) + len(constraint_points)
            fitting_count = len(fitting_points)
            constraint_count = len(constraint_points)
            
            logging.debug(f"{eye}眼数据统计:")
            logging.debug(f"  拟合点数量: {fitting_count}")
            logging.debug(f"  约束点数量: {constraint_count}")
            logging.debug(f"  总点数: {total_points}")
            
            # 计算平均可见性 - 直接从数据管理器获取原始数据
            visibility_values = []
            for fitting_type in FITTING_ENABLE + OUTSIDE_EYEBALL:
                points = self.data_manager.get_coordinate_point(eye, fitting_type)
                if points:
                    for point in points:
                        if isinstance(point, np.ndarray) and len(point) >= 4:
                            visibility_values.append(float(point[3]))
            
            mean_visibility = np.mean(visibility_values) if visibility_values else 0.0
            logging.debug(f"  平均可见性: {mean_visibility:.3f}")
            logging.debug(f"  可见性样本数: {len(visibility_values)}")
            
            quality = {
                "point_count": total_points,
                "fitting_point_count": fitting_count,
                "constraint_point_count": constraint_count,
                "visibility": mean_visibility
            }
            
            logging.debug(f"{eye}眼数据质量: {quality}")
            return quality
            
        except Exception as e:
            logging.error(f"计算{eye}眼数据质量时出错: {e}")
            return {
                "point_count": 0,
                "fitting_point_count": 0,
                "constraint_point_count": 0,
                "visibility": 0.0
            }

    def _is_valid_coordinate_point(self, point) -> bool:
        """验证坐标点是否有效"""
        try:
            # 检查是否为numpy数组
            if not isinstance(point, np.ndarray):
                return False
            
            # 检查数据类型是否为数值类型
            if point.dtype.kind not in 'fc':
                return False
            
            # 检查维度是否至少为3（x, y, z）
            if len(point) < 3:
                return False
            
            # 检查是否包含NaN或无穷大值
            if np.any(np.isnan(point)) or np.any(np.isinf(point)):
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
            sampled_fitting = random.sample(fitting_points, self.min_fitting_points)
        else:
            sampled_fitting = fitting_points
        
        # 约束点采样
        if len(constraint_points) >= self.min_constraint_points:
            sampled_constraint = random.sample(constraint_points, self.min_constraint_points)
        else:
            sampled_constraint = constraint_points
        
        return sampled_fitting, sampled_constraint

class FailSamplingStrategy(SamplingStrategy):
    """失败采样策略"""
    
    def sample(self, fitting_points: List[Tuple[np.ndarray, str]], 
               constraint_points: List[Tuple[np.ndarray, str]]) -> Tuple[List[Tuple[np.ndarray, str]], List[Tuple[np.ndarray, str]]]:
        logging.warning("采样失败，返回空列表")
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
def target_function(surface_residual: np.ndarray, center_residual: np.ndarray, 
                   depth_residual: float, alpha: float = 0.5, huber_kappa: float = 0.6, depth_lambda: float = 1.0) -> float:
    """目标函数：Φ(θ) = (1/2) * [α * Σ ρ1(f1,i) + (1-α) * Σρ2(f2,i) + λ * d^2]"""
    
    def huber_loss(f):
        if abs(f) <= huber_kappa:
            return 0.5 * f**2
        else:
            return huber_kappa * (abs(f) - 0.5 * huber_kappa)
    
    # 计算Huber损失
    surface_loss = sum(huber_loss(f) for f in surface_residual) if len(surface_residual) > 0 else 0.0
    center_loss = sum(huber_loss(f) for f in center_residual) if len(center_residual) > 0 else 0.0
    depth_loss = depth_residual**2
    
    # 计算总目标函数值
    total_value = (1/2) * (alpha * surface_loss + (1-alpha) * center_loss + depth_lambda * depth_loss)
    
    return total_value

# ==================== 置信度类 ====================

# ==================== 置信度计算类 ====================
class ConfidenceCalculator:
    """置信度计算器 - 基于几何约束和正态分布"""
    
    def __init__(self, anatomical_constraints: Dict = ANATOMICAL_CONSTRAINTS):
        self.constraints = anatomical_constraints
        self.residual_calculator = ResidualCalculator(anatomical_constraints)
        self._precompute_sigma_dict()
    
    def _precompute_sigma_dict(self):
        """预计算所有类型点的sigma字典"""
        self.sigma_dict = {}
        
        for residual_type in RESIDUAL_TYPE:
            self.sigma_dict[residual_type] = {}
            
            constraints_dict = self.constraints.get(residual_type, {})
            
            for point_type in constraints_dict:
                # 获取生理偏差中值
                mu = (constraints_dict[point_type][0] + constraints_dict[point_type][1]) / 2
                constraints = constraints_dict[point_type]
                
                # 计算最大偏移
                max_offset = max(abs(constraints[0] - mu), abs(constraints[1] - mu))
                
                # 解方程计算sigma平方
                sigma_squared = self._solve_sigma_squared(max_offset, 0.9)  # 使用0.9作为优秀阈值
                sigma = np.sqrt(sigma_squared)
                
                # 存储sigma和mu
                self.sigma_dict[residual_type][point_type] = {
                    "sigma": sigma,
                    "mu": mu
                }
    
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
                mu = sigma_info["mu"]
                sigma = sigma_info["sigma"]
                
                confidence_vector[i] = self.normal_distribution_vector(
                    residual_vector[i], mu, sigma
                )
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
                                    fitting_weight: float = 0.7, constraint_weight: float = 0.3) -> float:
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
    """Levenberg-Marquardt优化器"""
    
    def __init__(self, max_iterations: int = 30, lambda_init: float = 0.01, 
                 lambda_factor: float = 10.0, delta_convergence_threshold: float = 1,
                 max_lambda: float = 1e8, min_lambda: float = 1e-8, alpha: float = 0.5,
                 residual_convergence_threshold: float = 0.01,
                 residual_convergence_threshold_factor: float = 10.0,
                 max_step_size: float = 1,
                 huber_kappa: float = 0.6,
                 depth_epsilon: float = 5.0,
                 depth_sigma: float = 1.0,
                 depth_lambda: float = 1.0,
                 ):
        self.max_iterations = max_iterations
        self.lambda_init = lambda_init
        self.lambda_factor = lambda_factor
        self.max_lambda = max_lambda
        self.min_lambda = min_lambda
        self.delta_convergence_threshold = delta_convergence_threshold
        self.residual_convergence_threshold = residual_convergence_threshold
        self.residual_calculator = ResidualCalculator()
        self.confidence_calculator = ConfidenceCalculator()
        self.max_step_size = max_step_size
        self.last_target_value = float('inf')
        self.alpha = alpha
        self.huber_kappa = huber_kappa
        self.depth_epsilon = depth_epsilon
        self.depth_sigma = depth_sigma
        self.depth_lambda = depth_lambda
        
    def optimize(self, initial_params: FittingParameters, 
                fitting_points: List[Tuple[np.ndarray, str]],
                constraint_points: List[Tuple[np.ndarray, str]]) -> FittingResult:
        """Levenberg-Marquardt优化"""
        if not fitting_points:
            logging.warning("没有拟合点，无法进行优化")
            return FittingResult(
                parameters=initial_params,
                confidence=0.0,
                converged=False,
                iteration_count=0,
                final_residual_norm=float('inf'),
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
            logging.debug(f"------------------------------------------------------------------------------------------------------")
            # 只使用拟合点计算残差和雅可比矩阵
            surface_residual, center_residual = self.residual_calculator.calculate_residuals(
                current_params, fitting_points)
            
            logging.debug(f"第{iteration}次LM优化，\n计算表面残差：\n{surface_residual}\n 中心残差：\n{center_residual}")
            
            depth_residual, depth_gradient = self._depth_constraint_residual_and_gradient(current_params.center[2], fitting_points, self.depth_epsilon, self.depth_sigma, self.depth_lambda)
            
            old_residual = self._concatenate_residual(surface_residual, center_residual, depth_residual)
            
            # 计算雅可比矩阵（只针对拟合点）
            J = self._calculate_jacobian(surface_residual, center_residual, depth_gradient, current_params, fitting_points,self.alpha, self.huber_kappa, self.depth_lambda)

            
            # 求解正规方程
            delta = self._solve_normal_equations(J, old_residual, lambda_lm,self.alpha)
            
            logging.debug(f"第{iteration}次LM优化，未裁剪前计算的步长：{np.linalg.norm(delta)}")
            delta = self._clip_step_size(delta)
            logging.debug(f"第{iteration}次LM优化，裁剪后计算的步长：{np.linalg.norm(delta)}")
            
            # 更新参数
            new_params = current_params.copy()
            new_params.center += delta[:3]
            new_params.radius += delta[3]
            

            logging.debug(f"第{iteration}次LM优化初始参数: {current_params}")
            logging.debug(f"第{iteration}次LM优化准备更新参数: {new_params}")
            
            # 应用约束点限制参数范围（如果有约束点）
            if constraint_points:
                new_params = self._apply_constraint_limits(new_params, constraint_points)
                logging.debug(f"第{iteration}次LM优化，应用约束点限制参数范围后: {new_params}")
            
            # 计算新目标函数值
            new_surface_residual, new_center_residual = self.residual_calculator.calculate_residuals(
                new_params, fitting_points)
            logging.debug(f"第{iteration}次LM优化，\n计算新表面残差：\n{new_surface_residual}\n 新中心残差：\n{new_center_residual}")
            
            new_depth_residual, _ = self._depth_constraint_residual_and_gradient(new_params.center[2], fitting_points, self.depth_epsilon, self.depth_sigma, self.depth_lambda)
            new_residual = self._concatenate_residual(new_surface_residual, new_center_residual, new_depth_residual)
            new_target = target_function(new_surface_residual, new_center_residual, new_depth_residual,self.alpha, self.huber_kappa, self.depth_lambda)
            
            logging.debug(f"第{iteration}次LM优化，计算新目标函数值: {new_target}")
            
            logging.debug(f"目前的 置信度 {self.confidence_calculator.get_confidence_breakdown(current_params, fitting_points, constraint_points)}")
            
            # 检查更新是否有效
            if self._check_update_enable(new_target):
                current_params = new_params
                
                lambda_lm = max(lambda_lm/lamda_factor, min_lambda)
                
                # 检查收敛
                if self._check_convergence(delta, delta_convergence_threshold, old_residual, new_residual, residual_convergence_threshold):
                    logging.info(f"LM优化收敛，迭代次数: {iteration + 1}")
                    return FittingResult(
                        parameters=current_params,
                        confidence=self.confidence_calculator.calculate_geometry_confidence(current_params, fitting_points, constraint_points),
                        converged=True,
                        iteration_count=iteration + 1,
                        final_residual_norm=np.linalg.norm(surface_residual),
                        strategy_used="lm"
                    )
            else:
                lambda_lm = min(lambda_lm*lamda_factor, max_lambda)
                
            logging.debug(f"第{iteration}次LM优化，lambda_lm: {lambda_lm}")
        
        # 未收敛
        return FittingResult(
            parameters=current_params,
            confidence=self.confidence_calculator.calculate_geometry_confidence(current_params, fitting_points, constraint_points),
            converged=False,
            iteration_count=self.max_iterations,
            final_residual_norm=np.linalg.norm(surface_residual),
            strategy_used="lm"
        )
        
    def _concatenate_residual(self, surface_residual: np.ndarray,
                            center_residual: np.ndarray,
                            depth_residual: float) -> np.ndarray:
        """拼接残差，并带缩放"""
        return np.concatenate([
            np.sqrt(self.alpha) * surface_residual,
            np.sqrt(1 - self.alpha) * center_residual,
            [np.sqrt(self.depth_lambda) * depth_residual]
        ])   
        
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
        logging.debug(f"检查更新是否有效: {target_value} <= {self.last_target_value} = {improved}")
        return improved
    
    def _apply_constraint_limits(self, params: FittingParameters, 
                                constraint_points: List[Tuple[np.ndarray, str]]) -> FittingParameters:
        """应用约束点限制参数范围"""
        if not constraint_points:
            return params
        
        # 计算约束点到球心的距离
        distances = []
        for point, _ in constraint_points:
            dist = np.linalg.norm(point - params.center)
            distances.append(dist)
        
        # 根据约束点调整半径范围
        max_distance = max(distances + [ANATOMICAL_CONSTRAINTS["EYEBALL_RADIUS"][1]]) if distances else params.radius
        min_distance = min(distances + [ANATOMICAL_CONSTRAINTS["EYEBALL_RADIUS"][0]]) if distances else params.radius
        
        # 限制半径在合理范围内
        constrained_radius = np.clip(params.radius, min_distance, max_distance)
        
        # 创建新的参数对象
        new_params = params.copy()
        new_params.radius = constrained_radius
        
        logging.debug(f"约束点限制: 原始半径={params.radius:.3f}, 约束后半径={constrained_radius:.3f}")
        
        return new_params
    
    
    def _depth_constraint_residual_and_gradient(self, center_z: float, points: List[Tuple[np.ndarray, str]], 
                                epsilon: float = 5.0, sigma: float = 1.0, 
                                lambda_: float = 1.0) -> Tuple[float, float]:
        """计算深度约束残差"""
        if not points:
            return 0.0, 0.0
        
        # 找到所有点中的最大深度（最小z值）
        max_z = max(point[0][2] for point in points)
        
        # 计算u值
        u = max(0, max_z + epsilon - center_z)
        
        # 计算平滑约束
        exp_term = np.exp(u / sigma)
        return np.log(1 + exp_term), -(exp_term / (1 + exp_term)) * (1 / sigma)

    
    def _calculate_jacobian(self, surface_residual: np.ndarray, center_residual: np.ndarray, depth_gradient: float, params: FittingParameters, 
                            points: List[Tuple[np.ndarray, str]], 
                            alpha: float = 0.5,
                            huber_kappa: float = 0.6,
                            depth_lambda: float = 1.0) -> np.ndarray:
        """计算雅可比矩阵 (f1: surface, f2: center)"""
        
        def _judge_denom_valid(diff: np.ndarray) -> bool:
            return np.linalg.norm(diff) > 1e-8
        
        if not points:
            return np.zeros((0, 4))
        
        N = len(points)
        J = np.zeros((2 * N + 1, 4))  # f1 和 f2 和 d 拼接
        
        surface_weight = np.ones(N)
        for i,f in enumerate(surface_residual):
            if abs(f) > huber_kappa:
                surface_weight[i] = huber_kappa / abs(f)
                
        center_weight = np.ones(N)
        for i,f in enumerate(center_residual):
            if abs(f) > huber_kappa:
                center_weight[i] = huber_kappa / abs(f)
        
        # --- f1: 表面残差 ---
        for i in range(N):
            point, _ = points[i]
            diff = point - params.center
            dist = np.linalg.norm(diff)
            
            if _judge_denom_valid(diff):
                J[i, 0:3] = -np.sqrt(alpha) * (diff / dist) * surface_weight[i] # 对中心导数
                J[i, 3]   = -np.sqrt(alpha) * surface_weight[i]                 # 对半径导数
            else:
                J[i, :] = 0.0
        
        # --- f2: 中心残差 ---
        for i in range(N):
            point, _ = points[i]
            diff = point - params.center
            dist = np.linalg.norm(diff)
            
            if _judge_denom_valid(diff):
                J[N + i, 0:3] = -np.sqrt(1 - alpha) * (diff / dist) * center_weight[i]  # 只对中心
                J[N + i, 3]   = 0.0                                # 半径不参与
            else:
                J[N + i, :] = 0.0
        
        # --- f3: 深度约束残差 ---
        J[2 * N, 2] = depth_gradient * np.sqrt(depth_lambda)
        J[2 * N, 3] = 0.0
        
        return J

    
    def _solve_normal_equations(self, J: np.ndarray, residual: np.ndarray, lambda_lm: float,alpha: float = 0.5) -> np.ndarray:
        """求解正规方程"""
        # 现在只使用拟合点，所以雅可比矩阵J和残差的维度应该匹配
        # 合并残差：surface_residual和center_residual都是长度为N的数组
        
        # 检查维度匹配 - 现在J是(2*N, 4)，residual长度应该是2*N
        if len(residual) != J.shape[0]:
            logging.error(f"维度不匹配: J.shape[0]={J.shape[0]}, residual长度={len(residual)}")
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
            logging.warning("线性方程组求解失败，使用最小二乘法求解")
            delta = np.linalg.lstsq(H_lm, -g, rcond=None)[0]
        
        logging.debug(f"求解线性方程组结果: {delta}")
        return delta
    

class RANSACOptimizer(Optimizer):
    """RANSAC优化器"""
    
    def __init__(self, max_iterations: int = 50, threshold: float = 0.3, 
                 min_inlier_ratio: float = 0.7, base_optimizer: Optimizer = None):
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.min_inlier_ratio = min_inlier_ratio
        self.base_optimizer = base_optimizer or LevenbergMarquardtOptimizer()
        self.residual_calculator = ResidualCalculator()
    
    def optimize(self, initial_params: FittingParameters, 
                fitting_points: List[Tuple[np.ndarray, str]],
                constraint_points: List[Tuple[np.ndarray, str]]) -> FittingResult:
        
        best_result = None
        best_inlier_count = 0
        
        for iteration in range(self.max_iterations):
            # 随机采样
            if len(fitting_points) >= 4:
                import random
                sampled_points = random.sample(fitting_points, 4)
            else:
                sampled_points = fitting_points
            
            # 使用基础优化器
            result = self.base_optimizer.optimize(initial_params, sampled_points, constraint_points)
            
            # 计算内点数量
            inlier_count = self._count_inliers(result.parameters, fitting_points)
            
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_result = result
        
        if best_result is None:
            # 如果没有找到好的结果，返回初始参数
            return FittingResult(
                parameters=initial_params,
                confidence=0.0,
                converged=False,
                iteration_count=0,
                final_residual_norm=float('inf'),
                strategy_used="ransac"
            )
        
        return best_result
    
    def _count_inliers(self, params: FittingParameters, 
                      points: List[Tuple[np.ndarray, str]]) -> int:
        """计算内点数量"""
        if not points:
            return 0
        
        surface_residual, _ = self.residual_calculator.calculate_residuals(params, points)
        inlier_count = sum(1 for r in surface_residual if abs(r) < self.threshold)
        return inlier_count

class FailOptimizer(Optimizer):
    """失败优化器"""
    
    def optimize(self, initial_params: FittingParameters, 
                fitting_points: List[Tuple[np.ndarray, str]],
                constraint_points: List[Tuple[np.ndarray, str]]) -> FittingResult:
        logging.warning("优化失败")
        return FittingResult(
            parameters=FittingParameters(center=np.array([0, 0, 0]), radius=0.0),
            confidence=0.0,
            converged=False,
            iteration_count=0,
            final_residual_norm=float('inf'),
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
        logging.info(f"数据质量: {data_quality}")
        logging.debug(f"可用策略数量: {len(self.strategies)}")
        
        # 按可见性降序排列策略
        sorted_strategies = sorted(self.strategies, key=lambda s: s.min_visibility, reverse=True)
        
        for i, strategy in enumerate(sorted_strategies):
            logging.debug(f"策略 {i+1}: {strategy.name}")
            logging.debug(f"  最小点数: {strategy.min_points}, 最小可见性: {strategy.min_visibility}")
            logging.debug(f"  当前点数: {data_quality['point_count']}, 当前可见性: {data_quality['visibility']:.3f}")
            
            if (data_quality["point_count"] >= strategy.min_points and 
                data_quality["visibility"] >= strategy.min_visibility):
                logging.info(f"选择策略: {strategy.name}")
                return strategy
            else:
                logging.debug(f"  不满足条件，跳过")
        
        # 如果没有找到合适的策略，返回置信度最高的策略
        best_strategy = max(self.strategies, key=lambda s: s.confidence)
        logging.warning(f"没有找到满足条件的策略，使用置信度最高的策略: {best_strategy.name}")
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
    
    def _initialize_parameters(self, fitting_points: List[Tuple[np.ndarray, str]]) -> FittingParameters:
        """初始化参数"""
        if not fitting_points:
            return FittingParameters(center=np.array([0, 0, 0]), radius=ANATOMICAL_CONSTRAINTS["EYEBALL_RADIUS_DEFAULT"])
        
        # 简单的初始化策略：使用第一个点的位置作为球心
        first_point = fitting_points[0][0]
        return FittingParameters(center=first_point, radius=ANATOMICAL_CONSTRAINTS["EYEBALL_RADIUS_DEFAULT"])
    
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
            min_points=8
        ),
        # 中质量数据策略
        FittingStrategy(
            name="medium_quality",
            sampling_strategy=RandomSamplingStrategy(),
            optimizer=RANSACOptimizer(),
            confidence=0.8,
            min_visibility=0.7,
            min_points=4
        ),
        # 低质量数据策略
        FittingStrategy(
            name="low_quality",
            sampling_strategy=RandomSamplingStrategy(min_fitting_points=3),
            optimizer=RANSACOptimizer(max_iterations=30),
            confidence=0.9,
            min_visibility=0.5,
            min_points=3,
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
def create_fitting_controller(data_manager: RecgFitDataManager = RECG_FIT_DATA_MANAGER) -> FittingController:
    """创建拟合控制器"""
    data_provider = DataProvider(data_manager)
    strategies = create_default_strategies()
    strategy_selector = StrategySelector(strategies)
    return FittingController(data_provider, strategy_selector)

def fit_eye(eye: str, data_manager: RecgFitDataManager = RECG_FIT_DATA_MANAGER, params_only: bool = False) -> FittingResult:
    """简单的拟合函数"""
    controller = create_fitting_controller(data_manager)
    result = controller.fit_eye(eye)
    if params_only:
        return result.parameters
    return result

def fit_all_eyes(data_manager: RecgFitDataManager = RECG_FIT_DATA_MANAGER, params_only: bool = False) -> Dict[str, FittingResult]:
    """简单的拟合所有眼睛函数"""
    controller = create_fitting_controller(data_manager)
    results = controller.fit_all_eyes()
    if params_only:
        return {eye: result.parameters for eye, result in results.items()}
    return results