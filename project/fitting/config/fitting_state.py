from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from project.fitting.config.models import EllipsoidParams



@dataclass
class FittingResult:
    """拟合结果"""
    ellipsoid_params: EllipsoidParams
    center: np.ndarray
    inlier_count: int
    total_points: int
    inlier_ratio: float
    final_residual: float
    convergence_iterations: int
    converged: bool
    quality_score: float
    warnings: List[str]
    
    def __str__(self) -> str:
        """格式化输出拟合结果信息"""
        # 格式化中心点坐标
        center_str = f"({self.center[0]:.3f}, {self.center[1]:.3f}, {self.center[2]:.3f})"
        
        # 格式化半径（假设是球体，取第一个轴作为半径）
        radius = self.ellipsoid_params.axes[0] if len(self.ellipsoid_params.axes) > 0 else 0.0
        radius_str = f"{radius:.2f}mm"
        
        # 格式化收敛状态
        convergence_str = "✓ 已收敛" if self.converged else "✗ 未收敛"
        
        # 格式化内点比例
        inlier_ratio_str = f"{self.inlier_ratio:.1%}"
        
        # 格式化质量分数
        quality_str = f"{self.quality_score:.3f}"
        
        # 格式化残差
        residual_str = f"{self.final_residual:.2e}"
        
        # 构建警告信息
        warnings_str = ""
        if self.warnings:
            warnings_str = f"\n⚠️  警告: {'; '.join(self.warnings)}"
        
        # 构建完整的结果字符串
        result_str = (
            f"拟合结果:\n"
            f"  📍 中心点: {center_str}\n"
            f"  📏 半径: {radius_str}\n"
            f"  🔄 收敛状态: {convergence_str}\n"
            f"  📊 内点比例: {inlier_ratio_str} ({self.inlier_count}/{self.total_points})\n"
            f"  🎯 质量分数: {quality_str}\n"
            f"  📈 迭代次数: {self.convergence_iterations}\n"
            f"  📉 最终残差: {residual_str}"
            f"{warnings_str}"
        )
        
        return result_str

@dataclass
class FittingStatus:
    """拟合状态"""
    is_running: bool
    current_iteration: int
    max_iterations: int
    current_residual: float
    converged: bool
    error_message: Optional[str] = None

@dataclass
class ConvergenceStep:
    """收敛步骤"""
    iteration: int
    residual_norm: float
    parameter_change: float
    converged: bool

@dataclass
class QualityMetrics:
    """质量指标"""
    inlier_ratio: float
    residual_norm: float
    parameter_stability: float
    convergence_speed: float
    overall_score: float

@dataclass
class WeightResult:
    """权重计算结果"""
    weights: np.ndarray
    anatomical_weights: np.ndarray
    geometric_weights: np.ndarray
    structure_types: List[str]
    confidence_scores: np.ndarray
    calculation_time: float


@dataclass
class SamplingResult:
    """采样结果"""
    sampled_points: np.ndarray
    sampling_method: str
    inlier_indices: List[int]
    outlier_indices: List[int]
    sampling_quality: float
    sampling_time: float

@dataclass
class ConstraintValidationResult:
    """约束验证结果"""
    all_constraints_satisfied: bool
    constraint_violations: List[str]
    anatomical_valid: bool
    geometric_valid: bool
    validation_score: float
    suggestions: List[str]

@dataclass
class CurrentFit:
    """当前拟合状态"""
    center: np.ndarray
    radius: float
    iteration: int
    residual_norm: float

@dataclass
class ConstraintLMState:
    """约束LM优化状态"""
    iteration: int
    lambda_lm: float
    parameter_change: float
    residual_change: float
    trust_region_radius: float
    convergence_ratio: float
    converged: bool
    error_message: Optional[str] = None

@dataclass
class JacobianResult:
    """雅可比矩阵计算结果"""
    J_f: np.ndarray                    # 数据残差雅可比矩阵
    J_g: np.ndarray                    # 约束残差雅可比矩阵
    J_augmented: np.ndarray            # 增广雅可比矩阵
    calculation_time: float

@dataclass
class LMUpdateResult:
    """LM更新结果"""
    delta_theta: np.ndarray            # 参数更新向量
    lambda_lm: float                   # 当前阻尼因子
    trust_region_radius: float         # 信赖域半径
    convergence_ratio: float           # 收敛比例
    success: bool                      # 更新是否成功