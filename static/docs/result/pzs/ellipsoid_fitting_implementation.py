"""
椭球拟合眼部算法实现

RANSAC + 几何最小二乘（Geometric Least Squares）相结合：
- RANSAC 进行内点筛选与初值估计
- 在内点上使用几何残差进行最小二乘精修（可选鲁棒损失）

支持 MediaPipe 478 个关键点的输入，包含权重优化和先验约束。

作者：基于眼球中心拟合算法调研报告
日期：2024
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy.optimize import least_squares
from dataclasses import dataclass


@dataclass
class EllipsoidParams:
    """椭球参数数据结构"""
    center: np.ndarray  # 椭球中心 (3,)
    axes: np.ndarray    # 三主轴长度 (3,)
    rotation: np.ndarray  # 旋转矩阵 (3,3)
    
    def is_sphere(self, tolerance: float = 0.1) -> bool:
        """判断是否为球体（三轴长度相近）"""
        return np.std(self.axes) < tolerance
    
    def flattening_ratio(self) -> float:
        """计算扁平率（最短轴/最长轴）"""
        return np.min(self.axes) / np.max(self.axes)


class EllipsoidFitter:
    """椭球拟合器主类"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化椭球拟合器
        
        Args:
            config: 配置参数字典
        """
        self.config = config or self._get_default_config()
        self._validate_config()
    
    @staticmethod
    def _orthonormalize_rotation(R: np.ndarray) -> np.ndarray:
        """对 3x3 矩阵进行正交化，保证为合法旋转矩阵（SVD 投影）。"""
        try:
            U, _, Vt = np.linalg.svd(R)
            R_ortho = U @ Vt
            # 确保右手坐标系
            if np.linalg.det(R_ortho) < 0:
                U[:, -1] *= -1
                R_ortho = U @ Vt
            return R_ortho
        except Exception:
            return np.eye(3)
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            "threshold_mm": 1.5,           # 内点阈值（毫米）
            "max_trials": 200,             # RANSAC最大迭代次数
            "min_inlier_ratio": 0.8,      # 最小内点比例
            "flattening_ratio_range": (0.85, 0.95),  # 扁平率范围
            "axes_range_mm": (10.0, 14.0), # 轴长范围（毫米）
            "center_constraint_mm": 8.0,   # 瞳孔中心距离约束（毫米）
            "use_prior": True,             # 是否使用先验约束
            "weight_strategy": "anatomical", # 权重策略
            "ls_loss": "soft_l1",        # 几何最小二乘鲁棒损失（linear/soft_l1/huber/cauchy/arctan）
            "ls_f_scale": 1.0             # 鲁棒损失尺度
        }
    
    def _validate_config(self):
        """验证配置参数"""
        assert 0 < self.config["threshold_mm"] < 10, "阈值应在合理范围内"
        assert 10 < self.config["max_trials"] < 1000, "迭代次数应在合理范围内"
        assert 0 < self.config["min_inlier_ratio"] < 1, "内点比例应在(0,1)范围内"
    
    def fit_eyeball_center(
        self,
        key_coordinates: Dict[str, Union[np.ndarray, List[np.ndarray]]],
        eye_side: str,
        model_type: str = "ellipsoid",
        threshold: Optional[float] = None,
        max_trials: Optional[int] = None,
        use_prior: Optional[bool] = None
    ) -> Tuple[EllipsoidParams, np.ndarray]:
        """
        拟合单眼眼球中心，支持球体和椭球体模型
        
        Args:
            key_coordinates: 关键点坐标数据
            eye_side: 眼睛侧别 ("left" 或 "right")
            model_type: 模型类型 ("sphere" 或 "ellipsoid")
            threshold: 内点阈值（毫米）
            max_trials: RANSAC最大迭代次数
            use_prior: 是否使用解剖学先验约束
            
        Returns:
            ellipsoid_params: 椭球参数
            center: 眼球中心坐标 (3,)
        """
        # 更新配置参数
        if threshold is not None:
            self.config["threshold_mm"] = threshold
        if max_trials is not None:
            self.config["max_trials"] = max_trials
        if use_prior is not None:
            self.config["use_prior"] = use_prior
        
        # 提取眼睛关键点
        points, point_types = self._extract_eye_points(key_coordinates, eye_side)
        
        if len(points) < 9:  # 椭球拟合至少需要9个点
            raise ValueError(f"点数不足：{len(points)} < 9，无法进行椭球拟合")
        
        # 计算拟合权重
        weights = self._calculate_fitting_weights(points, point_types)
        
        # 执行RANSAC椭球拟合
        if model_type == "sphere":
            ellipsoid_params, center = self._fit_sphere_ransac(points, weights)
        else:
            ellipsoid_params, center = self._fit_ellipsoid_ransac(points, weights)
        
        # 应用先验约束
        if self.config["use_prior"]:
            ellipsoid_params = self._apply_anatomical_constraints(
                ellipsoid_params, key_coordinates, eye_side
            )
        
        return ellipsoid_params, center
    
    def _extract_eye_points(
        self, 
        key_coordinates: Dict[str, Union[np.ndarray, List[np.ndarray]]], 
        eye_side: str
    ) -> Tuple[np.ndarray, List[str]]:
        """
        提取眼睛关键点坐标和类型
        
        Args:
            key_coordinates: 关键点坐标数据
            eye_side: 眼睛侧别
            
        Returns:
            points: 点坐标数组 (N, 3)
            point_types: 点类型列表
        """
        points = []
        point_types = []
        
        # 瞳孔中心（最高权重）
        if f"{eye_side}_pupil" in key_coordinates:
            pupil_center = key_coordinates[f"{eye_side}_pupil"]
            if pupil_center is not None and not np.allclose(pupil_center, 0, equal_nan=False):
                points.append(pupil_center)
                point_types.append("pupil")
        
        # 虹膜边界点（高权重）
        if f"{eye_side}_iris" in key_coordinates:
            iris_points = key_coordinates[f"{eye_side}_iris"]
            if iris_points and len(iris_points) > 0:
                for point in iris_points:
                    if point is not None and not np.allclose(point, 0, equal_nan=False):
                        points.append(point)
                        point_types.append("iris")
        
        # 眼睛轮廓点（中等权重）
        if f"{eye_side}_eye_contour" in key_coordinates:
            contour_points = key_coordinates[f"{eye_side}_eye_contour"]
            if contour_points and len(contour_points) > 0:
                for point in contour_points:
                    if point is not None and not np.allclose(point, 0, equal_nan=False):
                        points.append(point)
                        point_types.append("contour")
        
        # 眼眶关键点（增强约束）
        if f"{eye_side}_eye_socket" in key_coordinates:
            socket_points = key_coordinates[f"{eye_side}_eye_socket"]
            if socket_points and len(socket_points) > 0:
                for point in socket_points:
                    if point is not None and not np.allclose(point, 0, equal_nan=False):
                        points.append(point)
                        point_types.append("socket")
        
        # 眼睑关键点（提升边界精度）
        if f"{eye_side}_eyelid" in key_coordinates:
            eyelid_points = key_coordinates[f"{eye_side}_eyelid"]
            if eyelid_points and len(eyelid_points) > 0:
                for point in eyelid_points:
                    if point is not None and not np.allclose(point, 0, equal_nan=False):
                        points.append(point)
                        point_types.append("eyelid")
        
        if not points:
            raise ValueError(f"未找到有效的{eye_side}眼关键点")
        
        return np.array(points), point_types
    
    def _calculate_fitting_weights(
        self,
        points: np.ndarray,
        point_types: List[str],
        confidence_scores: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        计算拟合点的权重，考虑点类型和置信度
        
        Args:
            points: 拟合点坐标 (N, 3)
            point_types: 点类型列表
            confidence_scores: 置信度分数 (N,)
            
        Returns:
            weights: 权重数组 (N,)
        """
        n_points = len(points)
        weights = np.ones(n_points)
        
        # 基于解剖学重要性的权重策略
        for i, point_type in enumerate(point_types):
            if point_type == "pupil":
                weights[i] = 1.0      # 瞳孔中心：最高权重
            elif point_type == "iris":
                weights[i] = 0.8      # 虹膜边界：高权重
            elif point_type == "contour":
                weights[i] = 0.6      # 眼睛轮廓：中等权重
            elif point_type == "socket":
                weights[i] = 0.7      # 眼眶：较高权重
            elif point_type == "eyelid":
                weights[i] = 0.5      # 眼睑：较低权重
            else:
                weights[i] = 0.4      # 其他：最低权重
        
        # 置信度调整（±20%范围）
        if confidence_scores is not None:
            confidence_adjustment = 0.2 * (confidence_scores - 0.5) / 0.5
            weights *= (1.0 + confidence_adjustment)
        
        # 归一化权重
        weights = weights / np.sum(weights) * n_points
        
        return weights
    
    def _fit_sphere_ransac(
        self, 
        points: np.ndarray, 
        weights: np.ndarray
    ) -> Tuple[EllipsoidParams, np.ndarray]:
        """
        使用RANSAC拟合球体
        
        Args:
            points: 点坐标 (N, 3)
            weights: 权重数组 (N,)
            
        Returns:
            ellipsoid_params: 椭球参数（球体）
            center: 球心坐标
        """
        best_inliers = []
        best_params = None
        best_score = 0
        
        threshold = self.config["threshold_mm"] / 1000.0  # 转换为米
        
        for _ in range(self.config["max_trials"]):
            # 随机选择4个点拟合球体
            idx = np.random.choice(len(points), 4, replace=False)
            sample_points = points[idx]
            
            try:
                # 初始估计
                center_init = np.mean(sample_points, axis=0)
                radius_init = np.mean(np.linalg.norm(sample_points - center_init, axis=1))
                
                # 最小二乘拟合
                params_init = np.append(center_init, radius_init)
                result = least_squares(
                    self._sphere_residuals, 
                    params_init, 
                    args=(sample_points,),
                    method='lm'
                )
                
                if result.success:
                    center, radius = result.x[:3], result.x[3]
                    
                    # 计算内点
                    distances = np.abs(np.linalg.norm(points - center, axis=1) - radius)
                    inliers = distances < threshold
                    inlier_count = np.sum(inliers)
                    
                    # 更新最佳结果
                    if inlier_count > len(best_inliers):
                        best_inliers = np.where(inliers)[0]
                        best_params = (center, radius)
                        best_score = inlier_count
                        
            except Exception:
                continue
        
        if best_params is None:
            raise RuntimeError("RANSAC球体拟合失败")
        
        # 用所有内点重新拟合
        center, radius = best_params
        if len(best_inliers) >= 4:
            inlier_points = points[best_inliers]
            inlier_weights = weights[best_inliers]
            
            # 加权最小二乘精细拟合
            center_init = np.average(inlier_points, axis=0, weights=inlier_weights)
            radius_init = np.average(
                np.linalg.norm(inlier_points - center_init, axis=1), 
                weights=inlier_weights
            )
            
            params_init = np.append(center_init, radius_init)
            result = least_squares(
                self._sphere_residuals, 
                params_init, 
                args=(inlier_points,),
                method='lm'
            )
            
            if result.success:
                center, radius = result.x[:3], result.x[3]
        
        # 构造椭球参数（球体）
        ellipsoid_params = EllipsoidParams(
            center=center,
            axes=np.array([radius, radius, radius]),
            rotation=np.eye(3)
        )
        
        return ellipsoid_params, center
    
    def _fit_ellipsoid_ransac(
        self, 
        points: np.ndarray, 
        weights: np.ndarray
    ) -> Tuple[EllipsoidParams, np.ndarray]:
        """
        使用RANSAC拟合椭球体
        
        Args:
            points: 点坐标 (N, 3)
            weights: 权重数组 (N,)
            
        Returns:
            ellipsoid_params: 椭球参数
            center: 椭球中心坐标
        """
        best_inliers = []
        best_params = None
        best_score = 0
        
        threshold = self.config["threshold_mm"] / 1000.0  # 转换为米
        
        for _ in range(self.config["max_trials"]):
            # 随机选择9个点拟合椭球体
            if len(points) < 9:
                idx = np.random.choice(len(points), len(points), replace=False)
            else:
                idx = np.random.choice(len(points), 9, replace=False)
            
            sample_points = points[idx]
            
            try:
                # 初始估计
                center_init = np.mean(sample_points, axis=0)
                
                # 计算协方差矩阵
                centered_points = sample_points - center_init
                cov_matrix = np.cov(centered_points.T)
                
                # 特征值分解
                eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
                axes_init = np.sqrt(np.abs(eigenvals)) * 2  # 转换为直径
                
                # 确保轴长在合理范围内
                axes_init = np.clip(axes_init, 0.01, 0.03)  # 10-30mm
                
                # 构造初始参数向量
                params_init = np.concatenate([center_init, axes_init, eigenvecs.flatten()])
                
                # 最小二乘拟合
                result = least_squares(
                    self._ellipsoid_residuals, 
                    params_init, 
                    args=(sample_points,),
                    method='trf',
                    loss=self.config.get("ls_loss", "soft_l1"),
                    f_scale=self.config.get("ls_f_scale", 1.0),
                    max_nfev=2000
                )
                
                if result.success:
                    params = result.x
                    center = params[:3]
                    axes = params[3:6]
                    rotation = params[6:].reshape(3, 3)
                    
                    # 计算内点
                    distances = self._ellipsoid_distances(points, center, axes, rotation)
                    inliers = distances < threshold
                    inlier_count = np.sum(inliers)
                    
                    # 更新最佳结果
                    if inlier_count > len(best_inliers):
                        best_inliers = np.where(inliers)[0]
                        best_params = (center, axes, rotation)
                        best_score = inlier_count
                        
            except Exception:
                continue
        
        if best_params is None:
            raise RuntimeError("RANSAC椭球体拟合失败")
        
        # 用所有内点重新拟合
        center, axes, rotation = best_params
        if len(best_inliers) >= 9:
            inlier_points = points[best_inliers]
            inlier_weights = weights[best_inliers]
            
            # 加权最小二乘精细拟合
            center_init = np.average(inlier_points, axis=0, weights=inlier_weights)
            
            centered_points = inlier_points - center_init
            cov_matrix = np.cov(centered_points.T, aweights=inlier_weights)
            
            eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
            axes_init = np.sqrt(np.abs(eigenvals)) * 2
            
            params_init = np.concatenate([center_init, axes_init, eigenvecs.flatten()])
            result = least_squares(
                self._ellipsoid_residuals, 
                params_init, 
                args=(inlier_points,),
                method='trf',
                loss=self.config.get("ls_loss", "soft_l1"),
                f_scale=self.config.get("ls_f_scale", 1.0),
                max_nfev=2000
            )
            
            if result.success:
                params = result.x
                center = params[:3]
                axes = params[3:6]
                rotation = params[6:].reshape(3, 3)
                rotation = self._orthonormalize_rotation(rotation)
        
        # 构造椭球参数
        ellipsoid_params = EllipsoidParams(
            center=center,
            axes=axes,
            rotation=rotation
        )
        
        return ellipsoid_params, center
    
    def _sphere_residuals(self, params: np.ndarray, points: np.ndarray) -> np.ndarray:
        """球体残差函数"""
        center, radius = params[:3], params[3]
        distances = np.linalg.norm(points - center, axis=1)
        return distances - radius
    
    def _ellipsoid_residuals(self, params: np.ndarray, points: np.ndarray) -> np.ndarray:
        """椭球体残差函数"""
        center = params[:3]
        axes = params[3:6]
        rotation = params[6:].reshape(3, 3)
        
        # 将点转换到椭球坐标系
        centered_points = points - center
        transformed_points = np.dot(centered_points, rotation)
        
        # 计算椭球距离
        # 防止轴长异常或为零
        safe_axes = np.clip(axes, 1e-6, None)
        normalized_points = transformed_points / safe_axes
        distances = np.linalg.norm(normalized_points, axis=1)
        
        return distances - 1.0
    
    def _ellipsoid_distances(
        self, 
        points: np.ndarray, 
        center: np.ndarray, 
        axes: np.ndarray, 
        rotation: np.ndarray
    ) -> np.ndarray:
        """计算点到椭球面的距离"""
        centered_points = points - center
        transformed_points = np.dot(centered_points, rotation)
        
        # 椭球距离近似
        normalized_points = transformed_points / axes
        distances = np.abs(np.linalg.norm(normalized_points, axis=1) - 1.0)
        
        return distances
    
    def _apply_anatomical_constraints(
        self,
        ellipsoid_params: EllipsoidParams,
        key_coordinates: Dict[str, Union[np.ndarray, List[np.ndarray]]],
        eye_side: str
    ) -> EllipsoidParams:
        """
        应用解剖学先验约束
        
        Args:
            ellipsoid_params: 椭球参数
            key_coordinates: 关键点坐标
            eye_side: 眼睛侧别
            
        Returns:
            约束后的椭球参数
        """
        # 瞳孔中心距离约束
        if f"{eye_side}_pupil" in key_coordinates:
            pupil_center = key_coordinates[f"{eye_side}_pupil"]
            if pupil_center is not None and not np.allclose(pupil_center, 0, equal_nan=False):
                distance = np.linalg.norm(ellipsoid_params.center - pupil_center)
                max_distance = self.config["center_constraint_mm"] / 1000.0
                
                if distance > max_distance:
                    # 调整椭球中心，使其更接近瞳孔中心
                    direction = (pupil_center - ellipsoid_params.center) / distance
                    new_center = pupil_center - direction * (max_distance * 0.8)
                    ellipsoid_params.center = new_center
        
        # 轴长范围约束
        min_axis, max_axis = self.config["axes_range_mm"]
        min_axis, max_axis = min_axis / 1000.0, max_axis / 1000.0
        
        ellipsoid_params.axes = np.clip(ellipsoid_params.axes, min_axis, max_axis)
        
        # 扁平率约束
        min_flattening, max_flattening = self.config["flattening_ratio_range"]
        current_flattening = ellipsoid_params.flattening_ratio()
        
        if current_flattening < min_flattening:
            # 调整最短轴
            min_axis_idx = np.argmin(ellipsoid_params.axes)
            max_axis_val = np.max(ellipsoid_params.axes)
            ellipsoid_params.axes[min_axis_idx] = max_axis_val * min_flattening
        
        return ellipsoid_params


def demo_ellipsoid_fitting():
    """演示椭球拟合功能"""
    
    # 创建模拟的眼部关键点数据
    np.random.seed(42)
    
    # 模拟眼球中心
    true_center = np.array([0.1, 0.05, 0.8])
    true_axes = np.array([0.012, 0.012, 0.011])  # 12mm, 12mm, 11mm
    
    # 生成模拟点
    n_points = 50
    points = []
    point_types = []
    
    # 瞳孔中心
    points.append(true_center + np.array([0, 0, true_axes[2] * 0.8]))
    point_types.append("pupil")
    
    # 虹膜边界点
    for i in range(8):
        angle = i * 2 * np.pi / 8
        radius = true_axes[0] * 0.9
        x = true_center[0] + radius * np.cos(angle)
        y = true_center[1] + radius * np.sin(angle)
        z = true_center[2] + true_axes[2] * 0.7
        points.append(np.array([x, y, z]))
        point_types.append("iris")
    
    # 眼睛轮廓点
    for i in range(16):
        angle = i * 2 * np.pi / 16
        radius = true_axes[0] * 1.1
        x = true_center[0] + radius * np.cos(angle)
        y = true_center[1] + radius * np.sin(angle)
        z = true_center[2] + true_axes[2] * 0.6
        points.append(np.array([x, y, z]))
        point_types.append("contour")
    
    # 添加噪声
    points = np.array(points)
    noise = np.random.normal(0, 0.001, points.shape)  # 1mm噪声
    points += noise
    
    # 构造关键点数据结构
    key_coordinates = {
        "left_pupil": points[0],
        "left_iris": points[1:9],
        "left_eye_contour": points[9:25]
    }
    
    # 创建拟合器
    fitter = EllipsoidFitter()
    
    try:
        # 拟合椭球体
        ellipsoid_params, center = fitter.fit_eyeball_center(
            key_coordinates, "left", "ellipsoid"
        )
        
        print("=== 椭球拟合结果 ===")
        print(f"拟合中心: {center * 1000:.2f} mm")
        print(f"真实中心: {true_center * 1000:.2f} mm")
        print(f"中心误差: {np.linalg.norm(center - true_center) * 1000:.2f} mm")
        
        print(f"\n拟合轴长: {ellipsoid_params.axes * 1000:.2f} mm")
        print(f"真实轴长: {true_axes * 1000:.2f} mm")
        print(f"轴长误差: {np.linalg.norm(ellipsoid_params.axes - true_axes) * 1000:.2f} mm")
        
        print(f"\n扁平率: {ellipsoid_params.flattening_ratio():.3f}")
        print(f"是否为球体: {ellipsoid_params.is_sphere()}")
        
        # 拟合球体对比
        sphere_params, sphere_center = fitter.fit_eyeball_center(
            key_coordinates, "left", "sphere"
        )
        
        print(f"\n=== 球体拟合对比 ===")
        print(f"球体中心误差: {np.linalg.norm(sphere_center - true_center) * 1000:.2f} mm")
        print(f"椭球体中心误差: {np.linalg.norm(center - true_center) * 1000:.2f} mm")
        print(f"精度提升倍数: {np.linalg.norm(sphere_center - true_center) / np.linalg.norm(center - true_center):.1f}")
        
    except Exception as e:
        print(f"拟合失败: {e}")


if __name__ == "__main__":
    demo_ellipsoid_fitting()
