"""Geometry helpers (placeholders)."""

from typing import TypedDict, Tuple, List, Dict
import numpy as np
from project.fitting.config.constants import KeyCoordinates, SingleEyeKeyCoordinates
from project.fitting.config.settings import SCREEN_WITH_RGBD

class EllipsoidParams(TypedDict):
    """
    椭球参数占位类型（仅类型标注用）：
    - axes: 三主轴长度 (a, b, c)，单位 m，np.ndarray(3,)
    - rotation: 世界到椭球主轴坐标的旋转矩阵 R，np.ndarray(3,3)
    """
    axes: np.ndarray           # (3,)
    rotation: np.ndarray       # (3, 3)
    center: np.ndarray        # (3,)
    
    def get_axes(self)->np.ndarray:
        return self["axes"]
    
    def get_rotation(self)->np.ndarray:
        return self["rotation"]
    
    def get_center(self)->np.ndarray:
        return self["center"]

class Plane(TypedDict, total=False):
    """
    平面类型（仅类型标注用）：
    - normal: 平面法向量，np.ndarray(3,)
    - point: 平面坐标系的原点在3D世界坐标系中的坐标，np.ndarray(3,)
    - x_axis: 平面坐标系的x轴向量，np.ndarray(3,)
    - y_axis: 平面坐标系的y轴向量，np.ndarray(3,)
    - top_left/top_right/bottom_left/bottom_right: 屏幕四角（3D）
    - width_m/height_m: 屏幕物理宽高（米）
    """
    normal: np.ndarray
    point: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    
    def __init__(self,rgb_d=False):
        if rgb_d:
            self["normal"] = SCREEN_WITH_RGBD["normal"]
            self["point"] = SCREEN_WITH_RGBD["point"]
            self["x_axis"] = SCREEN_WITH_RGBD["x_axis"]
            self["y_axis"] = SCREEN_WITH_RGBD["y_axis"]
            self["top_left"] = SCREEN_WITH_RGBD["top_left"]
            self["top_right"] = SCREEN_WITH_RGBD["top_right"]
            self["bottom_left"] = SCREEN_WITH_RGBD["bottom_left"]
            self["bottom_right"] = SCREEN_WITH_RGBD["bottom_right"]
            # 补充物理尺寸（米）
            self["width_m"] = float(SCREEN_WITH_RGBD.get("width_m", 0.0))
            self["height_m"] = float(SCREEN_WITH_RGBD.get("height_m", 0.0))
        else:
            self["normal"] = np.array([0,0,1])
            self["point"] = np.array([0,0,0])
            self["x_axis"] = np.array([1,0,0])
            self["y_axis"] = np.array([0,1,0])
            self["top_left"] = np.array([-0.30, 0.005, 0])
            self["top_right"] = np.array([0.30, 0.005, 0])
            self["bottom_left"] = np.array([-0.30, 0.345, 0])
            self["bottom_right"] = np.array([0.30, 0.345, 0])
            # 默认物理尺寸（米）
            self["width_m"] = 0.60
            self["height_m"] = 0.34
            
            
    def _intersection_of_vector_and_plane(self,vector: np.ndarray) -> np.ndarray | None:
        '''
        计算向量与平面的交点（假设向量起点为世界原点）。
        Args:
            vector: 方向向量 (3,)
            plane: 平面（包含 'normal'(3,), 'point'(3,)）
        Returns:
            交点 np.ndarray(3,)，若无唯一交点（平行或共面）返回 None
        '''
        v = np.asarray(vector, dtype=float).reshape(3)
        n = np.asarray(self["normal"], dtype=float).reshape(3)
        p0 = np.asarray(self["point"], dtype=float).reshape(3)

        denom = float(np.dot(n, v))
        if np.isclose(denom, 0.0):
            # 平行：若原点在平面上则无唯一交点（共面），否则无交点
            return None

        t = float(np.dot(n, p0) / denom)
        return v * t

    def _point_3d_to_2d(self,point: np.ndarray) -> np.ndarray:
        '''
        将3D点转换成在平面2D坐标系下的坐标
        Args:
            point: 3D点
            plane: 平面
        Returns:
            2D坐标
        '''
        x = np.dot(point - self["point"],self["x_axis"])
        y = np.dot(point - self["point"],self["y_axis"])
        return np.array([x,y])
    
    def intersection_on_plane(self,vector:np.ndarray)->np.ndarray | None:
        '''
        计算点与平面的交点
        Args:
            point: 点
        Returns:
            交点（在平面2D坐标系下）
        '''
        p3 = self._intersection_of_vector_and_plane(vector)
        if p3 is None:
            return None
        return self._point_3d_to_2d(p3)
        
            

class CenterFitter:
    def __init__(self, sphere: bool = True):
        print(f"[DEBUG] CenterFitter初始化: sphere={sphere}")
        # 模型类型
        self._sphere = sphere
        
        # 初始化椭球参数
        self._ellipsoid_params = EllipsoidParams()
        if sphere:
            self._r = 0.012  # 默认半径12mm
            self._ellipsoid_params["axes"] = np.array([self._r, self._r, self._r])
            self._ellipsoid_params["rotation"] = np.eye(3)
            self._ellipsoid_params["center"] = np.array([0.0, 0.0, 0.0])
        
        # 状态管理变量
        self._current_weights = None
        self._current_threshold = None
        self._weights_calculated = False
        
        # 从settings.py获取拟合参数
        from project.fitting.config.settings import FITTING_ALGORITHM_CONFIG
        self._fitting_config = FITTING_ALGORITHM_CONFIG
        
        # 从initial_weight.py获取权重参数
        from project.fitting.config.initial_weight import (
            ANATOMICAL_WEIGHT_PARAMS, GEOMETRIC_WEIGHT_PARAMS, GLOBAL_WEIGHT_CONFIG
        )
        self._anatomical_weights = ANATOMICAL_WEIGHT_PARAMS
        self._geometric_params = GEOMETRIC_WEIGHT_PARAMS
        self._global_config = GLOBAL_WEIGHT_CONFIG
        print(f"[DEBUG] CenterFitter初始化完成")
        
    # 主要接口函数
    def center_fitter(self, key_coordinates: SingleEyeKeyCoordinates, 
                     trials_times: int) -> Tuple[EllipsoidParams, np.ndarray]:
        """主拟合接口，实现调用_fit_center_only或者_int_center_fitter"""
        print(f"[DEBUG] center_fitter开始: trials_times={trials_times}")
        max_trials = self._fitting_config.get("max_trials", 100)
        
        if trials_times >= max_trials:
            print(f"[DEBUG] 超过最大实验次数，只拟合球心坐标")
            # 超过最大实验次数时，只拟合球心坐标，保持原有半径
            return self._fit_ellipsoid(key_coordinates, center_only=True)
        else:
            print(f"[DEBUG] 正常拟合模式")
            return self._fit_ellipsoid(key_coordinates, center_only=False)
    
    def _fit_ellipsoid(self, key_coordinates: SingleEyeKeyCoordinates, center_only: bool = False) -> Tuple[EllipsoidParams, np.ndarray]:
        """统一的椭球拟合函数，支持只拟合中心或完整拟合"""
        print(f"[DEBUG] _fit_ellipsoid开始: center_only={center_only}")
        # 1. 提取关键点数据
        points, structure_types = self._extract_key_points(key_coordinates)
        print(f"[DEBUG] 提取到 {len(points)} 个关键点")
        
        min_samples = 2 if center_only else self._fitting_config.get("min_samples", 4)
        if len(points) < min_samples:
            print(f"[DEBUG] 点数不足，返回默认参数")
            default_params = self._get_default_params()
            return default_params, default_params["center"]
        
        # 2. 初始化状态
        self._reset_fitting_state()
        
        # 3. RANSAC拟合
        best_params, best_inlier_count = self._run_ransac_fitting(points, structure_types, center_only)
        print(f"[DEBUG] RANSAC拟合完成: best_inlier_count={best_inlier_count}")
        
        # 4. 返回结果
        if best_params is not None:
            if center_only:
                # 只拟合中心时，保持原有半径
                current_radius = self._ellipsoid_params["axes"][0]
                result = EllipsoidParams()
                result["axes"] = np.array([current_radius, current_radius, current_radius])
                result["rotation"] = np.eye(3)
                result["center"] = best_params
                print(f"[DEBUG] 返回中心拟合结果")
                return result, best_params
            else:
                # 完整拟合
                ellipsoid_params = self._params_to_ellipsoid(best_params)
                print(f"[DEBUG] 返回完整拟合结果")
                return ellipsoid_params, ellipsoid_params["center"]
        else:
            print(f"[DEBUG] 拟合失败，返回默认参数")
            default_params = self._get_default_params()
            return default_params, default_params["center"]
    
    def _reset_fitting_state(self):
        """重置拟合状态"""
        print(f"[DEBUG] 重置拟合状态")
        self._current_weights = None
        self._current_threshold = self._fitting_config.get("ransac_threshold", 0.0015)
        self._weights_calculated = False
    
    def _run_ransac_fitting(self, points: List[np.ndarray], structure_types: List[str], center_only: bool) -> Tuple[np.ndarray, int]:
        """运行RANSAC拟合"""
        print(f"[DEBUG] _run_ransac_fitting开始: center_only={center_only}")
        best_params = None
        best_inlier_count = 0
        max_ransac_iterations = self._fitting_config.get("ransac_max_iterations", 50)
        
        for iteration in range(max_ransac_iterations):
            print(f"[DEBUG] RANSAC迭代 {iteration + 1}/{max_ransac_iterations}")
            # 单次RANSAC迭代
            params, inlier_count = self._run_single_ransac_iteration(points, structure_types, center_only)
            
            if params is not None and inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_params = params
                print(f"[DEBUG] 更新最佳结果: inlier_count={inlier_count}")
                
                # 更新阈值
                self._current_threshold = self._adaptive_threshold_adjustment(inlier_count, len(points))
        
        print(f"[DEBUG] RANSAC拟合完成: best_inlier_count={best_inlier_count}")
        return best_params, best_inlier_count
    
    def _run_single_ransac_iteration(self, points: List[np.ndarray], structure_types: List[str], center_only: bool) -> Tuple[np.ndarray, int]:
        """运行单次RANSAC迭代"""
        print(f"[DEBUG] _run_single_ransac_iteration开始")
        # 1. 采样点
        sampled_points, sampled_types = self._ransac_sampling(points, structure_types, center_only)
        print(f"[DEBUG] 采样到 {len(sampled_points)} 个点")
        
        min_samples = 2 if center_only else self._fitting_config.get("min_samples", 4)
        if len(sampled_points) < min_samples:
            print(f"[DEBUG] 采样点数不足")
            return None, 0
        
        # 2. 计算权重（如果未计算）
        if not self._weights_calculated:
            print(f"[DEBUG] 计算权重")
            self._calculate_weights(sampled_points, sampled_types, center_only)
        
        # 3. 初始参数估计
        initial_params = self._estimate_initial_params(sampled_points, center_only)
        print(f"[DEBUG] 初始参数: {initial_params}")
        
        # 4. 牛顿-高斯优化
        optimized_params, converged = self._newton_gauss_optimization(
            initial_params, sampled_points, center_only
        )
        print(f"[DEBUG] 牛顿-高斯优化: converged={converged}")
        
        if not converged:
            print(f"[DEBUG] 牛顿-高斯未收敛")
            return None, 0
        
        # 5. 计算内点数量
        inlier_count = self._count_inliers(optimized_params, points, structure_types, center_only)
        print(f"[DEBUG] 内点数量: {inlier_count}")
        
        return optimized_params, inlier_count
    
    def _ransac_sampling(self, points: List[np.ndarray], structure_types: List[str], center_only: bool) -> Tuple[List[np.ndarray], List[str]]:
        """RANSAC采样策略"""
        print(f"[DEBUG] _ransac_sampling开始: center_only={center_only}")
        n_points = len(points)
        min_samples = 2 if center_only else self._fitting_config.get("min_samples", 4)
        
        if n_points < min_samples:
            print(f"[DEBUG] 点数不足: {n_points} < {min_samples}")
            return [], []
        
        # 计算权重
        print(f"[DEBUG] 开始计算权重")
        self._calculate_weights(points, structure_types, center_only)
        print(f"[DEBUG] 权重计算完成: {self._current_weights}")
        
        # 检查是否有有效权重
        weight_sum = np.sum(self._current_weights)
        print(f"[DEBUG] 权重总和: {weight_sum}")
        if weight_sum == 0:
            print(f"[DEBUG] 权重总和为0")
            return [], []
        
        # 选择权重最高的点
        points_per_eye = self._fitting_config.get("points_per_eye", 4)
        selected_count = max(min_samples, min(points_per_eye, n_points))
        selected_indices = np.argsort(self._current_weights)[-selected_count:]
        print(f"[DEBUG] 选择的索引: {selected_indices}")
        
        # 修复：确保权重是标量进行比较
        sampled_points = []
        sampled_types = []
        for i in selected_indices:
            weight_val = float(self._current_weights[i])
            print(f"[DEBUG] 检查索引 {i}: weight={weight_val}")
            if weight_val > 0:
                sampled_points.append(points[i])
                sampled_types.append(structure_types[i])
        
        print(f"[DEBUG] 采样完成: {len(sampled_points)} 个点")
        return sampled_points, sampled_types
    
    def _calculate_weights(self, points: List[np.ndarray], structure_types: List[str], center_only: bool = False) -> None:
        """计算三层权重并缓存结果"""
        print(f"[DEBUG] _calculate_weights开始: center_only={center_only}")
        if not self._sphere:
            self._current_weights = np.ones(len(points))
            self._weights_calculated = True
            print(f"[DEBUG] 非球体模式，权重设为1")
            return
            
        n_points = len(points)
        self._current_weights = np.zeros(n_points)
        
        # 获取当前半径
        current_radius = self._ellipsoid_params["axes"][0]
        print(f"[DEBUG] 当前半径: {current_radius}")
        
        for i in range(n_points):
            try:
                print(f"[DEBUG] 计算第 {i} 个点权重")
                # 1. 解剖学先验权重
                w_anat = self._anatomical_weight(structure_types[i])
                print(f"[DEBUG] 解剖学权重: {w_anat}")
                
                # 2. 几何残差权重（基于当前椭球参数）
                # 修复：确保center不是None且是有效的数组
                center = self._ellipsoid_params["center"]
                print(f"[DEBUG] 当前中心: {center}")
                if center is not None and len(center) == 3:
                    print(f"[DEBUG] 中心有效，计算距离")
                    point = points[i]
                    print(f"[DEBUG] 点 {i}: {point}")
                    diff = point - center
                    print(f"[DEBUG] 差值: {diff}")
                    distance = abs(float(np.linalg.norm(diff)) - current_radius)
                    print(f"[DEBUG] 距离: {distance}")
                    geom_params = self._geometric_params.get(structure_types[i], {})
                    initial_sigma = geom_params.get('initial_sigma', 1.5)
                    w_geom = self._geometric_weight(distance, initial_sigma)
                    print(f"[DEBUG] 几何权重: {w_geom}")
                else:
                    print(f"[DEBUG] 中心无效，使用默认权重")
                    w_geom = 1.0
                
                # 3. 信号质量权重（默认1.0）
                w_snr = 1.0
                
                # 4. 综合权重
                final_weight = w_anat * w_geom * w_snr
                min_weight = self._global_config.get('min_weight', 0.1)
                print(f"[DEBUG] 最终权重: {final_weight}, 最小权重: {min_weight}")
                
                # 如果权重小于最小权重，则设为0
                if final_weight >= min_weight:
                    self._current_weights[i] = final_weight
                else:
                    self._current_weights[i] = 0.0
                
            except (KeyError, ValueError) as e:
                print(f"[DEBUG] 计算第{i}个点权重时出错: {e}")
                self._current_weights[i] = 0.0
        
        print(f"[DEBUG] 权重计算完成: {self._current_weights}")
        
        # 权重归一化（只对非零权重进行）
        if self._global_config.get('normalization', True):
            nonzero_weights = self._current_weights[self._current_weights > 0]
            print(f"[DEBUG] 非零权重: {nonzero_weights}")
            if len(nonzero_weights) > 0:
                mean_weight = np.mean(nonzero_weights)
                self._current_weights[self._current_weights > 0] = self._current_weights[self._current_weights > 0] / mean_weight
                print(f"[DEBUG] 归一化后权重: {self._current_weights}")
        
        self._weights_calculated = True
        print(f"[DEBUG] _calculate_weights完成")
    
    def _newton_gauss_optimization(self, initial_params: np.ndarray, 
                                 points: List[np.ndarray], center_only: bool = False) -> Tuple[np.ndarray, bool]:
        """牛顿-高斯算法优化"""
        print(f"[DEBUG] _newton_gauss_optimization开始: center_only={center_only}")
        if not self._sphere:
            return initial_params, False
        
        params = initial_params.copy()
        max_iterations = self._fitting_config.get("newton_max_iterations", 50)
        convergence_tol = self._fitting_config.get("convergence_tol", 1e-6)
        points_array = np.array(points)
        
        # 检查权重是否有效
        weight_sum = np.sum(self._current_weights)
        print(f"[DEBUG] 权重总和: {weight_sum}")
        if weight_sum == 0:
            print(f"[DEBUG] 权重为0，返回初始参数")
            return initial_params, False
        
        for iteration in range(max_iterations):
            print(f"[DEBUG] 牛顿-高斯迭代 {iteration + 1}")
            # 计算残差向量
            residuals = self._compute_residuals(params, points_array, center_only)
            print(f"[DEBUG] 残差: {residuals}")
            
            # 计算雅可比矩阵
            jacobian = self._compute_jacobian(params, points_array, center_only)
            print(f"[DEBUG] 雅可比矩阵形状: {jacobian.shape}")
            
            # 构造权重矩阵
            weight_matrix = np.diag(self._current_weights)
            
            # 解正规方程: (J^T * W * J) * Δ = -J^T * W * f(θ)
            jtwj = jacobian.T @ weight_matrix @ jacobian
            jtwf = jacobian.T @ weight_matrix @ residuals
            
            try:
                delta = np.linalg.solve(jtwj, -jtwf)
                print(f"[DEBUG] 参数增量: {delta}")
            except np.linalg.LinAlgError:
                print(f"[DEBUG] 线性代数求解失败")
                return params, False
            
            # 更新参数
            params_new = params + delta
            
            # 更新内部状态（用于下一次权重计算）
            if center_only:
                self._ellipsoid_params["center"] = params_new
            else:
                self._ellipsoid_params["center"] = params_new[:3]
                self._ellipsoid_params["axes"][0] = params_new[3]
            
            # 收敛检测
            delta_norm = float(np.linalg.norm(delta))
            print(f"[DEBUG] 增量范数: {delta_norm}, 收敛阈值: {convergence_tol}")
            if delta_norm < convergence_tol:
                print(f"[DEBUG] 收敛")
                return params_new, True
            
            params = params_new
        
        print(f"[DEBUG] 未收敛")
        return params, False
    
    def _compute_residuals(self, params: np.ndarray, points: np.ndarray, center_only: bool = False) -> np.ndarray:
        """计算残差向量 f(θ) = ||p_i - c|| - r"""
        print(f"[DEBUG] _compute_residuals: center_only={center_only}")
        if center_only:
            center = params
            radius = self._ellipsoid_params["axes"][0]
        else:
            center = params[:3]
            radius = params[3]
        
        print(f"[DEBUG] 中心: {center}, 半径: {radius}")
        distances = np.linalg.norm(points - center, axis=1)
        residuals = distances - radius
        print(f"[DEBUG] 残差: {residuals}")
        
        return residuals
    
    def _compute_jacobian(self, params: np.ndarray, points: np.ndarray, center_only: bool = False) -> np.ndarray:
        """计算雅可比矩阵 J"""
        print(f"[DEBUG] _compute_jacobian: center_only={center_only}")
        if center_only:
            center = params
            radius = self._ellipsoid_params["axes"][0]
            n_points = len(points)
            jacobian = np.zeros((n_points, 3))
            
            for i in range(n_points):
                point = points[i]
                distance = float(np.linalg.norm(point - center))  # 确保是标量
                print(f"[DEBUG] 点 {i}: distance={distance}")
                
                if distance > 1e-10:
                    jacobian[i, :] = -(point - center) / distance
                else:
                    jacobian[i, :] = [1.0, 0.0, 0.0]
        else:
            center = params[:3]
            n_points = len(points)
            jacobian = np.zeros((n_points, 4))
            
            for i in range(n_points):
                point = points[i]
                distance = float(np.linalg.norm(point - center))  # 确保是标量
                print(f"[DEBUG] 点 {i}: distance={distance}")
                
                if distance > 1e-10:
                    jacobian[i, :3] = -(point - center) / distance
                    jacobian[i, 3] = -1.0
                else:
                    jacobian[i, :3] = [1.0, 0.0, 0.0]
                    jacobian[i, 3] = -1.0
        
        print(f"[DEBUG] 雅可比矩阵计算完成")
        return jacobian
    
    def _count_inliers(self, params: np.ndarray, points: List[np.ndarray], 
                      structure_types: List[str], center_only: bool = False) -> int:
        """计算内点数量"""
        print(f"[DEBUG] _count_inliers: center_only={center_only}")
        if center_only:
            center = params
            current_radius = self._ellipsoid_params["axes"][0]
        else:
            center = params[:3]
            current_radius = params[3]
        
        print(f"[DEBUG] 中心: {center}, 半径: {current_radius}")
        inlier_count = 0
        for i, point in enumerate(points):
            distance = abs(float(np.linalg.norm(point - center)) - current_radius)
            
            # 根据结构类型调整阈值
            structure_type = structure_types[i]
            if structure_type in self._geometric_params:
                struct_threshold = self._geometric_params[structure_type].get('threshold', self._current_threshold)
            else:
                struct_threshold = self._current_threshold
            
            if distance <= struct_threshold:
                inlier_count += 1
        
        print(f"[DEBUG] 内点数量: {inlier_count}")
        return inlier_count
    
    def _adaptive_threshold_adjustment(self, inlier_count: int, total_points: int) -> float:
        """自适应阈值调整"""
        base_threshold = self._fitting_config.get("ransac_threshold", 0.0015)
        adjustment_factor = self._fitting_config.get("threshold_adjustment_factor", 1.5)
        inlier_ratio = inlier_count / total_points if total_points > 0 else 0
        
        if inlier_ratio > 0.8:
            # 拟合质量好，减小阈值
            return base_threshold * 0.8
        elif inlier_ratio < 0.5:
            # 拟合质量差，增大阈值
            return base_threshold * adjustment_factor
        else:
            return base_threshold
    
    def _anatomical_weight(self, structure_type: str) -> float:
        """解剖学先验权重"""
        try:
            if structure_type in self._anatomical_weights:
                median_distance = self._anatomical_weights[structure_type]['median_distance']
                sigma = self._anatomical_weights[structure_type]['sigma']
                if sigma <= 0:
                    return 0.0
                return np.exp(-(median_distance ** 2) / (2 * sigma ** 2))
            return 0.0
        except (KeyError, ValueError, TypeError):
            return 0.0
    
    def _geometric_weight(self, distance: float, sigma: float) -> float:
        """几何残差权重"""
        try:
            if sigma <= 0:
                return 0.0
            return np.exp(-(distance ** 2) / (2 * sigma ** 2))
        except (ValueError, TypeError):
            return 0.0
    
    def _extract_key_points(self, key_coordinates: SingleEyeKeyCoordinates) -> Tuple[List[np.ndarray], List[str]]:
        """提取关键点数据和对应的结构类型"""
        print(f"[DEBUG] _extract_key_points开始")
        points = []
        structure_types = []
        
        # 提取瞳孔中心点 - 修复：明确检查是否为None
        pupil_centers = key_coordinates.get("pupil_centers")
        if pupil_centers is not None:
            points.append(pupil_centers)
            structure_types.append("pupil_center")
            print(f"[DEBUG] 添加瞳孔中心点: {pupil_centers}")
        
        # 提取虹膜边界点 - 修复：明确检查列表是否为空
        iris_boundaries = key_coordinates.get("iris_boundaries")
        if iris_boundaries is not None and len(iris_boundaries) > 0:
            for point in iris_boundaries:
                if point is not None:
                    points.append(point)
                    structure_types.append("iris_boundary")
                    print(f"[DEBUG] 添加虹膜边界点: {point}")
        
        # 提取眼睛轮廓点 - 修复：明确检查列表是否为空
        eyes_contours = key_coordinates.get("eyes_contours")
        if eyes_contours is not None and len(eyes_contours) > 0:
            for point in eyes_contours:
                if point is not None:
                    points.append(point)
                    structure_types.append("eye_contour")
                    print(f"[DEBUG] 添加眼睛轮廓点: {point}")
        
        print(f"[DEBUG] 提取完成: {len(points)} 个点")
        return points, structure_types
    
    def _estimate_initial_params(self, points: List[np.ndarray], center_only: bool = False) -> np.ndarray:
        """估计初始参数 [cx, cy, cz, r]"""
        print(f"[DEBUG] _estimate_initial_params: center_only={center_only}")
        points_array = np.array(points)
        
        # 初始球心：所有点的质心
        center = np.mean(points_array, axis=0)
        print(f"[DEBUG] 初始中心: {center}")
        
        if center_only:
            return center
        else:
            # 初始半径：所有点到质心的平均距离
            distances = np.linalg.norm(points_array - center, axis=1)
            radius = np.mean(distances)
            print(f"[DEBUG] 初始半径: {radius}")
            return np.array([center[0], center[1], center[2], radius])
    
    def _params_to_ellipsoid(self, params: np.ndarray) -> EllipsoidParams:
        """将参数转换为椭球参数"""
        center = params[:3]
        radius = params[3]
        
        result = EllipsoidParams()
        result["axes"] = np.array([radius, radius, radius])
        result["rotation"] = np.eye(3)
        result["center"] = center
        
        # 更新内部状态
        self._ellipsoid_params = result
        
        return result
    
    def _get_default_params(self) -> EllipsoidParams:
        """获取默认参数"""
        return self._ellipsoid_params
    def __init__(self, sphere: bool = True):
        # 模型类型
        self._sphere = sphere
        
        # 初始化椭球参数
        self._ellipsoid_params = EllipsoidParams()
        if sphere:
            self._r = 0.012  # 默认半径12mm
            self._ellipsoid_params["axes"] = np.array([self._r, self._r, self._r])
            self._ellipsoid_params["rotation"] = np.eye(3)
            self._ellipsoid_params["center"] = np.array([0.0, 0.0, 0.0])
        
        # 状态管理变量
        self._current_weights = None
        self._current_threshold = None
        self._weights_calculated = False
        
        # 从settings.py获取拟合参数
        from project.fitting.config.settings import FITTING_ALGORITHM_CONFIG
        self._fitting_config = FITTING_ALGORITHM_CONFIG
        
        # 从initial_weight.py获取权重参数
        from project.fitting.config.initial_weight import (
            ANATOMICAL_WEIGHT_PARAMS, GEOMETRIC_WEIGHT_PARAMS, GLOBAL_WEIGHT_CONFIG
        )
        self._anatomical_weights = ANATOMICAL_WEIGHT_PARAMS
        self._geometric_params = GEOMETRIC_WEIGHT_PARAMS
        self._global_config = GLOBAL_WEIGHT_CONFIG
        
    # 主要接口函数
    def center_fitter(self, key_coordinates: SingleEyeKeyCoordinates, 
                     trials_times: int) -> Tuple[EllipsoidParams, np.ndarray]:
        """主拟合接口，实现调用_fit_center_only或者_int_center_fitter"""
        max_trials = self._fitting_config.get("max_trials", 100)
        
        if trials_times >= max_trials:
            # 超过最大实验次数时，只拟合球心坐标，保持原有半径
            return self._fit_ellipsoid(key_coordinates, center_only=True)
        else:
            return self._fit_ellipsoid(key_coordinates, center_only=False)
    
    def _fit_ellipsoid(self, key_coordinates: SingleEyeKeyCoordinates, center_only: bool = False) -> Tuple[EllipsoidParams, np.ndarray]:
        """统一的椭球拟合函数，支持只拟合中心或完整拟合"""
        # 1. 提取关键点数据
        points, structure_types = self._extract_key_points(key_coordinates)
        
        min_samples = 2 if center_only else self._fitting_config.get("min_samples", 4)
        if len(points) < min_samples:
            default_params = self._get_default_params()
            return default_params, default_params["center"]
        
        # 2. 初始化状态
        self._reset_fitting_state()
        
        # 3. RANSAC拟合
        best_params, best_inlier_count = self._run_ransac_fitting(points, structure_types, center_only)
        
        # 4. 返回结果
        if best_params is not None:
            if center_only:
                # 只拟合中心时，保持原有半径
                current_radius = self._ellipsoid_params["axes"][0]
                result = EllipsoidParams()
                result["axes"] = np.array([current_radius, current_radius, current_radius])
                result["rotation"] = np.eye(3)
                result["center"] = best_params
                return result, best_params
            else:
                # 完整拟合
                ellipsoid_params = self._params_to_ellipsoid(best_params)
                return ellipsoid_params, ellipsoid_params["center"]
        else:
            default_params = self._get_default_params()
            return default_params, default_params["center"]
    
    def _reset_fitting_state(self):
        """重置拟合状态"""
        self._current_weights = None
        self._current_threshold = self._fitting_config.get("ransac_threshold", 0.0015)
        self._weights_calculated = False
    
    def _run_ransac_fitting(self, points: List[np.ndarray], structure_types: List[str], center_only: bool) -> Tuple[np.ndarray, int]:
        """运行RANSAC拟合"""
        best_params = None
        best_inlier_count = 0
        max_ransac_iterations = self._fitting_config.get("ransac_max_iterations", 50)
        
        for iteration in range(max_ransac_iterations):
            # 单次RANSAC迭代
            params, inlier_count = self._run_single_ransac_iteration(points, structure_types, center_only)
            
            if params is not None and inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_params = params
                
                # 更新阈值
                self._current_threshold = self._adaptive_threshold_adjustment(inlier_count, len(points))
        
        return best_params, best_inlier_count
    
    def _run_single_ransac_iteration(self, points: List[np.ndarray], structure_types: List[str], center_only: bool) -> Tuple[np.ndarray, int]:
        """运行单次RANSAC迭代"""
        # 1. 采样点
        sampled_points, sampled_types = self._ransac_sampling(points, structure_types, center_only)
        
        min_samples = 2 if center_only else self._fitting_config.get("min_samples", 4)
        if len(sampled_points) < min_samples:
            return None, 0
        
        # 2. 计算权重（如果未计算）
        if not self._weights_calculated:
            self._calculate_weights(sampled_points, sampled_types, center_only)
        
        # 3. 初始参数估计
        initial_params = self._estimate_initial_params(sampled_points, center_only)
        
        # 4. 牛顿-高斯优化
        optimized_params, converged = self._newton_gauss_optimization(
            initial_params, sampled_points, center_only
        )
        
        if not converged:
            return None, 0
        
        # 5. 计算内点数量
        inlier_count = self._count_inliers(optimized_params, points, structure_types, center_only)
        
        return optimized_params, inlier_count
    
    def _ransac_sampling(self, points: List[np.ndarray], structure_types: List[str], center_only: bool) -> Tuple[List[np.ndarray], List[str]]:
        """RANSAC采样策略"""
        n_points = len(points)
        min_samples = 2 if center_only else self._fitting_config.get("min_samples", 4)
        
        if n_points < min_samples:
            return [], []
        
        # 计算权重
        self._calculate_weights(points, structure_types, center_only)
        
        # 检查是否有有效权重
        if np.sum(self._current_weights) == 0:
            return [], []
        
        # 选择权重最高的点
        points_per_eye = self._fitting_config.get("points_per_eye", 4)
        selected_count = max(min_samples, min(points_per_eye, n_points))
        selected_indices = np.argsort(self._current_weights)[-selected_count:]
        
        # 修复：确保权重是标量进行比较
        sampled_points = [points[i] for i in selected_indices if float(self._current_weights[i]) > 0]
        sampled_types = [structure_types[i] for i in selected_indices if float(self._current_weights[i]) > 0]
        
        return sampled_points, sampled_types

    def _calculate_weights(self, points: List[np.ndarray], structure_types: List[str], center_only: bool = False) -> None:
        """计算三层权重并缓存结果"""
        if not self._sphere:
            self._current_weights = np.ones(len(points))
            self._weights_calculated = True
            return
            
        n_points = len(points)
        self._current_weights = np.zeros(n_points)
        
        # 获取当前半径
        current_radius = self._ellipsoid_params["axes"][0]
        
        for i in range(n_points):
            try:
                # 1. 解剖学先验权重
                w_anat = self._anatomical_weight(structure_types[i])
                
                # 2. 几何残差权重（基于当前椭球参数）
                # 修复：确保center不是None且是有效的数组
                center = self._ellipsoid_params["center"]
                if center is not None and len(center) == 3:
                    distance = abs(float(np.linalg.norm(points[i] - center)) - current_radius)
                    geom_params = self._geometric_params.get(structure_types[i], {})
                    initial_sigma = geom_params.get('initial_sigma', 1.5)
                    w_geom = self._geometric_weight(distance, initial_sigma)
                else:
                    w_geom = 1.0
                
                # 3. 信号质量权重（默认1.0）
                w_snr = 1.0
                
                # 4. 综合权重
                final_weight = w_anat * w_geom * w_snr
                min_weight = self._global_config.get('min_weight', 0.1)
                
                # 如果权重小于最小权重，则设为0
                if final_weight >= min_weight:
                    self._current_weights[i] = final_weight
                else:
                    self._current_weights[i] = 0.0
                
            except (KeyError, ValueError) as e:
                print(f"计算第{i}个点权重时出错: {e}")
                self._current_weights[i] = 0.0
        
        # 权重归一化（只对非零权重进行）
        if self._global_config.get('normalization', True):
            nonzero_weights = self._current_weights[self._current_weights > 0]
            if len(nonzero_weights) > 0:
                mean_weight = np.mean(nonzero_weights)
                self._current_weights[self._current_weights > 0] = self._current_weights[self._current_weights > 0] / mean_weight
        
            self._weights_calculated = True
            """计算三层权重并缓存结果"""
            if not self._sphere:
                self._current_weights = np.ones(len(points))
                self._weights_calculated = True
                return
                
            n_points = len(points)
            self._current_weights = np.zeros(n_points)
            
            # 获取当前半径
            current_radius = self._ellipsoid_params["axes"][0]
            
            for i in range(n_points):
                try:
                    # 1. 解剖学先验权重
                    w_anat = self._anatomical_weight(structure_types[i])
                    
                    # 2. 几何残差权重（基于当前椭球参数）
                    if self._ellipsoid_params["center"] is not None:
                        distance = abs(np.linalg.norm(points[i] - self._ellipsoid_params["center"]) - current_radius)
                        geom_params = self._geometric_params.get(structure_types[i], {})
                        initial_sigma = geom_params.get('initial_sigma', 1.5)
                        w_geom = self._geometric_weight(distance, initial_sigma)
                    else:
                        w_geom = 1.0
                    
                    # 3. 信号质量权重（默认1.0）
                    w_snr = 1.0
                    
                    # 4. 综合权重
                    final_weight = w_anat * w_geom * w_snr
                    min_weight = self._global_config.get('min_weight', 0.1)
                    
                    # 如果权重小于最小权重，则设为0
                    if final_weight >= min_weight:
                        self._current_weights[i] = final_weight
                    else:
                        self._current_weights[i] = 0.0
                    
                except (KeyError, ValueError) as e:
                    print(f"计算第{i}个点权重时出错: {e}")
                    self._current_weights[i] = 0.0
            
            # 权重归一化（只对非零权重进行）
            if self._global_config.get('normalization', True):
                nonzero_weights = self._current_weights[self._current_weights > 0]
                if len(nonzero_weights) > 0:
                    mean_weight = np.mean(nonzero_weights)
                    self._current_weights[self._current_weights > 0] = self._current_weights[self._current_weights > 0] / mean_weight
            
            self._weights_calculated = True
    
    def _newton_gauss_optimization(self, initial_params: np.ndarray, 
                                 points: List[np.ndarray], center_only: bool = False) -> Tuple[np.ndarray, bool]:
        """牛顿-高斯算法优化"""
        if not self._sphere:
            return initial_params, False
        
        params = initial_params.copy()
        max_iterations = self._fitting_config.get("newton_max_iterations", 50)
        convergence_tol = self._fitting_config.get("convergence_tol", 1e-6)
        points_array = np.array(points)
        
        # 检查权重是否有效
        if np.sum(self._current_weights) == 0:
            return initial_params, False
        
        for iteration in range(max_iterations):
            # 计算残差向量
            residuals = self._compute_residuals(params, points_array, center_only)
            
            # 计算雅可比矩阵
            jacobian = self._compute_jacobian(params, points_array, center_only)
            
            # 构造权重矩阵
            weight_matrix = np.diag(self._current_weights)
            
            # 解正规方程: (J^T * W * J) * Δ = -J^T * W * f(θ)
            jtwj = jacobian.T @ weight_matrix @ jacobian
            jtwf = jacobian.T @ weight_matrix @ residuals
            
            try:
                delta = np.linalg.solve(jtwj, -jtwf)
            except np.linalg.LinAlgError:
                return params, False
            
            # 更新参数
            params_new = params + delta
            
            # 更新内部状态（用于下一次权重计算）
            if center_only:
                self._ellipsoid_params["center"] = params_new
            else:
                self._ellipsoid_params["center"] = params_new[:3]
                self._ellipsoid_params["axes"][0] = params_new[3]
            
            # 收敛检测
            if np.linalg.norm(delta) < convergence_tol:
                return params_new, True
            
            params = params_new
        
        return params, False
    
    def _compute_residuals(self, params: np.ndarray, points: np.ndarray, center_only: bool = False) -> np.ndarray:
        """计算残差向量 f(θ) = ||p_i - c|| - r"""
        if center_only:
            center = params
            radius = self._ellipsoid_params["axes"][0]
        else:
            center = params[:3]
            radius = params[3]
        
        distances = np.linalg.norm(points - center, axis=1)
        residuals = distances - radius
        
        return residuals
    
    def _compute_jacobian(self, params: np.ndarray, points: np.ndarray, center_only: bool = False) -> np.ndarray:
        """计算雅可比矩阵 J"""
        if center_only:
            center = params
            radius = self._ellipsoid_params["axes"][0]
            n_points = len(points)
            jacobian = np.zeros((n_points, 3))
            
            for i in range(n_points):
                point = points[i]
                distance = float(np.linalg.norm(point - center))  # 确保是标量
                
                if distance > 1e-10:
                    jacobian[i, :] = -(point - center) / distance
                else:
                    jacobian[i, :] = [1.0, 0.0, 0.0]
        else:
            center = params[:3]
            n_points = len(points)
            jacobian = np.zeros((n_points, 4))
            
            for i in range(n_points):
                point = points[i]
                distance = float(np.linalg.norm(point - center))  # 确保是标量
                
                if distance > 1e-10:
                    jacobian[i, :3] = -(point - center) / distance
                    jacobian[i, 3] = -1.0
                else:
                    jacobian[i, :3] = [1.0, 0.0, 0.0]
                    jacobian[i, 3] = -1.0
        
        return jacobian
    
    def _count_inliers(self, params: np.ndarray, points: List[np.ndarray], 
                      structure_types: List[str], center_only: bool = False) -> int:
        """计算内点数量"""
        if center_only:
            center = params
            current_radius = self._ellipsoid_params["axes"][0]
        else:
            center = params[:3]
            current_radius = params[3]
        
        inlier_count = 0
        for i, point in enumerate(points):
            distance = abs(np.linalg.norm(point - center) - current_radius)
            
            # 根据结构类型调整阈值
            structure_type = structure_types[i]
            if structure_type in self._geometric_params:
                struct_threshold = self._geometric_params[structure_type].get('threshold', self._current_threshold)
            else:
                struct_threshold = self._current_threshold
            
            if distance <= struct_threshold:
                inlier_count += 1
        
        return inlier_count
    
    def _adaptive_threshold_adjustment(self, inlier_count: int, total_points: int) -> float:
        """自适应阈值调整"""
        base_threshold = self._fitting_config.get("ransac_threshold", 0.0015)
        adjustment_factor = self._fitting_config.get("threshold_adjustment_factor", 1.5)
        inlier_ratio = inlier_count / total_points if total_points > 0 else 0
        
        if inlier_ratio > 0.8:
            # 拟合质量好，减小阈值
            return base_threshold * 0.8
        elif inlier_ratio < 0.5:
            # 拟合质量差，增大阈值
            return base_threshold * adjustment_factor
        else:
            return base_threshold
    
    def _anatomical_weight(self, structure_type: str) -> float:
        """解剖学先验权重"""
        try:
            if structure_type in self._anatomical_weights:
                median_distance = self._anatomical_weights[structure_type]['median_distance']
                sigma = self._anatomical_weights[structure_type]['sigma']
                if sigma <= 0:
                    return 0.0
                return np.exp(-(median_distance ** 2) / (2 * sigma ** 2))
            return 0.0
        except (KeyError, ValueError, TypeError):
            return 0.0
    
    def _geometric_weight(self, distance: float, sigma: float) -> float:
        """几何残差权重"""
        try:
            if sigma <= 0:
                return 0.0
            return np.exp(-(distance ** 2) / (2 * sigma ** 2))
        except (ValueError, TypeError):
            return 0.0
    
    def _extract_key_points(self, key_coordinates: SingleEyeKeyCoordinates) -> Tuple[List[np.ndarray], List[str]]:
        """提取关键点数据和对应的结构类型"""
        points = []
        structure_types = []
        
        # 提取瞳孔中心点
        if key_coordinates.get("pupil_centers"):
            pupil_center = key_coordinates["pupil_centers"]
            if pupil_center is not None:
                points.append(pupil_center)
                structure_types.append("pupil_center")
        
        # 提取虹膜边界点
        if key_coordinates.get("iris_boundaries"):
            iris_points = key_coordinates["iris_boundaries"]
            if isinstance(iris_points, list):
                for point in iris_points:
                    if point is not None:
                        points.append(point)
                        structure_types.append("iris_boundary")
        
        # 提取眼睛轮廓点
        if key_coordinates.get("eyes_contours"):
            contour_points = key_coordinates["eyes_contours"]
            if isinstance(contour_points, list):
                for point in contour_points:
                    if point is not None:
                        points.append(point)
                        structure_types.append("eye_contour")
        
        return points, structure_types
    
    def _estimate_initial_params(self, points: List[np.ndarray], center_only: bool = False) -> np.ndarray:
        """估计初始参数 [cx, cy, cz, r]"""
        points_array = np.array(points)
        
        # 初始球心：所有点的质心
        center = np.mean(points_array, axis=0)
        
        if center_only:
            return center
        else:
            # 初始半径：所有点到质心的平均距离
            distances = np.linalg.norm(points_array - center, axis=1)
            radius = np.mean(distances)
            return np.array([center[0], center[1], center[2], radius])
    
    def _params_to_ellipsoid(self, params: np.ndarray) -> EllipsoidParams:
        """将参数转换为椭球参数"""
        center = params[:3]
        radius = params[3]
        
        result = EllipsoidParams()
        result["axes"] = np.array([radius, radius, radius])
        result["rotation"] = np.eye(3)
        result["center"] = center
        
        # 更新内部状态
        self._ellipsoid_params = result
        
        return result
    
    def _get_default_params(self) -> EllipsoidParams:
        """获取默认参数"""
        return self._ellipsoid_params

def vector_of_2_points(point1:np.ndarray,point2:np.ndarray)->np.ndarray:
    return point2 - point1



def normalize_vector(vector:np.ndarray)->np.ndarray:
    '''
    归一化向量
    Args:
        vector: 向量
    Returns:
        归一化后的向量
    '''
    return vector / np.linalg.norm(vector)

def angle_between_vectors(vector1:np.ndarray,vector2:np.ndarray)->float:
    '''
    计算两个向量之间的夹角
    Args:
        vector1: 向量1
        vector2: 向量2
    Returns:
        夹角（弧度）
    '''
    return np.arccos(np.dot(vector1,vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2)))

# 对外接口

def intersect_pixel_on_screen(vector:np.ndarray,rgb_d=False)->np.ndarray:
    '''
    计算向量与屏幕的交点（像素坐标,(w,h)）
    Args:
        vector: 向量
        screen: 屏幕
    Returns:
        交点
    '''
    screen = Plane(rgb_d)
    point_2d_with_camera_as_origin = screen.intersection_on_plane(vector)
    if point_2d_with_camera_as_origin is None:
        return np.array([np.nan, np.nan])

    # 统一到以左上角为原点的2D坐标
    top_left_2d = screen._point_3d_to_2d(screen["top_left"])
    point_2d_with_top_left_as_origin = point_2d_with_camera_as_origin - top_left_2d

    x,y = point_2d_with_top_left_as_origin
    screen_width,screen_height = screen["width_m"],screen["height_m"]

    # 使用实际分辨率（来自设置）
    w,h = SCREEN_WITH_RGBD["resolution_px"]
    if screen_width <= 0 or screen_height <= 0 or w <= 0 or h <= 0:
        return np.array([np.nan, np.nan])

    return np.array([w*x/screen_width, h*y/screen_height])

def center_fitter(key_coordinates:SingleEyeKeyCoordinates,trials_times:int)->Tuple[EllipsoidParams, np.ndarray]:
    '''
    中心拟合器
    Args:
        key_coordinates: 关键点坐标
        trials_times: 尝试次数
    Returns:
        中心拟合结果：(椭球参数, 中心坐标)
    '''
    return CenterFitter().center_fitter(key_coordinates,trials_times)

def rotate_vector(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    '''
    使用罗德里格斯旋转公式绕轴旋转向量
    
    Args:
        vector: 要旋转的向量 (3,)
        axis: 旋转轴单位向量 (3,)
        angle: 旋转角度（弧度）
    
    Returns:
        旋转后的向量 (3,)
    '''
    # 确保轴向量是单位向量
    axis = normalize_vector(axis)
    
    # 罗德里格斯旋转公式：v' = v*cos(θ) + (k×v)*sin(θ) + k(k·v)(1-cos(θ))
    # 其中 k 是旋转轴，θ 是旋转角度
    
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    # 计算各个分量
    v_parallel = np.dot(vector, axis) * axis  # 平行于轴的分量
    v_perpendicular = vector - v_parallel     # 垂直于轴的分量
    
    # 计算旋转后的向量
    rotated_vector = (v_parallel + 
                     v_perpendicular * cos_angle + 
                     np.cross(axis, vector) * sin_angle)
    
    return rotated_vector
