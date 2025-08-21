"""
基于虹膜边界三维点的鲁棒球/椭球中心拟合（接口定义占位版本）

说明：
- 仅提供接口定义，不实现具体算法（函数体使用 pass）。
- 采用单例类，构造时以 key_coordinates 初始化。
- 采样次数、内点阈值、拟合函数（center_fitter）、椭球参数/眼球中心均为私有成员。
- 对外只暴露一个函数：ransac_sphere_eyeball(...)，返回左右眼的 (ellipsoid_params, center)。
- 在识别模块 main 循环调用 fitting.main 的场景下，类内维护 _max_trials 与"固定模型"缓存，
  以避免每帧都重新拟合形状参数，简化后续帧的眼球中心估计。
"""

from project.fitting.config.constants import KeyCoordinates, SingleEyeKeyCoordinates
from project.fitting.utils.geometry import EllipsoidParams
from typing import Dict, Optional, Tuple, Callable   
import numpy as np



# 识别模块输出的 key_coordinates 的宽松结构：
# 顶层键：'pupil_centers' | 'iris_boundaries' | 'eyes_contours'
# 次层键：'left' | 'right'
# 值：np.ndarray(3,) 或 List[np.ndarray(3,)]，并兼容 None




__all__ = ["ransac_sphere_eyeball","ransac_sphere_eyeball_restart"]


class _EyeCenterFitter:
    """
    单例类：维护跨帧的拟合配置与缓存（如固定椭球/上一帧中心），对外不直接暴露。

    初始化参数：
    - key_coordinates: KeyCoordinates
      约定包含：
        - 'pupil_centers': {'left': np.ndarray(3,), 'right': np.ndarray(3,)} 或 None/零向量
        - 'iris_boundaries': {'left': List[np.ndarray(3,)], 'right': List[np.ndarray(3,)]} 或 []
        - 'eyes_contours': {'left': List[np.ndarray(3,)], 'right': List[np.ndarray(3,)]} 或 []
      注：上游识别模块需保证这些键名与结构一致。

    - center_fitter: 可选自定义拟合器，签名统一为：
        center_fitter(key_coordinates: SingleEyeKeyCoordinates, trials_times: int)
            -> Tuple[EllipsoidParams, np.ndarray]  # (椭球参数, 中心坐标)
    """

    _instance: Optional["_EyeCenterFitter"] = None

    def __new__(cls, key_coordinates: KeyCoordinates, center_fitter: Optional[Callable[[SingleEyeKeyCoordinates, int], Tuple[EllipsoidParams, np.ndarray]]] = None) -> "_EyeCenterFitter":
        print(f"[DEBUG] _EyeCenterFitter.__new__: cls._instance={cls._instance}")
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            print(f"[DEBUG] 创建新的单例实例")
        else:
            print(f"[DEBUG] 返回现有单例实例")
        return cls._instance  # 单例

    def __init__(self, key_coordinates: KeyCoordinates = None, center_fitter: Optional[Callable[[SingleEyeKeyCoordinates, int], Tuple[EllipsoidParams, np.ndarray]]] = None) -> None:
        print(f"[DEBUG] _EyeCenterFitter.__init__开始")
        print(f"[DEBUG] key_coordinates类型: {type(key_coordinates)}")
        if key_coordinates:
            print(f"[DEBUG] key_coordinates键: {list(key_coordinates.keys())}")
        
        # 私有输入数据
        self._key_coordinates: KeyCoordinates = key_coordinates

        # 私有配置（仅接口占位：默认值可后续在实现时调整/覆盖）
        self._center_fitter: Optional[
            Callable[[SingleEyeKeyCoordinates, int], Tuple[EllipsoidParams, np.ndarray]]
        ] = center_fitter  # 自定义椭球/球拟合器
        self._trials_times: int = 0

        # 跨帧复用的固定"椭球形状"与中心（左右眼）
        self._fixed_shapes: Dict[str, Optional[EllipsoidParams]] = {"left": None, "right": None}
        self._eye_centers: Dict[str, Optional[np.ndarray]] = {"left": None, "right": None}

        # 历史参考（可选）
        self._last_center_left: Optional[np.ndarray] = None
        self._last_center_right: Optional[np.ndarray] = None
        
        print(f"[DEBUG] _EyeCenterFitter.__init__完成")

    def ransac_sphere_eyeball(self) -> Dict[str, Tuple[EllipsoidParams, np.ndarray]]:
        """
        对左右眼进行鲁棒球/椭球中心拟合。

        返回：
        - result: Dict[str, Tuple[EllipsoidParams, np.ndarray]]
          结构示例：
          {
            "left":  ({"axes": np.array([0.012, 0.012, 0.012]), "rotation": np.eye(3)}, np.array([0.10, 0.05, 0.80])),
            "right": ({"axes": np.array([0.012, 0.012, 0.012]), "rotation": np.eye(3)}, np.array([0.16, 0.05, 0.80]))
          }
        说明：
        - 若使用球模型，可令 axes 三者相等，rotation 取单位阵。
        """
        print(f"[DEBUG] ransac_sphere_eyeball开始")
        print(f"[DEBUG] 当前trials_times: {self._trials_times}")
        print(f"[DEBUG] center_fitter类型: {type(self._center_fitter)}")
        
        try:
            self._fit_one_eye("left")
            print(f"[DEBUG] 左眼拟合完成")
        except Exception as e:
            print(f"[DEBUG] 左眼拟合异常: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            self._fit_one_eye("right")
            print(f"[DEBUG] 右眼拟合完成")
        except Exception as e:
            print(f"[DEBUG] 右眼拟合异常: {e}")
            import traceback
            traceback.print_exc()
        
        result = {
            "left": (self._fixed_shapes["left"], self._eye_centers["left"]),
            "right": (self._fixed_shapes["right"], self._eye_centers["right"])
        }
        print(f"[DEBUG] ransac_sphere_eyeball返回结果: {result}")
        return result

    # ============ 私有辅助接口（占位）===========

    def _fit_one_eye(self, eye_key: str) -> Tuple[EllipsoidParams, np.ndarray]:
        """
        内部：对单眼进行中心拟合；可选复用固定形状；使用 _center_fitter。
        返回 (ellipsoid_params, center_xyz)。
        """
        print(f"[DEBUG] _fit_one_eye开始: eye_key={eye_key}")
        
        # 动态维护trials_times，每次拟合++
        self._trials_times += 1
        print(f"[DEBUG] 更新trials_times: {self._trials_times}")
        
        # 为每只眼睛创建单独的关键点坐标
        print(f"[DEBUG] 开始提取{eye_key}眼关键点坐标")
        eye_key_coordinates = self._extract_eye_key_coordinates(eye_key)
        print(f"[DEBUG] {eye_key}眼关键点坐标: {eye_key_coordinates}")
        
        # 调用center_fitter进行拟合
        print(f"[DEBUG] 调用center_fitter进行拟合")
        if self._center_fitter is None:
            print(f"[DEBUG] center_fitter为None，使用默认拟合器")
            from project.fitting.utils.geometry import center_fitter
            self._center_fitter = center_fitter
        
        try:
            self._fixed_shapes[eye_key], self._eye_centers[eye_key] = self._center_fitter(eye_key_coordinates, self._trials_times)
            print(f"[DEBUG] {eye_key}眼拟合完成: shape={self._fixed_shapes[eye_key]}, center={self._eye_centers[eye_key]}")
        except Exception as e:
            print(f"[DEBUG] {eye_key}眼拟合异常: {e}")
            import traceback
            traceback.print_exc()
            # 设置默认值
            self._fixed_shapes[eye_key] = EllipsoidParams()
            self._eye_centers[eye_key] = np.array([0.0, 0.0, 0.0])

    def _set_params(
        self,
        center_fitter: Optional[Callable[[SingleEyeKeyCoordinates, int], Tuple[EllipsoidParams, np.ndarray]]] = None,
    ) -> None:
        """
        内部：设置/覆盖私有配置（在对外接口调用时注入）。
        """
        print(f"[DEBUG] _set_params: center_fitter={center_fitter}")
        self._center_fitter = center_fitter

    def _update_key_coordinates(self, key_coordinates: KeyCoordinates) -> None:
        """
        内部：当外部传入新的关键点坐标时，更新内部状态（跨帧复用）。
        """
        print(f"[DEBUG] _update_key_coordinates开始")
        print(f"[DEBUG] 新key_coordinates类型: {type(key_coordinates)}")
        if key_coordinates:
            print(f"[DEBUG] 新key_coordinates键: {list(key_coordinates.keys())}")
            for key, value in key_coordinates.items():
                print(f"[DEBUG] {key}: {type(value)}")
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        print(f"[DEBUG]   {sub_key}: {type(sub_value)}")
                        if isinstance(sub_value, list):
                            print(f"[DEBUG]     list长度: {len(sub_value)}")
                            if len(sub_value) > 0:
                                print(f"[DEBUG]     第一个元素类型: {type(sub_value[0])}")
                        elif isinstance(sub_value, np.ndarray):
                            print(f"[DEBUG]     array形状: {sub_value.shape}")
        
        self._key_coordinates = key_coordinates
        print(f"[DEBUG] _update_key_coordinates完成")
        
    def reset_trials_time(self) -> None:
        """
        内部：重置采样次数。
        """
        print(f"[DEBUG] reset_trials_time: {self._trials_times} -> 0")
        self._trials_times = 0
        
    def _extract_eye_key_coordinates(self, eye_key: str) -> SingleEyeKeyCoordinates:
        """
        为指定眼睛提取关键点坐标
        """
        print(f"[DEBUG] _extract_eye_key_coordinates开始: eye_key={eye_key}")
        print(f"[DEBUG] self._key_coordinates: {self._key_coordinates}")
        
        eye_key_coordinates = {
            "pupil_centers": None,
            "iris_boundaries": [],
            "eyes_contours": []
        }
        
        # 提取瞳孔中心 - 修复：明确检查是否为None
        print(f"[DEBUG] 提取瞳孔中心")
        pupil_centers = self._key_coordinates.get("pupil_centers")
        print(f"[DEBUG] pupil_centers: {pupil_centers}")
        if pupil_centers is not None:
            print(f"[DEBUG] pupil_centers类型: {type(pupil_centers)}")
            eye_pupil = pupil_centers.get(eye_key)
            print(f"[DEBUG] eye_pupil: {eye_pupil}")
            if eye_pupil is not None:
                print(f"[DEBUG] eye_pupil类型: {type(eye_pupil)}")
                if isinstance(eye_pupil, np.ndarray):
                    print(f"[DEBUG] eye_pupil形状: {eye_pupil.shape}")
                eye_key_coordinates["pupil_centers"] = eye_pupil
        
        # 提取虹膜边界 - 修复：明确检查列表是否为空
        print(f"[DEBUG] 提取虹膜边界")
        iris_boundaries = self._key_coordinates.get("iris_boundaries")
        print(f"[DEBUG] iris_boundaries: {iris_boundaries}")
        if iris_boundaries is not None:
            print(f"[DEBUG] iris_boundaries类型: {type(iris_boundaries)}")
            eye_iris = iris_boundaries.get(eye_key)
            print(f"[DEBUG] eye_iris: {eye_iris}")
            if eye_iris is not None:
                print(f"[DEBUG] eye_iris类型: {type(eye_iris)}")
                if isinstance(eye_iris, list):
                    print(f"[DEBUG] eye_iris长度: {len(eye_iris)}")
                    if len(eye_iris) > 0:
                        print(f"[DEBUG] 第一个元素类型: {type(eye_iris[0])}")
                        if isinstance(eye_iris[0], np.ndarray):
                            print(f"[DEBUG] 第一个元素形状: {eye_iris[0].shape}")
                if len(eye_iris) > 0:
                    eye_key_coordinates["iris_boundaries"] = eye_iris
        
        # 提取眼睛轮廓 - 修复：明确检查列表是否为空
        print(f"[DEBUG] 提取眼睛轮廓")
        eyes_contours = self._key_coordinates.get("eyes_contours")
        print(f"[DEBUG] eyes_contours: {eyes_contours}")
        if eyes_contours is not None:
            print(f"[DEBUG] eyes_contours类型: {type(eyes_contours)}")
            eye_contour = eyes_contours.get(eye_key)
            print(f"[DEBUG] eye_contour: {eye_contour}")
            if eye_contour is not None:
                print(f"[DEBUG] eye_contour类型: {type(eye_contour)}")
                if isinstance(eye_contour, list):
                    print(f"[DEBUG] eye_contour长度: {len(eye_contour)}")
                    if len(eye_contour) > 0:
                        print(f"[DEBUG] 第一个元素类型: {type(eye_contour[0])}")
                        if isinstance(eye_contour[0], np.ndarray):
                            print(f"[DEBUG] 第一个元素形状: {eye_contour[0].shape}")
                if len(eye_contour) > 0:
                    eye_key_coordinates["eyes_contours"] = eye_contour
        
        print(f"[DEBUG] _extract_eye_key_coordinates完成: {eye_key_coordinates}")
        return eye_key_coordinates


def ransac_sphere_eyeball(
    key_coordinates: KeyCoordinates,
    center_fitter: Optional[Callable[[SingleEyeKeyCoordinates, int], Tuple[EllipsoidParams, np.ndarray]]] = None,
) -> Dict[str, Tuple[EllipsoidParams, np.ndarray]]:
    """
    对外唯一接口（模块级）：
    - 形如 def func(): return 构造器().func() 的封装风格
    - 构造单例、注入私有配置、执行拟合，返回左右眼 (ellipsoid_params, center)

    参数：
    - key_coordinates: KeyCoordinates
      示例：
      key_coordinates = {
        "pupil_centers": {
          "left":  np.array([0.12, 0.06, 0.82]),
          "right": np.array([0.18, 0.06, 0.82]),
        },
        "iris_boundaries": {
          "left":  [np.array([x, y, z]), ...],   # N 个点
          "right": [np.array([x, y, z]), ...],
        },
        "eyes_contours": {
          "left":  [np.array([x, y, z]), ...],   # M 个点（可选使用）
          "right": [np.array([x, y, z]), ...],
        },
      }

    - center_fitter: 可选，自定义"球/椭球中心拟合函数"，将注入为单例的私有 _center_fitter
        统一签名：
          center_fitter(key_coordinates: SingleEyeKeyCoordinates, trials_times: int)
            -> Tuple[EllipsoidParams, np.ndarray]
        返回示例：
          ({"axes": np.array([0.012, 0.011, 0.012]), "rotation": np.eye(3)}, np.array([0.10, 0.05, 0.80]))

    返回：
    - result: Dict[str, Tuple[EllipsoidParams, np.ndarray]]
      示例：
      {
        "left":  ({"axes": np.array([0.012, 0.012, 0.012]), "rotation": np.eye(3)}, np.array([0.10, 0.05, 0.80])),
        "right": ({"axes": np.array([0.012, 0.012, 0.012]), "rotation": np.eye(3)}, np.array([0.16, 0.05, 0.80])),
      }
    """
    print(f"[DEBUG] ransac_sphere_eyeball函数开始")
    print(f"[DEBUG] key_coordinates类型: {type(key_coordinates)}")
    print(f"[DEBUG] center_fitter类型: {type(center_fitter)}")
    
    try:
        fitter = _EyeCenterFitter(key_coordinates, center_fitter=center_fitter)
        print(f"[DEBUG] _EyeCenterFitter实例创建成功")
        
        fitter._update_key_coordinates(key_coordinates)
        print(f"[DEBUG] key_coordinates更新成功")
        
        result = fitter.ransac_sphere_eyeball()
        print(f"[DEBUG] ransac_sphere_eyeball执行成功")
        return result
    except Exception as e:
        print(f"[DEBUG] ransac_sphere_eyeball异常: {e}")
        import traceback
        traceback.print_exc()
        # 返回默认结果
        default_shape = EllipsoidParams()
        default_center = np.array([0.0, 0.0, 0.0])
        return {
            "left": (default_shape, default_center),
            "right": (default_shape, default_center)
        }

def ransac_sphere_eyeball_restart(key_coordinates: KeyCoordinates=None, center_fitter: Optional[Callable[[SingleEyeKeyCoordinates, int], Tuple[EllipsoidParams, np.ndarray]]] = None):
    """
    对外唯一接口（模块级）：
    - 形如 def func(): return 构造器().func() 的封装风格
    - 构造单例、注入私有配置、执行拟合，返回左右眼 (ellipsoid_params, center)
    """
    print(f"[DEBUG] ransac_sphere_eyeball_restart函数开始")
    try:
        fitter = _EyeCenterFitter(key_coordinates,center_fitter=center_fitter)
        fitter.reset_trials_time()
        result = fitter.ransac_sphere_eyeball()
        print(f"[DEBUG] ransac_sphere_eyeball_restart执行成功")
        return result
    except Exception as e:
        print(f"[DEBUG] ransac_sphere_eyeball_restart异常: {e}")
        import traceback
        traceback.print_exc()
        # 返回默认结果
        default_shape = EllipsoidParams()
        default_center = np.array([0.0, 0.0, 0.0])
        return {
            "left": (default_shape, default_center),
            "right": (default_shape, default_center)
        }