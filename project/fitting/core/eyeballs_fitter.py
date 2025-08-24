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

from project.fitting.config.models import KeyCoordinates, SingleEyeKeyCoordinates, EllipsoidParams
from typing import Dict, Optional, Tuple, Callable   
import numpy as np
import logging



# 识别模块输出的 key_coordinates 的宽松结构：
# 顶层键：'pupil_center' | 'iris_boundaries' | 'eyes_contours'
# 次层键：'left' | 'right'
# 值：np.ndarray(3,) 或 List[np.ndarray(3,)]，并兼容 None




__all__ = ["ransac_sphere_eyeball","ransac_sphere_eyeball_restart"]


class _EyeCenterFitter:
    """
    单例类：维护跨帧的拟合配置与缓存（如固定椭球/上一帧中心），对外不直接暴露。

    初始化参数：
    - key_coordinates: KeyCoordinates
      约定包含：
        - 'pupil_center': {'left': np.ndarray(3,), 'right': np.ndarray(3,)} 或 None/零向量
        - 'iris_boundaries': {'left': List[np.ndarray(3,)], 'right': List[np.ndarray(3,)]} 或 []
        - 'eyes_contours': {'left': List[np.ndarray(3,)], 'right': List[np.ndarray(3,)]} 或 []
      注：上游识别模块需保证这些键名与结构一致。

    - center_fitter: 可选自定义拟合器，签名统一为：
        center_fitter(key_coordinates: SingleEyeKeyCoordinates, trials_times: int)
            -> Tuple[EllipsoidParams, np.ndarray]  # (椭球参数, 中心坐标)
    """

    _instance: Optional["_EyeCenterFitter"] = None

    def __new__(cls, key_coordinates: KeyCoordinates, center_fitter: Optional[Callable[[SingleEyeKeyCoordinates, int], Tuple[EllipsoidParams, np.ndarray]]] = None) -> "_EyeCenterFitter":
        logging.debug(f"_EyeCenterFitter.__new__: cls._instance={cls._instance}")
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            logging.debug(f"创建新的单例实例")
        else:
            logging.debug(f"返回现有单例实例")
        return cls._instance  # 单例

    def __init__(self, key_coordinates: KeyCoordinates = None, center_fitter: Optional[Callable[[SingleEyeKeyCoordinates, int], Tuple[EllipsoidParams, np.ndarray]]] = None) -> None:
        logging.debug(f"_EyeCenterFitter.__init__开始")
        logging.debug(f"key_coordinates类型: {type(key_coordinates)}")
        if key_coordinates:
            logging.debug(f"key_coordinates键: {list(key_coordinates.keys())}")
        
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
        
        logging.debug(f"_EyeCenterFitter.__init__完成")

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
        logging.debug(f"ransac_sphere_eyeball开始")
        logging.debug(f"当前trials_times: {self._trials_times}")
        logging.debug(f"center_fitter类型: {type(self._center_fitter)}")
        
        try:
            self._fit_one_eye("left")
            logging.debug(f"左眼拟合完成")
        except Exception as e:
            logging.debug(f"左眼拟合异常: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            self._fit_one_eye("right")
            logging.debug(f"右眼拟合完成")
        except Exception as e:
            logging.debug(f"右眼拟合异常: {e}")
            import traceback
            traceback.print_exc()
        
        result = {
            "left": (self._fixed_shapes["left"], self._eye_centers["left"]),
            "right": (self._fixed_shapes["right"], self._eye_centers["right"])
        }
        logging.debug(f"ransac_sphere_eyeball返回结果: {result}")
        return result

    # ============ 私有辅助接口（占位）===========

    def _fit_one_eye(self, eye_key: str) -> Tuple[EllipsoidParams, np.ndarray]:
        """
        内部：对单眼进行中心拟合；可选复用固定形状；使用 _center_fitter。
        返回 (ellipsoid_params, center_xyz)。
        """
        logging.debug(f"_fit_one_eye开始: eye_key={eye_key}")
        
        # 动态维护trials_times，每次拟合++
        self._trials_times += 1
        logging.debug(f"更新trials_times: {self._trials_times}")
        
        # 为每只眼睛创建单独的关键点坐标
        logging.debug(f"开始提取{eye_key}眼关键点坐标")
        eye_key_coordinates = self._extract_eye_key_coordinates(eye_key)
        logging.debug(f"{eye_key}眼关键点坐标: {eye_key_coordinates}")
        
        # 调用center_fitter进行拟合
        logging.debug(f"调用center_fitter进行拟合")
        if self._center_fitter is None:
            logging.debug(f"center_fitter为None，使用默认拟合器")
            from project.fitting.utils.geometry import center_fitter
            self._center_fitter = center_fitter
        
        try:
            self._fixed_shapes[eye_key], self._eye_centers[eye_key] = self._center_fitter(eye_key_coordinates, self._trials_times)
            logging.debug(f"{eye_key}眼拟合完成: shape={self._fixed_shapes[eye_key]}, center={self._eye_centers[eye_key]}")
        except Exception as e:
            logging.debug(f"{eye_key}眼拟合异常: {e}")
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
        logging.debug(f"_set_params: center_fitter={center_fitter}")
        self._center_fitter = center_fitter

    def _update_key_coordinates(self, key_coordinates: KeyCoordinates) -> None:
        """
        内部：当外部传入新的关键点坐标时，更新内部状态（跨帧复用）。
        """
        logging.debug(f"_update_key_coordinates开始")
        logging.debug(f"新key_coordinates类型: {type(key_coordinates)}")
        if key_coordinates:
            logging.debug(f"新key_coordinates键: {list(key_coordinates.keys())}")
            for key, value in key_coordinates.items():
                logging.debug(f"{key}: {type(value)}")
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        logging.debug(f"  {sub_key}: {type(sub_value)}")
                        if isinstance(sub_value, list):
                            logging.debug(f"    list长度: {len(sub_value)}")
                            if len(sub_value) > 0:
                                logging.debug(f"    第一个元素类型: {type(sub_value[0])}")
                        elif isinstance(sub_value, np.ndarray):
                            logging.debug(f"    array形状: {sub_value.shape}")
        
        self._key_coordinates = key_coordinates
        logging.debug(f"_update_key_coordinates完成")
        
    def reset_trials_time(self) -> None:
        """
        内部：重置采样次数。
        """
        logging.debug(f"reset_trials_time: {self._trials_times} -> 0")
        self._trials_times = 0
        
    def _extract_eye_key_coordinates(self, eye_key: str) -> SingleEyeKeyCoordinates:
        """
        为指定眼睛提取关键点坐标
        """
        logging.debug(f"_extract_eye_key_coordinates开始: eye_key={eye_key}")
        logging.debug(f"self._key_coordinates: {self._key_coordinates}")
        
        eye_key_coordinates = {
            "pupil_center": None,
            "iris_boundaries": [],
            "eyes_contours": []
        }
        
        # 提取瞳孔中心 - 修复：明确检查是否为None
        logging.debug(f"提取瞳孔中心")
        pupil_center = self._key_coordinates.get("pupil_center")  # ✅ 使用正确的键名
        logging.debug(f"pupil_center: {pupil_center}")
        if pupil_center is not None:
            logging.debug(f"pupil_center类型: {type(pupil_center)}")
            eye_pupil = pupil_center.get(eye_key)
            logging.debug(f"eye_pupil: {eye_pupil}")
            if eye_pupil is not None:
                logging.debug(f"eye_pupil类型: {type(eye_pupil)}")
                if isinstance(eye_pupil, np.ndarray):
                    logging.debug(f"eye_pupil形状: {eye_pupil.shape}")
                eye_key_coordinates["pupil_center"] = eye_pupil  # ✅ 保持键名一致
        
        # 提取虹膜边界 - 修复：明确检查列表是否为空
        logging.debug(f"提取虹膜边界")
        iris_boundaries = self._key_coordinates.get("iris_boundaries")
        logging.debug(f"iris_boundaries: {iris_boundaries}")
        if iris_boundaries is not None:
            logging.debug(f"iris_boundaries类型: {type(iris_boundaries)}")
            eye_iris = iris_boundaries.get(eye_key)
            logging.debug(f"eye_iris: {eye_iris}")
            if eye_iris is not None:
                logging.debug(f"eye_iris类型: {type(eye_iris)}")
                if isinstance(eye_iris, list):
                    logging.debug(f"eye_iris长度: {len(eye_iris)}")
                    if len(eye_iris) > 0:
                        logging.debug(f"第一个元素类型: {type(eye_iris[0])}")
                        if isinstance(eye_iris[0], np.ndarray):
                            logging.debug(f"第一个元素形状: {eye_iris[0].shape}")
                if len(eye_iris) > 0:
                    eye_key_coordinates["iris_boundaries"] = eye_iris
        
        # 提取眼睛轮廓 - 修复：明确检查列表是否为空
        logging.debug(f"提取眼睛轮廓")
        eyes_contours = self._key_coordinates.get("eyes_contours")
        logging.debug(f"eyes_contours: {eyes_contours}")
        if eyes_contours is not None:
            logging.debug(f"eyes_contours类型: {type(eyes_contours)}")
            eye_contour = eyes_contours.get(eye_key)
            logging.debug(f"eye_contour: {eye_contour}")
            if eye_contour is not None:
                logging.debug(f"eye_contour类型: {type(eye_contour)}")
                if isinstance(eye_contour, list):
                    logging.debug(f"eye_contour长度: {len(eye_contour)}")
                    if len(eye_contour) > 0:
                        logging.debug(f"第一个元素类型: {type(eye_contour[0])}")
                        if isinstance(eye_contour[0], np.ndarray):
                            logging.debug(f"第一个元素形状: {eye_contour[0].shape}")
                if len(eye_contour) > 0:
                    eye_key_coordinates["eyes_contours"] = eye_contour
        
        logging.debug(f"_extract_eye_key_coordinates完成: {eye_key_coordinates}")
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
        "pupil_center": {
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
    logging.debug(f"ransac_sphere_eyeball函数开始")
    logging.debug(f"key_coordinates类型: {type(key_coordinates)}")
    logging.debug(f"center_fitter类型: {type(center_fitter)}")
    
    try:
        fitter = _EyeCenterFitter(key_coordinates, center_fitter=center_fitter)
        logging.debug(f"_EyeCenterFitter实例创建成功")
        
        fitter._update_key_coordinates(key_coordinates)
        logging.debug(f"key_coordinates更新成功")
        
        result = fitter.ransac_sphere_eyeball()
        logging.debug(f"ransac_sphere_eyeball执行成功")
        return result
    except Exception as e:
        logging.debug(f"ransac_sphere_eyeball异常: {e}")
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
    logging.debug(f"ransac_sphere_eyeball_restart函数开始")
    try:
        fitter = _EyeCenterFitter(key_coordinates,center_fitter=center_fitter)
        fitter.reset_trials_time()
        result = fitter.ransac_sphere_eyeball()
        logging.debug(f"ransac_sphere_eyeball_restart执行成功")
        return result
    except Exception as e:
        logging.debug(f"ransac_sphere_eyeball_restart异常: {e}")
        import traceback
        traceback.print_exc()
        # 返回默认结果
        default_shape = EllipsoidParams()
        default_center = np.array([0.0, 0.0, 0.0])
        return {
            "left": (default_shape, default_center),
            "right": (default_shape, default_center)
        }