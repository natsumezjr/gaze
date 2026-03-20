"""
Data models (主定义 + 兼容层 + 历史遗留)。

分层约定（不改变任何字段与行为）：
- **Native geometry（主定义）**：EllSeg 原生几何输出所需的数据结构（如 `Ellipse2D`, `EyeNativeGeometry2D`）。
- **Compatibility export（兼容层）**：为当前下游 fitting 保持兼容的临时导出格式（如 `CompatibilityEyePoints2D`,
  以及 `LEGACY_COMPAT_POINT_SCHEMA` 的点数约定）。其 optional 点目前保持空列表。
- **Legacy face landmarker（历史遗留）**：旧 Face Landmarker 的索引/字段，仅用于回溯旧数据或旧文件对齐，
  不参与当前 EllSeg 主链（如 `LEGACY_FACE_LANDMARK_INDICES`）。
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum
import cv2

# ================= 枚举常量 =================
class FittingType(Enum):
    """拟合类型枚举"""
    PUPIL = "pupil"
    IRIS = "iris"
    INNER_CANTHUS = "inner_canthus"
    UPPER_EYELID = "upper_eyelid"
    LOWER_EYELID = "lower_eyelid"
    OUTER_CANTHUS = "outer_canthus"

class EyeType(Enum):
    """眼睛类型枚举"""
    LEFT = "left"
    RIGHT = "right"

# =============================================================================
# Compatibility export layer (下游 fitting 兼容层)
# =============================================================================
# 兼容性常量（类型枚举值）
FITTING_TYPE = [ft.value for ft in FittingType]
EYE_TYPE = [et.value for et in EyeType]

# Compatibility schema：pupil/iris 为 EllSeg 兼容导出；canthus/eyelid 非 EllSeg 原生，仅保留键返回空。
# 主链的“主定义”是 native geometry（`Ellipse2D` 等），此 schema 仅为当前下游 fitting 临时兼容。
LEGACY_COMPAT_POINT_SCHEMA: Dict[str, int] = {
    "pupil": 1,
    "iris": 4,
    "inner_canthus": 1,
    "outer_canthus": 1,
    "upper_eyelid": 5,
    "lower_eyelid": 5,
}

# =============================================================================
# Legacy face landmarker layer (历史遗留，不参与当前 EllSeg 主链)
# =============================================================================
# 该索引来自历史 Face Landmarker(468/473/iris ring 等) 的定义，仅用于回溯旧数据/旧文件对齐。
LEGACY_FACE_LANDMARK_INDICES: Dict[str, Dict[str, List[int]]] = {
    "left": {
        "pupil": [468],
        "iris": [469, 470, 471, 472],
        "inner_canthus": [133],
        "upper_eyelid": [157, 158, 159, 160, 173],
        "lower_eyelid": [145, 153, 154, 155, 161],
        "outer_canthus": [246],
    },
    "right": {
        "pupil": [473],
        "iris": [474, 475, 476, 477],
        "inner_canthus": [362],
        "upper_eyelid": [384, 385, 386, 387, 398],
        "lower_eyelid": [374, 380, 381, 382, 390],
        "outer_canthus": [466],
    },
}

# =============================================================================
# Image / basic geometry types
# =============================================================================
@dataclass
class BGRImage:
    """BGR图像数据类"""
    data: np.ndarray  # BGR图像 (H,W,3)
    
    def __post_init__(self):
        if not isinstance(self.data, np.ndarray):
            raise ValueError("image必须是numpy数组")
        if len(self.data.shape) != 3 or self.data.shape[2] != 3:
            raise ValueError("image必须是(H,W,3)的numpy数组")
        if self.data.dtype != np.uint8:
            raise ValueError("image必须是uint8类型")
        
    @property
    def height(self) -> int:
        return self.data.shape[0]
    
    @property
    def width(self) -> int:
        return self.data.shape[1]
        
    def to_rgb(self) -> np.ndarray:
        return cv2.cvtColor(self.data, cv2.COLOR_BGR2RGB)
    
    def __str__(self) -> str:
        return f"BGRImage(H={self.data.shape[0]}, W={self.data.shape[1]})"

# ================= 基础坐标类型 =================
@dataclass
class Point2D:
    """2D像素坐标"""
    x: float
    y: float
    
    def to_ndarray(self) -> np.ndarray:
        return np.array([self.x, self.y])
    
    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y}
    
    @classmethod
    def from_ndarray(cls, arr: np.ndarray) -> 'Point2D':
        return cls(x=arr[0], y=arr[1])
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Point2D':
        return cls(x=float(d["x"]), y=float(d["y"]))
    
    def __str__(self) -> str:
        return f"Point2D({self.x:.6f}, {self.y:.6f})"


# =============================================================================
# Native geometry (EllSeg 主定义)
# =============================================================================
@dataclass
class Ellipse2D:
    """2D 椭圆参数（EllSeg 原生输出，像素坐标系）。"""

    cx: float
    cy: float
    major_axis: float
    minor_axis: float
    angle_deg: float
    confidence: Optional[float] = None


@dataclass
class EyeNativeGeometry2D:
    """
    EllSeg 原生几何输出（单眼，ROI 内坐标系）。
    - segmentation_mask: 分割 mask
    - pupil_center: 瞳孔中心（椭圆拟合中心）
    - pupil_ellipse: 瞳孔椭圆
    - iris_ellipse: 虹膜椭圆
    """

    segmentation_mask: Optional[np.ndarray] = None
    pupil_center: Optional[Point2D] = None
    pupil_ellipse: Optional["Ellipse2D"] = None
    iris_ellipse: Optional["Ellipse2D"] = None
    valid: bool = False


@dataclass
class CompatibilityEyePoints2D:
    """
    兼容当前下游 fitting 的导出格式（compatibility export layer）。
    - pupil: 1 点 = EllSeg pupil_center
    - iris: 4 点 = 由 fitted iris_ellipse 推导的 left/right/top/bottom
    - 其它 optional 键存在但为空（非 EllSeg 原生输出）
    """

    pupil: List[Point2D]
    iris: List[Point2D]
    inner_canthus: List[Point2D]
    outer_canthus: List[Point2D]
    upper_eyelid: List[Point2D]
    lower_eyelid: List[Point2D]


@dataclass
class Point3D:
    """3D坐标点（单位：mm，与双目 T 一致）"""
    x: float
    y: float
    z: float
    
    def to_ndarray(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
    
    def to_tuple(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.z
    
    @classmethod
    def from_ndarray(cls, arr: np.ndarray) -> 'Point3D':
        return cls(x=arr[0], y=arr[1], z=arr[2])
    
    @classmethod
    def normalize(cls, v: 'Point3D', eps: float = 1e-9) -> 'Point3D':
        if hasattr(v, 'to_ndarray'):
            v = v.to_ndarray()
        n = np.linalg.norm(v)
        if n < eps:
            return np.zeros_like(v, dtype=float)
        return cls.from_ndarray(v / n)
    
    def distance_to(self, other: 'Point3D') -> float:
        return np.linalg.norm(self.to_ndarray() - other.to_ndarray())
    
    def __str__(self) -> str:
        return f"Point3D({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"
    
    def __sub__(self, other: 'Point3D') -> 'Point3D':
        return Point3D(x=self.x - other.x, y=self.y - other.y, z=self.z - other.z)
    
    def __add__(self, other: 'Point3D') -> 'Point3D':
        return Point3D(x=self.x + other.x, y=self.y + other.y, z=self.z + other.z)
    
    def __mul__(self, other: Union[float, int]) -> 'Point3D':
        """标量乘法 (Point3D * float)"""
        return Point3D(x=self.x * other, y=self.y * other, z=self.z * other)
    
    def __rmul__(self, other: Union[float, int]) -> 'Point3D':
        """右标量乘法 (float * Point3D)"""
        return self.__mul__(other)

Vector3D = Point3D

@dataclass
class Landmark:
    """关键点（像素坐标 + legacy z + 可见性）。"""
    x: float  # 像素x坐标
    y: float  # 像素y坐标
    z: float  # legacy z（历史遗留字段，当前 EllSeg 主链不使用该语义）
    visibility: float  # 可见性 [0,1]
    
    def to_ndarray(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.visibility])
    
    @classmethod
    def from_ndarray(cls, arr: np.ndarray) -> 'Landmark':
        return cls(x=arr[0], y=arr[1], z=arr[2], visibility=arr[3])
    
    def to_point2d(self) -> Point2D:
        """提取2D像素坐标"""
        return Point2D(x=self.x, y=self.y)
    
    def is_visible(self, threshold: float = 0.5) -> bool:
        return self.visibility >= threshold
    
    def __str__(self) -> str:
        return f"Landmark({self.x:.1f}, {self.y:.1f}, {self.z:.3f}, v={self.visibility:.2f})"

@dataclass
class Point3DWithVisibility:
    """3D坐标点 + 可见性（最终输出，单位：mm）"""
    x: float  # mm
    y: float  # mm
    z: float  # mm
    visibility: float  # 可见性 [0,1]
    
    def to_ndarray(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.visibility])
    
    @classmethod
    def from_ndarray(cls, arr: np.ndarray) -> 'Point3DWithVisibility':
        return cls(x=arr[0], y=arr[1], z=arr[2], visibility=arr[3])
    
    @classmethod
    def from_point3d_and_visibility(cls, point3d: Point3D, visibility: float) -> 'Point3DWithVisibility':
        return cls(x=point3d.x, y=point3d.y, z=point3d.z, visibility=visibility)
    
    def to_point3d(self) -> Point3D:
        return Point3D(x=self.x, y=self.y, z=self.z)
    
    def is_visible(self, threshold: float = 0.5) -> bool:
        return self.visibility >= threshold
    
    def __str__(self) -> str:
        return f"Point3DWithVisibility({self.x:.3f}, {self.y:.3f}, {self.z:.3f}, v={self.visibility:.2f})"
    
@dataclass
class KeyCoordinates:
    """关键点坐标管理 - 最终简化版本"""
    left_eye: Dict[str, List[Point3DWithVisibility]]  # 左眼各类型的关键点
    right_eye: Dict[str, List[Point3DWithVisibility]]  # 右眼各类型的关键点
    
    def __init__(self):
        # 初始化空字典
        self.left_eye = {ft.value: [] for ft in FittingType}
        self.right_eye = {ft.value: [] for ft in FittingType}
    
    def add_point(self, eye: str, fitting_type: str, point: Point3DWithVisibility):
        """添加单个关键点"""
        if eye == "left":
            self.left_eye[fitting_type].append(point)
        elif eye == "right":
            self.right_eye[fitting_type].append(point)
        else:
            raise ValueError(f"无效的眼睛类型: {eye}")
    
    def set_points(self, eye: str, fitting_type: str, points: List[Point3DWithVisibility]):
        """设置指定眼睛和类型的关键点列表"""
        if eye == "left":
            self.left_eye[fitting_type] = points
        elif eye == "right":
            self.right_eye[fitting_type] = points
        else:
            raise ValueError(f"无效的眼睛类型: {eye}")
    
    def get_points(self, eye: str, fitting_type: str) -> List[Point3DWithVisibility]:
        """获取指定眼睛和类型的关键点"""
        eye_data = self.left_eye if eye == "left" else self.right_eye
        return eye_data.get(fitting_type, [])
    
    def clear(self):
        """清空所有数据"""
        for eye_data in [self.left_eye, self.right_eye]:
            for key in eye_data:
                eye_data[key].clear()
    
    def to_dict(self) -> Dict[str, Dict[str, List[np.ndarray]]]:
        """转换为字典格式（兼容旧接口）"""
        result = {}
        for eye in ["left", "right"]:
            result[eye] = {}
            eye_data = self.left_eye if eye == "left" else self.right_eye
            for fitting_type, points in eye_data.items():
                result[eye][fitting_type] = [point.to_ndarray() for point in points]
        return result
    
    def __str__(self) -> str:
        result = "KeyCoordinates:\n"
        for eye in ["left", "right"]:
            result += f"  {eye} eye:\n"
            eye_data = self.left_eye if eye == "left" else self.right_eye
            for fitting_type, points in eye_data.items():
                result += f"    {fitting_type}: {len(points)} points\n"
        return result


# ================= Kappa 补偿类型 =================
@dataclass
class Kappa:
    """Kappa (κ) 补偿参数 - 轴角表示（axis-angle）
    表示从光轴到视轴的旋转，单位：弧度
    """
    x: float  # αx (radians)
    y: float  # αy (radians)
    z: float  # αz (radians)，通常为 0（lock_roll=True）
    
    def to_ndarray(self) -> np.ndarray:
        """转换为 numpy 数组"""
        return np.array([self.x, self.y, self.z], dtype=float)
    
    @classmethod
    def from_ndarray(cls, arr: np.ndarray) -> 'Kappa':
        """从 numpy 数组创建"""
        arr = np.asarray(arr, dtype=float)
        if arr.shape != (3,):
            raise ValueError(f"Kappa 数组必须是形状 (3,)，得到 {arr.shape}")
        return cls(x=float(arr[0]), y=float(arr[1]), z=float(arr[2]))
    
    def to_degrees(self) -> 'Kappa':
        """转换为度数（返回新的 Kappa 对象）"""
        return Kappa(
            x=np.degrees(self.x),
            y=np.degrees(self.y),
            z=np.degrees(self.z)
        )
    
    def __str__(self) -> str:
        return f"Kappa(x={np.degrees(self.x):.2f}°, y={np.degrees(self.y):.2f}°, z={np.degrees(self.z):.2f}°)"


@dataclass
class KappaEstimationResult:
    """Kappa 估计结果"""
    kappa: Kappa  # 估计的 kappa 值（弧度）
    num_samples: int  # 使用的样本数量
    mean_ang_err_deg: float  # 平均角度误差（度）
    median_ang_err_deg: float  # 中位数角度误差（度）
    max_ang_err_deg: float  # 最大角度误差（度）
    per_sample_err_deg: List[float]  # 每个样本的角度误差（度）
    
    @property
    def kappa_deg(self) -> Kappa:
        """获取度数的 kappa"""
        return self.kappa.to_degrees()
    
    def __str__(self) -> str:
        return (f"KappaEstimationResult(kappa={self.kappa}, "
                f"num_samples={self.num_samples}, "
                f"mean_err={self.mean_ang_err_deg:.2f}°, "
                f"median_err={self.median_ang_err_deg:.2f}°, "
                f"max_err={self.max_ang_err_deg:.2f}°)")


@dataclass
class KappaFitEvaluation:
    """Kappa 拟合评估结果"""
    mean_ang_err_deg: float  # 平均角度误差（度）
    median_ang_err_deg: float  # 中位数角度误差（度）
    max_ang_err_deg: float  # 最大角度误差（度）
    num_samples: int  # 评估的样本数量
    
    def __str__(self) -> str:
        return (f"KappaFitEvaluation(num_samples={self.num_samples}, "
                f"mean_err={self.mean_ang_err_deg:.2f}°, "
                f"median_err={self.median_ang_err_deg:.2f}°, "
                f"max_err={self.max_ang_err_deg:.2f}°)")


# ================= 拟合模块类型 =================
@dataclass
class GazeSamples:
    """One calibration sample.
    Provide either (target_ray) OR (target_pixel, K). If both provided, target_ray is used.
    """
    c_eye: Point3D           # (3,) 单位：mm
    c_pupil: Point3D         # (3,) 单位：mm
    target_point: Optional[np.ndarray] = None  # (3,), in camera coords (optional, 单位：mm)
    target_ray: Optional[np.ndarray] = None    # (3,), unit direction ray to target
    weight: float = 1.0

    def optical_axis(self) -> np.ndarray:
        return Vector3D.normalize(self.c_pupil - self.c_eye).to_ndarray()

    def target_dir(self) -> Optional[np.ndarray]:
        if self.target_ray is not None:
            return Vector3D.normalize(self.target_ray).to_ndarray()
        if self.target_point is not None:
            return Vector3D.normalize(self.target_point - self.c_eye).to_ndarray()
        return None


# ================= 标定前端类型 =================
@dataclass
class CalibrationRequest:
    """标定请求 - FittingScheduler → UI"""
    frame_id: int
    eye_type: str  # left/right
    intersection: Point2D
    
    def __init__(self, frame_id: int, eye_type: str, intersection: Point2D):
        self.frame_id = frame_id
        self.eye_type = eye_type
        self.intersection = intersection
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "eye_type": self.eye_type,
            "intersection": self.intersection.to_dict()
        }
        
        
@dataclass
class CalibrationResponse:
    """标定响应 - UI → FittingScheduler"""
    frame_id: int
    eye_type: str  # left/right
    target_pixel: Point2D
    background_color: str
    
    def __init__(self, frame_id: int, eye_type: str, target_pixel: Point2D, background_color: str):
        self.frame_id = frame_id
        self.eye_type = eye_type
        self.target_pixel = target_pixel
        self.background_color = background_color
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "eye_type": self.eye_type,
            "target_pixel": self.target_pixel.to_dict(),
            "background_color": self.background_color
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CalibrationResponse':
        target_pixel = Point2D.from_dict(d["target_pixel"])
        return cls(
            frame_id=int(d["frame_id"]),
            eye_type=str(d["eye_type"]),
            target_pixel=target_pixel,
            background_color=str(d["background_color"])
        )

# 导出所有类型
__all__ = [
    # 枚举
    'FittingType', 'EyeType', 'FITTING_TYPE', 'EYE_TYPE', 'LEGACY_FACE_LANDMARK_INDICES',
    'LEGACY_COMPAT_POINT_SCHEMA',
    # EllSeg 原生几何
    'Ellipse2D', 'EyeNativeGeometry2D', 'CompatibilityEyePoints2D',
    # 图像类型
    'BGRImage',
    # 基础类型
    'Point2D', 'Point3D', 'Vector3D', 'Landmark', 'Point3DWithVisibility',
    # 复合类型
    'KeyCoordinates',
    # Kappa 类型
    'Kappa', 'KappaEstimationResult', 'KappaFitEvaluation',
    # 拟合类型
    'GazeSamples',
    # 标定类型
    'CalibrationRequest', 'CalibrationResponse',
]
