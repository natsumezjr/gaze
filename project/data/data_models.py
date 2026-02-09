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

# 兼容性常量
FITTING_TYPE = [ft.value for ft in FittingType]
EYE_TYPE = [et.value for et in EyeType]

# ================= 识别模块图像类型 =================
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
    
@dataclass
class DepthMap:
    """深度图数据类"""
    data: np.ndarray  # 深度图 (H,W)
    unit: str = "camera_unit"
    
    def __post_init__(self):
        if not isinstance(self.data, np.ndarray):
            raise ValueError("depth_map必须是numpy数组")
        if len(self.data.shape) != 2:
            raise ValueError("depth_map必须是2维numpy数组")
        if self.data.dtype != np.float32:
            raise ValueError("depth_map必须是float32类型")
        if self.unit not in ["camera_unit", "meter"]:
            raise ValueError("unit必须是camera_unit或meter")
    
    @property
    def height(self) -> int:
        return self.data.shape[0]
    
    @property
    def width(self) -> int:
        return self.data.shape[1]
    
    def to_meters(self, depth_scale: float = 0.001) -> "DepthMap":
        if self.unit == "meter":
            return DepthMap(data=self.data, unit="meter")
        elif self.unit == "camera_unit":
            return DepthMap(data=self.data * depth_scale, unit="meter")
        else:
            raise ValueError("unit必须是camera_unit或meter")
        
    def validate_with_image(self, image: BGRImage) -> bool:
        if self.height != image.height or self.width != image.width:
            return False
        return True
    
    def copy(self) -> "DepthMap":
        return DepthMap(data=self.data.copy(), unit=self.unit)
    
    def __str__(self) -> str:
        return f"DepthMap(H={self.data.shape[0]}, W={self.data.shape[1]}, unit={self.unit})"

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

@dataclass
class Point3D:
    """3D坐标点（米单位）"""
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
    """关键点（像素坐标 + MediaPipe估计的z + 可见性）"""
    x: float  # 像素x坐标
    y: float  # 像素y坐标
    z: float  # MediaPipe估计的z坐标
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
    """3D坐标点 + 可见性（最终输出）"""
    x: float  # 米单位
    y: float  # 米单位
    z: float  # 米单位
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
    c_eye: Point3D           # (3,)
    c_pupil: Point3D         # (3,)
    target_point: Optional[np.ndarray] = None  # (3,), in camera coords (optional)
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
    'FittingType', 'EyeType', 'FITTING_TYPE', 'EYE_TYPE',
    # 图像类型
    'BGRImage', 'DepthMap',
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
