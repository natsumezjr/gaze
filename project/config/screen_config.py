from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np
import logging
from project.data.data_models import Vector3D, Point3D, Point2D
from project.config.logging_config import setup_logging

logger = setup_logging(__name__, logging.DEBUG)


# =========================
# 坐标系约定（务必读）
# =========================
"""
【相机坐标系 Camera frame】—— 你的人眼/3D 都在这个坐标系里
- 原点：相机光心（摄像机原点）
- +X：相机视角图像向右（相机看到的右边）
- +Y：相机视角图像向下（常见视觉/图像坐标习惯）
- +Z：相机朝向场景前方（离相机更远）

注意：这里的“左/右”一律以【相机视角】定义，不用人的左右来想，避免镜像混乱。
相机和屏幕正对人时，人左右与相机画面左右常常镜像。

【屏幕局部坐标系 Screen frame】—— 只是用于把“落点”映射到屏幕像素
- 可以选：
  A) 原点在屏幕左上角 (推荐，和像素一致)
     x 向右，y 向下
  B) 原点在屏幕中心（某些算法喜欢）
     x 向右，y 向下
- 这个局部原点【不等于】相机原点。屏幕在相机坐标系下的位置由四角点决定。

【屏幕像素坐标 Pixel (u,v)】
- 原点在屏幕左上角
- u 向右增大，v 向下增大
"""


# ==============
# 四角点数据类
# ==============
@dataclass
class ScreenCorners:
    """屏幕四角在【相机坐标系】下的3D坐标（单位：mm 或 m，但必须与眼球/视线一致）"""
    top_left: Point3D
    top_right: Point3D
    bottom_left: Point3D
    bottom_right: Point3D


# =========================
# ScreenConfig 主类
# =========================
@dataclass
class ScreenConfig:
    """
    通过屏幕四角点（相机坐标系下）定义屏幕平面，并提供：
    - 视线射线与屏幕平面求交 -> 屏幕像素坐标
    - 像素 -> 屏幕平面3D点（相机坐标系下）
    """

    # --- 输入：屏幕四角（相机坐标系） ---
    corners: Optional[ScreenCorners] = None

    # --- 输出：平面/坐标轴（相机坐标系） ---
    # 平面上一点（默认取 top_left）
    point: Point3D = field(default_factory=lambda: Point3D(0.0, 0.0, 0.0))
    # 平面法向（单位向量）
    normal: Vector3D = field(default_factory=lambda: Vector3D(0.0, 0.0, 1.0))
    # 屏幕x轴（单位向量，指向屏幕“右”）
    x_axis: Vector3D = field(default_factory=lambda: Vector3D(1.0, 0.0, 0.0))
    # 屏幕y轴（单位向量，指向屏幕“下”）
    y_axis: Vector3D = field(default_factory=lambda: Vector3D(0.0, 1.0, 0.0))

    # --- 屏幕分辨率（像素） ---
    resolution_px: Tuple[int, int] = (0, 0)  # (W, H)

    # --- 屏幕物理尺寸（与 corners 单位一致；如果 corners=mm，这里也是 mm） ---
    # 注意：如果用 corners 推导，会自动覆盖这两个值
    width: float = 0.0
    height: float = 0.0

    # --- 屏幕局部坐标系原点选择 ---
    # True: 屏幕局部原点 = 左上角 (0,0) —— 推荐
    # False: 屏幕局部原点 = 屏幕中心 (0,0)
    origin_at_top_left: bool = True

    # --- 几何参数 ---
    intersection_tolerance: float = 1e-8
    parallel_threshold: float = 1e-6
    max_distance: float = 1e6  # 与你的单位一致：mm时可设大一点
    clamp_to_screen: bool = True  # 交点超屏幕时是否裁剪

    def __post_init__(self):
        if self.corners is not None:
            self.set_corners(self.corners)

    # ------------------------
    # 由四角点推导平面与坐标轴
    # ------------------------
    def set_corners(self, corners: ScreenCorners):
        self.corners = corners
        self._update_from_corners(corners)

    def _update_from_corners(self, corners: ScreenCorners):
        TL = corners.top_left.to_ndarray().astype(float)
        TR = corners.top_right.to_ndarray().astype(float)
        BL = corners.bottom_left.to_ndarray().astype(float)
        BR = corners.bottom_right.to_ndarray().astype(float)

        x_vec = TR - TL
        y_vec = BL - TL

        w = float(np.linalg.norm(x_vec))
        h = float(np.linalg.norm(y_vec))
        if w < 1e-9 or h < 1e-9:
            raise ValueError("Invalid screen corners: width/height too small.")

        x_axis = x_vec / w
        y_axis = y_vec / h

        # 法向 = x × y
        n = np.cross(x_axis, y_axis)
        n_norm = float(np.linalg.norm(n))
        if n_norm < 1e-9:
            raise ValueError("Invalid screen corners: normal is near zero (points may be collinear).")
        n = n / n_norm

        # 可选：统一法向方向（看你的相机坐标定义）
        # 如果你希望 normal 指向相机，可以加规则，比如让 normal·(TL-相机原点) < 0
        # 相机原点是 (0,0,0)，TL 在相机坐标下。
        # 下面示例：让 normal 朝向相机（即 normal 与 TL 的方向相反）
        # if float(np.dot(n, TL)) > 0:
        #     n = -n

        self.point = Point3D.from_ndarray(TL)         # 屏幕参考点：TL（相机坐标系下）
        self.x_axis = Vector3D.from_ndarray(x_axis)   # 屏幕向右
        self.y_axis = Vector3D.from_ndarray(y_axis)   # 屏幕向下
        self.normal = Vector3D.from_ndarray(n)        # 屏幕法向（相机坐标系下）

        # 用四角点推导屏幕物理宽高（单位与 corners 相同）
        self.width = w
        self.height = h

        # 记录一下四角一致性（可选诊断）
        w2 = float(np.linalg.norm(BR - BL))
        h2 = float(np.linalg.norm(BR - TR))
        logger.info(f"Screen size from corners: width={w:.3f}, height={h:.3f} (bottom width={w2:.3f}, right height={h2:.3f})")

    # ------------------------
    # 平面方程 ax+by+cz+d=0
    # ------------------------
    def get_plane_equation(self) -> Tuple[float, float, float, float]:
        N = self.normal.to_ndarray().astype(float)
        P0 = self.point.to_ndarray().astype(float)
        a, b, c = float(N[0]), float(N[1]), float(N[2])
        d = -float(np.dot(N, P0))
        return a, b, c, d

    # ------------------------
    # 射线与平面求交（相机坐标系）
    # ------------------------
    def _ray_plane_intersection(self, ray_origin: Point3D, ray_direction: Vector3D) -> Optional[Point3D]:
        O = ray_origin.to_ndarray().astype(float)
        D = ray_direction.to_ndarray().astype(float)
        N = self.normal.to_ndarray().astype(float)
        P0 = self.point.to_ndarray().astype(float)

        denom = float(np.dot(N, D))
        if abs(denom) < self.parallel_threshold:
            logger.warning("Ray is parallel to screen plane (no intersection).")
            return None

        t = float(np.dot(N, (P0 - O))) / denom
        if t < 0:
            logger.warning(f"Intersection behind ray origin (t={t:.6f}).")
            return None
        if t > self.max_distance:
            logger.warning(f"Intersection too far (t={t:.3f} > {self.max_distance}).")
            return None

        P = O + t * D
        return Point3D.from_ndarray(P)

    # ------------------------
    # 3D点 -> 屏幕局部2D (x,y)
    # ------------------------
    def _point3d_to_screen_2d(self, point_3d: Point3D) -> Point2D:
        """
        把相机坐标系下的3D点投影到屏幕局部2D坐标系。
        注意：这是“屏幕局部坐标”，与相机原点无关。
        """
        P = point_3d.to_ndarray().astype(float)

        TL = self.point.to_ndarray().astype(float)  # 我们把 point 定义为 top_left（相机坐标系下）
        x_axis = self.x_axis.to_ndarray().astype(float)
        y_axis = self.y_axis.to_ndarray().astype(float)

        rel = P - TL
        x = float(np.dot(x_axis, rel))
        y = float(np.dot(y_axis, rel))

        # 如果你选择“屏幕中心为局部原点”，这里平移半宽高
        if not self.origin_at_top_left:
            x -= self.width * 0.5
            y -= self.height * 0.5

        return Point2D(x, y)

    # ------------------------
    # 屏幕局部2D (x,y) -> 像素 (u,v)
    # ------------------------
    def _screen_2d_to_pixel(self, p2d: Point2D) -> Point2D:
        """
        屏幕局部坐标 -> 像素坐标
        - 屏幕局部原点若在左上角：x∈[0,width], y∈[0,height]
        - 若在中心：x∈[-w/2, w/2], y∈[-h/2, h/2]
        """
        if self.resolution_px[0] <= 0 or self.resolution_px[1] <= 0:
            raise ValueError("resolution_px is not set.")

        if self.origin_at_top_left:
            u_norm = p2d.x / self.width
            v_norm = p2d.y / self.height
        else:
            u_norm = (p2d.x + self.width * 0.5) / self.width
            v_norm = (p2d.y + self.height * 0.5) / self.height

        u = u_norm * self.resolution_px[0]
        v = v_norm * self.resolution_px[1]

        if self.clamp_to_screen:
            u = max(0.0, min(u, self.resolution_px[0] - 1))
            v = max(0.0, min(v, self.resolution_px[1] - 1))

        return Point2D(u, v)

    # ------------------------
    # 像素 (u,v) -> 屏幕局部2D (x,y)
    # ------------------------
    def _pixel_to_screen_2d(self, pixel_point: Point2D) -> Point2D:
        """
        像素坐标 -> 屏幕局部物理坐标（与像素方向一致）

        这里“左右/上下”全部按【屏幕像素】定义：
        - u 向右增大，v 向下增大
        - 与相机视角的图像方向一致（不使用人的左右概念）

        注意：这一步只是在屏幕平面上给出 (x,y) 的物理坐标，
        并不涉及相机原点；屏幕在相机坐标系下的位置由 corners 决定。
        """
        Wpx, Hpx = self.resolution_px
        if Wpx <= 0 or Hpx <= 0:
            raise ValueError("resolution_px is not set.")

        u_norm = float(pixel_point.x) / float(Wpx)
        v_norm = float(pixel_point.y) / float(Hpx)

        if self.origin_at_top_left:
            x = u_norm * self.width
            y = v_norm * self.height
        else:
            x = u_norm * self.width - self.width * 0.5
            y = v_norm * self.height - self.height * 0.5

        return Point2D(x, y)

    # ------------------------
    # 屏幕局部2D (x,y) -> 相机坐标系下3D点
    # ------------------------
    def _screen_2d_to_point3d(self, p2d: Point2D) -> Point3D:
        """
        由屏幕局部坐标恢复屏幕平面上的3D点（相机坐标系）
        """
        TL = self.point.to_ndarray().astype(float)  # top_left in camera frame
        x_axis = self.x_axis.to_ndarray().astype(float)
        y_axis = self.y_axis.to_ndarray().astype(float)

        x = float(p2d.x)
        y = float(p2d.y)

        # 若局部原点在中心，则转回 TL 原点坐标
        if not self.origin_at_top_left:
            x += self.width * 0.5
            y += self.height * 0.5

        P = TL + x * x_axis + y * y_axis
        return Point3D.from_ndarray(P)

    # ------------------------
    # 主接口：视线 -> 屏幕像素
    # ------------------------
    def calculate_gaze_intersection(
        self,
        pupil_position: Optional[Point3D] = None,
        eyeball_position: Optional[Point3D] = None,
        gaze_direction: Optional[Vector3D] = None,
    ) -> Point2D:
        """
        计算视线射线与屏幕平面交点，并返回屏幕像素坐标 (u,v)。

        - 眼球中心、瞳孔位置、视线向量都必须在【相机坐标系】下
        """
        if gaze_direction is None:
            if pupil_position is None or eyeball_position is None:
                raise ValueError("Need gaze_direction or (pupil_position and eyeball_position).")
            gaze_direction = Vector3D(
                pupil_position.x - eyeball_position.x,
                pupil_position.y - eyeball_position.y,
                pupil_position.z - eyeball_position.z,
            )

        if eyeball_position is None:
            raise ValueError("eyeball_position is required as ray origin.")

        P3 = self._ray_plane_intersection(eyeball_position, gaze_direction)
        if P3 is None:
            logger.warning("No intersection. Return screen center pixel.")
            return Point2D(self.resolution_px[0] * 0.5, self.resolution_px[1] * 0.5)

        p2d = self._point3d_to_screen_2d(P3)
        uv = self._screen_2d_to_pixel(p2d)
        return uv

    # ------------------------
    # 逆接口：像素 -> 屏幕平面3D点（相机坐标系）
    # ------------------------
    def pixel_to_3d_intersection(self, pixel_point: Point2D) -> Point3D:
        """
        将屏幕像素坐标转换为屏幕平面上的3D点（相机坐标系）。
        """
        p2d = self._pixel_to_screen_2d(pixel_point)
        P3 = self._screen_2d_to_point3d(p2d)
        return P3

    # ------------------------
    # 扩展：像素点 + 眼球中心 -> 视线方向（相机坐标系）
    # ------------------------
    def calculate_gaze_direction_from_pixel(self, pixel_point: Point2D, eyeball_position: Point3D) -> Vector3D:
        target_3d = self.pixel_to_3d_intersection(pixel_point)
        d = np.array([target_3d.x - eyeball_position.x,
                      target_3d.y - eyeball_position.y,
                      target_3d.z - eyeball_position.z], dtype=float)
        n = float(np.linalg.norm(d))
        if n < 1e-12:
            return Vector3D(0.0, 0.0, 0.0)
        d /= n
        return Vector3D.from_ndarray(d)


# ------------------------
# 分辨率获取（保留你原逻辑）
# ------------------------
def get_current_resolution_px() -> tuple[int, int]:
    """获取当前主显示器分辨率（像素）。优先使用 pywin32，失败回退到 ctypes。"""
    try:
        import win32api  # type: ignore
        width = int(win32api.GetSystemMetrics(0))
        height = int(win32api.GetSystemMetrics(1))
        if width > 0 and height > 0:
            return width, height
    except Exception as e:
        logger.error(f"win32api 获取屏幕分辨率失败: {e}", exc_info=True)

    try:
        import ctypes
        user32 = ctypes.windll.user32
        width = int(user32.GetSystemMetrics(0))
        height = int(user32.GetSystemMetrics(1))
        if width > 0 and height > 0:
            return width, height
    except Exception as e:
        logger.error(f"ctypes 获取屏幕分辨率失败: {e}", exc_info=True)

    return (0, 0)


# =========================
# 示例：如何配置四角点（相机坐标系）
# =========================
def _example_build_config() -> ScreenConfig:
    Wpx, Hpx = get_current_resolution_px()

    logger.debug(f"Screen resolution: {Wpx}x{Hpx}")

    # ✅ 注意：这些角点必须是【相机坐标系】下的3D，单位与眼球/瞳孔一致（常用 mm）
    # 下面只是示例：假设屏幕在相机前方 z=600mm，宽350mm，高225mm
    corners = ScreenCorners(
        top_left=Point3D(205.0, 17.8,-5),
        top_right=Point3D(-145.0, 17.8,-5),
        bottom_left=Point3D(205.0, 237.8,-5),
        bottom_right=Point3D(-145.0, 237.8,-5),
    )

    cfg = ScreenConfig(
        corners=corners,
        resolution_px=(Wpx, Hpx),
        origin_at_top_left=True,   # True: 屏幕局部原点左上角；False: 屏幕中心
        clamp_to_screen=True,
    )
    a, b, c, d = cfg.get_plane_equation()
    logger.info(f"Screen plane equation: {a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0")
    return cfg

SCREEN_CONFIG = _example_build_config()

def main() -> None:
    cfg = _example_build_config()

    # 假设 mm
    pupil_position = Point3D(15.0, 170.0, 400.0)
    eyeball_position = Point3D(15.0, 169.0, 409.5)

    uv = cfg.calculate_gaze_intersection(pupil_position=pupil_position, eyeball_position=eyeball_position)
    print("gaze pixel:", uv)

    P3 = cfg.pixel_to_3d_intersection(uv)
    print("pixel->3D on screen plane:", P3)


if __name__ == "__main__":
    main()