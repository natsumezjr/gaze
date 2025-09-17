from __future__ import annotations
from dataclasses import dataclass, field
from project.data.data_models import Vector3D, Point3D, Point2D
import numpy as np
from project.config.logging_config import setup_logging, get_logger
import logging
setup_logging(level=logging.DEBUG)
logger = get_logger(__name__)


DEFAULT_SCREEN_WIDTH_M: float = 0.30
DEFAULT_SCREEN_HEIGHT_M: float = 0.225


@dataclass
class ScreenConfig:
    """屏幕配置类 - 按照算法步骤重新组织"""
    # 屏幕平面定义（步骤1：平面定义）
    normal: Vector3D = field(default_factory=lambda: Vector3D(0.0, 0.0, 1.0))
    point: Point3D = field(default_factory=lambda: Point3D(0.0, 0.0, 0.0))
    x_axis: Vector3D = field(default_factory=lambda: Vector3D(1.0, 0.0, 0.0))
    y_axis: Vector3D = field(default_factory=lambda: Vector3D(0.0, 1.0, 0.0))
    
    # 屏幕物理参数
    width_m: float = 0.30
    height_m: float = 0.225
    resolution_px: tuple[int, int] = (0, 0)
    
    # 屏幕边界定义（用于像素映射）
    top_left_m: Point3D = field(default_factory=lambda: Point3D(0.15, 0.005, 0.0))

    xmin_m: float = -0.15
    xmax_m: float = 0.15
    ymin_m: float = 0.005
    ymax_m: float = 0.005 + 0.30
    
    
    
    # 单位转换
    to_mm: bool = True
    
    # 几何计算参数
    intersection_tolerance: float = 1e-8
    parallel_threshold: float = 1e-6
    boundary_margin: float = 0.01
    max_distance: float = 1000.0
    
    def __post_init__(self):
        if self.to_mm:
            self.width_mm = self.width_m * 1000
            self.height_mm = self.height_m * 1000
            self.top_left_mm = Point3D(self.top_left_m.x * 1000, self.top_left_m.y * 1000, self.top_left_m.z * 1000)
            self.xmin_mm = self.xmin_m * 1000
            self.xmax_mm = self.xmax_m * 1000
            self.ymin_mm = self.ymin_m * 1000
            self.ymax_mm = self.ymax_m * 1000
    
    def set_point(self, point: Point3D):
        """设置屏幕平面中心点"""
        self.point = point
    
    def set_normal(self, normal: Vector3D):
        """设置屏幕平面法向量"""
        self.normal = normal
    
    def set_x_axis(self, x_axis: Vector3D):
        """设置屏幕平面X轴方向"""
        self.x_axis = x_axis
    
    def set_y_axis(self, y_axis: Vector3D):
        """设置屏幕平面Y轴方向"""
        self.y_axis = y_axis
        
    def set_top_left(self, top_left: Point3D):
        """设置屏幕左上角坐标"""
        if self.to_mm:
            self.top_left_mm = top_left
        else:
            self.top_left_m = top_left

    def _ray_plane_intersection(self, ray_origin: Point3D, ray_direction: Vector3D) -> Point3D | None:
        """
        步骤2：射线求交 - 计算视线与屏幕平面的交点
        
        使用数学公式：
        t = N · (P_0 - O) / (N · D)
        P_intersect = O + t * D
        
        Args:
            ray_origin: 射线起点（眼球中心）
            ray_direction: 射线方向（视线向量）
        
        Returns:
            交点坐标，如果无交点则返回None
        """
        # 转换为numpy数组
        O = ray_origin.to_ndarray()
        D = ray_direction.to_ndarray()
        N = self.normal.to_ndarray()
        P_0 = self.point.to_ndarray()
        
        logger.debug(f"射线起点: {O}")
        logger.debug(f"射线方向: {D}")
        logger.debug(f"平面法向量: {N}")
        logger.debug(f"平面点: {P_0}")
        
        # 计算分母：N · D
        denom = float(np.dot(N, D))
        
        # 检查射线是否与平面平行
        if abs(denom) < self.parallel_threshold:
            logger.warning("视线与屏幕平面平行，无交点")
            return None
        
        # 计算参数 t = N · (P_0 - O) / (N · D)
        t = float(np.dot(N, P_0 - O)) / denom
        
        # 检查交点是否在射线前方
        if t < 0:
            logger.warning(f"交点在射线起点后方 (t={t:.6f})")
            return None
        
        # 检查交点距离是否合理
        if t > self.max_distance:
            logger.warning(f"交点距离过远 (t={t:.6f} > {self.max_distance})")
            return None
        
        # 计算交点坐标：P = O + t * D
        intersection_point = O + t * D
        logger.info(f"交点坐标: {intersection_point}")
        
        return Point3D.from_ndarray(intersection_point)
    
    def _point3d_to_screen_2d(self, point_3d: Point3D) -> Point2D:
        """
        步骤3：坐标转换 - 将3D交点转换为屏幕2D坐标
        
        使用数学公式：
        p_2D = [x_axis · (P_intersect - P_0), y_axis · (P_intersect - P_0)]
        
        Args:
            point_3d: 3D交点坐标
        
        Returns:
            屏幕2D坐标
        """
        P_intersect = point_3d.to_ndarray()
        P_0 = self.point.to_ndarray()
        x_axis = self.x_axis.to_ndarray()
        y_axis = self.y_axis.to_ndarray()
        
        logger.debug(f"3D交点: {P_intersect}")
        logger.debug(f"平面中心: {P_0}")
        logger.debug(f"X轴: {x_axis}")
        logger.debug(f"Y轴: {y_axis}")
        
        # 计算在屏幕坐标系中的位置
        relative_pos = P_intersect - P_0
        x_coord = float(np.dot(x_axis, relative_pos))
        y_coord = float(np.dot(y_axis, relative_pos))
        
        point_2d = Point2D(x_coord, y_coord)
        logger.debug(f"屏幕2D坐标: {point_2d}")
        
        return point_2d
    
    def _screen_2d_to_pixel(self, point_2d: Point2D) -> Point2D:
        """
        步骤4：像素映射 - 将物理坐标映射到像素坐标
        
        使用数学公式：
        u = (x_max - x) / (x_max - x_min) * W_screen
        v = (y - y_min) / (y_max - y_min) * H_screen
        
        Args:
            point_2d: 屏幕2D坐标
        
        Returns:
            像素坐标
        """
        if self.to_mm:
            x_max = self.xmax_mm
            x_min = self.xmin_mm
            y_min = self.ymin_mm
            y_max = self.ymax_mm
        else:
            x_min = self.xmin_m
            x_max = self.xmax_m
            y_min = self.ymin_m
            y_max = self.ymax_m

        
        logger.debug(f"屏幕边界: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
        logger.debug(f"屏幕分辨率: {self.resolution_px}")
        
        # 左上角归一化坐标
        u_norm = (x_max - point_2d.x) / (x_max - x_min)
        v_norm = (point_2d.y - y_min) / (y_max - y_min)
        
        # 映射到像素坐标
        u_px = u_norm * self.resolution_px[0]
        v_px = v_norm * self.resolution_px[1]
        
        pixel_point = Point2D(u_px, v_px)
        logger.debug(f"归一化坐标: ({u_norm:.4f}, {v_norm:.4f})")
        logger.debug(f"像素坐标: {pixel_point}")
        
        return pixel_point
    
    def _validate_intersection(self, pixel_point: Point2D) -> Point2D:
        """
        步骤5&6：边界检查和有效性验证
        
        Args:
            pixel_point: 像素坐标
        
        Returns:
            验证后的像素坐标
        """
        # 边界裁剪
        x_clipped = max(0, min(pixel_point.x, self.resolution_px[0] - 1))
        y_clipped = max(0, min(pixel_point.y, self.resolution_px[1] - 1))
        
        clipped_point = Point2D(x_clipped, y_clipped)
        
        # 检查是否在边界附近
        if (pixel_point.x != x_clipped or pixel_point.y != y_clipped):
            logger.warning(f"交点超出屏幕边界，已裁剪: {pixel_point} -> {clipped_point}")
        
        logger.info(f"最终像素坐标: {clipped_point}")
        return clipped_point
    
    def calculate_gaze_intersection(self, pupil_position: Point3D, eyeball_position: Point3D) -> Point2D:
        """
        主接口：计算视线与屏幕的交点（像素坐标）
        
        按照算法步骤执行：
        1. 平面定义
        2. 射线求交
        3. 坐标转换
        4. 像素映射
        5. 边界检查
        6. 有效性验证
        
        Args:
            pupil_position: 瞳孔位置（起点）
            eyeball_position: 眼球位置
        
        Returns:
            注视点像素坐标
        """
        # 计算视线方向
        gaze_direction = Vector3D(
            pupil_position.x - eyeball_position.x,
            pupil_position.y - eyeball_position.y,
            pupil_position.z - eyeball_position.z
        )
        
        logger.info(f"瞳孔位置: {pupil_position}")
        logger.info(f"眼球位置: {eyeball_position}")
        logger.info(f"视线方向: {gaze_direction}")
        
        # 步骤2：射线求交
        intersection_3d = self._ray_plane_intersection(eyeball_position, gaze_direction)
        if intersection_3d is None:
            logger.warning("视线与屏幕无交点，返回屏幕中心")
            return Point2D(self.resolution_px[0] // 2, self.resolution_px[1] // 2)
        
        # 步骤3：坐标转换
        screen_2d = self._point3d_to_screen_2d(intersection_3d)
        
        # 步骤4：像素映射
        pixel_point = self._screen_2d_to_pixel(screen_2d)
        
        # 步骤5&6：边界检查和有效性验证
        final_point = self._validate_intersection(pixel_point)
        
        return final_point
    
    def pixel_to_3d_intersection(self, pixel_point: Point2D, eyeball_position: Point3D) -> Point3D | None:
        """
        逆运算：将像素坐标转换为3D交点
        
        按照逆算法步骤执行：
        1. 像素坐标验证
        2. 像素到物理坐标转换
        3. 2D到3D坐标转换
        4. 构造射线并返回3D交点
        
        Args:
            pixel_point: 像素坐标
            eyeball_position: 眼球位置（用于构造射线）
        
        Returns:
            3D交点坐标，如果无效则返回None
        """
        # 步骤1：像素坐标验证
        if not self._validate_pixel_coordinates(pixel_point):
            logger.warning(f"像素坐标超出屏幕范围: {pixel_point}")
            return None
        
        # 步骤2：像素到物理坐标转换
        screen_2d = self._pixel_to_screen_2d(pixel_point)
        
        # 步骤3：2D到3D坐标转换
        intersection_3d = self._screen_2d_to_point3d(screen_2d)
        
        # 步骤4：构造射线并返回3D交点
        logger.info(f"像素坐标 {pixel_point} 转换为3D交点: {intersection_3d}")
        return intersection_3d

    def _validate_pixel_coordinates(self, pixel_point: Point2D) -> bool:
        """
        步骤1：验证像素坐标是否在有效范围内
        
        Args:
            pixel_point: 像素坐标
        
        Returns:
            是否有效
        """
        if (pixel_point.x < 0 or pixel_point.x >= self.resolution_px[0] or
            pixel_point.y < 0 or pixel_point.y >= self.resolution_px[1]):
            return False
        return True

    def _pixel_to_screen_2d(self, pixel_point: Point2D) -> Point2D:
        """
        步骤2：像素坐标转换为屏幕2D坐标（逆像素映射）
        
        使用逆数学公式：
        x = x_max - u_norm * (x_max - x_min)
        y = y_min + v_norm * (y_max - y_min)
        其中 u_norm = u_px / W_screen, v_norm = v_px / H_screen
        
        Args:
            pixel_point: 像素坐标
        
        Returns:
            屏幕2D坐标
        """
        if self.to_mm:
            x_max = self.xmax_mm
            x_min = self.xmin_mm
            y_min = self.ymin_mm
            y_max = self.ymax_mm
        else:
            x_min = self.xmin_m
            x_max = self.xmax_m
            y_min = self.ymin_m
            y_max = self.ymax_m
        
        logger.debug(f"像素坐标: {pixel_point}")
        logger.debug(f"屏幕边界: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
        
        # 归一化坐标
        u_norm = pixel_point.x / self.resolution_px[0]
        v_norm = pixel_point.y / self.resolution_px[1]
        
        # 转换为物理坐标
        x_coord = x_max - u_norm * (x_max - x_min)
        y_coord = y_min + v_norm * (y_max - y_min)
        
        screen_2d = Point2D(x_coord, y_coord)
        logger.debug(f"归一化坐标: ({u_norm:.4f}, {v_norm:.4f})")
        logger.debug(f"屏幕2D坐标: {screen_2d}")
        
        return screen_2d

    def _screen_2d_to_point3d(self, point_2d: Point2D) -> Point3D:
        """
        步骤3：屏幕2D坐标转换为3D坐标（逆坐标转换）
        
        使用逆数学公式：
        P_intersect = P_0 + x_coord * x_axis + y_coord * y_axis
        
        Args:
            point_2d: 屏幕2D坐标
        
        Returns:
            3D交点坐标
        """
        P_0 = self.point.to_ndarray()
        x_axis = self.x_axis.to_ndarray()
        y_axis = self.y_axis.to_ndarray()
        
        logger.debug(f"屏幕2D坐标: {point_2d}")
        logger.debug(f"平面中心: {P_0}")
        logger.debug(f"X轴: {x_axis}")
        logger.debug(f"Y轴: {y_axis}")
        
        # 计算3D坐标：P = P_0 + x * x_axis + y * y_axis
        intersection_3d = P_0 + point_2d.x * x_axis + point_2d.y * y_axis
        
        point_3d = Point3D.from_ndarray(intersection_3d)
        logger.debug(f"3D交点坐标: {point_3d}")
        
        return point_3d

    def calculate_gaze_direction_from_pixel(self, pixel_point: Point2D, eyeball_position: Point3D) -> Vector3D | None:
        """
        扩展功能：从像素坐标计算视线方向向量
        
        Args:
            pixel_point: 像素坐标
            eyeball_position: 眼球位置
        
        Returns:
            视线方向向量，如果无效则返回None
        """
        # 获取3D交点
        intersection_3d = self.pixel_to_3d_intersection(pixel_point, eyeball_position)
        if intersection_3d is None:
            return None
        
        # 计算视线方向
        gaze_direction = Vector3D(
            intersection_3d.x - eyeball_position.x,
            intersection_3d.y - eyeball_position.y,
            intersection_3d.z - eyeball_position.z
        )
        
        # 归一化方向向量
        direction_array = gaze_direction.to_ndarray()
        norm = np.linalg.norm(direction_array)
        if norm > 0:
            normalized_direction = direction_array / norm
            gaze_direction = Vector3D.from_ndarray(normalized_direction)
        
        logger.info(f"从像素坐标 {pixel_point} 计算的视线方向: {gaze_direction}")
        return gaze_direction
    


def get_current_resolution_px() -> tuple[int, int]:
    """获取当前主显示器分辨率（像素）。优先使用 pywin32，失败回退到 ctypes。"""
    # 方案一：pywin32
    try:
        import win32api  # type: ignore
        width = int(win32api.GetSystemMetrics(0))  # SM_CXSCREEN
        height = int(win32api.GetSystemMetrics(1))  # SM_CYSCREEN
        if width > 0 and height > 0:
            return width, height
    except Exception:
        pass
    # 方案二：ctypes（不依赖 pywin32）
    try:
        import ctypes
        user32 = ctypes.windll.user32
        width = int(user32.GetSystemMetrics(0))
        height = int(user32.GetSystemMetrics(1))
        if width > 0 and height > 0:
            return width, height
    except Exception:
        pass
    return (0, 0)


# 获取屏幕配置
_SCREEN_WIDTH_M, _SCREEN_HEIGHT_M = DEFAULT_SCREEN_WIDTH_M, DEFAULT_SCREEN_HEIGHT_M
_CURRENT_RES_W, _CURRENT_RES_H = get_current_resolution_px()

# 创建屏幕配置实例
SCREEN_CONFIG = ScreenConfig(
    width_m=_SCREEN_WIDTH_M,
    height_m=_SCREEN_HEIGHT_M,
    resolution_px=(_CURRENT_RES_W, _CURRENT_RES_H)
)


def main() -> None:
    print("SCREEN_CONFIG:", SCREEN_CONFIG)
    pupil_position = Point3D(15.0, 170, 400)
    eyeball_position = Point3D(15.0, 169, 409.5)
    point2d = SCREEN_CONFIG.calculate_gaze_intersection(pupil_position, eyeball_position)
    print("point2d:", point2d)
    point3d = SCREEN_CONFIG.pixel_to_3d_intersection(point2d, eyeball_position)
    print("point3d:", point3d)


if __name__ == "__main__":
    main()