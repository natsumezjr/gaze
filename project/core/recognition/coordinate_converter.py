# 坐标转换模块
import numpy as np
import cv2
from typing import List, Dict, Optional
import logging
from project.data.data_models import (
    Point2D, Point3D, Landmark, Point3DWithVisibility, 
    BGRImage, DepthMap
)

# 配置日志
from project.config.logging_config import setup_logging 
logger = setup_logging(__name__)


# 默认参数（深度单位：米；低于此值视为无效，支持近距离人脸约 5cm+）
DEPTH_VALIDATION_THRESHOLD = 0.05
COORDINATE_QUALITY_THRESHOLD = 0.8

def pixel_to_3d(pixel: Point2D, depth_map: DepthMap, camera_params: Dict) -> Point3D:
    """
    像素坐标结合深度转三维坐标
    
    Args:
        pixel: 像素坐标
        depth_map: 深度图
        camera_params: 相机参数字典
    
    Returns:
        point_3d: 三维坐标点（米单位）
    
    Raises:
        ValueError: 输入参数无效或深度值无效时抛出
    """
    
    # 提取相机参数
    intrinsic_params = camera_params.get("intrinsic_params", {})
    fx = intrinsic_params.get("fx", 925.0)
    fy = intrinsic_params.get("fy", 925.0)
    cx = intrinsic_params.get("cx", 640.0)
    cy = intrinsic_params.get("cy", 360.0)
    
    x_pixel, y_pixel = pixel.x, pixel.y
    height, width = depth_map.height, depth_map.width
    
    # 检查像素坐标是否在图像范围内，如果超出则限制到边界
    if not (0.0 <= x_pixel < width and 0.0 <= y_pixel < height):
        logger.warning(f"像素坐标({y_pixel:.2f}, {x_pixel:.2f})超出图像范围({height}, {width})，已限制到边界")
        x_pixel = max(0.0, min(x_pixel, width - 1))
        y_pixel = max(0.0, min(y_pixel, height - 1))
    
    # 获取深度值（使用插值）
    depth_meters = get_depth_interpolated(depth_map, pixel)
    
    # 验证深度值
    if not np.isfinite(depth_meters) or depth_meters <= DEPTH_VALIDATION_THRESHOLD:
        raise ValueError(f"无效的深度值：{depth_meters}，必须大于{DEPTH_VALIDATION_THRESHOLD}米")
    
    # 转换公式：X = (x_pixel - cx) × Z / fx, Y = (y_pixel - cy) × Z / fy, Z = depth
    x_3d = (x_pixel - cx) * depth_meters / fx
    y_3d = (y_pixel - cy) * depth_meters / fy
    z_3d = depth_meters
    
    return Point3D(x=x_3d, y=y_3d, z=z_3d)

def get_depth_interpolated(depth_map: DepthMap, pixel: Point2D) -> float:
    """
    使用OpenCV remap进行高效插值获取深度值
    
    Args:
        depth_map: 深度图
        pixel: 像素坐标
    
    Returns:
        depth_value: 插值后的深度值
    """
    x, y = pixel.x, pixel.y
    height, width = depth_map.height, depth_map.width
    
    # 边界处理
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    
    # 创建坐标映射数组
    map_x = np.array([[x]], dtype=np.float32)
    map_y = np.array([[y]], dtype=np.float32)
    
    # 使用OpenCV remap进行插值
    interpolated = cv2.remap(depth_map.data, map_x, map_y, 
                            cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=np.nan)
    
    return float(interpolated[0, 0])

def batch_convert_landmarks(landmarks: List[Landmark], depth_map: DepthMap, 
                           camera_params: Dict) -> List[Point3DWithVisibility]:
    """
    批量转换关键点为三维坐标
    
    Args:
        landmarks: 关键点列表
        depth_map: 深度图
        camera_params: 相机参数字典
    
    Returns:
        points_3d: 三维坐标点列表（带可见性）
    """
    points_3d = []
    success_count = 0
    _debug_first_z_logged = [False]
    if not hasattr(batch_convert_landmarks, "_depth_log_count"):
        batch_convert_landmarks._depth_log_count = 0
    batch_convert_landmarks._depth_log_count += 1
    _log_depth_this_batch = batch_convert_landmarks._depth_log_count <= 3 or batch_convert_landmarks._depth_log_count % 60 == 0

    if _log_depth_this_batch:
        # [深度调试] 在关键点处采样深度；若为 nan 则 [深度追溯] 说明可能原因
        for idx in [0, 468, 473]:
            if idx < len(landmarks):
                lm = landmarks[idx]
                pixel = lm.to_point2d()
                raw_z = get_depth_interpolated(depth_map, pixel)
                valid = np.isfinite(raw_z) and raw_z > DEPTH_VALIDATION_THRESHOLD
                logger.info(
                    "[深度调试] 关键点 %d 像素(%.0f,%.0f) 采样深度=%.4f m 有效=%s (depth_map %dx%d unit=%s)",
                    idx, pixel.x, pixel.y, raw_z, valid, depth_map.height, depth_map.width, getattr(depth_map, "unit", "?"),
                )
                if not np.isfinite(raw_z) or (isinstance(raw_z, float) and np.isnan(raw_z)):
                    logger.info(
                        "[深度追溯] 关键点 %d 像素(%.0f,%.0f) 深度=nan: 立体匹配在该像素无有效视差(或深度图与图像未对齐)，可检查标定/光照/遮挡及 image 与 depth 是否同源同尺寸",
                        idx, pixel.x, pixel.y,
                    )
        valid_mask = np.isfinite(depth_map.data) & (depth_map.data > 0)
        if np.any(valid_mask):
            v = depth_map.data[valid_mask]
            logger.info(
                "[深度调试] 深度图统计: min=%.4f max=%.4f mean=%.4f m nan占比=%.1f%%",
                float(np.min(v)), float(np.max(v)), float(np.mean(v)),
                100.0 * (1.0 - np.sum(valid_mask) / depth_map.data.size),
            )

    for i, landmark in enumerate(landmarks):
        try:
            # 提取像素坐标
            pixel = landmark.to_point2d()
            
            # 调试信息：检查坐标值
            if i < 10:  # 打印前10个关键点的调试信息
                logger.debug(f"关键点 {i}: 原始landmark=({landmark.x:.2f}, {landmark.y:.2f}), 像素坐标=({pixel.x:.2f}, {pixel.y:.2f})")
            
            # 转换为3D坐标
            point_3d = pixel_to_3d(pixel, depth_map, camera_params)
            
            # [深度调试] 首次成功转换时打印该点 Z(米)，用于确认进入 coordinate_converter 的深度
            if not _debug_first_z_logged[0]:
                logger.info("[深度调试] coordinate_converter 首次成功 pixel_to_3d 得到 Z(米)=%.4f  (期望约0.4)", point_3d.z)
                _debug_first_z_logged[0] = True
            
            # 创建带可见性的3D点
            point_with_visibility = Point3DWithVisibility(
                x=point_3d.x, y=point_3d.y, z=point_3d.z,
                visibility=landmark.visibility
            )
            
            points_3d.append(point_with_visibility)
            success_count += 1
            
        except Exception as e:
            logger.warning(f"关键点 {i} 转换失败: {e}")
    
    logger.debug(f"批量转换完成：成功 {success_count}/{len(landmarks)} 个关键点")
    return points_3d

def validate_3d_coordinates(points_3d: List[Point3DWithVisibility]) -> bool:
    """
    验证三维坐标的有效性
    
    Args:
        points_3d: 三维坐标点列表
    
    Returns:
        is_valid: 坐标是否有效
    """
    if not isinstance(points_3d, list) or len(points_3d) == 0:
        return False
    
    valid_count = 0
    for point in points_3d:
        if not isinstance(point, Point3DWithVisibility):
            continue
        
        # 检查数值有效性
        if not (np.isfinite(point.x) and np.isfinite(point.y) and 
                np.isfinite(point.z) and np.isfinite(point.visibility)):
            continue
        
        # 检查坐标范围（根据实际应用调整）
        if not (-10.0 <= point.x <= 10.0 and -10.0 <= point.y <= 10.0 and 0.1 <= point.z <= 10.0):
            continue
        
        valid_count += 1
    
    # 计算有效坐标比例
    valid_ratio = valid_count / len(points_3d)
    is_valid = valid_ratio >= COORDINATE_QUALITY_THRESHOLD
    
    logger.debug(f"坐标验证结果：{valid_count}/{len(points_3d)} 有效，比例：{valid_ratio:.2f}")
    
    return is_valid

def calculate_coordinate_quality(points_3d: List[Point3DWithVisibility]) -> float:
    """
    计算坐标质量分数
    
    Args:
        points_3d: 三维坐标点列表
    
    Returns:
        quality_score: 质量分数，范围[0,1]
    """
    if not points_3d:
        return 0.0
    
    valid_count = 0
    total_count = len(points_3d)
    
    for point in points_3d:
        if (isinstance(point, Point3DWithVisibility) and 
            np.isfinite(point.x) and np.isfinite(point.y) and np.isfinite(point.z)):
            valid_count += 1
    
    return valid_count / total_count

def convert_landmarks_to_key_coordinates(landmarks: List[Landmark], depth_map: DepthMap, 
                                       camera_params: Dict) -> Dict[str, Dict[str, List[Point3DWithVisibility]]]:
    """
    将关键点转换为拟合模块所需的格式
    
    Args:
        landmarks: 关键点列表
        depth_map: 深度图
        camera_params: 相机参数字典
    
    Returns:
        key_coordinates: 关键点坐标字典
    """
    from project.core.recognition.landmark_extractor import get_landmark_indices
    from project.data.data_models import FITTING_TYPE, EYE_TYPE
    
    # 先转换为3D坐标
    points_3d = batch_convert_landmarks(landmarks, depth_map, camera_params)
    
    # 获取关键点索引
    indices = get_landmark_indices()
    
    # 创建结果字典
    result = {}
    for eye in EYE_TYPE:
        result[eye] = {}
        for fitting_type in FITTING_TYPE:
            result[eye][fitting_type] = []
    
    # 填充数据
    for eye in EYE_TYPE:
        for fitting_type in FITTING_TYPE:
            landmark_indices = indices[eye][fitting_type]
            for idx in landmark_indices:
                if idx < len(points_3d):
                    result[eye][fitting_type].append(points_3d[idx])
    
    return result

# 测试函数
def test_coordinate_converter():
    """测试坐标转换器功能"""
    print("=== 坐标转换器测试 ===")
    
    # 创建测试数据
    test_camera_params = {
        "intrinsic_params": {
            "fx": 925.0,
            "fy": 925.0,
            "cx": 640.0,
            "cy": 360.0
        },
        "depth_scale": 0.001
    }
    
    # 创建模拟深度图
    depth_map = DepthMap(data=np.ones((720, 1280), dtype=np.float32) * 0.5, unit="meter")
    
    # 创建测试关键点
    test_landmarks = [
        Landmark(x=640, y=360, z=0.1, visibility=1.0),
        Landmark(x=384, y=504, z=0.2, visibility=0.8),
        Landmark(x=1024, y=144, z=0.3, visibility=0.9)
    ]
    
    # 测试像素坐标转三维坐标
    print("\n1. 测试像素坐标转三维坐标")
    pixel = Point2D(x=640, y=360)
    point_3d = pixel_to_3d(pixel, depth_map, test_camera_params)
    print(f"像素坐标 {pixel} -> 三维坐标 {point_3d}")
    
    # 测试批量转换
    print("\n2. 测试批量转换")
    points_3d = batch_convert_landmarks(test_landmarks, depth_map, test_camera_params)
    print(f"批量转换结果：{len(points_3d)} 个有效坐标")
    
    # 测试坐标验证
    print("\n3. 测试坐标验证")
    is_valid = validate_3d_coordinates(points_3d)
    print(f"坐标验证结果：{'通过' if is_valid else '失败'}")
    
    # 测试质量评估
    print("\n4. 测试质量评估")
    quality_score = calculate_coordinate_quality(points_3d)
    print(f"质量分数：{quality_score:.2f}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_coordinate_converter()