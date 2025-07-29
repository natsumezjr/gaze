import numpy as np
from typing import List, Tuple

def pixel_to_3d(pixel_coords: Tuple[int, int], depth_map_meters: np.ndarray, camera_params: dict) -> np.ndarray:
    """
    输入：像素坐标(x,y)，以米为单位深度映射图，相机内参
    输出：像素的相机坐标系下的3D坐标()
    
    :param pixel_coords: 像素坐标
    示例
    pixel_coords = (640, 360)  # (x_pixel, y_pixel) 像素坐标
    
    :param depth_map_meters: 深度图（米单位）
    示例
    depth_map_meters = np.ndarray(
        shape=(720, 1280),  # 深度图尺寸 (高度, 宽度)
        dtype=np.float32,   # 深度数据类型 (浮点数，米单位)
        data=[[depth_meters, ...], ...]  # 深度值数组（米）
    )
    
    :param camera_params: 相机内参
    示例
    camera_params = {
        "intrinsic_params": {
            "fx": 925.0,  # 焦距x
            "fy": 925.0,  # 焦距y
            "cx": 640.0,  # 主点x坐标
            "cy": 360.0   # 主点y坐标
        },
        "depth_scale": 0.001  # 深度值缩放因子
    }
    
    返回值示例：
    np.array([0.12, 0.06, 0.82])   # (0.12m, 0.06m, 0.82m)
    
    调用关系：
    被调用：
    - detector.py 中的 get_eye_centers() 调用此函数
    - detector.py 中的 get_pupil_centers() 调用此函数
    - batch_convert_landmarks() 调用此函数
    
    调用：
    - 无直接调用其他函数
    
    可能用到的库函数：numpy
    """
    # 检查输入参数
    if not isinstance(pixel_coords, tuple) or len(pixel_coords) != 2:
        raise ValueError("pixel_coords 必须是包含2个元素的元组")
    
    if not isinstance(depth_map_meters, np.ndarray) or depth_map_meters.ndim != 2:
        raise ValueError("depth_map_meters 必须是2维numpy数组")
    
    if not isinstance(camera_params, dict):
        raise ValueError("camera_params 必须是字典")
    
    # 提取像素坐标
    x_pixel, y_pixel = pixel_coords
    height, width = depth_map_meters.shape
    
    # 检查像素坐标是否在图像范围内
    if not (0 <= x_pixel < width and 0 <= y_pixel < height):
        raise ValueError("像素坐标超出图像范围")
    
    # 获取深度值（已经是米单位）
    depth_meters = depth_map_meters[y_pixel, x_pixel]
    
    # 检查深度值是否有效
    if depth_meters <= 0 or np.isnan(depth_meters):
        raise ValueError("无效的深度值")
    
    # 提取相机内参
    intrinsic_params = camera_params.get("intrinsic_params", {})
    fx = intrinsic_params.get("fx", 1.0)
    fy = intrinsic_params.get("fy", 1.0)
    cx = intrinsic_params.get("cx", width / 2)
    cy = intrinsic_params.get("cy", height / 2)
    
    # 使用相机内参计算三维坐标（OpenCV坐标系）
    # 公式：X = (x - cx) * depth / fx, Y = (y - cy) * depth / fy, Z = depth
    x_3d = (x_pixel - cx) * depth_meters / fx
    y_3d = (y_pixel - cy) * depth_meters / fy
    z_3d = depth_meters
    
    return np.array([x_3d, y_3d, z_3d])

def batch_convert_landmarks(landmarks: List[Tuple[float, float, float]], depth_map_meters: np.ndarray, camera_params: dict) -> List[np.ndarray]:
    """
    批量转换关键点为三维坐标
    
    :param landmarks: 关键点列表
    示例
    landmarks = [
        (x1, y1, z1),  # 关键点1坐标 (像素x, 像素y, 相对深度z)
        (x2, y2, z2),  # 关键点2坐标
        ...,
        (xn, yn, zn)   # 关键点n坐标
    ]
    
    :param depth_map_meters: 深度图（米单位）
    示例
    depth_map_meters = np.ndarray(
        shape=(720, 1280),  # 深度图尺寸 (高度, 宽度)
        dtype=np.float32,   # 深度数据类型（浮点数，米单位）
        data=[[depth_meters, ...], ...]  # 深度值数组（米）
    )
    
    :param camera_params: 相机内参
    示例
    camera_params = {
        "intrinsic_params": {
            "fx": 925.0,  # 焦距x
            "fy": 925.0,  # 焦距y
            "cx": 640.0,  # 主点x坐标
            "cy": 360.0   # 主点y坐标
        },
        "depth_scale": 0.001  # 深度值缩放因子
    }
    
    调用关系：
    被调用：
    - detector.py 中的 get_eye_centers() 调用此函数
    - detector.py 中的 get_pupil_centers() 调用此函数
    
    调用：
    - pixel_to_3d() (内部函数)
    
    可能用到的库函数：numpy
    """
    # 检查输入参数
    if not isinstance(landmarks, list) or len(landmarks) == 0:
        raise ValueError("landmarks 必须是非空列表")
    
    if not isinstance(depth_map_meters, np.ndarray) or depth_map_meters.ndim != 2:
        raise ValueError("depth_map_meters 必须是2维numpy数组")
    
    if not isinstance(camera_params, dict):
        raise ValueError("camera_params 必须是字典")
    
    # 批量转换关键点
    coords_3d = []
    height, width = depth_map_meters.shape
    
    for landmark in landmarks:
        if not isinstance(landmark, tuple) or len(landmark) != 3:
            continue  # 跳过无效的关键点
        
        x_pixel, y_pixel, z_relative = landmark
        
        # 检查像素坐标是否在图像范围内
        if not (0 <= x_pixel < width and 0 <= y_pixel < height):
            continue  # 跳过超出范围的关键点
        
        try:
            # 使用pixel_to_3d函数转换单个关键点
            coord_3d = pixel_to_3d((int(x_pixel), int(y_pixel)), depth_map_meters, camera_params)
            coords_3d.append(coord_3d)
        except (ValueError, IndexError):
            # 如果转换失败，跳过该关键点
            continue
    
    return coords_3d

def validate_3d_coordinates(coords_3d: List[np.ndarray]) -> bool:
    """
    验证三维坐标有效性
    
    :param coords_3d: 三维坐标列表
    示例
    coords_3d = [
        np.array([x1, y1, z1]),  # 关键点1的三维坐标 (米)
        np.array([x2, y2, z2]),  # 关键点2的三维坐标
        ...,
        np.array([xn, yn, zn])   # 关键点n的三维坐标
    ]
    
    调用关系：
    被调用：
    - 无直接被调用
    
    调用：
    - pixel_to_3d() (内部函数)
    
    可能用到的库函数：numpy
    """
    # 检查输入参数
    if not isinstance(coords_3d, list):
        return False
    
    if len(coords_3d) == 0:
        return False
    
    # 验证每个三维坐标
    for coord in coords_3d:
        if not isinstance(coord, np.ndarray):
            return False
        
        if coord.shape != (3,):
            return False
        
        # 检查坐标值是否有效
        if not np.all(np.isfinite(coord)):
            return False
        
        # 检查坐标范围是否合理（假设在合理的三维空间范围内）
        x, y, z = coord
        if not (-100.0 <= x <= 100.0 and -100.0 <= y <= 100.0 and 0.0 <= z <= 100.0):
            return False
    
    return True
