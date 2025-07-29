import numpy as np
import cv2
from typing import List, Dict, Tuple

class FaceDetector:
    def __init__(self, camera_params: dict):
        """
        初始化检测器
        
        :param camera_params: 相机内参字典
        示例
        camera_params = {
            # 相机基本信息
            "camera_name": "Intel RealSense D435i",  # 相机型号名称
            "camera_type": "RGB-D",                  # 相机类型（RGB-D表示同时有RGB和深度）
            
            # 相机内参矩阵参数（用于坐标转换）
            "intrinsic_params": {
                "fx": 925.0,  # 焦距x（像素单位），用于X方向坐标转换
                "fy": 925.0,  # 焦距y（像素单位），用于Y方向坐标转换  
                "cx": 640.0,  # 主点x坐标（像素），图像中心X坐标
                "cy": 360.0   # 主点y坐标（像素），图像中心Y坐标
            },
            
            # 图像分辨率信息
            "image_resolution": {
                "width": 1280,   # 图像宽度（像素）
                "height": 720    # 图像高度（像素）
            },
            
            # 深度相关参数
            "depth_scale": 0.001,  # 深度值缩放因子（米/单位）
            "min_depth": 0.1,      # 最小有效深度（米）
            "max_depth": 10.0,     # 最大有效深度（米）
            
            # 畸变系数（可选，用于图像校正）
            "distortion_coeffs": {
                "k1": 0.0,  # 径向畸变系数1
                "k2": 0.0,  # 径向畸变系数2
                "p1": 0.0,  # 切向畸变系数1
                "p2": 0.0,  # 切向畸变系数2
                "k3": 0.0   # 径向畸变系数3
            },
            
            # 标定信息（可选）
            "calibration_date": "2024-01-15",     # 标定日期
            "calibration_method": "OpenCV",        # 标定方法
            "notes": "相机内参通过OpenCV标定获得"  # 备注信息
        }
        
        传入参数的函数：
        load_camera_params(file_path: str) -> dict
            传入参数：file_path (配置文件路径)
            返回：相机参数字典
        
        可能用到的库函数：无
        """
        self.camera_params = camera_params
        self._rgb_image = None
        self._depth_map_meters = None
        self._landmarks = None
        self._detection_success = False

def detect_face(self, rgb_image: np.ndarray, depth_map: np.ndarray) -> bool:
    """
    检测人脸和关键点
    
    :param rgb_image: RGB图像，numpy数组格式，形状为(H, W, 3)
    注意：输入图像可以是BGR或RGB格式，函数内部会统一转换为RGB格式处理
    示例
    rgb_image = np.array([
        [[255, 0, 0], [0, 255, 0], ...],  # 第一行像素，RGB格式
        [[0, 0, 255], [128, 128, 128], ...],  # 第二行像素
        ...
    ], dtype=np.uint8, shape=(720, 1280, 3))
    
    :param depth_map: 深度图，numpy数组格式，形状为(H, W)
    注意：深度值为相机原始单位，函数内部会乘以depth_scale转换为米
    示例
    depth_map = np.array([
        [1000, 1005, 1010, ...],  # 第一行深度值（相机原始单位）
        [1002, 1008, 1015, ...],  # 第二行深度值
        ...
    ], dtype=np.uint16, shape=(720, 1280))
    
    :return: 检测是否成功
    返回值示例：
    True   # 检测成功，找到人脸
    False  # 检测失败，未找到人脸
    
    处理流程：
    1. 统一RGB格式处理（BGR转RGB）
    2. 深度图转换为米单位
    3. 使用MediaPipe检测人脸关键点
    4. 验证关键点质量
    5. 存储处理后的数据供后续方法调用
    
    传入参数的函数：
    extract_landmarks(rgb_image: np.ndarray) -> List[Tuple[float, float, float]]:
        传入参数：rgb_image (RGB图像)
        返回：468个关键点(x,y,z)列表
    
    validate_landmarks(landmarks: List[Tuple[float, float, float]]) -> bool
        传入参数：landmarks (关键点列表,(x,y,z))
        返回：关键点是否有效
    
    可能用到的库函数：mediapipe, opencv
    """
    # 1. 统一RGB格式处理
    self._rgb_image = self._convert_to_rgb(rgb_image)
    
    # 2. 深度图转换为米单位
    self._depth_map_meters = self._convert_depth_to_meters(depth_map)
    
    # 3. 检测人脸关键点
    from .landmark_extractor import extract_landmarks, validate_landmarks
    landmarks = extract_landmarks(self._rgb_image)
    
    # 4. 验证关键点质量
    if validate_landmarks(landmarks):
        self._landmarks = landmarks
        self._detection_success = True
        return True
    else:
        self._detection_success = False
        return False

def get_eye_centers(self) -> Dict[str, np.ndarray]:
    """
    获取左右眼球中心三维坐标
    
    :return: 左右眼球中心的三维坐标字典
    返回格式：
    {
        'left': np.array([x_left, y_left, z_left]),   # 左眼球中心坐标
        'right': np.array([x_right, y_right, z_right]) # 右眼球中心坐标
    }
    
    坐标说明（OpenCV标准坐标系）：
    - x, y, z: 相机坐标系下的三维坐标（米）
    - x: 水平方向，向右为正（图像宽度方向）
    - y: 垂直方向，向下为正（图像高度方向）
    - z: 深度方向，向前为正（相机光轴方向，指向物体）
    
    返回值示例：
    {
        'left': np.array([0.1, 0.05, 0.8]),   # 左眼球中心：(0.1m, 0.05m, 0.8m)
        'right': np.array([0.15, 0.05, 0.8])  # 右眼球中心：(0.15m, 0.05m, 0.8m)
    }
    
    处理流程：
    1. 从已检测的关键点中提取眼球区域关键点
    2. 计算眼球区域的几何中心
    3. 结合深度信息转换为三维坐标
    4. 返回左右眼球的中心坐标
    
    前置条件：
    - 必须先调用detect_face()方法进行人脸检测
    - 检测结果必须成功且置信度足够高
    
    传入参数的函数：
    get_eye_landmarks(landmarks: List[Tuple[float, float, float]], eye_type: str) -> List[Tuple[float, float, float]]
        传入参数：landmarks (关键点列表), eye_type ('left'或'right')
        返回：指定眼睛的关键点列表
    
    pixel_to_3d(pixel_coords: Tuple[int, int], depth_map_meters: np.ndarray, camera_params: dict) -> np.ndarray
        传入参数：pixel_coords (像素坐标), depth_map_meters (米单位深度图), camera_params (相机参数)
        返回：三维坐标数组（米单位）
    
    可能用到的库函数：numpy
    """
    if not self._detection_success or self._landmarks is None:
        return {'left': None, 'right': None}
    
    # 从已检测的关键点中提取眼球区域关键点
    from .landmark_extractor import get_eye_landmarks
    from .coordinate_converter import pixel_to_3d
    
    # 获取左右眼关键点
    left_eye_landmarks = get_eye_landmarks(self._landmarks, 'left')
    right_eye_landmarks = get_eye_landmarks(self._landmarks, 'right')
    
    # 计算眼球中心
    if left_eye_landmarks:
        left_center = self._calculate_center(left_eye_landmarks)
        left_3d = pixel_to_3d((int(left_center[0]), int(left_center[1])), self._depth_map_meters, self.camera_params)
    else:
        left_3d = None
    
    if right_eye_landmarks:
        right_center = self._calculate_center(right_eye_landmarks)
        right_3d = pixel_to_3d((int(right_center[0]), int(right_center[1])), self._depth_map_meters, self.camera_params)
    else:
        right_3d = None
    
    return {'left': left_3d, 'right': right_3d}

def get_pupil_centers(self) -> Dict[str, np.ndarray]:
    """
    获取左右瞳孔中心三维坐标
    
    :return: 左右瞳孔中心的三维坐标字典
    返回格式：
    {
        'left': np.array([x_left, y_left, z_left]),   # 左瞳孔中心坐标
        'right': np.array([x_right, y_right, z_right]) # 右瞳孔中心坐标
    }
    
    坐标说明（OpenCV标准坐标系）：
    - x, y, z: 相机坐标系下的三维坐标（米）
    - x: 水平方向，向右为正（图像宽度方向）
    - y: 垂直方向，向下为正（图像高度方向）
    - z: 深度方向，向前为正（相机光轴方向，指向瞳孔）
    
    返回值示例：
    {
        'left': np.array([0.12, 0.06, 0.82]),   # 左瞳孔中心：(0.12m, 0.06m, 0.82m)
        'right': np.array([0.18, 0.06, 0.82])   # 右瞳孔中心：(0.18m, 0.06m, 0.82m)
    }
    
    处理流程：
    1. 从已检测的关键点中提取瞳孔中心关键点（索引468和473）
    2. 获取瞳孔中心的像素坐标
    3. 结合深度图获取该点的深度值
    4. 使用相机内参转换为三维坐标
    5. 返回左右瞳孔的三维坐标
    
    前置条件：
    - 必须先调用detect_face()方法进行人脸检测
    - 检测结果必须成功且置信度足够高
    - 瞳孔中心关键点必须被正确检测到
    
    传入参数的函数：
    get_pupil_landmarks(landmarks: List[Tuple[float, float, float]]) -> Dict[str, Tuple[float, float, float]]
        传入参数：landmarks (关键点列表)
        返回：{'left': (x,y,z), 'right': (x,y,z)}
    
    pixel_to_3d(pixel_coords: Tuple[int, int], depth_map_meters: np.ndarray, camera_params: dict) -> np.ndarray
        传入参数：pixel_coords (像素坐标), depth_map_meters (米单位深度图), camera_params (相机参数)
        返回：三维坐标数组（米单位）
    
    可能用到的库函数：numpy
    """
    pass

def get_iris_boundaries(self) -> Dict[str, List[np.ndarray]]:
    """
    获取左右虹膜边界点三维坐标
    
    :return: 左右虹膜边界点的三维坐标字典
    返回格式：
    {
        'left': [
            np.array([x_0, y_0, z_0]),   # 左虹膜边界点0
            np.array([x_1, y_1, z_1]),   # 左虹膜边界点1
            np.array([x_2, y_2, z_2]),   # 左虹膜边界点2
            np.array([x_3, y_3, z_3])    # 左虹膜边界点3
        ],
        'right': [
            np.array([x_0, y_0, z_0]),   # 右虹膜边界点0
            np.array([x_1, y_1, z_1]),   # 右虹膜边界点1
            np.array([x_2, y_2, z_2]),   # 右虹膜边界点2
            np.array([x_3, y_3, z_3])    # 右虹膜边界点3
        ]
    }
    
    坐标说明（OpenCV标准坐标系）：
    - x, y, z: 相机坐标系下的三维坐标（米）
    - x: 水平方向，向右为正（图像宽度方向）
    - y: 垂直方向，向下为正（图像高度方向）
    - z: 深度方向，向前为正（相机光轴方向，指向虹膜）
    
    返回值示例：
    {
        'left': [
            np.array([0.11, 0.05, 0.81]),  # 左虹膜边界点0
            np.array([0.13, 0.05, 0.81]),  # 左虹膜边界点1
            np.array([0.11, 0.07, 0.81]),  # 左虹膜边界点2
            np.array([0.13, 0.07, 0.81])   # 左虹膜边界点3
        ],
        'right': [
            np.array([0.17, 0.05, 0.81]),  # 右虹膜边界点0
            np.array([0.19, 0.05, 0.81]),  # 右虹膜边界点1
            np.array([0.17, 0.07, 0.81]),  # 右虹膜边界点2
            np.array([0.19, 0.07, 0.81])   # 右虹膜边界点3
        ]
    }
    
    处理流程：
    1. 从已检测的关键点中提取虹膜边界点（索引469-472为左眼，474-477为右眼）
    2. 获取每个边界点的像素坐标
    3. 结合深度图获取各点的深度值
    4. 使用相机内参转换为三维坐标
    5. 返回左右虹膜的边界点三维坐标列表
    
    前置条件：
    - 必须先调用detect_face()方法进行人脸检测
    - 检测结果必须成功且置信度足够高
    - 虹膜边界点必须被正确检测到
    
    传入参数的函数：
    get_iris_landmarks(landmarks: List[Tuple[float, float, float]]) -> Dict[str, List[Tuple[float, float, float]]]
        传入参数：landmarks (关键点列表)
        返回：{'left': [(x,y,z)...], 'right': [(x,y,z)...]}
    
    pixel_to_3d(pixel_coords: Tuple[int, int], depth_map_meters: np.ndarray, camera_params: dict) -> np.ndarray
        传入参数：pixel_coords (像素坐标), depth_map_meters (米单位深度图), camera_params (相机参数)
        返回：三维坐标数组（米单位）
    
    可能用到的库函数：numpy
    """
    pass

    def get_detection_confidence(self) -> float:
        """
        获取检测置信度
        :return: 置信度分数
        可能用到的库函数：无
        """
        pass

    def get_detection_status(self) -> str:
        """
        获取检测状态信息
        :return: 状态字符串
        可能用到的库函数：无
        """
        if self._detection_success:
            return "检测成功"
        else:
            return "检测失败"
    
    def _convert_to_rgb(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        统一转换为RGB格式
        
        :param rgb_image: 输入图像（BGR或RGB格式）
        :return: RGB格式图像
        """
        if len(rgb_image.shape) == 3 and rgb_image.shape[2] == 3:
            # 检查是否为BGR格式（OpenCV默认格式）
            if rgb_image.dtype == np.uint8:
                return cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            else:
                return rgb_image
        else:
            raise ValueError("输入图像必须是3通道RGB图像")
    
    def _convert_depth_to_meters(self, depth_map: np.ndarray) -> np.ndarray:
        """
        将深度图从相机原始单位转换为米
        
        :param depth_map: 原始深度图
        :return: 米单位的深度图
        """
        depth_scale = self.camera_params.get("depth_scale", 1.0)
        return depth_map.astype(np.float32) * depth_scale
    
    def _get_camera_intrinsics(self) -> Tuple[float, float, float, float]:
        """
        获取相机内参，支持动态图像尺寸
        
        :return: (fx, fy, cx, cy) 相机内参
        """
        intrinsic_params = self.camera_params.get("intrinsic_params", {})
        
        # 如果相机参数中没有内参，使用默认值
        fx = intrinsic_params.get("fx", 1.0)
        fy = intrinsic_params.get("fy", 1.0)
        
        # 主点坐标：如果未指定，使用图像中心
        if self._rgb_image is not None:
            height, width = self._rgb_image.shape[:2]
            cx = intrinsic_params.get("cx", width / 2)
            cy = intrinsic_params.get("cy", height / 2)
        else:
            cx = intrinsic_params.get("cx", 640.0)
            cy = intrinsic_params.get("cy", 360.0)
        
        return fx, fy, cx, cy
    
    def _calculate_center(self, points: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
        """
        计算多个点的中心点
        
        :param points: 点列表
        :return: 中心点坐标
        """
        if not points:
            return (0.0, 0.0, 0.0)
        
        # 分别计算x, y, z的平均值
        x_sum = sum(point[0] for point in points)
        y_sum = sum(point[1] for point in points)
        z_sum = sum(point[2] for point in points)
        
        center_x = x_sum / len(points)
        center_y = y_sum / len(points)
        center_z = z_sum / len(points)
        
        return (center_x, center_y, center_z)
