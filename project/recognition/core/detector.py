import numpy as np
from typing import List, Dict

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
        pass

def detect_face(self, rgb_image: np.ndarray, depth_map: np.ndarray) -> bool:
    """
    检测人脸和关键点
    
    :param rgb_image: RGB图像，numpy数组格式，形状为(H, W, 3)
    其中RGB图像分3层。最外层是图像层，第二层是行，第三层是具体像素。
    每个像素的值为RGB三个通道的值，范围为0-255的整数。
    具体的图像宽高在shape中给出。
    示例
    rgb_image = np.array([
        [[255, 0, 0], [0, 255, 0], ...],  # 第一行像素，RGB格式
        [[0, 0, 255], [128, 128, 128], ...],  # 第二行像素
        ...
    ], dtype=np.uint8, shape=(720, 1280, 3))
    
    :param depth_map: 深度图，numpy数组格式，形状为(H, W)
    
    各点深度值为相机到该点的距离，单位为米。
    示例
    depth_map = np.array([
        [1000, 1005, 1010, ...],  # 第一行深度值
        [1002, 1008, 1015, ...],  # 第二行深度值
        ...
    ], dtype=np.uint16, shape=(720, 1280))
    注意：深度值单位为相机原始单位，需要乘以depth_scale转换为米。
    
    :return: 检测是否成功
    返回值示例：
    True   # 检测成功，找到人脸
    False  # 检测失败，未找到人脸
    
    处理流程：
    1. 使用MediaPipe检测人脸关键点
    2. 验证关键点质量
    3. 提取瞳孔中心和虹膜边界点
    4. 结合深度图进行坐标转换
    5. 存储检测结果供后续方法调用
    
    传入参数的函数：
    extract_landmarks(rgb_image: np.ndarray) -> List[Tuple[float, float, float]]:
        传入参数：rgb_image (RGB图像)
        返回：468个关键点(x,y,z)列表
    
    validate_landmarks(landmarks: List) -> bool
        传入参数：landmarks (关键点列表)
        返回：关键点是否有效
    
    可能用到的库函数：mediapipe, opencv
    """
    pass

def get_eye_centers(self) -> Dict[str, np.ndarray]:
    """
    获取左右眼球中心三维坐标
    
    :return: 左右眼球中心的三维坐标字典
    返回格式：
    {
        'left': np.array([x_left, y_left, z_left]),   # 左眼球中心坐标
        'right': np.array([x_right, y_right, z_right]) # 右眼球中心坐标
    }
    
    坐标说明：
    - x, y, z: 相机坐标系下的三维坐标（米）
    - x: 水平方向，向右为正
    - y: 垂直方向，向下为正  
    - z: 深度方向，向前为正（相机到物体的距离）
    
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
    get_eye_landmarks(landmarks: List, eye_type: str) -> List
        传入参数：landmarks (关键点列表), eye_type ('left'或'right')
        返回：指定眼睛的关键点列表
    
    pixel_to_3d(pixel_coords: Tuple[int, int], depth_map: np.ndarray, camera_params: dict) -> np.ndarray
        传入参数：pixel_coords (像素坐标), depth_map (深度图), camera_params (相机参数)
        返回：三维坐标数组
    
    可能用到的库函数：numpy
    """
    pass

def get_pupil_centers(self) -> Dict[str, np.ndarray]:
    """
    获取左右瞳孔中心三维坐标
    
    :return: 左右瞳孔中心的三维坐标字典
    返回格式：
    {
        'left': np.array([x_left, y_left, z_left]),   # 左瞳孔中心坐标
        'right': np.array([x_right, y_right, z_right]) # 右瞳孔中心坐标
    }
    
    坐标说明：
    - x, y, z: 相机坐标系下的三维坐标（米）
    - x: 水平方向，向右为正
    - y: 垂直方向，向下为正
    - z: 深度方向，向前为正（相机到瞳孔的距离）
    
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
    get_pupil_landmarks(landmarks: List) -> Dict[str, object]
        传入参数：landmarks (关键点列表)
        返回：{'left': landmark, 'right': landmark}
    
    pixel_to_3d(pixel_coords: Tuple[int, int], depth_map: np.ndarray, camera_params: dict) -> np.ndarray
        传入参数：pixel_coords (像素坐标), depth_map (深度图), camera_params (相机参数)
        返回：三维坐标数组
    
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
    
    坐标说明：
    - x, y, z: 相机坐标系下的三维坐标（米）
    - x: 水平方向，向右为正
    - y: 垂直方向，向下为正
    - z: 深度方向，向前为正（相机到虹膜的距离）
    
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
    get_iris_landmarks(landmarks: List) -> Dict[str, List]
        传入参数：landmarks (关键点列表)
        返回：{'left': [landmark...], 'right': [landmark...]}
    
    pixel_to_3d(pixel_coords: Tuple[int, int], depth_map: np.ndarray, camera_params: dict) -> np.ndarray
        传入参数：pixel_coords (像素坐标), depth_map (深度图), camera_params (相机参数)
        返回：三维坐标数组
    
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
        pass
