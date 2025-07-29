import numpy as np
import cv2
import base64
from typing import List, Dict, Tuple
import io

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
        
        可能用到的库函数：无
        """
        self.camera_params = camera_params
        self._bgr_image = None
        self._depth_map_meters = None
        self._landmarks = None
        self._detection_success = False

    def _decode_base64_image(self, b64_string: str) -> np.ndarray:
        """
        将base64编码的jpg/png图像字符串解码为numpy数组（BGR格式，OpenCV默认）
        """
        img_bytes = base64.b64decode(b64_string)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img

    def detect_face(self, bgr_image_b64: str, depth_map: np.ndarray) -> bool:
        """
        检测人脸和关键点
        
        :param bgr_image_b64: base64编码的jpg/png图像字符串（BGR格式，OpenCV默认）
        :param depth_map: 深度图，numpy数组格式，形状为(H, W)
        
        示例：
        bgr_image_b64 = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDA..."  # base64字符串
        
        :return: 检测是否成功
        """
        bgr_image = self._decode_base64_image(bgr_image_b64)
        self._bgr_image = bgr_image
        self._depth_map_meters = self._convert_depth_to_meters(depth_map)
        from .landmark_extractor import extract_landmarks, validate_landmarks
        landmarks = extract_landmarks(bgr_image_b64)
        if validate_landmarks(landmarks):
            self._landmarks = landmarks
            self._detection_success = True
            return True
        else:
            self._detection_success = False
            return False

    def get_eye_centers(self) -> Dict[str, np.ndarray]:
        """
        输入：一张包含人脸的RGB图片和深度图（已经通过detect_face()处理）
        处理：处理流程见下
        输出：左右眼球中心的三维坐标字典
        
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
        
        调用关系：
        被调用：
        - main.py 中的主循环调用此函数
        
        调用：
        - get_eye_landmarks() (landmark_extractor.py)
        - pixel_to_3d() (coordinate_converter.py)
        - _calculate_center() (内部方法)
        
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
        
        调用关系：
        被调用：
        - main.py 中的主循环调用此函数
        
        调用：
        - get_pupil_landmarks() (landmark_extractor.py)
        - pixel_to_3d() (coordinate_converter.py)
        - _calculate_center() (内部方法)
        
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
        
        调用关系：
        被调用：
        - main.py 中的主循环调用此函数
        
        调用：
        - get_iris_landmarks() (landmark_extractor.py)
        - pixel_to_3d() (coordinate_converter.py)
        
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
        if self._bgr_image is not None:
            height, width = self._bgr_image.shape[:2]
            cx = intrinsic_params.get("cx", width / 2)
            cy = intrinsic_params.get("cy", height / 2)
        else:
            cx = intrinsic_params.get("cx", 640.0)
            cy = intrinsic_params.get("cy", 360.0)
        
        return fx, fy, cx, cy
    
    def _enhance_depth_with_landmarks(self) -> np.ndarray:
        """
        使用MediaPipe预估深度增强RGB-D深度图
        
        处理流程：
        1. 遍历所有landmarks的(x,y,z)坐标
        2. 获取对应位置的RGB-D深度值
        3. 将MediaPipe预估深度z转换为米单位
        4. 根据置信度权重融合两种深度信息
        5. 更新_depth_map_meters中的对应位置
        
        :return: 增强后的深度图（米单位）
        示例
        enhanced_depth = np.ndarray(
            shape=(720, 1280),  # 深度图尺寸 (高度, 宽度)
            dtype=np.float32,   # 深度数据类型 (浮点数，米单位)
            data=[[depth_meters, ...], ...]  # 融合后的深度值数组（米）
        )
        
        调用关系：
        被调用：
        - detect_face() 调用此函数
        
        调用：
        - _calculate_depth_confidence() (内部方法)
        - _calculate_estimated_confidence() (内部方法)
        - _convert_estimated_depth_to_meters() (内部方法)
        
        可能用到的库函数：numpy
        """
        pass

    def _calculate_depth_confidence(self, depth_d: float) -> float:
        """
        计算RGB-D深度置信度
        
        :param depth_d: RGB-D深度值（米单位）
        :return: 置信度值（0-1范围）
        """
        pass

    def _calculate_estimated_confidence(self, z: float) -> float:
        """
        计算MediaPipe预估深度置信度
        
        :param z: MediaPipe预估的相对深度值（-1到1范围）
        :return: 置信度值（0-1范围）
        """
        pass

    def _convert_estimated_depth_to_meters(self, z: float) -> float:
        """
        将MediaPipe相对深度转换为米单位
        
        :param z: MediaPipe预估的相对深度值（-1到1范围）
        :return: 转换后的深度值（米单位）
        """
        pass

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
