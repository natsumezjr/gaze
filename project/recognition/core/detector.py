import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional

class FaceDetector:
    """
    人脸检测器核心类
    
    功能：
    - 从RGB-D图像中检测人脸和关键点
    - 提取眼球中心、瞳孔中心和虹膜边界点的三维坐标
    - 提供检测置信度和状态信息
    
    使用方法：
    1. 初始化检测器并传入相机参数
    2. 调用detect_face()方法检测人脸
    3. 调用get_eye_centers()等方法获取三维坐标
    """
    
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super(FaceDetector, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, camera_params: dict):
        """
        初始化检测器
        
        Args:
            camera_params: 相机内参字典，包含以下必需字段：
                - intrinsic_params: 相机内参 {fx, fy, cx, cy}
                - depth_scale: 深度值缩放因子（米/单位）
                - min_depth: 最小有效深度（米）
                - max_depth: 最大有效深度（米）
        
        示例:
            camera_params = {
                "camera_name": "Intel RealSense D435i",
                "camera_type": "RGB-D",
                "intrinsic_params": {
                    "fx": 925.0, "fy": 925.0,
                    "cx": 640.0, "cy": 360.0
                },
                "image_resolution": {"width": 1280, "height": 720},
                "depth_scale": 0.001,
                "min_depth": 0.1,
                "max_depth": 10.0,
                "distortion_coeffs": {
                    "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0
                }
            }
        """
        # 验证相机参数
        self._validate_camera_params(camera_params)
        
        self.camera_params = camera_params
        self._bgr_image = None
        self._depth_map_meters = None
        self._landmarks = None
        self._detection_success = False
        self._detection_confidence = 0.0
        self._error_message = ""

    def detect_face(self, bgr_image: np.ndarray, depth_map: np.ndarray) -> bool:
        """
        检测人脸和关键点
        
        :param bgr_image: BGR格式的numpy数组，形状为(H, W, 3)
        :param depth_map: 深度图，numpy数组格式，形状为(H, W)
        
        示例：
        bgr_image = np.ndarray(shape=(720, 1280, 3), dtype=np.uint8)  # BGR图像
        
        :return: 检测是否成功
        """
        try:
            # 验证输入参数
            self._validate_input_images(bgr_image, depth_map)
            
            # 保存输入数据
            self._bgr_image = bgr_image
            self._depth_map_meters = self._convert_depth_to_meters(depth_map)
            
            # 导入landmark_extractor模块
            from project.recognition.core.landmark_extractor import extract_landmarks, validate_landmarks
            
            # 提取关键点
            landmarks = extract_landmarks(bgr_image)
            
            if landmarks and validate_landmarks(landmarks):
                # 将List[List[float]]转换为List[Tuple[float, float, float]]格式以便内部使用
                self._landmarks = [(point[0], point[1], point[2]) for point in landmarks]
                self._detection_success = True
                self._detection_confidence = self._calculate_detection_confidence()
                self._error_message = ""
                return True
            else:
                self._detection_success = False
                self._landmarks = None
                self._detection_confidence = 0.0
                self._error_message = "关键点提取失败或质量不佳"
                return False
                
        except Exception as e:
            self._detection_success = False
            self._landmarks = None
            self._detection_confidence = 0.0
            self._error_message = f"检测过程中发生错误: {str(e)}"
            return False

    def get_eyes_contours(self) -> Dict[str, List[np.ndarray]]:
        """
        获取左右眼轮廓点三维坐标
        
        :return: 左右眼轮廓点的三维坐标字典
        返回格式：
        {
            'left': [
                np.array([x_0, y_0, z_0]),   # 左眼轮廓点0
                np.array([x_1, y_1, z_1]),   # 左眼轮廓点1
                ...,
                np.array([x_n, y_n, z_n])    # 左眼轮廓点n
            ],
            'right': [
                np.array([x_0, y_0, z_0]),   # 右眼轮廓点0
                np.array([x_1, y_1, z_1]),   # 右眼轮廓点1
                ...,
                np.array([x_n, y_n, z_n])    # 右眼轮廓点n
            ]
        }
        
        坐标说明（OpenCV标准坐标系）：
        - x, y, z: 相机坐标系下的三维坐标（米）
        - x: 水平方向，向右为正（图像宽度方向）
        - y: 垂直方向，向下为正（图像高度方向）
        - z: 深度方向，向前为正（相机光轴方向，指向眼睛）
        
        返回值示例：
        {
            'left': [
                np.array([0.11, 0.05, 0.81]),  # 左眼轮廓点0
                np.array([0.13, 0.05, 0.81]),  # 左眼轮廓点1
                np.array([0.11, 0.07, 0.81]),  # 左眼轮廓点2
                np.array([0.13, 0.07, 0.81])   # 左眼轮廓点3
            ],
            'right': [
                np.array([0.17, 0.05, 0.81]),  # 右眼轮廓点0
                np.array([0.19, 0.05, 0.81]),  # 右眼轮廓点1
                np.array([0.17, 0.07, 0.81]),  # 右眼轮廓点2
                np.array([0.19, 0.07, 0.81])   # 右眼轮廓点3
            ]
        }
        
        处理流程：
        1. 从已检测的关键点中提取眼轮廓点
        2. 获取每个轮廓点的像素坐标
        3. 结合深度图获取各点的深度值
        4. 使用相机内参转换为三维坐标
        5. 返回左右眼的轮廓点三维坐标列表
        
        前置条件：
        - 必须先调用detect_face()方法进行人脸检测
        - 检测结果必须成功且置信度足够高
        - 眼轮廓点必须被正确检测到
        
        调用关系：
        被调用：
        - main.py 中的主循环调用此函数
        
        调用：
        - get_eye_contours_landmarks() (landmark_extractor.py)
        - pixel_to_3d() (coordinate_converter.py)
        
        可能用到的库函数：numpy
        """
        # 检查前置条件
        if not self._detection_success or self._landmarks is None:
            return {'left': [], 'right': []}
        
        try:
            # 导入所需模块
            from project.recognition.core.landmark_extractor import get_eye_contours_landmarks
            from project.recognition.core.coordinate_converter import batch_convert_landmarks, pixel_to_3d
            
            # 获取左右眼轮廓关键点
            eye_contours_landmarks = get_eye_contours_landmarks(self._landmarks)
            
            # 检查是否成功获取眼轮廓关键点
            if not eye_contours_landmarks.get('left') or not eye_contours_landmarks.get('right'):
                return {'left': [], 'right': []}
            
            # 尝试使用批量转换
            try:
                left_eye_3d = batch_convert_landmarks(
                    eye_contours_landmarks['left'],
                    self._depth_map_meters,
                    self.camera_params
                )
                
                right_eye_3d = batch_convert_landmarks(
                    eye_contours_landmarks['right'],
                    self._depth_map_meters,
                    self.camera_params
                )
                
                # 如果批量转换成功且结果不为空，直接返回
                if left_eye_3d and right_eye_3d:
                    return {
                        'left': left_eye_3d,
                        'right': right_eye_3d
                    }
            except Exception:
                # 批量转换失败，使用单点转换
                pass
            
            # 单点转换方法
            left_eye_3d = []
            right_eye_3d = []
            
            # 转换左眼轮廓点
            for point in eye_contours_landmarks['left']:
                try:
                    x, y, _ = point
                    point_3d = pixel_to_3d(
                        (int(x), int(y)),
                        self._depth_map_meters,
                        self.camera_params
                    )
                    left_eye_3d.append(point_3d)
                except Exception:
                    # 如果转换失败，添加零向量
                    left_eye_3d.append(np.array([0.0, 0.0, 0.0]))
            
            # 转换右眼轮廓点
            for point in eye_contours_landmarks['right']:
                try:
                    x, y, _ = point
                    point_3d = pixel_to_3d(
                        (int(x), int(y)),
                        self._depth_map_meters,
                        self.camera_params
                    )
                    right_eye_3d.append(point_3d)
                except Exception:
                    # 如果转换失败，添加零向量
                    right_eye_3d.append(np.array([0.0, 0.0, 0.0]))
            
            return {
                'left': left_eye_3d,
                'right': right_eye_3d
            }
            
        except Exception as e:
            # 如果转换过程中出现异常，返回空列表
            print(f"眼轮廓点转换异常: {e}")
            return {'left': [], 'right': []}
    
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
        - get_pupil_centers_landmarks() (landmark_extractor.py)
        - pixel_to_3d() (coordinate_converter.py)
        
        可能用到的库函数：numpy
        """
        # 检查前置条件
        if not self._detection_success or self._landmarks is None:
            return {
                'left': np.array([0.0, 0.0, 0.0]), 
                'right': np.array([0.0, 0.0, 0.0])
            }
        
        try:
            # 导入所需模块
            from project.recognition.core.landmark_extractor import get_pupil_centers_landmarks
            from project.recognition.core.coordinate_converter import pixel_to_3d
            
            # 获取左右瞳孔中心关键点（像素坐标）
            pupil_landmarks = get_pupil_centers_landmarks(self._landmarks)
            
            # 检查是否成功获取瞳孔关键点
            if not pupil_landmarks.get('left') or not pupil_landmarks.get('right'):
                return {
                    'left': np.array([0.0, 0.0, 0.0]), 
                    'right': np.array([0.0, 0.0, 0.0])
                }
            
            # 转换为三维坐标
            left_x, left_y, _ = pupil_landmarks['left']
            right_x, right_y, _ = pupil_landmarks['right']
            
            left_pupil_3d = pixel_to_3d(
                (int(left_x), int(left_y)),
                self._depth_map_meters,
                self.camera_params
            )
            
            right_pupil_3d = pixel_to_3d(
                (int(right_x), int(right_y)),
                self._depth_map_meters,
                self.camera_params
            )
            
            return {
                'left': left_pupil_3d,
                'right': right_pupil_3d
            }
            
        except Exception as e:
            # 如果转换失败，返回零向量
            print(f"瞳孔中心坐标转换失败: {e}")
            return {
                'left': np.array([0.0, 0.0, 0.0]), 
                'right': np.array([0.0, 0.0, 0.0])
            }

    def get_iris_boundaries(self) -> Dict[str, List[np.ndarray]]:
        """
        获取左右虹膜边界点三维坐标
        
        :return: 左右虹膜边界点的三维坐标字典
        返回格式：
        {
            'left_iris': [
                np.array([x_0, y_0, z_0]),   # 左虹膜边界点0
                np.array([x_1, y_1, z_1]),   # 左虹膜边界点1
                np.array([x_2, y_2, z_2]),   # 左虹膜边界点2
                np.array([x_3, y_3, z_3])    # 左虹膜边界点3
            ],
            'right_iris': [
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
            'left_iris': [
                np.array([0.11, 0.05, 0.81]),  # 左虹膜边界点0
                np.array([0.13, 0.05, 0.81]),  # 左虹膜边界点1
                np.array([0.11, 0.07, 0.81]),  # 左虹膜边界点2
                np.array([0.13, 0.07, 0.81])   # 左虹膜边界点3
            ],
            'right_iris': [
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
        - get_iris_boundaries_landmarks() (landmark_extractor.py)
        - pixel_to_3d() (coordinate_converter.py)
        
        可能用到的库函数：numpy
        """
        # 检查前置条件
        if not self._detection_success or self._landmarks is None:
            return {'left': [], 'right': []}
        
        try:
            # 导入所需模块
            from project.recognition.core.landmark_extractor import get_iris_boundaries_landmarks
            from project.recognition.core.coordinate_converter import batch_convert_landmarks, pixel_to_3d
            
            # 获取左右虹膜边界关键点
            iris_landmarks = get_iris_boundaries_landmarks(self._landmarks)
            
            # 检查是否成功获取虹膜关键点
            if not iris_landmarks.get('left') or not iris_landmarks.get('right'):
                return {'left': [], 'right': []}
            
            # 尝试使用批量转换
            try:
                left_iris_3d = batch_convert_landmarks(
                    iris_landmarks['left'],
                    self._depth_map_meters,
                    self.camera_params
                )
                
                right_iris_3d = batch_convert_landmarks(
                    iris_landmarks['right'],
                    self._depth_map_meters,
                    self.camera_params
                )
                
                # 如果批量转换成功且结果不为空，直接返回
                if left_iris_3d and right_iris_3d:
                    return {
                        'left': left_iris_3d,
                        'right': right_iris_3d
                    }
            except Exception:
                # 批量转换失败，使用单点转换
                pass
            
            # 单点转换方法
            left_iris_3d = []
            right_iris_3d = []
            
            # 转换左虹膜边界点
            for point in iris_landmarks['left']:
                try:
                    x, y, _ = point
                    point_3d = pixel_to_3d(
                        (int(x), int(y)),
                        self._depth_map_meters,
                        self.camera_params
                    )
                    left_iris_3d.append(point_3d)
                except Exception:
                    # 如果转换失败，添加零向量
                    left_iris_3d.append(np.array([0.0, 0.0, 0.0]))
            
            # 转换右虹膜边界点
            for point in iris_landmarks['right']:
                try:
                    x, y, _ = point
                    point_3d = pixel_to_3d(
                        (int(x), int(y)),
                        self._depth_map_meters,
                        self.camera_params
                    )
                    right_iris_3d.append(point_3d)
                except Exception:
                    # 如果转换失败，添加零向量
                    right_iris_3d.append(np.array([0.0, 0.0, 0.0]))
            
            return {
                'left': left_iris_3d,
                'right': right_iris_3d
            }
            
        except Exception as e:
            # 如果转换过程中出现异常，返回空列表
            print(f"虹膜边界点转换异常: {e}")
            return {'left': [], 'right': []}

    def get_detection_confidence(self) -> float:
        """
        获取检测置信度
        :return: 置信度分数
        可能用到的库函数：无
        """
        return self._detection_confidence

    def get_detection_status(self) -> str:
        """
        获取检测状态信息
        :return: 状态字符串
        可能用到的库函数：无
        """
        if self._detection_success:
            return f"检测成功 (置信度: {self._detection_confidence:.2f})"
        else:
            return f"检测失败: {self._error_message}"
    
    # ==================== 私有方法 ====================
    
    def _validate_camera_params(self, camera_params: dict) -> None:
        """
        验证相机参数的有效性
        
        Args:
            camera_params: 相机参数字典
            
        Raises:
            ValueError: 相机参数不完整或无效
        """
        required_keys = ['intrinsic_params', 'depth_scale']
        for key in required_keys:
            if key not in camera_params:
                raise ValueError(f"相机参数缺少必需字段: {key}")
        
        intrinsic_params = camera_params['intrinsic_params']
        required_intrinsic_keys = ['fx', 'fy', 'cx', 'cy']
        for key in required_intrinsic_keys:
            if key not in intrinsic_params:
                raise ValueError(f"相机内参缺少必需字段: {key}")
            if not isinstance(intrinsic_params[key], (int, float)) or intrinsic_params[key] <= 0:
                raise ValueError(f"相机内参 {key} 必须为正数")
        
        if not isinstance(camera_params['depth_scale'], (int, float)) or camera_params['depth_scale'] <= 0:
            raise ValueError("depth_scale 必须为正数")
    
    def _validate_input_images(self, bgr_image: np.ndarray, depth_map: np.ndarray) -> None:
        """
        验证输入图像的有效性
        
        Args:
            bgr_image: BGR图像
            depth_map: 深度图
            
        Raises:
            ValueError: 图像格式不正确
        """
        # 验证BGR图像
        if not isinstance(bgr_image, np.ndarray):
            raise ValueError("bgr_image 必须是 numpy.ndarray 类型")
        
        if len(bgr_image.shape) != 3 or bgr_image.shape[2] != 3:
            raise ValueError(f"bgr_image 形状必须为 (H, W, 3)，当前为 {bgr_image.shape}")
        
        if bgr_image.dtype != np.uint8:
            raise ValueError(f"bgr_image 数据类型必须为 uint8，当前为 {bgr_image.dtype}")
        
        # 验证深度图
        if not isinstance(depth_map, np.ndarray):
            raise ValueError("depth_map 必须是 numpy.ndarray 类型")
        
        if len(depth_map.shape) != 2:
            raise ValueError(f"depth_map 形状必须为 (H, W)，当前为 {depth_map.shape}")
        
        # 验证图像尺寸匹配
        if bgr_image.shape[:2] != depth_map.shape:
            raise ValueError(f"BGR图像和深度图尺寸不匹配: {bgr_image.shape[:2]} vs {depth_map.shape}")
    
    def _convert_depth_to_meters(self, depth_map: np.ndarray) -> np.ndarray:
        """
        将深度图从相机原始单位转换为米
        
        Args:
            depth_map: 原始深度图
            
        Returns:
            np.ndarray: 米单位的深度图
        """
        depth_scale = self.camera_params.get("depth_scale", 1.0)
        return depth_map.astype(np.float32) * depth_scale
    
    def _get_camera_intrinsics(self) -> Tuple[float, float, float, float]:
        """
        获取相机内参，支持动态图像尺寸
        
        Returns:
            Tuple[float, float, float, float]: (fx, fy, cx, cy) 相机内参
        """
        intrinsic_params = self.camera_params.get("intrinsic_params", {})
        
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
    
    def _calculate_detection_confidence(self) -> float:
        """
        计算检测置信度
        
        Returns:
            float: 置信度分数，范围0.0-1.0
        """
        if not self._landmarks:
            return 0.0
        
        try:
            # 转换为numpy数组进行计算
            landmarks_array = np.array(self._landmarks)
            
            # 计算关键点的分布标准差，标准差越大，说明关键点分布越广，质量越好
            x_std = np.std(landmarks_array[:, 0])
            y_std = np.std(landmarks_array[:, 1])
            z_std = np.std(landmarks_array[:, 2])
            
            # 标准化标准差，将其映射到0-1范围
            # 假设正常人脸的标准差在一定范围内
            x_quality = min(1.0, x_std / 100.0)
            y_quality = min(1.0, y_std / 100.0)
            z_quality = min(1.0, z_std / 0.5)
            
            # 综合评分，权重可以根据实际情况调整
            landmark_quality = (x_quality * 0.4 + y_quality * 0.4 + z_quality * 0.2)
            
            # 最终置信度，结合关键点质量和其他因素
            confidence = landmark_quality * 0.8 + 0.2  # 基础置信度0.2，最高1.0
            
            return min(1.0, max(0.0, confidence))  # 确保在0-1范围内
            
        except Exception:
            return 0.0