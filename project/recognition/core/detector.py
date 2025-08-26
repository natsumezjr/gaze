import numpy as np
import cv2
from typing import Tuple, Optional, List
import logging

from project.recg_fit_data.data_manager import (
    CoordinatePoint, RecgFitDataManager, RECG_FIT_DATA_MANAGER, 
    EYE_TYPE, FITTING_TYPE
)

class FaceDetector:
    """人脸检测器核心类"""
    
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(FaceDetector, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, camera_params: dict, rgb_d: bool = False):
        """初始化检测器"""

        self._validate_camera_params(camera_params)
        self.camera_params = camera_params
        self._bgr_image = None
        self._depth_map_meters = None
        self._landmarks = None
        self._detection_success = False
        self._rgb_d = rgb_d
        
        # 初始化RecgFitDataManager
        self._data_manager = RECG_FIT_DATA_MANAGER

            

    def detect_face(self, bgr_image: np.ndarray, depth_map: np.ndarray) -> bool:
        """检测人脸和关键点"""
        try:
            self._validate_input_images(bgr_image, depth_map)
            self._bgr_image = bgr_image
            
            from project.recognition.core.landmark_extractor import extract_landmarks, validate_landmarks
            
            landmarks = extract_landmarks(bgr_image)
            
            if landmarks and validate_landmarks(landmarks):
                self._landmarks = landmarks
                self._detection_success = True
                
                if not self._rgb_d:
                    modified_depth_map = self._set_depth_map_without_rgb_d(depth_map)
                    self._depth_map_meters = self._convert_depth_to_meters(modified_depth_map)
                else:
                    self._depth_map_meters = self._convert_depth_to_meters(depth_map)
                
                return True
            else:
                self._detection_success = False
                self._landmarks = None
                self._depth_map_meters = None
                return False
                
        except Exception as e:
            logging.error(f"人脸检测失败: {e}")
            self._detection_success = False
            self._landmarks = None
            return False

    def update_fitting_data(self) -> None:
        """
        获取拟合所需的关键点数据
        
        """
        if not self._detection_success or self._landmarks is None:
            return
        
        try:
            from project.recognition.core.landmark_extractor import get_fitting_landmarks
            from project.recognition.core.coordinate_converter import batch_convert_landmarks
            
            # 获取原始拟合数据
            fitting_data = get_fitting_landmarks(self._landmarks)
            
            # 清空RecgFitDataManager中的数据
            self._data_manager.clear_all_data()
            
            # 处理每个眼睛的数据
            for eye in EYE_TYPE:
                eye_coordinates = {}
                
                # 处理每个拟合类型
                for fitting_type in FITTING_TYPE:
                    coordinate_points = fitting_data[eye][fitting_type]
                    
                    if coordinate_points is not None and len(coordinate_points) > 0:
                        # 转换坐标点
                        converted_points = self._convert_landmarks_to_3d(coordinate_points)
                        if converted_points:
                            eye_coordinates[fitting_type] = converted_points
                        else:
                            eye_coordinates[fitting_type] = []
                    else:
                        eye_coordinates[fitting_type] = []
                
                # 设置该眼睛的所有坐标数据
                self._data_manager.set_eye_coordinates(eye, eye_coordinates)

            
        except Exception as e:
            logging.error(f"获取拟合数据失败: {e}")
            return None

    def get_data_manager(self) -> RecgFitDataManager:
        """
        获取RecgFitDataManager实例
        
        :return: RecgFitDataManager实例
        """
        return self._data_manager

    def _convert_landmarks_to_3d(self, landmarks: List[List[float]]) -> List[CoordinatePoint]:
        """
        将2D关键点转换为3D坐标点
        
        Args:
            landmarks: 2D关键点列表，每个点为[x, y, z, visibility]
            
        Returns:
            List[CoordinatePoint]: 转换后的3D坐标点列表
        """
        converted_points = []
        
        try:
            from project.recognition.core.coordinate_converter import batch_convert_landmarks
            
            # 尝试批量转换
            points_3d = batch_convert_landmarks(
                landmarks,
                self._depth_map_meters,
                self.camera_params
            )
            
            if points_3d:
                return points_3d
            
        except Exception as e:
            logging.warning(f"批量转换失败，使用单点转换: {e}")
        
        # 批量转换失败，使用单点转换
        for landmark in landmarks:
            try:
                if len(landmark) >= 4:
                    x, y, z, visibility = landmark[0], landmark[1], landmark[2], landmark[3]
                    point_3d = self._pixel_to_3d((int(x), int(y)))
                    if point_3d is not None:
                        # 创建4D坐标点 [x, y, z, visibility]
                        coordinate_point = np.array([point_3d[0], point_3d[1], point_3d[2], visibility], dtype=np.float32)
                        converted_points.append(coordinate_point)
            except Exception as e:
                logging.warning(f"单点转换失败: {e}")
                continue
        
        return converted_points

    def _pixel_to_3d(self, pixel_coord: Tuple[int, int]) -> Optional[np.ndarray]:
        """将像素坐标转换为3D坐标"""
        try:
            from project.recognition.core.coordinate_converter import pixel_to_3d
            return pixel_to_3d(pixel_coord, self._depth_map_meters, self.camera_params)
        except Exception as e:
            logging.warning(f"像素转3D坐标失败: {e}")
            return None

    def _set_depth_map_without_rgb_d(self, depth_map: np.ndarray) -> np.ndarray:
        """当不使用RGB-D相机时，为所有关键点设置深度值"""
        if self._landmarks is None:
            return depth_map
        
        modified_depth_map = depth_map.copy().astype(np.float32)
        
        for landmark in self._landmarks:
            if len(landmark) >= 3:
                x, y, z = landmark[0], landmark[1], landmark[2]
                x_idx, y_idx = int(round(x)), int(round(y))
                
                if (0 <= x_idx < depth_map.shape[1] and 0 <= y_idx < depth_map.shape[0]):
                    original_depth = depth_map[y_idx, x_idx]
                    new_depth = original_depth + z * 100
                    modified_depth_map[y_idx, x_idx] = new_depth
        
        return modified_depth_map
    
    def _convert_depth_to_meters(self, depth_map: np.ndarray) -> np.ndarray:
        """将深度图从相机原始单位转换为米"""
        # 检查深度图是否已经是米单位（通过检查深度值范围）
        max_depth = np.max(depth_map)
        min_depth = np.min(depth_map)
        logging.debug(f"深度图范围: min={min_depth:.6f}, max={max_depth:.6f}")
        
        if max_depth > 10.0:  # 如果最大深度值大于10，可能是毫米单位
            depth_scale = self.camera_params.get("depth_scale", 1.0)
            logging.debug(f"检测到毫米单位，乘以depth_scale={depth_scale}")
            return depth_map.astype(np.float32) * depth_scale
        else:
            # 深度图已经是米单位，直接返回
            logging.debug(f"检测到米单位，直接返回")
            return depth_map.astype(np.float32)
    
    def _validate_camera_params(self, camera_params: dict) -> None:
        """验证相机参数的有效性"""
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
        """验证输入图像的有效性"""
        if not isinstance(bgr_image, np.ndarray) or len(bgr_image.shape) != 3 or bgr_image.shape[2] != 3:
            raise ValueError("bgr_image 形状必须为 (H, W, 3)")
        
        if not isinstance(depth_map, np.ndarray) or len(depth_map.shape) != 2:
            raise ValueError("depth_map 形状必须为 (H, W)")
        
        if bgr_image.shape[:2] != depth_map.shape:
            raise ValueError(f"BGR图像和深度图尺寸不匹配: {bgr_image.shape[:2]} vs {depth_map.shape}")
      
__all__ = ['FaceDetector']