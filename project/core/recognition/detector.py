# 人脸检测模块
import numpy as np
from typing import Dict, List, Optional
from project.data.data_models import (
    BGRImage, DepthMap, Landmark, KeyCoordinates,
    FITTING_TYPE, EYE_TYPE
)

# 配置日志
from project.config.logging_config import setup_logging 
logger = setup_logging(__name__)


class FaceDetector:
    """人脸检测器核心类"""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(FaceDetector, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, camera_params: Dict, rgb_d: bool = False):
        """初始化检测器"""
        self._validate_camera_params(camera_params)
        self.camera_params = camera_params
        self._bgr_image: Optional[BGRImage] = None
        self._depth_map: Optional[DepthMap] = None
        self._landmarks: List[Landmark] = []
        self._detection_success = False
        self._rgb_d = rgb_d
        
        logger.info(f"FaceDetector初始化完成，RGB-D模式: {rgb_d}")
    
    def detect_face(self, bgr_image: BGRImage, depth_map: DepthMap) -> bool:
        """检测人脸和关键点"""
        try:
            self._validate_input_images(bgr_image, depth_map)
            self._bgr_image = bgr_image
            
            # 导入关键点提取模块
            from project.core.recognition.landmark_extractor import extract_landmarks, validate_landmarks
            
            # 提取关键点
            landmarks = extract_landmarks(bgr_image)
            # #region agent log
            if not landmarks:
                try:
                    _sh = getattr(getattr(bgr_image, "data", None), "shape", None) or (0, 0)
                    _payload = {"sessionId": "6d5179", "timestamp": __import__("time").time() * 1000, "location": "detector.py:detect_face", "message": "empty landmarks", "data": {"image_shape": list(_sh)[:2] if hasattr(_sh, "__len__") else []}, "hypothesisId": "H1"}
                    open("debug-6d5179.log", "a").write(__import__("json").dumps(_payload) + "\n")
                except Exception:
                    pass
            # #endregion
            if landmarks:
                logger.info(f"提取到 {len(landmarks)} 个关键点")
                is_valid = validate_landmarks(landmarks)
                logger.info(f"关键点验证结果: {is_valid}")
                
                if is_valid:
                    self._landmarks = landmarks
                    self._detection_success = True
                    
                    # 处理深度图：统一转为米，使用配置的 depth_scale（camera_unit→米，默认 mm→m 0.001）
                    depth_scale = self.camera_params.get("depth_scale", 0.001)
                    # [深度调试] to_meters 前：camera_unit 原始值
                    raw = depth_map.data
                    raw_valid = np.isfinite(raw) & (raw > 0)
                    if np.any(raw_valid):
                        rv = raw[raw_valid]
                        logger.info(
                            "[深度调试] to_meters 前 unit=%s depth_scale=%s: min=%.3f max=%.3f mean=%.3f (若实际约0.4m则期望camera_unit约400)",
                            depth_map.unit, depth_scale, float(np.min(rv)), float(np.max(rv)), float(np.mean(rv)),
                        )
                    if not self._rgb_d:
                        modified_depth_map = self._set_depth_map_without_rgb_d(depth_map)
                        self._depth_map = modified_depth_map.to_meters(depth_scale)
                    else:
                        self._depth_map = depth_map.to_meters(depth_scale)
                    # [深度调试] to_meters 后：应为米
                    out = self._depth_map.data
                    out_valid = np.isfinite(out) & (out > 0)
                    if np.any(out_valid):
                        ov = out[out_valid]
                        logger.info(
                            "[深度调试] to_meters 后 unit=m: min=%.4f max=%.4f mean=%.4f (期望约0.4)",
                            float(np.min(ov)), float(np.max(ov)), float(np.mean(ov)),
                        )
                    logger.info(f"人脸检测成功，提取到 {len(landmarks)} 个关键点")
                    return True
                else:
                    logger.warning(f"关键点验证失败，期望478个，实际{len(landmarks)}个")
            else:
                logger.warning("未提取到关键点")
                
            self._detection_success = False
            self._landmarks = []
            self._depth_map = None
            logger.warning("人脸检测失败或关键点无效")
            return False
                
        except Exception as e:
            logger.error(f"人脸检测失败: {e}", exc_info=True)
            self._detection_success = False
            self._landmarks = []
            self._depth_map = None
            return False
    
    def get_fitting_data(self) -> KeyCoordinates:
        """
        获取拟合所需的关键点数据
        
        Returns:
            key_coordinates: 关键点坐标数据
        """
        if not self._detection_success or not self._landmarks or not self._depth_map:
            logger.warning("检测未成功或无有效数据")
            return KeyCoordinates()
        
        try:
            # 导入坐标转换模块
            from project.core.recognition.coordinate_converter import batch_convert_landmarks
            from project.core.recognition.landmark_extractor import get_landmark_indices
            
            # 批量转换关键点为3D坐标
            points_3d = batch_convert_landmarks(self._landmarks, self._depth_map, self.camera_params)
            
            # 创建KeyCoordinates对象
            key_coordinates = KeyCoordinates()
            
            # 获取关键点索引
            indices = get_landmark_indices()
            
            # 填充数据
            for eye in EYE_TYPE:
                for fitting_type in FITTING_TYPE:
                    # 获取该类型的关键点索引
                    landmark_indices = indices[eye][fitting_type]
                    eye_points = []
                    
                    for idx in landmark_indices:
                        if idx < len(points_3d):
                            eye_points.append(points_3d[idx])
                    
                    # 设置坐标
                    key_coordinates.set_points(eye, fitting_type, eye_points)
            
            
            logger.debug(f"拟合数据获取完成：左眼 {len(key_coordinates.left_eye)} 种类型，右眼 {len(key_coordinates.right_eye)} 种类型")
            return key_coordinates
            
        except Exception as e:
            logger.error(f"获取拟合数据失败: {e}", exc_info=True)
            return KeyCoordinates()
    
    def get_landmarks(self) -> List[Landmark]:
        """获取原始关键点数据"""
        return self._landmarks.copy() if self._landmarks else []
    
    def get_bgr_image(self) -> Optional[BGRImage]:
        """获取BGR图像"""
        return self._bgr_image
    
    def get_depth_map(self) -> Optional[DepthMap]:
        """获取深度图"""
        return self._depth_map
    
    def is_detection_successful(self) -> bool:
        """检查检测是否成功"""
        return self._detection_success
    
    def clear_data(self):
        """清空所有数据"""
        self._bgr_image = None
        self._depth_map = None
        self._landmarks = []
        self._detection_success = False
        logger.debug("数据已清空")
    
    def _validate_camera_params(self, camera_params: Dict) -> None:
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
    
    def _set_depth_map_without_rgb_d(self, depth_map: DepthMap) -> DepthMap:
        """当不使用RGB-D相机时，为所有关键点设置深度值"""
        if not self._landmarks:
            return depth_map
        
        # 创建修改后的深度图
        modified_data = depth_map.data.copy().astype(np.float32)
        
        for landmark in self._landmarks:
            if len(landmark.to_ndarray()) >= 3:
                x, y, z = landmark.x, landmark.y, landmark.z
                x_idx, y_idx = int(round(x)), int(round(y))
                
                if (0 <= x_idx < depth_map.width and 0 <= y_idx < depth_map.height):
                    original_depth = depth_map.data[y_idx, x_idx]
                    # 使用MediaPipe估计的z值来修补深度
                    new_depth = original_depth + z * 100  # z是相对深度，需要放大
                    modified_data[y_idx, x_idx] = new_depth
        
        return DepthMap(data=modified_data, unit=depth_map.unit)
    
    
    def _validate_input_images(self, bgr_image: BGRImage, depth_map: DepthMap) -> None:
        """验证输入图像的有效性"""
        if not isinstance(bgr_image, BGRImage):
            raise ValueError("bgr_image必须是BGRImage类型")
        
        if not isinstance(depth_map, DepthMap):
            raise ValueError("depth_map必须是DepthMap类型")
        
        if not depth_map.validate_with_image(bgr_image):
            raise ValueError(f"BGR图像和深度图尺寸不匹配: {bgr_image.shape} vs {depth_map.shape}")
    
    def get_detection_info(self) -> Dict:
        """获取检测信息"""
        return {
            'detection_success': self._detection_success,
            'landmark_count': len(self._landmarks) if self._landmarks else 0,
            'has_bgr_image': self._bgr_image is not None,
            'has_depth_map': self._depth_map is not None,
            'rgb_d_mode': self._rgb_d
        }

# 测试函数
def test_face_detector():
    """测试人脸检测器功能"""
    print("=== 人脸检测器测试 ===")
    
    # 创建测试相机参数
    camera_params = {
        "intrinsic_params": {
            "fx": 925.0,
            "fy": 925.0,
            "cx": 640.0,
            "cy": 360.0
        },
        "depth_scale": 0.001
    }
    
    # 测试RGB-D模式
    print("\n1. 测试RGB-D模式")
    detector_rgbd = FaceDetector(camera_params, rgb_d=True)
    bgr_image = BGRImage(data=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    depth_map_rgbd = DepthMap(data=np.ones((480, 640), dtype=np.float32) * 500, unit="camera_unit")  # 毫米单位
    
    success_rgbd = detector_rgbd.detect_face(bgr_image, depth_map_rgbd)
    print(f"RGB-D检测结果: {'成功' if success_rgbd else '失败'}")
    if success_rgbd:
        final_depth_map = detector_rgbd.get_depth_map()
        print(f"最终深度图单位: {final_depth_map.unit}")
        print(f"深度图范围: {np.min(final_depth_map.data):.3f}m - {np.max(final_depth_map.data):.3f}m")
    
    # 测试非RGB-D模式
    print("\n2. 测试非RGB-D模式")
    detector_normal = FaceDetector(camera_params, rgb_d=False)
    depth_map_normal = DepthMap(data=np.ones((480, 640), dtype=np.float32) * 0.5, unit="meter")  # 米单位
    
    success_normal = detector_normal.detect_face(bgr_image, depth_map_normal)
    print(f"非RGB-D检测结果: {'成功' if success_normal else '失败'}")
    if success_normal:
        final_depth_map = detector_normal.get_depth_map()
        print(f"最终深度图单位: {final_depth_map.unit}")
        print(f"深度图范围: {np.min(final_depth_map.data):.3f}m - {np.max(final_depth_map.data):.3f}m")
        
        # 测试获取拟合数据
        print("\n3. 测试获取拟合数据")
        key_coordinates = detector_normal.get_fitting_data()
        print(f"拟合数据: {key_coordinates}")
        
        # 测试获取关键点
        print("\n4. 测试获取关键点")
        landmarks = detector_normal.get_landmarks()
        print(f"关键点数量: {len(landmarks)}")
        
        # 测试检测信息
        print("\n5. 测试检测信息")
        info = detector_normal.get_detection_info()
        print(f"检测信息: {info}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_face_detector()