# 数据管理器
import numpy as np
import cv2
import json
import os
import threading
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from project.data.data_models import (
    Point3DWithVisibility, KeyCoordinates, FITTING_TYPE, EYE_TYPE, DepthMap, BGRImage
)
from project.config.settings import CAMERA_PARAMS_PATH

# 配置日志
from project.config.logging_config import setup_logging
logger = setup_logging(__name__)


import cv2
import numpy as np
import threading
from datetime import datetime
from typing import Dict, Optional, Tuple

class CameraDataManager:
    """
    相机数据管理器单例类
    负责相机参数管理、摄像头控制和图像数据管理
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """单例模式实现"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config_file_path: Optional[str] = None, rgb_d: bool = True):
        """初始化（只在第一次创建时执行）"""
        if not hasattr(self, '_initialized'):
            self._rgb_d = rgb_d
            self._config_file_path = config_file_path or CAMERA_PARAMS_PATH
            self._data_dict = {}  # 帧数据存储：{frame_id: frame_data}
            self._resolution = None  # 分辨率 (width, height)
            self._camera_params = None  # 相机参数缓存
            ok = self.initialize_camera()
            if not ok:
                self._cap = None
            # 加载配置用于清理
            from project.config.settings import FRAME_ID_PERIOD_SECONDS, FRAME_ID_CLEANUP_ENABLED
            self.cleanup_enabled = FRAME_ID_CLEANUP_ENABLED
            self._initialized = False
    
    # ==================== 相机控制接口 ====================
    
    def get_cap(self) -> Optional[cv2.VideoCapture]:
        """获取摄像头对象"""
        return self._cap
    
    def initialize_camera(self) -> bool:
        """初始化摄像头"""
        try:
            if self._rgb_d:
                logger.info("初始化RGB-D摄像头")
                self._cap = self._open_rgb_d()
            else:
                logger.info("初始化RGB摄像头")
                self._cap = cv2.VideoCapture(0)
            
            if self._cap is None or not self._cap.isOpened():
                logger.warning("无法初始化摄像头")
                return False
            
            logger.info(f"摄像头初始化成功，RGB-D模式: {self._rgb_d}")
            return True
        except Exception as e:
            logger.error(f"摄像头初始化失败: {e}", exc_info=True)
            return False
    
    def release_camera(self):
        """释放摄像头资源"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("摄像头资源已释放")
    
    # ==================== 相机参数管理接口 ====================
    
    def load_camera_params(self) -> Dict:
        """
        加载相机参数并返回最终字典
        
        Returns:
            Dict: 验证和调整后的相机参数字典
        """
        if self._camera_params is not None:
            return self._camera_params
        
        try:
            if os.path.exists(self._config_file_path):
                with open(self._config_file_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logger.info(f"从配置文件加载参数: {self._config_file_path}")
            else:
                logger.warning("配置文件不存在，使用默认参数")
                config = self._get_default_config()
                
            # 缓存参数
            self._camera_params = config
            return config
            
        except Exception as e:
            logger.error(f"加载相机参数失败: {e}", exc_info=True)
            return self._get_default_config()
    
    def get_image_resolution(self) -> Tuple[int, int]:
        """
        获取图像分辨率
        
        Returns:
            Tuple[int, int]: (width, height) 元组
        """
        if self._cap is None or not self._cap.isOpened():
            return (640, 480)  # 默认分辨率
        
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)
    
    def get_intrinsics(self) -> Dict:
        """
        获取相机内参
        
        Returns:
            Dict: 内参字典 {"fx": float, "fy": float, "cx": float, "cy": float}
        """
        config = self.load_camera_params()
        return config.get("intrinsic_params", {})
    
    def get_depth_scale(self) -> float:
        """
        获取深度缩放因子
        
        Returns:
            float: 深度缩放因子
        """
        config = self.load_camera_params()
        return config.get("depth_scale", 0.001)
    
    def get_camera_params(self) -> Dict:
        """
        获取完整的相机参数
        
        Returns:
            Dict: 完整的相机参数字典
        """
        return self.load_camera_params()
    
    # ==================== 图像数据管理接口 ====================
    
    def add_frame(self, frame_id: int, bgr_image: np.ndarray, 
                  depth_map: np.ndarray = None) -> None:
        """
        添加一帧数据
        
        Args:
            frame_id: 帧ID
            bgr_image: BGR格式的图像数据 (H, W, 3)
            depth_map: 深度图数据 (H, W)，可选
        """
        # 自动设置分辨率（使用第一帧的分辨率）
        if self._resolution is None:
            h, w = bgr_image.shape[:2]
            self._resolution = (w, h)
        
        if depth_map is None:
            depth_map = self._read_depth_map_from_rgb_d(bgr_image)


         
        frame_data = {
            "bgr_image": bgr_image,
            "depth_map": depth_map,
            "timestamp": datetime.now().isoformat()
        }
        
        # 存储数据
        self._data_dict[frame_id] = frame_data
    
    def get_image(self, frame_id: int) -> Optional[BGRImage]:
        """
        获取指定帧的BGR图像
        
        Args:
            frame_id: 帧ID
            
        Returns:
            Optional[BGRImage]: BGR图像对象，如果不存在则返回None
        """
        frame_data = self._data_dict.get(frame_id)
        if frame_data and "bgr_image" in frame_data:
            return BGRImage(data=frame_data["bgr_image"])
        logger.warning(f"获取帧数据失败: frame_id={frame_id}")
        return None
    
    def get_depth(self, frame_id: int) -> Optional[DepthMap]:
        """
        获取指定帧的深度图
        
        Args:
            frame_id: 帧ID
            
        Returns:
            Optional[DepthMap]: 深度图对象，如果不存在则返回None
        """
        frame_data = self._data_dict.get(frame_id)
        if frame_data and "depth_map" in frame_data and frame_data["depth_map"] is not None:
            return DepthMap(data=frame_data["depth_map"], unit="camera_unit")
        logger.warning(f"获取深度图数据失败: frame_id={frame_id}")
        return None

    def clear_data(self):
        """清空所有帧数据"""
        self._data_dict.clear()
        logger.info("所有帧数据已清空")
    
    def cleanup_old_data(self, threshold: int):
        """清理时间戳小于阈值的旧数据（与 FrameIdManager 同步）"""
        if not self.cleanup_enabled:
            return
        
        original_count = len(self._data_dict)
        # 保留时间戳 >= threshold 的数据
        self._data_dict = {fid: data for fid, data in self._data_dict.items() if fid >= threshold}
        removed_count = original_count - len(self._data_dict)
        
        if removed_count > 0:
            pass
    
    # ==================== 私有方法 ====================
    
    def _open_rgb_d(self) -> Optional[cv2.VideoCapture]:
        """打开 RGB-D 摄像头，优先 1080P（双目 3840x1080 或单路 1920x1080）"""
        import platform
        try:
            if platform.system() == "Windows":
                cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(1)
            if not cap.isOpened():
                logger.warning("RGB-D摄像头打开失败")
                return None
            # 设置 1080P：双目并排 3840x1080 或单路 1920x1080
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            if actual_w < 2000:  # 相机可能不支持 3840，尝试单路 1080P
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"RGB-D摄像头打开成功，分辨率 {actual_w}x{actual_h}")
            return cap
        except Exception as e:
            logger.error(f"打开RGB-D摄像头失败: {e}", exc_info=True)
            return None
        
    def _read_depth_map_from_rgb_d(self, bgr_image: np.ndarray) -> np.ndarray:
        """从RGB-D摄像头读取深度图"""
        if self._rgb_d:
            # RGB-D 双目相机输出 3840x1080，暂无可用的深度 API 时使用占位深度图
            h, w = bgr_image.shape[:2]
            depth_scale = self.get_depth_scale()
            depth_map = np.ones((h, w), dtype=np.float32) * (0.4 / depth_scale)
            return depth_map
        else:
            logger.warning("非RGB-D摄像头，使用默认深度图")
            h, w = bgr_image.shape[:2]
            depth_scale = self.get_depth_scale()
            depth_map = np.ones((h, w), dtype=np.float32) * (0.4 / depth_scale)
            return depth_map
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            "camera_name": "Auto Detected Camera",
            "camera_type": "RGB-D" if self._rgb_d else "RGB",
            "intrinsic_params": {
                "fx": 925.0,
                "fy": 925.0,
                "cx": 320.0,
                "cy": 240.0
            },
            "image_resolution": {
                "width": 640,
                "height": 480
            },
            "depth_scale": 0.001
        }
    

    
    def _check_required_fields(self, config: Dict) -> Dict:
        """检查必需字段"""
        required_fields = [
            "camera_name", "camera_type", "intrinsic_params", 
            "image_resolution", "depth_scale"
        ]
        
        for field in required_fields:
            if field not in config:
                logger.warning(f"缺少必需字段: {field}")
                if field == "camera_name":
                    config[field] = "Auto Detected Camera"
                elif field == "camera_type":
                    config[field] = "RGB-D" if self._rgb_d else "RGB"
                elif field == "intrinsic_params":
                    config[field] = {"fx": 925.0, "fy": 925.0, "cx": 320.0, "cy": 240.0}
                elif field == "image_resolution":
                    config[field] = {"width": 640, "height": 480}
                elif field == "depth_scale":
                    config[field] = 0.001
        
        return config

    
    def __del__(self):
        """析构函数，释放资源"""
        self.release_camera()
    


# 全局单例实例
CAMERA_DATA_MANAGER = CameraDataManager()



class RecgFitDataManager:
    """识别拟合数据管理器 - 重构版本"""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(RecgFitDataManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, key_coordinates: KeyCoordinates = None, 
                 debug_log: bool = True, convert2mm: bool = True):
        """
        初始化数据管理器
        
        Args:
            key_coordinates: 关键点坐标数据
            debug_log: 是否启用调试日志
            convert2mm: 是否转换为毫米单位
        """
        try:
            # 更新参数（单例模式，每次调用都会更新）
            self._key_coordinates = key_coordinates
            self._debug_log = debug_log
            self._convert2mm = convert2mm
            
            # 添加调试信息
            logger.info(f"RecgFitDataManager初始化 - debug_log: {self._debug_log}, key_coordinates: {key_coordinates is not None}")

            if key_coordinates is not None:
                self._convert_coords_2mm()
                
                logger.info("RecgFitDataManager初始化完成")
                
                if self._debug_log:
                    self._log_debug()
            else:
                logger.info("RecgFitDataManager初始化完成（无关键点数据）")
            
        except Exception as e:
            logger.error(f"RecgFitDataManager初始化失败: {e}")
            raise
        
    def _convert_coords_2mm(self) -> None:
        """将坐标从米转换为毫米"""

        if not self._convert2mm:
            return
        
        conversion_count = 0
        
        for eye in EYE_TYPE:
            for fitting_type in FITTING_TYPE:
                points = self._key_coordinates.get_points(eye, fitting_type)
                if points:
                    # 转换每个点的坐标（前3维）为毫米
                    converted_points = []

                    for point in points:
                        if point.z > 5:
                            logger.warning(f"检测到异常z值: {point.z}，跳过转换")
                            return
                        
                        # 记录转换前的坐标
                        if self._debug_log and conversion_count < 5:
                            pass
                        
                        converted_point = Point3DWithVisibility(
                            x=point.x * 1000,  # 米转毫米
                            y=point.y * 1000,
                            z=point.z * 1000,
                            visibility=point.visibility  # 可见性不变
                        )
                        
                        # 记录转换后的坐标
                        if self._debug_log and conversion_count < 5:
                            pass
                        
                        converted_points.append(converted_point)
                        conversion_count += 1
                    
                    # 更新坐标
                    self._key_coordinates.set_points(eye, fitting_type, converted_points)
        
    
    # ==================== Getter方法 ====================    
    def get_key_coordinates(self) -> KeyCoordinates:
        """获取完整的关键点坐标数据"""
        return self._key_coordinates
    
    def get_eye_coordinates(self, eye: str) -> Dict[str, List[Point3DWithVisibility]]:
        """
        获取指定眼睛的关键点坐标数据
        
        Args:
            eye: 眼睛类型 ("left" 或 "right")
            
        Returns:
            Dict[str, List[Point3DWithVisibility]]: 单眼关键点坐标字典
        """
        if not self._check_eye_type(eye):
            raise ValueError(f"无效的眼睛类型: {eye}")
        
        
        result = {}
        for fitting_type in FITTING_TYPE:
            result[fitting_type] = self._key_coordinates.get_points(eye, fitting_type)
        
        return result
    
    def get_coordinate_point(self, eye: str, fitting_type: str) -> List[Point3DWithVisibility]:
        """
        获取指定眼睛和类型的坐标点
        
        Args:
            eye: 眼睛类型 ("left" 或 "right")
            fitting_type: 拟合类型
            
        Returns:
            List[Point3DWithVisibility]: 坐标点列表
        """
        if not self._check_eye_type(eye):
            raise ValueError(f"无效的眼睛类型: {eye}")
        
        if fitting_type not in FITTING_TYPE:
            raise ValueError(f"无效的拟合类型: {fitting_type}")
        
        return self._key_coordinates.get_points(eye, fitting_type)
    
    def get_point_count(self, eye: str) -> int:
        """获取指定眼睛的坐标点数量"""
        if not self._check_eye_type(eye):
            raise ValueError(f"无效的眼睛类型: {eye}")
        
        total_count = 0
        for fitting_type in FITTING_TYPE:
            points = self.get_coordinate_point(eye, fitting_type)
            total_count += len(points)
        
        return total_count
    
    def get_mean_visibility(self, eye: str) -> float:
        """获取指定眼睛的平均可见性"""
        if not self._check_eye_type(eye):
            raise ValueError(f"无效的眼睛类型: {eye}")
        
        visibility_values = []
        for fitting_type in FITTING_TYPE:
            points = self.get_coordinate_point(eye, fitting_type)
            for point in points:
                visibility_values.append(point.visibility)
        
        if visibility_values:
            return np.mean(visibility_values)
        else:
            return 0.0
        
    def get_data_summary(self) -> Dict[str, Dict[str, int]]:
        """获取数据摘要"""
        summary = {}
        for eye in EYE_TYPE:
            summary[eye] = {}
            for fitting_type in FITTING_TYPE:
                points = self.get_coordinate_point(eye, fitting_type)
                summary[eye][fitting_type] = len(points)
        
        return summary
    
    # ==================== Setter方法 ====================
    
    def add_key_coordinates(self, key_coordinates: KeyCoordinates) -> None:
        """
        添加完整的关键点坐标数据
        
        Args:
            key_coordinates: 关键点坐标数据
        """
        self._key_coordinates = key_coordinates
        self._convert_coords_2mm()
        self._log_debug()
    

    
    def clear_all_data(self) -> None:
        """清空所有眼睛的数据"""
        self._key_coordinates.clear()
        self._log_debug()

    
    # ==================== 验证方法 ====================

        
    def _check_eye_type(self, eye: str) -> bool:
        """检查眼睛类型是否有效"""
        return eye in EYE_TYPE
        
    def _log_debug(self) -> None:
        """输出调试日志（已禁用）"""
        pass
    
    def __str__(self) -> str:
        """字符串表示"""
        summary = self.get_data_summary()
        result = "RecgFitDataManager:\n"
        for eye in EYE_TYPE:
            result += f"  {eye} eye:\n"
            for fitting_type in FITTING_TYPE:
                count = summary[eye][fitting_type]
                result += f"    {fitting_type}: {count} points\n"
        return result

# 全局实例

CAMERA_DATA_MANAGER = CameraDataManager()


# 导出
__all__ = ['RecgFitDataManager', 'CameraDataManager', 'CAMERA_DATA_MANAGER']