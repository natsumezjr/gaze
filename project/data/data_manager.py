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
    Point3DWithVisibility, KeyCoordinates, FITTING_TYPE, EYE_TYPE, BGRImage
)
from project.config.settings import CAMERA_PARAMS_PATH
from project.config.stereo_config import (
    DEFAULT_DEPTH_SCALE,
    PLACEHOLDER_DEPTH_METERS,
    SGBM_BLOCK_SIZE,
    SGBM_DISP12_MAX_DIFF,
    SGBM_MIN_DISPARITY,
    SGBM_NUM_DISPARITIES,
    SGBM_P1,
    SGBM_P2,
    SGBM_PREFILTER_CAP,
    SGBM_SPECKLE_RANGE,
    SGBM_SPECKLE_WINDOW_SIZE,
    SGBM_UNIQUENESS_RATIO,
    STEREO_FRAME_SIZES,
)

# 配置日志
from project.config.logging_config import setup_logging
logger = setup_logging(__name__)


def _median_filter_nan_safe(depth: np.ndarray, ksize: int = 5) -> np.ndarray:
    """对深度图做 ksize×ksize 中值滤波，nan 不参与中值，输出保持 float32。"""
    from numpy.lib.stride_tricks import sliding_window_view
    pad = ksize // 2
    padded = np.pad(
        depth.astype(np.float64),
        ((pad, pad), (pad, pad)),
        mode="constant",
        constant_values=np.nan,
    )
    windows = sliding_window_view(padded, (ksize, ksize))
    out = np.nanmedian(windows, axis=(-2, -1)).astype(np.float32)
    return out


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
            # 立体深度：remap 与 SGBM 缓存（懒初始化）
            self._stereo_map1x = None
            self._stereo_map1y = None
            self._stereo_map2x = None
            self._stereo_map2y = None
            self._stereo_sgbm = None
            self._stereo_Q = None
            self._stereo_image_size = None
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

    # 识别模块 2D→3D 固定深度默认值（mm），不做单位换算
    DEFAULT_FACE_DEPTH_MM = 500

    def pixel_to_xy_mm(self, u: float, v: float, z_mm: float) -> Tuple[float, float]:
        """
        像素坐标 + 深度(Z, mm) → 相机坐标系下的 X,Y（单位 mm）。
        内参来自 load_camera_params：若有 intrinsic_params 则用，否则从 JSON 的 stereo.P1 直接解析，不依赖 rgb_d。
        """
        config = self.load_camera_params()
        fx = fy = cx = cy = None
        stereo = config.get("stereo", {})
        if isinstance(stereo.get("P1"), list) and len(stereo["P1"]) >= 2:
            P1 = stereo["P1"]
            fx = float(P1[0][0])
            fy = float(P1[1][1])
            cx = float(P1[0][2])
            cy = float(P1[1][2])
        if fx is None or fy is None or cx is None or cy is None or fx <= 0 or fy <= 0:
            return (float("nan"), float("nan"))
        x_mm = (u - cx) * z_mm / fx
        y_mm = (v - cy) * z_mm / fy
        return (x_mm, y_mm)
    
    # ==================== 图像数据管理接口 ====================
    
    def add_frame(self, frame_id: int, bgr_image: np.ndarray,
                  depth_map: np.ndarray = None) -> None:
        """
        添加一帧数据。

        Args:
            frame_id: 帧ID
            bgr_image: BGR格式的图像数据 (H, W, 3)
            depth_map: 深度图数据 (H, W)，可选；仅当 store_depth=True 时写入
            store_depth: 是否计算/存储深度图。识别路径传 False 即可只存图像，不触发立体深度
        """
        if self._resolution is None:
            h, w = bgr_image.shape[:2]
            self._resolution = (w, h)

        if depth_map is None and self._rgb_d:
            result = self._compute_stereo_depth(bgr_image)
            if result is not None:
                rect_left, _rect_right, depth_mm = result
                self._data_dict[frame_id] = {
                    "bgr_image": rect_left,
                    "depth_map": depth_mm,
                    "timestamp": datetime.now().isoformat(),
                }
                return

        if depth_map is None:
            depth_map = self._set_default_depth_map(bgr_image)

        self._data_dict[frame_id] = {
            "bgr_image": bgr_image,
            "depth_map": depth_map,
            "timestamp": datetime.now().isoformat(),
        }
    
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

    def get_depth_at_pixel(self, frame_id: int, u: float, v: float) -> Optional[float]:
        """
        获取指定帧在像素 (u, v) 处的深度（毫米）。与 get_image 同帧的 depth_map 对齐。
        无深度图、越界或无效时返回 None。
        """
        frame_data = self._data_dict.get(frame_id)
        if not frame_data or "depth_map" not in frame_data:
            return None
        depth_mm = frame_data["depth_map"]
        h, w = depth_mm.shape[:2]
        x, y = int(round(u)), int(round(v))
        if x < 0 or x >= w or y < 0 or y >= h:
            return None
        val = float(depth_mm[y, x])
        return val if np.isfinite(val) and val > 0 else None

    def get_depth_neighborhood_median(
        self,
        frame_id: int,
        u: float,
        v: float,
        half_win: int = 1,
        z_min_mm: float = 200.0,
        z_max_mm: float = 1500.0,
    ) -> Optional[float]:
        """
        取像素 (u,v) 邻域内有效深度的中值（用于替代单点无效时的 fallback）。
        邻域为 (2*half_win+1)×(2*half_win+1)，默认 3×3。只考虑 [z_min_mm, z_max_mm] 内的值。
        若无有效值则返回 NaN。
        """
        frame_data = self._data_dict.get(frame_id)
        if not frame_data or "depth_map" not in frame_data:
            return np.nan
        depth_mm = frame_data["depth_map"]
        h, w = depth_mm.shape[:2]
        cx, cy = int(round(u)), int(round(v))
        y0 = max(0, cy - half_win)
        y1 = min(h, cy + half_win + 1)
        x0 = max(0, cx - half_win)
        x1 = min(w, cx + half_win + 1)
        region = depth_mm[y0:y1, x0:x1]
        valid = (
            np.isfinite(region)
            & (region >= z_min_mm)
            & (region <= z_max_mm)
            & (region > 0)
        )
        if not np.any(valid):
            return np.nan
        return float(np.median(region[valid]))

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
    
    def compute_stereo_depth(
        self, frame_bgr: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        从双目并排整帧计算左/右目校正图与深度图（毫米）。
        供测试或外部调用；若帧尺寸不支持或无双目标定则返回 None。
        返回 (rect_left_bgr, rect_right_bgr, depth_mm)。
        """
        return self._compute_stereo_depth(frame_bgr)

    def _ensure_stereo_processor(self) -> bool:
        """若存在 stereo 标定则构建并缓存 remap 与 SGBM，返回是否可用。"""
        if self._stereo_sgbm is not None:
            return True
        config = self.load_camera_params()
        if "stereo" not in config or "left" not in config or "right" not in config:
            return False
        left = config["left"]
        right = config["right"]
        stereo = config["stereo"]
        image_size = tuple(stereo["image_size"])
        K1 = np.array(left["K"], dtype=np.float64)
        dist1 = np.array(left["dist"], dtype=np.float64).reshape(-1, 1)
        K2 = np.array(right["K"], dtype=np.float64)
        dist2 = np.array(right["dist"], dtype=np.float64).reshape(-1, 1)
        R1 = np.array(stereo["R1"], dtype=np.float64)
        R2 = np.array(stereo["R2"], dtype=np.float64)
        P1 = np.array(stereo["P1"], dtype=np.float64)
        P2 = np.array(stereo["P2"], dtype=np.float64)
        Q = np.array(stereo["Q"], dtype=np.float64)
        self._stereo_map1x, self._stereo_map1y = cv2.initUndistortRectifyMap(
            K1, dist1, R1, P1, image_size, cv2.CV_32FC1
        )
        self._stereo_map2x, self._stereo_map2y = cv2.initUndistortRectifyMap(
            K2, dist2, R2, P2, image_size, cv2.CV_32FC1
        )
        self._stereo_sgbm = cv2.StereoSGBM_create(
            minDisparity=SGBM_MIN_DISPARITY,
            numDisparities=SGBM_NUM_DISPARITIES,
            blockSize=SGBM_BLOCK_SIZE,
            P1=SGBM_P1,
            P2=SGBM_P2,
            disp12MaxDiff=SGBM_DISP12_MAX_DIFF,
            uniquenessRatio=SGBM_UNIQUENESS_RATIO,
            speckleWindowSize=SGBM_SPECKLE_WINDOW_SIZE,
            speckleRange=SGBM_SPECKLE_RANGE,
            preFilterCap=SGBM_PREFILTER_CAP,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        self._stereo_Q = Q
        self._stereo_image_size = image_size
        return True

    def _compute_stereo_depth(
        self, frame_bgr: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """从双目并排整帧得到 (左目校正图, 右目校正图, 深度图 mm)。不支持或无双目标定时返回 None。"""
        h, w = frame_bgr.shape[:2]
        if (w, h) not in STEREO_FRAME_SIZES:
            return None
        if not self._ensure_stereo_processor():
            return None
        half = w // 2
        left_bgr = frame_bgr[:, :half]
        right_bgr = frame_bgr[:, half:]
        # 标定与运行一致(3840×1080 → 半幅 1920×1080, stereo.image_size=(1920,1080))时无需 resize，直接 remap(1920)
        w_cal, h_cal = self._stereo_image_size
        if left_bgr.shape[1] != w_cal or left_bgr.shape[0] != h_cal:
            left_bgr = cv2.resize(left_bgr, (w_cal, h_cal), interpolation=cv2.INTER_LINEAR)
        if right_bgr.shape[1] != w_cal or right_bgr.shape[0] != h_cal:
            right_bgr = cv2.resize(right_bgr, (w_cal, h_cal), interpolation=cv2.INTER_LINEAR)
        rect_left = cv2.remap(left_bgr, self._stereo_map1x, self._stereo_map1y, cv2.INTER_LINEAR)
        rect_right = cv2.remap(right_bgr, self._stereo_map2x, self._stereo_map2y, cv2.INTER_LINEAR)
        gray_l = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)
        disp_raw = self._stereo_sgbm.compute(gray_l, gray_r)  # int16, Q16 定点：真实视差 = raw/16
        disp = disp_raw.astype(np.float32) / 16.0
        invalid_disp = disp <= 0
        disp_safe = disp.copy()
        disp_safe[invalid_disp] = 0.0
        points_3d = cv2.reprojectImageTo3D(disp_safe, self._stereo_Q)
        depth_mm = points_3d[:, :, 2].astype(np.float32)
        # 优先级1：限制有效深度范围，人脸典型 300~1200mm，消灭极远野值（如 67830）
        depth_invalid = (
            invalid_disp
            | ~np.isfinite(depth_mm)
            | (depth_mm <= 200)
            | (depth_mm >= 1500)
        )
        depth_mm[depth_invalid] = np.nan
        # 优先级3：轻量 5×5 中值平滑（保留边界、抗野值），nan 不参与中值
        depth_mm = _median_filter_nan_safe(depth_mm, ksize=5)
        # [深度调试] 立体输出 Z（已按 Q16 换算视差后 reproject），单位由 Q 决定（标定为 mm）
        valid = np.isfinite(depth_mm) & (depth_mm > 0)
        if np.any(valid):
            v = depth_mm[valid]
            h_d, w_d = depth_mm.shape
            center_z = depth_mm[h_d // 2, w_d // 2]
            center_disp = float(disp[h_d // 2, w_d // 2]) if disp[h_d // 2, w_d // 2] > 0 else float("nan")
            logger.info(
                "[深度调试] 立体输出 Z: min=%.3f max=%.3f mean=%.3f | nan占比=%.1f%% | 中心Z=%.3f 中心disp=%.2f (200~1500mm+中值平滑)",
                float(np.min(v)), float(np.max(v)), float(np.mean(v)),
                100.0 * (depth_mm.size - np.sum(valid)) / depth_mm.size,
                float(center_z) if np.isfinite(center_z) else float("nan"),
                center_disp,
            )
        return (rect_left, rect_right, depth_mm)

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
        
    def _set_default_depth_map(self, bgr_image: np.ndarray) -> np.ndarray:
        """从RGB-D摄像头读取深度图；无双目标定或非立体分辨率时使用占位深度图。"""
        h, w = bgr_image.shape[:2]
        depth_scale = self.get_depth_scale()
        depth_map = np.ones((h, w), dtype=np.float32) * (PLACEHOLDER_DEPTH_METERS / depth_scale)
        if not self._rgb_d:
            logger.warning("非RGB-D摄像头，使用默认深度图")
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
            "depth_scale": DEFAULT_DEPTH_SCALE
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
                    config[field] = DEFAULT_DEPTH_SCALE
        
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
    
    def __init__(self, key_coordinates: KeyCoordinates = None, debug_log: bool = True):
        """
        初始化数据管理器。输入 KeyCoordinates 约定为 mm，不做单位换算。

        Args:
            key_coordinates: 关键点坐标数据（x,y,z 单位 mm）
            debug_log: 是否启用调试日志
        """
        try:
            self._key_coordinates = key_coordinates
            self._debug_log = debug_log

            logger.info(f"RecgFitDataManager初始化 - debug_log: {self._debug_log}, key_coordinates: {key_coordinates is not None}")

            if key_coordinates is not None:
                logger.info("RecgFitDataManager初始化完成")
                if self._debug_log:
                    self._log_debug()
            else:
                logger.info("RecgFitDataManager初始化完成（无关键点数据）")

        except Exception as e:
            logger.error(f"RecgFitDataManager初始化失败: {e}")
            raise

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
        添加完整的关键点坐标数据（约定 x,y,z 单位 mm，不做单位换算）。

        Args:
            key_coordinates: 关键点坐标数据
        """
        self._key_coordinates = key_coordinates
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