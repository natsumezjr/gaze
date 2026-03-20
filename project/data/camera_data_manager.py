"""
Camera / Frame / Triangulation data manager.

从 `data_manager.py` 拆分出来以收紧职责边界；对外接口与行为保持不变。
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from project.config.logging_config import setup_logging
from project.config.settings import CAMERA_PARAMS_PATH
from project.config.stereo_config import STEREO_FRAME_SIZES
from project.data.data_models import (
    BGRImage,
    EYE_TYPE,
    FITTING_TYPE,
    KeyCoordinates,
    Point2D,
    Point3DWithVisibility,
)

logger = setup_logging(__name__)


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
        if not hasattr(self, "_initialized"):
            self._rgb_d = rgb_d
            self._config_file_path = config_file_path or CAMERA_PARAMS_PATH
            self._data_dict = {}
            self._resolution = None
            self._camera_params = None
            ok = self.initialize_camera()
            if not ok:
                self._cap = None
            # 加载配置用于清理
            from project.config.settings import FRAME_ID_CLEANUP_ENABLED

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

            logger.info("摄像头初始化成功，RGB-D模式: %s", self._rgb_d)
            return True
        except Exception as e:
            logger.error("摄像头初始化失败: %s", e, exc_info=True)
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
                with open(self._config_file_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                logger.info("从配置文件加载参数: %s", self._config_file_path)
            else:
                logger.warning("配置文件不存在，使用默认参数")
                config = self._get_default_config()
            # 缓存参数
            self._camera_params = config
            return config

        except Exception as e:
            logger.error("加载相机参数失败: %s", e, exc_info=True)
            return self._get_default_config()

    def get_camera_params(self) -> Dict:
        """获取完整的相机参数"""
        return self.load_camera_params()

    def get_stereo_raw_params(
        self,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        从 camera_params.json 中读取并返回原始双目几何（用于稀疏三角化）：
        K1, dist1, K2, dist2, R, T
        若缺少 left/right/stereo 或 K/dist/R/T 则返回 None。
        """
        config = self.load_camera_params()
        if "left" not in config or "right" not in config or "stereo" not in config:
            logger.warning("get_stereo_raw_params: config 缺少 left/right/stereo")
            return None
        left = config["left"]
        right = config["right"]
        stereo = config["stereo"]
        for key in ("K", "dist"):
            if key not in left or key not in right:
                logger.warning("get_stereo_raw_params: left/right 缺少 %s", key)
                return None
        if "R" not in stereo or "T" not in stereo:
            logger.warning("get_stereo_raw_params: stereo 缺少 R 或 T")
            return None
        K1 = np.array(left["K"], dtype=np.float64)
        dist1 = np.array(left["dist"], dtype=np.float64).reshape(-1, 1)
        K2 = np.array(right["K"], dtype=np.float64)
        dist2 = np.array(right["dist"], dtype=np.float64).reshape(-1, 1)
        R = np.array(stereo["R"], dtype=np.float64)
        T = np.array(stereo["T"], dtype=np.float64).reshape(3, 1)
        return (K1, dist1, K2, dist2, R, T)

    def triangulate_one_point(self, pt_l: Point2D, pt_r: Point2D) -> Point3DWithVisibility:
        """
        使用原始相机几何做单点三角化：
        1) cv2.undistortPoints
        2) P1=[I|0], P2=[R|T]
        3) cv2.triangulatePoints
        4) 返回 Point3DWithVisibility(x,y,z,visibility)

        visibility 为 1.0；若输入 2D 或三角化结果非有限 / z<=0，则 visibility=0。
        坐标单位与 T 一致（mm）。
        """
        # 输入 2D 基本检查
        if pt_l is None or pt_r is None:
            return Point3DWithVisibility(x=0.0, y=0.0, z=0.0, visibility=0.0)
        if not (
            np.isfinite(pt_l.x)
            and np.isfinite(pt_l.y)
            and np.isfinite(pt_r.x)
            and np.isfinite(pt_r.y)
        ):
            return Point3DWithVisibility(x=0.0, y=0.0, z=0.0, visibility=0.0)

        raw = self.get_stereo_raw_params()
        if raw is None:
            return Point3DWithVisibility(x=0.0, y=0.0, z=0.0, visibility=0.0)

        K1, dist1, K2, dist2, R, T = raw
        P1 = np.hstack([np.eye(3), np.zeros((3, 1))]).astype(np.float64)
        P2 = np.hstack([R, T]).astype(np.float64)
        pts_l = np.array([[[pt_l.x, pt_l.y]]], dtype=np.float32)
        pts_r = np.array([[[pt_r.x, pt_r.y]]], dtype=np.float32)
        und_l = cv2.undistortPoints(pts_l, K1, dist1, P=None)
        und_r = cv2.undistortPoints(pts_r, K2, dist2, P=None)
        homog = cv2.triangulatePoints(P1, P2, und_l, und_r)
        w = homog[3, 0]
        if not np.isfinite(w) or abs(w) < 1e-9:
            return Point3DWithVisibility(x=0.0, y=0.0, z=0.0, visibility=0.0)
        x = homog[0, 0] / w
        y = homog[1, 0] / w
        z = homog[2, 0] / w
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)) or z <= 0:
            return Point3DWithVisibility(x=0.0, y=0.0, z=0.0, visibility=0.0)
        return Point3DWithVisibility(x=x, y=y, z=z, visibility=1.0)

    def triangulate_fitting_points(
        self,
        left_dict: Dict[str, Dict[str, List[Point2D]]],
        right_dict: Dict[str, Dict[str, List[Point2D]]],
    ) -> KeyCoordinates:
        """
        对左右同语义点逐点三角化，输出 KeyCoordinates。
        遍历 eye in EYE_TYPE, fitting_type in FITTING_TYPE，对应列表按索引一一配对；
        若左右数量不一致则取较小长度。坐标单位与 T 一致（mm）。
        非法三角化结果（可见性=0 或坐标非有限）会被保留为 visibility=0 的点。
        """
        key_coordinates = KeyCoordinates()
        total_triangulated = 0
        invalid_count = 0
        z_values: List[float] = []
        for eye in EYE_TYPE:
            for fitting_type in FITTING_TYPE:
                left_pts = left_dict.get(eye, {}).get(fitting_type, [])
                right_pts = right_dict.get(eye, {}).get(fitting_type, [])
                n = min(len(left_pts), len(right_pts))
                points: List[Point3DWithVisibility] = []
                for i in range(n):
                    pt = self.triangulate_one_point(left_pts[i], right_pts[i])
                    points.append(pt)
                    total_triangulated += 1
                    if pt.visibility <= 0:
                        invalid_count += 1
                    elif np.isfinite(pt.z):
                        z_values.append(pt.z)
                key_coordinates.set_points(eye, fitting_type, points)
        if z_values:
            logger.debug(
                "triangulate_fitting_points: triangulated point count=%d, invalid count=%d, z range=[%.2f, %.2f] mm",
                total_triangulated,
                invalid_count,
                min(z_values),
                max(z_values),
            )
        else:
            logger.debug(
                "triangulate_fitting_points: triangulated point count=%d, invalid count=%d, z range=N/A",
                total_triangulated,
                invalid_count,
            )
        return key_coordinates

    # ==================== 图像数据管理接口 ====================

    def add_frame(self, frame_id: int, bgr_image: np.ndarray) -> None:
        """添加一帧数据，仅存储原始 BGR 图像。"""
        if self._resolution is None:
            h, w = bgr_image.shape[:2]
            self._resolution = (w, h)
        self._data_dict[frame_id] = {
            "bgr_frame": bgr_image.copy(),
            "timestamp": datetime.now().isoformat(),
        }

    def get_image(self, frame_id: int) -> Optional[BGRImage]:
        """获取指定帧的 BGR 图像（整帧）。"""
        frame_data = self._data_dict.get(frame_id)
        if frame_data and "bgr_frame" in frame_data:
            return BGRImage(data=frame_data["bgr_frame"])
        logger.warning("获取帧数据失败: frame_id=%s", frame_id)
        return None

    def get_stereo_pair(self, frame_id: int) -> Optional[Tuple[BGRImage, BGRImage]]:
        """
        若该帧存在且分辨率为双目并排（STEREO_FRAME_SIZES），则拆分为左/右半幅并返回 (left, right)；
        否则返回 None。
        """
        frame_data = self._data_dict.get(frame_id)
        if not frame_data or "bgr_frame" not in frame_data:
            return None
        frame = frame_data["bgr_frame"]
        h, w = frame.shape[:2]
        if (w, h) not in STEREO_FRAME_SIZES:
            return None
        half = w // 2
        left = BGRImage(data=frame[:, :half].copy())
        right = BGRImage(data=frame[:, half:].copy())
        return (left, right)

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
            logger.info("RGB-D摄像头打开成功，分辨率 %dx%d", actual_w, actual_h)
            return cap
        except Exception as e:
            logger.error("打开RGB-D摄像头失败: %s", e, exc_info=True)
            return None

    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            "camera_name": "Auto Detected Camera",
            "camera_type": "RGB-D" if self._rgb_d else "RGB",
            "intrinsic_params": {"fx": 925.0, "fy": 925.0, "cx": 320.0, "cy": 240.0},
            "image_resolution": {"width": 640, "height": 480},
            "depth_scale": 0.001,
        }

    def _check_required_fields(self, config: Dict) -> Dict:
        """检查必需字段"""
        required_fields = ["camera_name", "camera_type", "intrinsic_params", "image_resolution", "depth_scale"]

        for field in required_fields:
            if field not in config:
                logger.warning("缺少必需字段: %s", field)
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


__all__ = ["CameraDataManager"]

