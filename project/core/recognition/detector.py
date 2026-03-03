# 人脸检测模块 - 仅拟合点 2D→3D，无深度图
import numpy as np
from typing import Dict, List, Optional, TYPE_CHECKING
from project.data.data_models import (
    BGRImage, Landmark, KeyCoordinates, Point3DWithVisibility,
    FITTING_TYPE, EYE_TYPE, FITTING_LANDMARK_INDICES,
)

# 配置日志
from project.config.logging_config import setup_logging
logger = setup_logging(__name__)

# 当使用邻域中值或默认深度 fallback 时，用 MediaPipe 相对 z 做微调：Z_final = Z + K_MEDIAPIPE_Z_SCALE * z_mediapipe
K_MEDIAPIPE_Z_SCALE_MM = 20.0
# 无有效深度时使用的低可见性，避免“500mm 平面墙”被当作高置信度
FALLBACK_VISIBILITY = 0.2

if TYPE_CHECKING:
    from project.data.data_manager import CameraDataManager


class FaceDetector:
    """人脸检测器 - 仅校验拟合点有效，2D→3D 依赖 CameraDataManager（单位 mm）"""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(FaceDetector, cls).__new__(cls)
        return cls._instance

    def __init__(self, camera_data_manager: "CameraDataManager"):
        """初始化：仅接受 CameraDataManager，内参与 pixel_to_xy_mm 由其提供。"""
        if camera_data_manager is None:
            raise ValueError("camera_data_manager 不能为 None")
        self._camera_data_manager = camera_data_manager
        self._bgr_image: Optional[BGRImage] = None
        self._landmarks: List[Landmark] = []
        self._key_coordinates: Optional[KeyCoordinates] = None
        self._detection_success = False
        self._current_frame_id: Optional[int] = None
        logger.info("FaceDetector 初始化完成（依赖 CameraDataManager）")

    def detect_face(self, bgr_image: BGRImage, frame_id: Optional[int] = None) -> bool:
        """检测：提取 478 关键点，仅对拟合点做 2D→3D；当且仅当所有拟合点存在且 x,y,z 非 nan 时返回 True。
        frame_id 非空且该帧有立体深度时使用每像素深度，否则使用 DEFAULT_FACE_DEPTH_MM。"""
        try:
            if not isinstance(bgr_image, BGRImage) or bgr_image.data is None:
                raise ValueError("bgr_image 必须是有效的 BGRImage")
            self._bgr_image = bgr_image
            self._current_frame_id = frame_id

            from project.core.recognition.landmark_extractor import extract_landmarks

            landmarks = extract_landmarks(bgr_image)
            if not landmarks:
                logger.warning("未提取到关键点")
                self._set_fail()
                return False

            logger.info(f"提取到 {len(landmarks)} 个关键点")
            key_coordinates, all_valid = self._fitting_points_2d_to_3d(landmarks)
            if not all_valid:
                logger.warning("拟合点未全部有效（缺失或 x/y/z 为 nan）")
                self._set_fail()
                return False

            self._landmarks = landmarks
            self._key_coordinates = key_coordinates
            self._detection_success = True
            logger.info("人脸检测成功，拟合点全部有效")
            return True

        except Exception as e:
            logger.error(f"人脸检测失败: {e}", exc_info=True)
            self._set_fail()
            return False

    def _fitting_points_2d_to_3d(self, landmarks: List[Landmark]) -> tuple:
        """
        仅对拟合点做 2D→3D（单位 mm）。返回 (KeyCoordinates, all_valid)。
        深度优先级：单点深度 -> 5×5 邻域中值 -> 默认深度；后两种时 visibility=0.2 并可选融合 MediaPipe z。
        """
        key_coordinates = KeyCoordinates()
        default_z_mm = getattr(
            self._camera_data_manager,
            "DEFAULT_FACE_DEPTH_MM",
            500,
        )
        indices = FITTING_LANDMARK_INDICES
        all_valid = True

        for eye in EYE_TYPE:
            for fitting_type in FITTING_TYPE:
                landmark_indices = indices[eye][fitting_type]
                points = []
                for idx in landmark_indices:
                    if idx >= len(landmarks):
                        all_valid = False
                        continue
                    lm = landmarks[idx]
                    z_mm = None
                    used_fallback = False
                    if self._current_frame_id is not None:
                        z_mm = self._camera_data_manager.get_depth_at_pixel(
                            self._current_frame_id, lm.x, lm.y
                        )
                        if z_mm is None:
                            z_mm = self._camera_data_manager.get_depth_neighborhood_median(
                                self._current_frame_id, lm.x, lm.y
                            )
                            used_fallback = z_mm is not None
                        if z_mm is None:
                            z_mm = default_z_mm
                            used_fallback = True
                    else:
                        z_mm = default_z_mm
                        used_fallback = True
                    if used_fallback:
                        z_mm = z_mm + K_MEDIAPIPE_Z_SCALE_MM * lm.z
                    visibility = FALLBACK_VISIBILITY if used_fallback else lm.visibility
                    x_mm, y_mm = self._camera_data_manager.pixel_to_xy_mm(lm.x, lm.y, z_mm)
                    if not (np.isfinite(x_mm) and np.isfinite(y_mm)):
                        all_valid = False
                    pt = Point3DWithVisibility(
                        x=x_mm, y=y_mm, z=z_mm,
                        visibility=visibility,
                    )
                    if not (np.isfinite(pt.x) and np.isfinite(pt.y) and np.isfinite(pt.z)):
                        all_valid = False
                    points.append(pt)
                key_coordinates.set_points(eye, fitting_type, points)

        return key_coordinates, all_valid

    def _set_fail(self) -> None:
        self._detection_success = False
        self._landmarks = []
        self._key_coordinates = None

    def get_fitting_data(self) -> KeyCoordinates:
        """获取拟合所需关键点（mm）。未检测成功时返回空 KeyCoordinates。"""
        if not self._detection_success or not self._landmarks:
            logger.warning("检测未成功或无有效数据")
            return KeyCoordinates()
        if self._key_coordinates is not None:
            return self._key_coordinates
        key_coordinates, _ = self._fitting_points_2d_to_3d(self._landmarks)
        return key_coordinates

    def get_landmarks(self) -> List[Landmark]:
        """获取原始关键点数据"""
        return self._landmarks.copy() if self._landmarks else []

    def is_detection_successful(self) -> bool:
        return self._detection_success

    def get_detection_info(self) -> Dict:
        return {
            "detection_success": self._detection_success,
            "landmark_count": len(self._landmarks) if self._landmarks else 0,
            "has_bgr_image": self._bgr_image is not None,
        }


def test_face_detector() -> None:
    """测试：使用 CameraDataManager，仅传 image，无深度图。"""
    print("=== 人脸检测器测试 ===")
    from project.data.data_manager import CameraDataManager
    from project.data.data_models import BGRImage

    camera_manager = CameraDataManager(rgb_d=False)
    detector = FaceDetector(camera_manager)
    bgr_image = BGRImage(data=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))

    success = detector.detect_face(bgr_image)
    print(f"检测结果: {'成功' if success else '失败'}")
    if success:
        key_coords = detector.get_fitting_data()
        print(f"拟合数据: {key_coords}")
        landmarks = detector.get_landmarks()
        print(f"关键点数量: {len(landmarks)}")
        info = detector.get_detection_info()
        print(f"检测信息: {info}")
    print("=== 测试完成 ===")


if __name__ == "__main__":
    test_face_detector()
