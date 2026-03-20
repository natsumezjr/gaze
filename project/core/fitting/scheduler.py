# 拟合调度器 - 事件驱动的拟合任务调度
import threading
from concurrent.futures import ThreadPoolExecutor

from project.core.fitting.main import FittingManager
from project.managers import CALLBACK_MANAGER, DATA_PIPELINE_MANAGER, FRAME_ID_MANAGER
from project.events import RECOGNITION_COMPLETE, FITTING_START, SYSTEM_STOP
from project.events.event_types import (
    CALIBRATION_POINT_SUBMIT,
    CALIBRATION_COMPLETE,
    CALIBRATION_START_REQUEST,
    ROUGH_GAZE_UPDATE,
    REQUEST_CALIBRATION_UI_CLOSE,
)
from project.core.fitting.kappa_calibrator import KAPPA_STORAGE, SAMPLES_STORAGE, build_samples_from_arrays, estimate_kappa
from project.core.fitting.fitting_strategy import fit_all_eyes
from project.config.screen_config import SCREEN_CONFIG
from project.data.data_models import Point3D, CalibrationRequest, CalibrationResponse
from project.data.data_models import EYE_TYPE
from project.events.event_types import GAZE_POINT_UPDATE
from project.config.logging_config import setup_logging

logger = setup_logging(__name__)


def _run_calibration_fitting_and_emit(frame_id: int, data):
    """在拟合池中执行标定路径的拟合，并发送 ROUGH_GAZE_UPDATE / GAZE_POINT_UPDATE（供调度器提交，避免阻塞识别线程）"""
    try:
        fitting_results = fit_all_eyes(data)
        for eye in EYE_TYPE:
            if eye not in fitting_results:
                continue
            result = fitting_results[eye]
            eyeball_center = Point3D.from_ndarray(result.parameters.center)
            pupil_points = data.get_coordinate_point(eye, "pupil")
            if not pupil_points:
                continue
            pupil_center = Point3D(pupil_points[0].x, pupil_points[0].y, pupil_points[0].z)
            intersection = SCREEN_CONFIG.calculate_gaze_intersection(pupil_center, eyeball_center)
            if intersection:
                CALLBACK_MANAGER.emit(GAZE_POINT_UPDATE, point=intersection, color="#0000FF")
                calibration_request = CalibrationRequest(
                    frame_id=frame_id,
                    eye_type=eye,
                    intersection=intersection,
                )
                CALLBACK_MANAGER.emit(ROUGH_GAZE_UPDATE, calibration_request=calibration_request)
    except Exception as e:
        logger.error(f"标定路径拟合或发送事件失败: {e}", exc_info=True)


class FittingScheduler:
    """事件驱动的拟合调度器"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, thread_pool_executor: ThreadPoolExecutor):
        self.thread_pool_executor = thread_pool_executor
        self.running = False
        self._setup_event_handlers()

    def _setup_event_handlers(self):
        """设置事件处理器"""
        CALLBACK_MANAGER.register(RECOGNITION_COMPLETE, self._on_recognition_complete)
        CALLBACK_MANAGER.register(SYSTEM_STOP, self._on_system_stop)
        CALLBACK_MANAGER.register(CALIBRATION_POINT_SUBMIT, self._on_calibration_point_submit)
        CALLBACK_MANAGER.register(CALIBRATION_COMPLETE, self._on_calibration_complete)

    def _on_recognition_complete(self, frame_id: int):
        """识别完成事件处理 - 自动触发拟合"""
        if not self.running:
            return

        try:
            data = DATA_PIPELINE_MANAGER.get_recognition_data(frame_id)
            if data is None:
                logger.warning(f"frame_id {frame_id} 的数据为空，跳过拟合")
                return

            recognized_ids = FRAME_ID_MANAGER.get_recognized_frame_ids()
            if frame_id in recognized_ids:
                if FRAME_ID_MANAGER.remove_recognized_frame_id(frame_id):
                    FRAME_ID_MANAGER.add_fitting_frame_id(frame_id)

            if not KAPPA_STORAGE.is_kappa_valid():
                logger.info(f"kappa 未有效，发送标定启动请求，frame_id: {frame_id}")
                CALLBACK_MANAGER.emit(CALIBRATION_START_REQUEST, frame_id=frame_id)
                self.thread_pool_executor.submit(
                    _run_calibration_fitting_and_emit,
                    frame_id=frame_id,
                    data=data,
                )
                return

            logger.info(f"收到识别完成事件，开始拟合 frame_id: {frame_id}")
            fitting_manager = FittingManager(data)
            self.thread_pool_executor.submit(fitting_manager.run)
            CALLBACK_MANAGER.emit(FITTING_START, frame_id)

        except Exception as e:
            logger.error(f"处理识别完成事件时出错: {e}", exc_info=True)

    def _on_calibration_point_submit(self, calibration_response: CalibrationResponse):
        """处理标定点提交事件"""
        try:
            frame_id = calibration_response.frame_id
            eye_type = calibration_response.eye_type

            data = DATA_PIPELINE_MANAGER.get_recognition_data(frame_id)
            if data is None:
                logger.warning(f"无法获取 frame_id {frame_id} 的数据，跳过标定点保存")
                return

            fitting_results = fit_all_eyes(data)
            if eye_type not in fitting_results:
                logger.warning(f"eye_type {eye_type} 不在拟合结果中，跳过")
                return

            result = fitting_results[eye_type]
            eyeball_center = Point3D.from_ndarray(result.parameters.center)
            pupil_points = data.get_coordinate_point(eye_type, "pupil")
            if not pupil_points:
                logger.warning(f"无法获取 {eye_type} 眼的 pupil 数据，跳过")
                return

            pupil_center = Point3D(pupil_points[0].x, pupil_points[0].y, pupil_points[0].z)
            SAMPLES_STORAGE.append(frame_id, eye_type, pupil_center, eyeball_center, calibration_response.target_pixel)

        except Exception as e:
            logger.error(f"处理标定点提交事件时出错: {e}", exc_info=True)

    def _on_calibration_complete(self, calibration_result: dict):
        """处理标定完成事件"""
        try:
            logger.info("开始处理标定完成事件，估计 kappa...")
            all_samples = SAMPLES_STORAGE.get_all_samples()

            for eye in ["left", "right"]:
                pupils_list = []
                eyeball_list = []
                pixel_list = []

                for frame_id, frame_data in all_samples.items():
                    if eye in frame_data:
                        pupils_list.extend(frame_data[eye]["pupils"])
                        eyeball_list.extend(frame_data[eye]["eyeball"])
                        pixel_list.extend(frame_data[eye]["pixel"])

                if len(pupils_list) == 0:
                    logger.warning(f"{eye} 眼没有标定数据，跳过")
                    continue

                target_points = []
                for i, pixel in enumerate(pixel_list):
                    if i < len(eyeball_list):
                        target_3d = SCREEN_CONFIG.pixel_to_3d_intersection(pixel, eyeball_list[i])
                        if target_3d:
                            target_points.append(target_3d)

                if len(target_points) != len(pupils_list):
                    logger.warning(f"{eye} 眼数据不匹配，跳过")
                    continue

                samples = build_samples_from_arrays(eyeball_list, pupils_list, target_points)
                kappa, result = estimate_kappa(samples)
                KAPPA_STORAGE.set_kappa(eye, kappa)
                logger.info(f"{eye} 眼 kappa 估计完成: {result}")

            SAMPLES_STORAGE.clear()
            logger.info("标定完成，kappa 已保存")
            CALLBACK_MANAGER.emit(REQUEST_CALIBRATION_UI_CLOSE)
            logger.info("已发送标定 UI 关闭请求")

        except Exception as e:
            logger.error(f"处理标定完成事件时出错: {e}", exc_info=True)

    def _on_system_stop(self):
        """系统停止事件处理"""
        self.running = False
        CALLBACK_MANAGER.unregister(RECOGNITION_COMPLETE, self._on_recognition_complete)
        CALLBACK_MANAGER.unregister(SYSTEM_STOP, self._on_system_stop)
        CALLBACK_MANAGER.unregister(CALIBRATION_POINT_SUBMIT, self._on_calibration_point_submit)
        CALLBACK_MANAGER.unregister(CALIBRATION_COMPLETE, self._on_calibration_complete)
        logger.info("拟合调度器已停止（线程池由主线程关闭）")

    def start(self):
        """启动调度器"""
        self.running = True
        logger.info("事件驱动拟合调度器已启动")
