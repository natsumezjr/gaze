from project.data.data_manager import CameraDataManager
from project.data.data_models import Landmark, EYE_TYPE, FITTING_TYPE
from project.core.recognition.detector import FaceDetector
import logging
from time import sleep
import cv2
from project.managers import FRAME_ID_MANAGER, FrameIdManager, DATA_PIPELINE_MANAGER, CALLBACK_MANAGER
from project.data.data_manager import RecgFitDataManager
from project.core.visualization import KeypointVisualizer
from project.events import RECOGNITION_COMPLETE
# 配置日志
from project.config.logging_config import setup_logging
logger = setup_logging(__name__, logging.DEBUG)



class RecognitionException(Exception):
    """识别异常"""
    def __init__(self, message, error_code=None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class RecognitionWarning(RecognitionException):
    """警告级别异常 - 可恢复的问题"""
    def __init__(self, message, error_code=None, retry_count=0, sleep_duration=None):
        super().__init__(message, error_code)
        self.retry_count = retry_count
        self.severity = "WARNING"
        self.sleep_duration = sleep_duration  # 可选的睡眠时间

class RecognitionError(RecognitionException):
    """错误级别异常 - 严重问题，需要停止处理"""
    def __init__(self, message, error_code=None, should_stop=True):
        super().__init__(message, error_code)
        self.severity = "ERROR"
        self.should_stop = should_stop  # 是否应该停止执行

class ErrorHandler:
    """错误处理策略 - 通过回调函数处理具体行为"""
    def __init__(self, max_warnings=10, max_retries=3):
        self.max_warnings = max_warnings
        self.max_retries = max_retries
        self.warning_count = 0
        self.retry_count = 0
        
        # 回调函数 - 由外部设置具体行为
        self.on_warning = None  # 警告时的睡眠回调
        self.on_error = None     # 错误时的停止回调
    
    def set_callbacks(self, on_warning=None, on_error=None):
        """设置回调函数"""
        self.on_warning = on_warning
        self.on_error = on_error
    
    def handle_warning(self, warning: RecognitionWarning) -> bool:
        """处理警告，返回是否继续执行"""
        self.warning_count += 1
        logger.warning(f"警告 {self.warning_count}/{self.max_warnings}: {warning.message}")
        
        if self.warning_count >= self.max_warnings:
            logger.error(f"警告次数超过限制 ({self.max_warnings})，停止执行")
            return False
        
        # 执行警告回调（如睡眠）
        if self.on_warning and warning.sleep_duration:
            # #region agent log
            try:
                open(r"d:\my_projects\gaze\.cursor\debug.log", "a").write(
                    __import__("json").dumps({"id": "log_warn_cb", "timestamp": __import__("time").time() * 1000, "location": "main.py:on_warning", "message": "calling on_warning with 1 arg", "data": {"sleep_duration": warning.sleep_duration}, "hypothesisId": "A"}) + "\n"
                )
            except Exception:
                pass
            # #endregion
            self.on_warning(warning.sleep_duration)
        
        if self.retry_count < self.max_retries:
            self.retry_count += 1
            logger.info(f"尝试重试 ({self.retry_count}/{self.max_retries})")
            return True
        
        return True
    
    def handle_error(self, error: RecognitionError) -> bool:
        """处理错误，返回是否继续执行"""
        logger.error(f"严重错误: {error.message}")
        
        # 执行错误回调（如停止运行）
        if self.on_error and error.should_stop:
            self.on_error()
        
        return False
    
    def reset_warnings(self):
        """重置警告计数"""
        self.warning_count = 0
        self.retry_count = 0

class RecognitionManager:
    def __init__(self, frame_id_manager: FrameIdManager = FRAME_ID_MANAGER, rgb_d=True):
        self.rgb_d = rgb_d
        self.camera_manager = CameraDataManager(rgb_d=rgb_d)
        self.face_detector = FaceDetector(self.camera_manager)
        self.running = False
        self.interval = 0.1
        self.error_handler = ErrorHandler()
        self.frame_id_manager = frame_id_manager
        self.data_pipeline = DATA_PIPELINE_MANAGER
        # 初始化可视化器
        self.visualizer = KeypointVisualizer(self.camera_manager.get_camera_params())
        # 设置回调函数
        self.error_handler.set_callbacks(
            on_warning=self._handle_warning,
            on_error=self._handle_error
        )
        self._setup_event_handlers()
        
    def _setup_event_handlers(self):
        """设置事件处理器"""
        from project.events import SYSTEM_STOP
        CALLBACK_MANAGER.register(SYSTEM_STOP, self._on_system_stop)
    
    def _cleanup_event_handlers(self):
        """清理事件处理器"""
        from project.events import SYSTEM_STOP
        CALLBACK_MANAGER.unregister(SYSTEM_STOP, self._on_system_stop)
    
    def _handle_warning(self, sleep_duration=None):
        """警告时的睡眠处理 - 接受 ErrorHandler 传入的 sleep_duration"""
        # #region agent log
        try:
            open(r"d:\my_projects\gaze\.cursor\debug.log", "a").write(
                __import__("json").dumps({"id": "log_handle_warn", "timestamp": __import__("time").time() * 1000, "location": "main.py:_handle_warning", "message": "callback received", "data": {"sleep_duration": sleep_duration}, "hypothesisId": "B"}) + "\n"
            )
        except Exception:
            pass
        # #endregion
        if sleep_duration is not None and sleep_duration > 0:
            sleep(sleep_duration)
    
    def _handle_error(self):
        """错误时的停止处理"""
        self.running = False
    
    def _check_camera_initialization(self):
        """检查摄像头初始化"""
        self.cap = self.camera_manager.get_cap()
        if self.cap is None or not self.cap.isOpened():
            # 如果摄像头对象无效，尝试重新初始化
            if not self.camera_manager.initialize_camera():
                raise RecognitionError("摄像头初始化失败", "CAM_INIT_FAILED")
    
    def _check_frame_reading(self, ret, frame):
        """检查帧读取"""
        if not ret:
            raise RecognitionWarning("无法读取摄像头数据", "FRAME_READ_FAILED", sleep_duration=self.interval)
    
    def _check_data_availability(self, image):
        """检查数据可用性（仅图像，不依赖深度图）"""
        if not image:
            raise RecognitionWarning("无法获取图像", "DATA_UNAVAILABLE", sleep_duration=self.interval)

    def _check_face_detection(self, image, frame_id=None):
        """检查人脸检测（仅图像，拟合点全部有效则通过）。传入 frame_id 时使用该帧立体深度作为 z。"""
        if not self.face_detector.detect_face(image, frame_id=frame_id):
            raise RecognitionWarning("人脸检测失败", "FACE_DETECTION_FAILED", sleep_duration=self.interval)

    @staticmethod
    def _scale_landmarks_for_display(landmarks, src_w: int, src_h: int, dst_w: int, dst_h: int):
        """将原始图像坐标下的关键点缩放到显示尺寸，用于与 resize 后的画面对齐"""
        if not landmarks or src_w <= 0 or src_h <= 0:
            return landmarks
        sx, sy = dst_w / src_w, dst_h / src_h
        return [Landmark(x=lm.x * sx, y=lm.y * sy, z=lm.z, visibility=lm.visibility) for lm in landmarks]

    @staticmethod
    def _show_frame(win_name: str, image, display_params_logged: list, frame_to_show=None,
                    max_display_w: int = 1280, show: bool = True):
        """
        按固定比例得到显示帧与尺寸，可选地执行一次日志、resizeWindow、imshow、waitKey(1)。
        frame_to_show 不为 None 时显示该帧（如叠加关键点后的图），否则显示由 image 生成的帧。
        show=False 时只计算并返回 (frame_display, display_w, display_h, w, h)，不 imshow。
        返回 (frame_display, display_w, display_h, w, h)。
        """
        frame_for_draw = image.data if hasattr(image, "data") else image
        h, w = frame_for_draw.shape[:2]
        if w > max_display_w:
            scale = max_display_w / w
            display_w, display_h = max_display_w, int(round(h * scale))
            frame_display = cv2.resize(frame_for_draw, (display_w, display_h), interpolation=cv2.INTER_LINEAR)
        else:
            display_w, display_h = w, h
            frame_display = frame_for_draw
        if show:
            if not display_params_logged[0]:
                logger.info(
                    "[显示参数] 图像尺寸 h=%d w=%d 显示尺寸 %dx%d 宽高比=%.3f (窗口将固定为该尺寸)",
                    h, w, display_w, display_h, display_w / display_h if display_h else 0,
                )
                display_params_logged[0] = True
            try:
                cv2.resizeWindow(win_name, display_w, display_h)
            except cv2.error:
                pass
            cv2.imshow(win_name, frame_to_show if frame_to_show is not None else frame_display)
            cv2.waitKey(1)
        return frame_display, display_w, display_h, w, h

    def run(self, warmup_frames: int = 20, max_frames: int = None):
        """主运行循环"""
        self.running = True
        
        try:
            self._check_camera_initialization()
            self.cap = self.camera_manager.get_cap()
            
        except RecognitionError as e:
            logger.error(f"摄像头初始化失败: {e}", exc_info=True)
            self.error_handler.handle_error(e)
            return
    

        _win_name = "Gaze-Camera"  # ASCII 标题避免 Windows 下中文乱码
        cv2.namedWindow(_win_name, cv2.WINDOW_NORMAL)
        _display_params_logged = [False]
        frames_processed = 0
        while self.running:
            try:
                frame_id = self._cycle_start()
                ret, frame = self.cap.read()
                self._check_frame_reading(ret, frame)
                frames_processed += 1
                if frames_processed < warmup_frames:
                    continue
                if max_frames is not None and frames_processed >= max_frames:
                    break
                self.camera_manager.add_frame(frame_id, frame)
                
                image = self.camera_manager.get_image(frame_id)
                self._check_data_availability(image)
                self._show_frame(_win_name, image, _display_params_logged)
                # 尝试检测人脸（仅图像，拟合点全部有效则通过）；传入 frame_id 以使用立体深度
                self._check_face_detection(image, frame_id)
                logger.info("人脸检测成功")
                key_coordinates = self.face_detector.get_fitting_data()
                # 坐标 debug：拟合点 3D (mm) 与可见性，仅 debug 级别输出
                for eye in EYE_TYPE:
                    for fitting_type in FITTING_TYPE:
                        points = key_coordinates.get_points(eye, fitting_type)
                        for idx, pt in enumerate(points):
                            logger.debug(
                                "坐标 frame_id=%s eye=%s type=%s idx=%d x=%.3f y=%.3f z=%.3f v=%.3f",
                                frame_id, eye, fitting_type, idx, pt.x, pt.y, pt.z, pt.visibility,
                            )

                recg_fit_data_manager = RecgFitDataManager(key_coordinates, debug_log=True)
                self.data_pipeline.store_recognition_data(frame_id, recg_fit_data_manager)
                # 检测成功后用 2D 关键点绘制叠加层（显示与上面保持同一缩放）
                frame_display, display_w, display_h, w, h = self._show_frame(
                    _win_name, image, _display_params_logged, show=False
                )
                landmarks_2d = self.face_detector.get_landmarks()
                scaled = self._scale_landmarks_for_display(landmarks_2d, w, h, display_w, display_h)
                visualized_frame = self.visualizer.draw_2d_fitting_landmarks(frame_display, scaled)
                self._show_frame(_win_name, image, _display_params_logged, frame_to_show=visualized_frame)
                self._cycle_update()
                
            except KeyboardInterrupt:
                raise KeyboardInterrupt
                
            except RecognitionWarning as w:
                # 处理其他警告（如帧读取失败等），但继续循环
                self.error_handler.handle_warning(w)
                continue
            except RecognitionError as e:
                logger.error(f"识别模块严重错误: {e}", exc_info=True)
                if not self.error_handler.handle_error(e):
                    self.running = False
                    break
            except Exception as e:
                logger.error(f"识别循环中发生未预期的异常: {e}", exc_info=True)
                # 继续循环，不终止
                continue
        # 识别线程退出时清理资源（在识别线程中执行，避免阻塞事件处理器）
        logger.info("识别线程主循环停止")
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("摄像头资源已释放")
        cv2.destroyAllWindows()
        logger.info("OpenCV窗口已关闭")
            
    def _on_system_stop(self):
        """系统停止事件处理 - 只设置标志，资源清理在识别线程退出时执行"""
        self.running = False
        # 只注销事件处理器，资源清理在识别线程退出时执行（避免阻塞事件处理链）
        self._cleanup_event_handlers()
        logger.info("识别模块停止标志已设置（资源清理将在识别线程退出时执行）")
    
    def restart(self):
        """重启识别"""
        self.running = True
        self.run()
        
    def _cycle_update(self) -> None:
        """循环更新 - 发送事件而不是信号量"""
        self.error_handler.reset_warnings()
        frame_id = self.frame_id_manager.get_recognizing_frame_id()
        self.frame_id_manager.add_recognized_frame_id(frame_id)
        # 发送识别完成事件（非阻塞）
        CALLBACK_MANAGER.emit(RECOGNITION_COMPLETE, frame_id)
        
        self.frame_id_manager.generate_recognizing_frame_id()
        # 使用可中断的sleep，检查running标志
        elapsed = 0
        while elapsed < self.interval and self.running:
            sleep(min(0.01, self.interval - elapsed))  # 每10ms检查一次
            elapsed += 0.01
        
    def _cycle_start(self) -> int:
        """循环开始"""
        return self.frame_id_manager.get_recognizing_frame_id()
        
if __name__ == "__main__":
    recognition_manager = RecognitionManager()
    recognition_manager.run(warmup_frames=20, max_frames=30)