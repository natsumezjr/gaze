from project.data.data_manager import CameraDataManager
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
logger = setup_logging(__name__)



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
        self.face_detector = FaceDetector(self.camera_manager.get_camera_params(), rgb_d=rgb_d)
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
    
    def _handle_warning(self):
        """警告时的睡眠处理"""
        pass
    
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
    
    def _check_data_availability(self, image, depth_map):
        """检查数据可用性"""
        if not (image and depth_map):
            raise RecognitionWarning("无法获取图像或深度图", "DATA_UNAVAILABLE", sleep_duration=self.interval)
    
    def _check_face_detection(self, image, depth_map):
        """检查人脸检测"""
        if not self.face_detector.detect_face(image, depth_map):
            raise RecognitionWarning("人脸检测失败", "FACE_DETECTION_FAILED", sleep_duration=self.interval)
    
    def run(self):
        """主运行循环"""
        self.running = True
        
        try:
            self._check_camera_initialization()
            self.cap = self.camera_manager.get_cap()
            
        except RecognitionError as e:
            logger.error(f"摄像头初始化失败: {e}", exc_info=True)
            self.error_handler.handle_error(e)
            return
    

        while self.running:
            try:
                frame_id = self._cycle_start()
                ret, frame = self.cap.read()
                self._check_frame_reading(ret, frame)
                
                self.camera_manager.add_frame(frame_id, frame)
                
                image = self.camera_manager.get_image(frame_id)
                depth_map = self.camera_manager.get_depth(frame_id)
                self._check_data_availability(image, depth_map)
                
                # 尝试检测人脸，但即使失败也继续显示摄像头画面
                self._check_face_detection(image, depth_map)
                logger.info("人脸检测成功")
                key_coordinates = self.face_detector.get_fitting_data()

                recg_fit_data_manager = RecgFitDataManager(key_coordinates, debug_log=True)
                self.data_pipeline.store_recognition_data(frame_id, recg_fit_data_manager)
                
                # 可视化关键点并显示
                if frame is not None:
                    visualized_frame = self.visualizer.draw_keypoints(frame, key_coordinates)
                    cv2.imshow('眼动追踪系统 - 摄像头画面', visualized_frame)
                    cv2.waitKey(1)  # 非阻塞等待，允许其他处理继续
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
        
        