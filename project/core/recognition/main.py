from project.data.data_manager import CameraDataManager
from project.core.recognition.detector import FaceDetector
import logging
from time import sleep
import cv2
from project.managers import SEMAPHORE_MANAGER, SemaphoreManager, FRAME_ID_MANAGER, FrameIdManager, DATA_PIPELINE_MANAGER
from project.data.data_manager import RecgFitDataManager
# 配置日志
from project.config.logging_config import setup_logging, get_logger
setup_logging()
logger = get_logger(__name__)


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
    def __init__(self, semaphore_manager: SemaphoreManager = SEMAPHORE_MANAGER, frame_id_manager: FrameIdManager = FRAME_ID_MANAGER, rgb_d=False):
        self.rgb_d = rgb_d
        self.camera_manager = CameraDataManager(rgb_d=rgb_d)
        self.face_detector = FaceDetector(self.camera_manager.get_camera_params(), rgb_d=rgb_d)
        self.running = False
        self.interval = 0.1
        self.error_handler = ErrorHandler()
        self.semaphore_manager = semaphore_manager
        self.frame_id_manager = frame_id_manager
        self.data_pipeline = DATA_PIPELINE_MANAGER
        # 设置回调函数
        self.error_handler.set_callbacks(
            on_warning=self._handle_warning,
            on_error=self._handle_error
        )
    
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
            if not self.camera_manager._initialize_camera():
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
        logger.debug("RecognitionManager.run() 开始执行")
        self.running = True
        
        try:
            logger.debug("检查摄像头初始化...")
            self._check_camera_initialization()
            logger.debug("摄像头初始化检查完成")
            self.cap = self.camera_manager.get_cap()
            logger.debug("获取摄像头对象成功")
        except RecognitionError as e:
            logger.error(f"摄像头初始化失败: {e}")
            self.error_handler.handle_error(e)
            return
    

        logger.debug("开始主循环...")

        while self.running:
            try:
                logger.debug("开始新的识别周期...")
                frame_id = self._cycle_start()
                logger.debug(f"读取帧，frame_id: {frame_id}")
                ret, frame = self.cap.read()
                self._check_frame_reading(ret, frame)
                
                # 显示摄像头画面
                if frame is not None:
                    cv2.imshow('眼动追踪系统 - 摄像头画面', frame)
                    cv2.waitKey(1)  # 非阻塞等待，允许其他处理继续
                
                self.camera_manager.add_frame(frame_id, frame)
                
                image = self.camera_manager.get_image(frame_id)
                depth_map = self.camera_manager.get_depth(frame_id)
                
                self._check_data_availability(image, depth_map)
                self._check_face_detection(image, depth_map)
                
                logger.info("人脸检测成功")
                key_coordinates = self.face_detector.get_fitting_data()

                recg_fit_data_manager = RecgFitDataManager(key_coordinates, debug_log=True)
                self.data_pipeline.store_recognition_data(frame_id, recg_fit_data_manager)
                

                
                
            except RecognitionWarning as w:
                if not self.error_handler.handle_warning(w):
                    self.running = False
                    break          
            except RecognitionError as e:
                if not self.error_handler.handle_error(e):
                    self.running = False
                    break
            finally:
                self._cycle_update()


            
    def stop(self):
        """停止识别"""
        self.running = False
        logger.info("识别线程主循环停止")
        self.cap.release()
        logger.info("摄像头资源已释放")
        cv2.destroyAllWindows()
        logger.info("OpenCV窗口已关闭")
    
    def restart(self):
        """重启识别"""
        self.running = True
        self.run()
        
    def _cycle_update(self) -> None:
        """循环更新"""
        frame_id = self.frame_id_manager.get_recognizing_frame_id()
        self.frame_id_manager.add_recognized_frame_id(frame_id)
        logger.debug(f"添加已识别frame_id: {frame_id}")
        self.frame_id_manager.increment_recognizing_frame_id()
        self.semaphore_manager.signal_recognition_complete()
        sleep(self.interval)
        
    def _cycle_start(self) -> int:
        """循环开始"""
        return self.frame_id_manager.get_recognizing_frame_id()
        
        