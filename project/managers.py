# 全局管理器模块
# 包含系统中使用的各种全局管理器，避免循环依赖
import threading
import time
from project.data.data_manager import RecgFitDataManager
from project.config.logging_config import setup_logging, get_logger
from project.config.settings import FRAME_ID_PERIOD_SECONDS, FRAME_ID_CLEANUP_ENABLED
setup_logging()
logger = get_logger(__name__)

class FrameIdManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._lock = threading.Lock()  # 添加锁保护
            # 加载配置
            self.period_seconds = FRAME_ID_PERIOD_SECONDS
            self.cleanup_enabled = FRAME_ID_CLEANUP_ENABLED
            # 初始化 frame_id（使用微秒时间戳）
            self.recognizing_frame_id = self._generate_frame_id()
            self.recognized_frame_ids = []
            self.fitting_frame_ids = []
            self.fitted_frame_ids = []
            self.tracking_frame_id = self._generate_frame_id()
            self.tracked_frame_ids = []
            # 清理相关
            self._last_cleanup_time = None
            self._initialized = True
        
    def _generate_frame_id(self) -> int:
        """生成微秒级时间戳作为 frame_id"""
        return int(time.time() * 1_000_000)
    
    def get_recognizing_frame_id(self):
        with self._lock:
            return self.recognizing_frame_id
    
    def generate_recognizing_frame_id(self):
        """生成新的识别帧ID（微秒时间戳）"""
        with self._lock:
            self.recognizing_frame_id = self._generate_frame_id()
        
    def choose_recognized_to_fitting(self) -> int:
        with self._lock:
            if not self.recognized_frame_ids:
                return None
            frame_id = self.recognized_frame_ids.pop(0)  # 使用pop(0)获取第一个元素
            self.fitting_frame_ids.append(frame_id)
            return frame_id
    
    def get_recognized_frame_ids(self):
        with self._lock:
            return self.recognized_frame_ids.copy()
    
    def add_recognized_frame_id(self, frame_id):
        with self._lock:
            self.recognized_frame_ids.append(frame_id)
            self._check_and_cleanup()
    
    def remove_recognized_frame_id(self, frame_id):
        """从已识别列表中移除指定的 frame_id"""
        with self._lock:
            if frame_id in self.recognized_frame_ids:
                self.recognized_frame_ids.remove(frame_id)
                logger.debug(f"从已识别列表中移除 frame_id: {frame_id}")
                return True
            return False
    
    def get_fitting_frame_ids(self):
        with self._lock:
            return self.fitting_frame_ids.copy()
    
    def add_fitting_frame_id(self, frame_id):
        with self._lock:
            self.fitting_frame_ids.append(frame_id)
            self._check_and_cleanup()
    
    def get_fitted_frame_ids(self):
        with self._lock:
            return self.fitted_frame_ids.copy()
    
    def add_fitted_frame_id(self, frame_id):
        with self._lock:
            self.fitted_frame_ids.append(frame_id)
            self._check_and_cleanup()
    
    def get_tracking_frame_id(self):
        with self._lock:
            return self.tracking_frame_id
    
    def increment_tracking_frame_id(self):
        """生成新的追踪帧ID（微秒时间戳）"""
        with self._lock:
            self.tracking_frame_id = self._generate_frame_id()
    
    def get_tracked_frame_ids(self):
        with self._lock:
            return self.tracked_frame_ids.copy()
    
    def add_tracked_frame_id(self, frame_id):
        with self._lock:
            self.tracked_frame_ids.append(frame_id)
            self._check_and_cleanup()
    
    def _cleanup_list(self, list_name: str, threshold: int):
        """清理列表中时间戳小于阈值的元素（最旧的半个周期）"""
        frame_list = getattr(self, list_name)
        
        # 由于 frame_id 是时间戳，可以直接比较
        # 保留时间戳 >= threshold 的元素（保留新的半个周期）
        original_count = len(frame_list)
        frame_list[:] = [fid for fid in frame_list if fid >= threshold]
        removed_count = original_count - len(frame_list)
        
        if removed_count > 0:
            logger.debug(f"清理 {list_name}: 移除 {removed_count} 个旧数据，保留 {len(frame_list)} 个")
    
    def _cleanup_old_data(self):
        """清理半个周期的旧数据，并同步清理其他管理器"""
        if not self.cleanup_enabled:
            return
        
        current_time = self._generate_frame_id()
        period_microseconds = int(self.period_seconds * 1_000_000)
        half_period = period_microseconds // 2
        cleanup_threshold = current_time - half_period
        
        # 清理各个列表
        self._cleanup_list('recognized_frame_ids', cleanup_threshold)
        self._cleanup_list('fitting_frame_ids', cleanup_threshold)
        self._cleanup_list('fitted_frame_ids', cleanup_threshold)
        self._cleanup_list('tracked_frame_ids', cleanup_threshold)
        
        # 同步清理其他管理器
        try:
            DATA_PIPELINE_MANAGER.cleanup_old_data(cleanup_threshold)
        except Exception as e:
            logger.warning(f"清理 DataPipelineManager 时出错: {e}")
        
        try:
            from project.data.data_manager import CameraDataManager
            camera_manager = CameraDataManager()
            camera_manager.cleanup_old_data(cleanup_threshold)
        except Exception as e:
            logger.warning(f"清理 CameraDataManager 时出错: {e}")
        
        logger.debug(f"数据清理完成，阈值: {cleanup_threshold} (当前时间: {current_time})")
    
    def _check_and_cleanup(self):
        """检查是否需要清理（每个周期清理一次）"""
        if not self.cleanup_enabled:
            return
        
        current_time = self._generate_frame_id()
        period_microseconds = int(self.period_seconds * 1_000_000)
        
        if (self._last_cleanup_time is None or 
            current_time - self._last_cleanup_time >= period_microseconds):
            self._cleanup_old_data()
            self._last_cleanup_time = current_time


class CallbackManager:
    """回调管理器 - 统一管理所有线程间的回调"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._callbacks = {}  # {event_name: [callback1, callback2, ...]}
            self._callback_lock = threading.Lock()
            self._initialized = True
    
    def register(self, event_name: str, callback: callable):
        """注册回调函数
        
        Args:
            event_name: 事件名称
            callback: 回调函数
        """
        with self._callback_lock:
            if event_name not in self._callbacks:
                self._callbacks[event_name] = []
            if callback not in self._callbacks[event_name]:
                self._callbacks[event_name].append(callback)
                logger.debug(f"注册回调: {event_name}")
    
    def unregister(self, event_name: str, callback: callable = None):
        """注销回调函数
        
        Args:
            event_name: 事件名称
            callback: 要注销的回调函数，如果为None则注销该事件的所有回调
        """
        with self._callback_lock:
            if event_name in self._callbacks:
                if callback:
                    if callback in self._callbacks[event_name]:
                        self._callbacks[event_name].remove(callback)
                        logger.debug(f"注销回调: {event_name}")
                else:
                    del self._callbacks[event_name]
                    logger.debug(f"注销所有回调: {event_name}")
    
    def emit(self, event_name: str, *args, **kwargs):
        """触发回调（线程安全）
        
        Args:
            event_name: 事件名称
            *args: 位置参数
            **kwargs: 关键字参数
        """
        with self._callback_lock:
            callbacks = self._callbacks.get(event_name, []).copy()
        
        # 在锁外执行回调，避免死锁
        if not callbacks:
            logger.debug(f"事件 {event_name} 没有注册的回调")
            return
        
        for callback in callbacks:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"回调执行失败 {event_name}: {e}", exc_info=True)
    
    def has_callbacks(self, event_name: str) -> bool:
        """检查是否有注册的回调
        
        Args:
            event_name: 事件名称
            
        Returns:
            bool: 是否有注册的回调
        """
        with self._callback_lock:
            return event_name in self._callbacks and len(self._callbacks[event_name]) > 0

class DataPipelineManager:
    """数据管道管理器 - 管理识别到拟合的数据传递"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._frame_data = {}  # {frame_id: RecgFitDataManager}
            self._data_lock = threading.Lock()
            # 加载配置用于清理
            from project.config.settings import FRAME_ID_PERIOD_SECONDS, FRAME_ID_CLEANUP_ENABLED
            self.period_seconds = FRAME_ID_PERIOD_SECONDS
            self.cleanup_enabled = FRAME_ID_CLEANUP_ENABLED
            self._initialized = True
    
    def store_recognition_data(self, frame_id: int, data_manager: RecgFitDataManager):
        """存储识别数据"""
        with self._data_lock:
            self._frame_data[frame_id] = data_manager
            logger.debug(f"存储识别数据: frame_id={frame_id}")
    
    def get_recognition_data(self, frame_id: int):
        """获取识别数据"""
        with self._data_lock:
            data = self._frame_data.get(frame_id)
            if data:
                logger.debug(f"获取识别数据: frame_id={frame_id}")
            return data
    
    def remove_data(self, frame_id: int):
        """移除数据（避免内存泄漏）"""
        with self._data_lock:
            if frame_id in self._frame_data:
                del self._frame_data[frame_id]
                logger.debug(f"移除数据: frame_id={frame_id}")
    
    def get_latest_data(self):
        """获取最新的识别数据"""
        with self._data_lock:
            if not self._frame_data:
                return None
            latest_frame_id = max(self._frame_data.keys())
            return self._frame_data[latest_frame_id]
    
    def cleanup_old_data(self, threshold: int):
        """清理时间戳小于阈值的旧数据（与 FrameIdManager 同步）"""
        if not self.cleanup_enabled:
            return
        
        with self._data_lock:
            original_count = len(self._frame_data)
            # 保留时间戳 >= threshold 的数据
            self._frame_data = {fid: data for fid, data in self._frame_data.items() if fid >= threshold}
            removed_count = original_count - len(self._frame_data)
            
            if removed_count > 0:
                logger.debug(f"DataPipelineManager 清理: 移除 {removed_count} 个旧数据，保留 {len(self._frame_data)} 个")

# 全局实例
FRAME_ID_MANAGER = FrameIdManager()
DATA_PIPELINE_MANAGER = DataPipelineManager()
CALLBACK_MANAGER = CallbackManager()