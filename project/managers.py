# 全局管理器模块
# 包含系统中使用的各种全局管理器，避免循环依赖
import threading
from project.data.data_manager import RecgFitDataManager
from project.config.logging_config import setup_logging, get_logger
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
            self.recognizing_frame_id = 0
            self.recognized_frame_ids = []
            self.fitting_frame_ids = []
            self.fitted_frame_ids = []
            self.tracking_frame_id = 0
            self.tracked_frame_ids = []
            self._initialized = True
        
    def get_recognizing_frame_id(self):
        with self._lock:
            return self.recognizing_frame_id
    
    def increment_recognizing_frame_id(self):
        with self._lock:
            self.recognizing_frame_id += 1
        
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
    
    def get_fitting_frame_ids(self):
        with self._lock:
            return self.fitting_frame_ids.copy()
    
    def add_fitting_frame_id(self, frame_id):
        with self._lock:
            self.fitting_frame_ids.append(frame_id)
    
    def get_fitted_frame_ids(self):
        with self._lock:
            return self.fitted_frame_ids.copy()
    
    def add_fitted_frame_id(self, frame_id):
        with self._lock:
            self.fitted_frame_ids.append(frame_id)
    
    def get_tracking_frame_id(self):
        with self._lock:
            return self.tracking_frame_id
    
    def increment_tracking_frame_id(self):
        with self._lock:
            self.tracking_frame_id += 1
    
    def get_tracked_frame_ids(self):
        with self._lock:
            return self.tracked_frame_ids.copy()
    
    def add_tracked_frame_id(self, frame_id):
        with self._lock:
            self.tracked_frame_ids.append(frame_id)

class SemaphoreManager:
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
            self.fitting_enable = threading.Semaphore(0)
            self.tracking_enable = threading.Semaphore(0)
            self._initialized = True
        
    def wait_for_fitting_start(self):
        self.fitting_enable.acquire()
                
    def signal_recognition_complete(self):
        self.fitting_enable.release()
        
    def wait_for_recognition_complete(self):
        """等待识别完成信号"""
        self.fitting_enable.acquire()
        
    def wait_for_tracking_start(self):
        self.tracking_enable.acquire()
        
    def signal_fitting_complete(self):
        self.tracking_enable.release()

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

# 全局实例
FRAME_ID_MANAGER = FrameIdManager()
SEMAPHORE_MANAGER = SemaphoreManager()
DATA_PIPELINE_MANAGER = DataPipelineManager()
CALLBACK_MANAGER = CallbackManager()