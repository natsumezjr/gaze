"""事件处理器基类 - 提供事件驱动模块的模板"""

import threading
from typing import Any, Optional
from project.managers import CALLBACK_MANAGER
from project.events import (
    RECOGNITION_COMPLETE,
    FITTING_COMPLETE,
    FITTING_START,
    SYSTEM_STOP,
    ERROR_OCCURRED,
)
from project.config.logging_config import setup_logging
logger = setup_logging(__name__)


class EventDrivenModule:
    """事件驱动模块基类模板"""
    
    def __init__(self):
        self.running = False
        self._lock = threading.Lock()
        self._setup_event_handlers()
    
    def _setup_event_handlers(self):
        """设置事件处理器 - 子类重写此方法"""
        # 默认注册系统停止事件
        CALLBACK_MANAGER.register(SYSTEM_STOP, self._on_system_stop)
    
    def _on_system_stop(self):
        """系统停止事件处理"""
        with self._lock:
            self.running = False
        logger.debug(f"{self.__class__.__name__} 收到系统停止事件")
    
    def start(self):
        """启动模块"""
        with self._lock:
            self.running = True
        self._run()
    
    def _run(self):
        """主运行循环 - 子类必须重写此方法"""
        raise NotImplementedError("子类必须实现 _run() 方法")
    
    def stop(self):
        """停止模块"""
        with self._lock:
            self.running = False
        self._cleanup()
    
    def _cleanup(self):
        """清理资源 - 子类可以重写此方法"""
        # 默认注销系统停止事件
        CALLBACK_MANAGER.unregister(SYSTEM_STOP, self._on_system_stop)
        logger.debug(f"{self.__class__.__name__} 已清理资源")
    
    def is_running(self) -> bool:
        """检查模块是否在运行"""
        with self._lock:
            return self.running

