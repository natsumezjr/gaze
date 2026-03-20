# 运行时控制 - 停止事件与系统停止请求
import threading
from project.managers import CALLBACK_MANAGER
from project.events import SYSTEM_STOP
from project.config.logging_config import setup_logging

logger = setup_logging(__name__)

# 停止事件 - 用于协调各模块的退出
stop_event = threading.Event()


def request_system_stop():
    """请求系统停止 - 统一入口，避免重复调用"""
    if not stop_event.is_set():
        stop_event.set()
        CALLBACK_MANAGER.emit(SYSTEM_STOP)
        logger.info("系统停止请求已发送")
