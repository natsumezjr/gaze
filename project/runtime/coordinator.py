# 线程协调器 - 统一创建、启动、停止所有工作线程
import threading
import time
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from project.core.recognition.main import RecognitionManager
from project.managers import FRAME_ID_MANAGER
from project.config.settings import MAX_FITTING_THREADS, BRIDGE_ENABLED
from project.config.logging_config import setup_logging

logger = setup_logging(__name__)


class ThreadCoordinator:
    """统一管理所有工作线程的创建、启动和停止"""

    def __init__(self, rgb_d: bool = True):
        self.rgb_d = rgb_d
        self.recognition_manager: Optional[RecognitionManager] = None
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.recognition_thread: Optional[threading.Thread] = None
        self.bridge_server = None
        self.fitting_scheduler = None

    def start(self):
        """按固定顺序启动所有工作线程"""
        # 1) 创建拟合线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=MAX_FITTING_THREADS)

        # 2) 创建并启动 FittingScheduler
        from project.core.fitting.scheduler import FittingScheduler
        self.fitting_scheduler = FittingScheduler(self.thread_pool)
        self.fitting_scheduler.start()
        logger.info("拟合调度器已启动")

        # 3) 启动 BridgeServer（若启用）
        if BRIDGE_ENABLED:
            try:
                from project.bridge.server import run_bridge_server
                self.bridge_server = run_bridge_server()
                self.bridge_server.start()
                logger.info("WebSocket桥接服务器已启动")
            except Exception as e:
                logger.error(f"启动桥接服务器失败: {e}", exc_info=True)

        # 4) 启动 RecognitionThread
        self.recognition_manager = RecognitionManager(FRAME_ID_MANAGER, rgb_d=self.rgb_d)
        self.recognition_thread = threading.Thread(
            target=self.recognition_manager.run,
            name="RecognitionThread"
        )
        self.recognition_thread.start()
        logger.info("识别线程已启动")

    def stop(self, calibration_ui=None):
        """按固定顺序停止所有工作线程"""
        from project.runtime.control import stop_event

        logger.info("等待所有模块完成资源清理...")

        # 1) 关闭 UI（如果存在）
        if calibration_ui is not None:
            logger.info("程序结束，关闭 kappa UI...")
            try:
                calibration_ui.close()
            except Exception as e:
                logger.error(f"关闭 UI 失败: {e}", exc_info=True)

        # 2) 给事件处理一些时间完成
        time.sleep(0.1)

        # 3) 停止 BridgeServer
        if self.bridge_server is not None:
            logger.info("正在停止桥接服务器...")
            try:
                self.bridge_server.stop()
            except Exception as e:
                logger.error(f"停止桥接服务器失败: {e}", exc_info=True)

        # 4) 关闭拟合线程池
        if self.thread_pool is not None:
            logger.info("正在关闭拟合线程池...")
            self.thread_pool.shutdown(wait=True)
            logger.info("拟合线程池已关闭")

        # 5) 等待 RecognitionThread 结束
        if self.recognition_thread is not None and self.recognition_thread.is_alive():
            logger.info("等待识别线程结束...")
            self.recognition_thread.join(timeout=2.0)

        logger.info("所有工作线程已停止")
