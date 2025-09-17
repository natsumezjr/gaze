# 眼动追踪系统主入口
# 系统启动、模块协调、全局状态管理
from project.core.fitting.main import FittingManager
from project.core.recognition.main import RecognitionManager
from project.core.track.main import TrackingManager
from project.managers import FRAME_ID_MANAGER, SEMAPHORE_MANAGER, DATA_PIPELINE_MANAGER, SemaphoreManager
import threading
import signal
from project.config.settings import (MAX_FITTING_THREADS, 
    MAX_THREADS)
from concurrent.futures import ThreadPoolExecutor
import time
import sys
from project.config.logging_config import setup_logging, get_logger
import logging
import cv2
# 设置统一的日志配置
setup_logging(level=logging.DEBUG, log_to_file=True)
logger = get_logger(__name__)


class FittingScheduler:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, thread_pool_executor: ThreadPoolExecutor, semaphore_manager: SemaphoreManager = SEMAPHORE_MANAGER):
        self.semaphore_manager = semaphore_manager
        self.thread_pool_executor = thread_pool_executor
        self.running = True
    
    def schedule_fitting(self):
        """拟合调度循环"""
        logger.info("FittingScheduler.schedule_fitting() 开始执行")
        try:
            while self.running:
                try:
                    logger.info("等待识别完成信号...")
                    self.semaphore_manager.wait_for_recognition_complete()
                    if not self.running:
                        break
                    logger.info("收到识别完成信号，开始拟合...")
                    frame_id = FRAME_ID_MANAGER.choose_recognized_to_fitting()
                    if frame_id is None:
                        logger.debug("没有可用的已识别帧，继续等待...")
                        time.sleep(0.01)  # 短暂等待，避免忙等待
                        continue
                    logger.info(f"开始拟合frame_id: {frame_id}")
                    recg_fit_data_manager = DATA_PIPELINE_MANAGER.get_recognition_data(frame_id)
                    if recg_fit_data_manager is None:
                        logger.warning(f"无法获取frame_id {frame_id}的识别数据")
                        continue
                    fitting_manager = FittingManager(recg_fit_data_manager, self.semaphore_manager)
                    self.thread_pool_executor.submit(fitting_manager.run)
                except Exception as e:
                    if self.running:
                        logger.error(f"拟合调度异常: {e}")
                    break
        except KeyboardInterrupt:
            logger.info("拟合调度器收到中断信号")
        finally:
            logger.info("拟合调度器已停止")
            
    def stop(self):
        self.running = False
    
# 创建停止事件
stop_event = threading.Event()

def stop_signal_handler(signum, frame):
    logger.info("收到停止信号，开始优雅关闭...")
    stop_event.set()

# 注册信号处理器
signal.signal(signal.SIGINT, stop_signal_handler)
signal.signal(signal.SIGTERM, stop_signal_handler)
# 创建kappa校准事件
kappa_calibrate_event = threading.Event()
from project.client.kappa.ui import EyeCalibrationApp
EYE_CALIBRATION_APP = EyeCalibrationApp()

def main():
    def stop():
        logger.info("正在停止程序...")
        recognition_manager.stop()
        logger.info("识别线程已停止")
        logger.info("OpenCV窗口已关闭")
        fitting_scheduler.stop()
        logger.info("拟合调度线程已停止")
        thread_pool.shutdown(wait=True)
        logger.info("线程池已停止")
        logger.info("程序已停止")
    
    rgb_d = False
   
    recognition_manager = RecognitionManager(SEMAPHORE_MANAGER,FRAME_ID_MANAGER, rgb_d=rgb_d)
    thread_pool = ThreadPoolExecutor(max_workers=MAX_FITTING_THREADS)
    
    try:
        fitting_scheduler = FittingScheduler(thread_pool, SEMAPHORE_MANAGER)
        
        # 启动识别线程
        logger.debug("启动识别线程...")
        recognition_future = thread_pool.submit(recognition_manager.run)
        logger.debug(f"识别线程已提交: {recognition_future}")
        
        # 启动拟合调度线程
        logger.debug("启动拟合调度线程...")
        fitting_future = thread_pool.submit(fitting_scheduler.schedule_fitting)
        logger.debug(f"拟合调度线程已提交: {fitting_future}")
        
        # 主线程优雅等待 - 使用事件而不是sleep
        logger.info("系统运行中... (按 Ctrl+C 停止)")
        
        while not stop_event.is_set():
            # 检查线程是否还在运行
            if recognition_future.done() or fitting_future.done():
                logger.warning("检测到线程异常退出")
                break
            
            if kappa_calibrate_event.is_set():
                thread_pool.submit(EYE_CALIBRATION_APP.run)
            # 等待停止信号，最多等待1秒
            stop_event.wait(timeout=1.0)
    except Exception as e:
        logger.error(f"程序异常: {e}")
        stop_event.set()
    finally:
        stop()
        logger.info("程序结束")

if __name__ == "__main__":
    main()
    
    

            
            


