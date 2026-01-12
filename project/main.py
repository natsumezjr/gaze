# 眼动追踪系统主入口
# 系统启动、模块协调、全局状态管理
from project.core.fitting.main import FittingManager
from project.core.recognition.main import RecognitionManager
from project.managers import FRAME_ID_MANAGER, CALLBACK_MANAGER
from project.events import RECOGNITION_COMPLETE, FITTING_START, SYSTEM_STOP
from project.events.event_types import CALIBRATION_POINT_SUBMIT, CALIBRATION_COMPLETE, CALIBRATION_START_REQUEST, ROUGH_GAZE_UPDATE
import threading
from typing import Optional
from project.config.settings import MAX_FITTING_THREADS
from concurrent.futures import ThreadPoolExecutor
from project.config.logging_config import setup_logging, get_logger
import logging
from project.managers import DATA_PIPELINE_MANAGER
from project.core.fitting.kappa_calibrator import KAPPA_STORAGE, SAMPLES_STORAGE, build_samples_from_arrays, estimate_kappa
from project.core.fitting.fitting_strategy import fit_all_eyes
from project.config.screen_config import SCREEN_CONFIG
from project.data.data_models import Point3D, Point2D, CalibrationRequest, CalibrationResponse
from project.data.data_models import EYE_TYPE
from project.client.kappa.ui import EyeCalibrationApp
# 设置统一的日志配置
setup_logging(level=logging.DEBUG, log_to_file=True)
logger = get_logger(__name__)


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
        CALLBACK_MANAGER.register(CALIBRATION_POINT_SUBMIT,self._on_calibration_point_submit)
        CALLBACK_MANAGER.register(CALIBRATION_COMPLETE,self._on_calibration_complete)
        
        logger.debug("拟合调度器事件处理器已注册")
    
    def _on_recognition_complete(self, frame_id: int):
        """识别完成事件处理 - 自动触发拟合"""
        if not self.running:
            return
        
        try:
            # 直接使用事件传入的 frame_id 获取数据
            data = DATA_PIPELINE_MANAGER.get_recognition_data(frame_id)
            if data is None:
                logger.warning(f"frame_id {frame_id} 的数据为空，跳过拟合")
                return
            
            recognized_ids = FRAME_ID_MANAGER.get_recognized_frame_ids()
            if frame_id in recognized_ids:
                # 从已识别列表中移除
                if FRAME_ID_MANAGER.remove_recognized_frame_id(frame_id):
                    # 添加到拟合列表
                    FRAME_ID_MANAGER.add_fitting_frame_id(frame_id)
            else:
                logger.debug(f"frame_id {frame_id} 不在已识别列表中，可能已被处理")
            
            # 检查 kappa 是否 valid
            if not KAPPA_STORAGE.is_kappa_valid():
                logger.info(f"kappa 未有效，发送标定启动请求，frame_id: {frame_id}")
                CALLBACK_MANAGER.emit(CALIBRATION_START_REQUEST, frame_id=frame_id)
                # 执行拟合获取 intersection
                logger.debug(f"执行拟合以获取 intersection，frame_id: {frame_id}")
                fitting_results = fit_all_eyes(data)
                
                # 为每个眼睛发送 CalibrationRequest
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
                        # 在 UI 上显示实现点（蓝色）
                        try:
                            ui = EyeCalibrationApp.get_instance()
                            if ui:
                                ui.add_gaze_point(intersection, color="#0000FF")  # 蓝色
                        except Exception as e:
                            logger.debug(f"显示实现点失败: {e}")
                        
                        # 发送 CalibrationRequest 给 UI
                        calibration_request = CalibrationRequest(
                            frame_id=frame_id,
                            eye_type=eye,
                            intersection=intersection
                        )
                        CALLBACK_MANAGER.emit(ROUGH_GAZE_UPDATE, calibration_request=calibration_request)
                        logger.debug(f"发送 CalibrationRequest: frame_id={frame_id}, eye={eye}, intersection={intersection}")
                
                return
            
            # kappa valid，正常提交拟合任务
            logger.info(f"收到识别完成事件，开始拟合 frame_id: {frame_id}")
            
            # 提交拟合任务（非阻塞）
            fitting_manager = FittingManager(data)
            self.thread_pool_executor.submit(fitting_manager.run)
            
            # 发送拟合开始事件
            CALLBACK_MANAGER.emit(FITTING_START, frame_id)
            logger.debug(f"拟合任务已提交: frame_id={frame_id}")
            
        except Exception as e:
            logger.error(f"处理识别完成事件时出错: {e}", exc_info=True)
    
    def _on_calibration_point_submit(self, calibration_response: CalibrationResponse):
        """处理标定点提交事件"""
        try:
            # 从 CalibrationResponse 中提取数据
            frame_id = calibration_response.frame_id
            eye_type = calibration_response.eye_type
            target_pixel = calibration_response.target_pixel
            background_color = calibration_response.background_color
            
            # 使用 frame_id 从 DATA_PIPELINE_MANAGER 获取对应的数据
            data = DATA_PIPELINE_MANAGER.get_recognition_data(frame_id)
            if data is None:
                logger.warning(f"无法获取 frame_id {frame_id} 的数据，跳过标定点保存")
                return
            
            # 执行拟合获取 eyeball 和 pupil center
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
            
            # 保存到 SamplesStorage
            SAMPLES_STORAGE.append(frame_id, eye_type, pupil_center, eyeball_center, target_pixel)
            logger.debug(f"保存标定点: frame_id={frame_id}, eye={eye_type}, target_pixel={target_pixel}, background_color={background_color}")
                
        except Exception as e:
            logger.error(f"处理标定点提交事件时出错: {e}", exc_info=True)
    
    def _on_calibration_complete(self, calibration_result: dict):
        """处理标定完成事件"""
        try:
            logger.info("开始处理标定完成事件，估计 kappa...")
            
            # 获取所有样本
            all_samples = SAMPLES_STORAGE.get_all_samples()
            
            # 按眼睛分组收集所有数据
            for eye in ["left", "right"]:
                pupils_list = []
                eyeball_list = []
                pixel_list = []
                
                # 从所有 frame_id 中收集该眼睛的数据
                for frame_id, frame_data in all_samples.items():
                    if eye in frame_data:
                        pupils_list.extend(frame_data[eye]["pupils"])
                        eyeball_list.extend(frame_data[eye]["eyeball"])
                        pixel_list.extend(frame_data[eye]["pixel"])
                
                if len(pupils_list) == 0:
                    logger.warning(f"{eye} 眼没有标定数据，跳过")
                    continue
                
                # 将 pixel 转换为 3D 目标点
                target_points = []
                for i, pixel in enumerate(pixel_list):
                    if i < len(eyeball_list):
                        target_3d = SCREEN_CONFIG.pixel_to_3d_intersection(pixel, eyeball_list[i])
                        if target_3d:
                            target_points.append(target_3d)
                
                if len(target_points) != len(pupils_list):
                    logger.warning(f"{eye} 眼数据不匹配，跳过")
                    continue
                
                # 构建样本并估计 kappa
                samples = build_samples_from_arrays(eyeball_list, pupils_list, target_points)
                kappa, result = estimate_kappa(samples)
                
                # 保存到 KappaStorage
                KAPPA_STORAGE.set_kappa(eye, kappa)
                logger.info(f"{eye} 眼 kappa 估计完成: {result}")
            
            # 清空样本存储
            SAMPLES_STORAGE.clear()
            logger.info("标定完成，kappa 已保存")
            
            # 直接关闭 UI（使用单例）
            calibration_ui = EyeCalibrationApp.get_instance()
            if calibration_ui is not None:
                logger.info("标定完成，关闭 kappa UI...")
                try:
                    calibration_ui.root.quit()
                except:
                    pass
                try:
                    calibration_ui.close()
                except:
                    pass
                logger.info("kappa UI 已关闭")
            
        except Exception as e:
            logger.error(f"处理标定完成事件时出错: {e}", exc_info=True)
    
    def _on_system_stop(self):
        """系统停止事件处理 - 只负责停止调度器，不关闭线程池"""
        self.running = False
        
        # 直接关闭 UI（使用单例）
        calibration_ui = EyeCalibrationApp.get_instance()
        if calibration_ui is not None:
            try:
                calibration_ui.root.quit()
            except:
                pass
            try:
                calibration_ui.close()
            except:
                pass
        
        CALLBACK_MANAGER.unregister(RECOGNITION_COMPLETE, self._on_recognition_complete)
        CALLBACK_MANAGER.unregister(SYSTEM_STOP, self._on_system_stop)
        CALLBACK_MANAGER.unregister(CALIBRATION_POINT_SUBMIT,self._on_calibration_point_submit)
        CALLBACK_MANAGER.unregister(CALIBRATION_COMPLETE,self._on_calibration_complete)
        logger.info("拟合调度器已停止（线程池由主线程关闭）")
    
    def start(self):
        """启动调度器"""
        self.running = True
        logger.info("事件驱动拟合调度器已启动")
    


# 创建停止事件
stop_event = threading.Event()

def request_system_stop():
    """请求系统停止 - 统一入口，避免重复调用"""
    if not stop_event.is_set():
        stop_event.set()
        CALLBACK_MANAGER.emit(SYSTEM_STOP)
        logger.info("系统停止请求已发送")

def main():
    """主函数 - 纯事件驱动架构，所有停止逻辑由各模块的 _on_system_stop() 处理"""
    rgb_d = False
    recognition_manager = RecognitionManager(FRAME_ID_MANAGER, rgb_d=rgb_d)
    thread_pool = ThreadPoolExecutor(max_workers=MAX_FITTING_THREADS)
    
    
    try:
        # 在主线程中直接创建 UI（单例）
        logger.info("在主线程中创建 kappa UI...")
        
        # 创建拟合调度器（不需要传递 UI 引用）
        fitting_scheduler = FittingScheduler(thread_pool)
        
        # 启动拟合调度器（注册事件处理器）
        fitting_scheduler.start()
        
        # 启动识别线程（在 thread_pool 中运行）
        logger.debug("启动识别线程...")
        recognition_future = thread_pool.submit(recognition_manager.run)
        logger.debug(f"识别线程已提交: {recognition_future}")
        
        def check_system_status():
            """检查系统状态（定期调用）"""
            # 获取 UI 单例
            ui = EyeCalibrationApp.get_instance()
            if ui is None:
                return
            
            # 检查线程是否还在运行
            if recognition_future.done():
                logger.warning("检测到识别线程异常退出")
                try:
                    ui.root.quit()
                except:
                    pass
                return
            
            # 如果还没停止，继续调度
            if not stop_event.is_set():
                try:
                    # UI 存在，使用 UI 的 after 来调度
                    ui.root.after(100, check_system_status)
                except:
                    # UI 可能已被关闭，不再调度
                    pass
            else:
                # 停止事件已设置，退出 UI
                try:
                    ui.root.quit()
                except:
                    pass
                
        calibration_ui = EyeCalibrationApp.get_instance()
        # 启动检查循环
        calibration_ui.root.after(100, check_system_status)
        
        # 主线程运行 UI 的 mainloop
        logger.info("系统运行中... (按 Ctrl+C 停止)")
        try:
            calibration_ui.root.mainloop()
        except Exception as e:
            logger.error(f"UI mainloop 异常: {e}")
        
        # mainloop 退出后，等待停止事件
        while not stop_event.is_set():
            # 检查线程是否还在运行
            if recognition_future.done():
                logger.warning("检测到识别线程异常退出")
                break
            stop_event.wait(timeout=0.1)
        
    except KeyboardInterrupt:
        logger.info("收到键盘中断信号")
        request_system_stop()
    except Exception as e:
        logger.error(f"程序异常: {e}", exc_info=True)
        request_system_stop()
    finally:
        # 等待所有模块完成清理（通过事件驱动）
        # 各模块的 _on_system_stop() 会自动处理资源清理
        logger.info("等待所有模块完成资源清理...")
        
        # 关闭 UI（如果还存在）
        if 'calibration_ui' in locals() and calibration_ui is not None:
            logger.info("程序结束，关闭 kappa UI...")
            try:
                calibration_ui.close()
            except:
                pass
        
        # 给事件处理一些时间完成
        import time
        time.sleep(0.1)
        
        # 主线程负责关闭线程池（等待所有任务完成，包括识别线程）
        if 'thread_pool' in locals() and thread_pool is not None:
            logger.info("正在关闭线程池...")
            thread_pool.shutdown(wait=True)
            logger.info("线程池已关闭")
        
        logger.info("程序结束")

if __name__ == "__main__":
    main()
    
    

            
            


