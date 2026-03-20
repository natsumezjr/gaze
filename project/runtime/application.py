# 应用入口 - 协调启动/关闭流程
from project.client.kappa.ui import EyeCalibrationApp
from project.runtime.coordinator import ThreadCoordinator
from project.runtime.control import stop_event, request_system_stop
from project.config.logging_config import setup_logging

logger = setup_logging(__name__)


def _create_check_system_status(coordinator: ThreadCoordinator):
    """创建 check_system_status 闭包，绑定 coordinator 的 recognition_thread"""

    def check_system_status():
        ui = EyeCalibrationApp.get_instance()
        if ui is None:
            return
        if coordinator.recognition_thread is not None and not coordinator.recognition_thread.is_alive():
            logger.warning("检测到识别线程退出")
            try:
                ui.root.quit()
            except Exception as e:
                logger.error(f"check_system_status UI quit 失败: {e}", exc_info=True)
            return
        if not stop_event.is_set():
            try:
                ui.root.after(100, check_system_status)
            except Exception as e:
                logger.error(f"check_system_status after 调度失败: {e}", exc_info=True)
        else:
            try:
                ui.root.quit()
            except Exception as e:
                logger.error(f"check_system_status 停止时 UI quit 失败: {e}", exc_info=True)

    return check_system_status


def run_application(rgb_d: bool = True):
    """运行应用 - 封装完整的启动、主循环、清理流程"""
    coordinator = None
    calibration_ui = None

    try:
        # 1) 在主线程中创建 UI（单例）
        logger.info("在主线程中创建 kappa UI...")
        calibration_ui = EyeCalibrationApp.get_instance()

        # 2) 创建并启动线程协调器（拟合调度器、桥接服务器、识别线程）
        coordinator = ThreadCoordinator(rgb_d=rgb_d)
        coordinator.start()

        # 3) 启动 check_system_status 定时器
        check_system_status = _create_check_system_status(coordinator)
        calibration_ui.root.after(100, check_system_status)

        # 4) 主线程运行 UI 的 mainloop
        logger.info("系统运行中... (按 Ctrl+C 停止)")
        try:
            calibration_ui.root.mainloop()
        except Exception as e:
            logger.error(f"UI mainloop 异常: {e}", exc_info=True)

        # 5) mainloop 退出后，等待停止事件
        while not stop_event.is_set():
            if coordinator.recognition_thread is not None and not coordinator.recognition_thread.is_alive():
                logger.warning("检测到识别线程退出")
                break
            stop_event.wait(timeout=0.1)

    except KeyboardInterrupt:
        logger.info("收到键盘中断信号")
        request_system_stop()
    except Exception as e:
        logger.error(f"程序异常: {e}", exc_info=True)
        request_system_stop()
    finally:
        if coordinator is not None:
            coordinator.stop(calibration_ui=calibration_ui)
        logger.info("程序结束")
