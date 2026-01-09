# 回调接口定义模块
"""
回调接口定义

本模块定义了系统中所有线程间通信的回调接口规范。

回调事件类型：
1. recognition_complete: 识别完成
   - 参数: frame_id (int), data_manager (RecgFitDataManager)
   - 触发者: RecognitionManager
   - 接收者: FittingScheduler

2. calibration_start_request: 请求启动校准
   - 参数: 无
   - 触发者: FittingManager
   - 接收者: EyeCalibrationApp

3. calibration_data_request: 请求校准数据
   - 参数: target_pixel (Point2D)
   - 触发者: FittingManager
   - 接收者: EyeCalibrationApp

4. calibration_data_submit: 提交校准数据（拟合线程 -> UI）
   - 参数: left_eye_data (Point2D), right_eye_data (Point2D)
   - 触发者: FittingManager
   - 接收者: EyeCalibrationApp

5. calibration_data_ready: 校准数据就绪（UI -> 拟合线程）
   - 参数: left_eye_data (Point2D), right_eye_data (Point2D)
   - 触发者: EyeCalibrationApp
   - 接收者: FittingManager

6. calibration_complete: 校准完成
   - 参数: calibration_result (dict)
   - 触发者: EyeCalibrationApp
   - 接收者: FittingManager

7. kappa_ready: Kappa角计算完成
   - 参数: 无
   - 触发者: FittingManager
   - 接收者: 主线程/其他模块
"""

from typing import Protocol, TYPE_CHECKING
from project.data.data_models import Point2D
from project.data.data_manager import RecgFitDataManager

if TYPE_CHECKING:
    from typing import Dict, Any

# 回调接口类型定义（用于类型提示）
class RecognitionCompleteCallback(Protocol):
    """识别完成回调接口"""
    def __call__(self, frame_id: int, data_manager: RecgFitDataManager) -> None:
        """识别完成回调
        
        Args:
            frame_id: 帧ID
            data_manager: 识别数据管理器
        """
        ...

class CalibrationStartRequestCallback(Protocol):
    """校准启动请求回调接口"""
    def __call__(self) -> None:
        """请求启动校准"""
        ...

class CalibrationDataRequestCallback(Protocol):
    """校准数据请求回调接口"""
    def __call__(self, target_pixel: Point2D) -> None:
        """请求校准数据
        
        Args:
            target_pixel: 目标像素坐标
        """
        ...

class CalibrationDataSubmitCallback(Protocol):
    """提交校准数据回调接口（拟合线程 -> UI）"""
    def __call__(self, left_eye_data: Point2D, right_eye_data: Point2D) -> None:
        """提交校准数据
        
        Args:
            left_eye_data: 左眼数据
            right_eye_data: 右眼数据
        """
        ...

class CalibrationDataReadyCallback(Protocol):
    """校准数据就绪回调接口（UI -> 拟合线程）"""
    def __call__(self, left_eye_data: Point2D, right_eye_data: Point2D) -> None:
        """校准数据就绪
        
        Args:
            left_eye_data: 左眼数据
            right_eye_data: 右眼数据
        """
        ...

class CalibrationCompleteCallback(Protocol):
    """校准完成回调接口"""
    def __call__(self, calibration_result: 'Dict[str, Any]') -> None:
        """校准完成
        
        Args:
            calibration_result: 校准结果字典，包含 left_eye 和 right_eye
        """
        ...

class KappaReadyCallback(Protocol):
    """Kappa角就绪回调接口"""
    def __call__(self) -> None:
        """Kappa角计算完成"""
        ...

# 回调事件名称常量
class CallbackEvents:
    """回调事件名称常量"""
    RECOGNITION_COMPLETE = "recognition_complete"
    CALIBRATION_START_REQUEST = "calibration_start_request"
    CALIBRATION_DATA_REQUEST = "calibration_data_request"
    CALIBRATION_DATA_SUBMIT = "calibration_data_submit"
    CALIBRATION_DATA_READY = "calibration_data_ready"
    CALIBRATION_COMPLETE = "calibration_complete"
    KAPPA_READY = "kappa_ready"
