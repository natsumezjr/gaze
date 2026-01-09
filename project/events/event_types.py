"""事件类型定义 - 系统所有事件的类型常量"""

from enum import Enum
from typing import Dict


class MainEventTypes(Enum):
    """事件类型枚举"""
    RECOGNITION_COMPLETE = "recognition_complete"
    FITTING_COMPLETE = "fitting_complete"
    FITTING_START = "fitting_start"
    TRACKING_START = "tracking_start"
    SYSTEM_STOP = "system_stop"
    ERROR_OCCURRED = "error_occurred"
    
# 回调事件名称常量
class KapaCallbackEventTypes(Enum):
    """回调事件名称常量"""
    CALIBRATION_START_REQUEST = "calibration_start_request"
    CALIBRATION_POINT_SUBMIT = "calibration_point_submit"
    CALIBRATION_COMPLETE = "calibration_complete"
    ROUGH_GAZE_UPDATE = "rough_gaze_update"

    
          


# 事件类型常量（字符串形式，便于使用）
RECOGNITION_COMPLETE = MainEventTypes.RECOGNITION_COMPLETE.value
FITTING_COMPLETE = MainEventTypes.FITTING_COMPLETE.value
FITTING_START = MainEventTypes.FITTING_START.value
TRACKING_START = MainEventTypes.TRACKING_START.value
SYSTEM_STOP = MainEventTypes.SYSTEM_STOP.value
ERROR_OCCURRED = MainEventTypes.ERROR_OCCURRED.value

CALIBRATION_START_REQUEST = KapaCallbackEventTypes.CALIBRATION_START_REQUEST.value
CALIBRATION_POINT_SUBMIT = KapaCallbackEventTypes.CALIBRATION_POINT_SUBMIT.value
CALIBRATION_COMPLETE = KapaCallbackEventTypes.CALIBRATION_COMPLETE.value
ROUGH_GAZE_UPDATE = KapaCallbackEventTypes.ROUGH_GAZE_UPDATE.value

# 事件描述字典
EVENTS: Dict[str, str] = {
    RECOGNITION_COMPLETE: "识别完成事件",
    FITTING_COMPLETE: "拟合完成事件",
    FITTING_START: "拟合开始事件",
    TRACKING_START: "追踪开始事件",
    SYSTEM_STOP: "系统停止事件",
    ERROR_OCCURRED: "错误发生事件",
}

