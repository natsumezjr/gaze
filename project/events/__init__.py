"""事件定义模块 - 集中管理所有系统事件"""

from project.events.event_types import (
    MainEventTypes,
    EVENTS,
    RECOGNITION_COMPLETE,
    FITTING_COMPLETE,
    FITTING_START,
    TRACKING_START,
    SYSTEM_STOP,
    ERROR_OCCURRED,
)

__all__ = [
    'MainEventTypes',
    'EVENTS',
    'RECOGNITION_COMPLETE',
    'FITTING_COMPLETE',
    'FITTING_START',
    'TRACKING_START',
    'SYSTEM_STOP',
    'ERROR_OCCURRED',
]

