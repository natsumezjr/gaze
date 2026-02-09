# 前端适配器抽象 - 后端通过事件与适配器通信，适配器可为 Tk / Unity / Web 等实现
from abc import ABC, abstractmethod
from typing import Optional

from project.data.data_models import Point2D
from project.data.data_models import CalibrationRequest


class FrontendAdapter(ABC):
    """
    前端适配器接口：所有「需要前端显示或输入」的交互均通过此接口。
    后端只发事件，由注册了该事件的适配器实现具体 UI 行为。
    便于后续替换为 Unity / Web 等实现。
    """

    @abstractmethod
    def show_calibration_start(self, frame_id: Optional[int] = None) -> None:
        """收到标定启动请求时，显示/启动标定流程。"""
        pass

    @abstractmethod
    def update_rough_gaze(self, calibration_request: CalibrationRequest) -> None:
        """更新当前帧的粗略注视点（用于标定时的蓝点/动画）。"""
        pass

    @abstractmethod
    def show_gaze_point(self, point: Point2D, color: str = "#0000FF") -> None:
        """在界面上显示一个视线点（如拟合后的蓝点）。"""
        pass

    @abstractmethod
    def request_close(self) -> None:
        """请求关闭标定/前端窗口（如标定完成或系统停止）。"""
        pass
