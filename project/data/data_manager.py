"""
兼容入口：保留 `project.data.data_manager` 的 import 语义（不改变外部行为）。

实现已拆分为：
- `project.data.camera_data_manager.CameraDataManager`
- `project.data.recg_fit_data_manager.RecgFitDataManager`
"""

from project.data.camera_data_manager import CameraDataManager
from project.data.recg_fit_data_manager import RecgFitDataManager

# 全局实例（保持原语义不变）
CAMERA_DATA_MANAGER = CameraDataManager()


__all__ = ["RecgFitDataManager", "CameraDataManager", "CAMERA_DATA_MANAGER"]

