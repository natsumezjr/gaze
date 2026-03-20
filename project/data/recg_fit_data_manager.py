"""
Recognition / fitting compatibility container.

从 `data_manager.py` 拆分出来以收紧职责边界；对外接口与行为保持不变。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from project.config.logging_config import setup_logging
from project.data.data_models import (
    EYE_TYPE,
    FITTING_TYPE,
    KeyCoordinates,
    Point3DWithVisibility,
)

logger = setup_logging(__name__)


class RecgFitDataManager:
    """
    识别拟合数据管理器 - EllSeg 兼容版。

    - key_coordinates: 三角化后的 3D 点（mm），来自 compatibility export 的 pupil/iris 三角化。
      pupil 是 EllSeg 原生 pupil_center 的三角化结果；
      iris 4 点是由 fitted iris_ellipse 推导的 cardinal 点的三角化结果。
      这样做仅为兼容当前下游 fitting，后续 fitting 模块会再重构。
    - native_geometry: EllSeg 原生几何（主存储），供后续使用。
    """

    def __init__(
        self,
        key_coordinates: KeyCoordinates = None,
        debug_log: bool = True,
        native_geometry: Optional[Dict[str, Dict]] = None,
    ):
        """
        Args:
            key_coordinates: 三角化后的关键点（compatibility export 三角化结果）
            debug_log: 是否启用调试日志
            native_geometry: EllSeg 原生几何 {"left": {...}, "right": {...}}
        """
        try:
            self._key_coordinates = key_coordinates
            self._debug_log = debug_log
            self._native_geometry = native_geometry or {}

            logger.debug(
                "RecgFitDataManager初始化 - debug_log=%s key_coordinates=%s native_geometry=%s",
                self._debug_log,
                key_coordinates is not None,
                bool(self._native_geometry),
            )

            if key_coordinates is not None:
                if self._native_geometry:
                    logger.debug("native geometry stored into RecgFitDataManager")
                if self._debug_log:
                    self._log_debug()
        except Exception as e:
            logger.error("RecgFitDataManager初始化失败: %s", e)
            raise

    # ==================== Getter方法 ====================
    def get_key_coordinates(self) -> KeyCoordinates:
        """获取完整的关键点坐标数据"""
        return self._key_coordinates

    def get_eye_coordinates(self, eye: str) -> Dict[str, List[Point3DWithVisibility]]:
        """获取指定眼睛的关键点坐标数据"""
        if not self._check_eye_type(eye):
            raise ValueError(f"无效的眼睛类型: {eye}")

        result = {}
        for fitting_type in FITTING_TYPE:
            result[fitting_type] = self._key_coordinates.get_points(eye, fitting_type)
        return result

    def get_coordinate_point(self, eye: str, fitting_type: str) -> List[Point3DWithVisibility]:
        """获取指定眼睛和类型的坐标点"""
        if not self._check_eye_type(eye):
            raise ValueError(f"无效的眼睛类型: {eye}")
        if fitting_type not in FITTING_TYPE:
            raise ValueError(f"无效的拟合类型: {fitting_type}")
        return self._key_coordinates.get_points(eye, fitting_type)

    def get_point_count(self, eye: str) -> int:
        """获取指定眼睛的坐标点数量"""
        if not self._check_eye_type(eye):
            raise ValueError(f"无效的眼睛类型: {eye}")

        total_count = 0
        for fitting_type in FITTING_TYPE:
            points = self.get_coordinate_point(eye, fitting_type)
            total_count += len(points)
        return total_count

    def get_mean_visibility(self, eye: str) -> float:
        """获取指定眼睛的平均可见性"""
        if not self._check_eye_type(eye):
            raise ValueError(f"无效的眼睛类型: {eye}")

        visibility_values = []
        for fitting_type in FITTING_TYPE:
            points = self.get_coordinate_point(eye, fitting_type)
            for point in points:
                visibility_values.append(point.visibility)
        if visibility_values:
            return np.mean(visibility_values)
        return 0.0

    def get_data_summary(self) -> Dict[str, Dict[str, int]]:
        """获取数据摘要"""
        summary = {}
        for eye in EYE_TYPE:
            summary[eye] = {}
            for fitting_type in FITTING_TYPE:
                points = self.get_coordinate_point(eye, fitting_type)
                summary[eye][fitting_type] = len(points)
        return summary

    # ==================== Setter方法 ====================

    def add_key_coordinates(self, key_coordinates: KeyCoordinates) -> None:
        """添加完整的关键点坐标数据（约定 x,y,z 单位 mm，不做单位换算）。"""
        self._key_coordinates = key_coordinates
        self._log_debug()

    def set_native_eye_geometry(self, eye: str, geometry: Dict) -> None:
        """设置单眼 EllSeg 原生几何。"""
        if eye not in EYE_TYPE:
            raise ValueError(f"无效的眼睛类型: {eye}")
        self._native_geometry[eye] = geometry

    def get_native_eye_geometry(self, eye: str) -> Optional[Dict]:
        """获取单眼 EllSeg 原生几何。"""
        if eye not in EYE_TYPE:
            raise ValueError(f"无效的眼睛类型: {eye}")
        return self._native_geometry.get(eye)

    def export_compatibility_points(self, eye: str) -> Dict[str, List]:
        """
        从原生几何导出兼容点格式（pupil=1, iris=4）。
        pupil 来自 pupil_center；iris 4 点由 iris_ellipse 推导。
        仅为兼容当前下游 fitting，非 EllSeg 原生输出。
        """
        from project.core.recognition.eye_model_ellseg import ellipse_to_cardinal_points

        geom = self.get_native_eye_geometry(eye)
        if not geom:
            return {
                "pupil": [],
                "iris": [],
                "inner_canthus": [],
                "outer_canthus": [],
                "upper_eyelid": [],
                "lower_eyelid": [],
            }
        pupil = [geom["pupil_center"]] if geom.get("pupil_center") else []
        el = geom.get("iris_ellipse")
        iris = list(ellipse_to_cardinal_points(el)) if el is not None else []
        return {
            "pupil": pupil,
            "iris": iris,
            "inner_canthus": [],
            "outer_canthus": [],
            "upper_eyelid": [],
            "lower_eyelid": [],
        }

    def clear_all_data(self) -> None:
        """清空所有眼睛的数据"""
        self._key_coordinates.clear()
        self._log_debug()

    # ==================== 验证方法 ====================

    def _check_eye_type(self, eye: str) -> bool:
        """检查眼睛类型是否有效"""
        return eye in EYE_TYPE

    def _log_debug(self) -> None:
        """输出调试日志（已禁用）"""
        pass

    def __str__(self) -> str:
        """字符串表示"""
        summary = self.get_data_summary()
        result = "RecgFitDataManager:\n"
        for eye in EYE_TYPE:
            result += f"  {eye} eye:\n"
            for fitting_type in FITTING_TYPE:
                count = summary[eye][fitting_type]
                result += f"    {fitting_type}: {count} points\n"
        return result


__all__ = ["RecgFitDataManager"]

