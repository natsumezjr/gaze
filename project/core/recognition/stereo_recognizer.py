from __future__ import annotations

import logging
from typing import Tuple, TYPE_CHECKING, Dict, Any, Optional, List

from project.config.logging_config import setup_logging
from project.data.data_models import BGRImage, KeyCoordinates, Point2D
from project.core.recognition.roi.state_machine import RoiStateMachine
from project.core.recognition.landmark_extractor import (
    EyeLandmarkExtractor,
    extract_fitting_dict_for_pair,
)
from project.core.recognition.eye_model_ellseg import EyeRoiModel
from project.core.recognition.face_detector import YuNetDetector

if TYPE_CHECKING:
    from project.data.data_manager import CameraDataManager

logger = setup_logging(__name__, logging.DEBUG)


class StereoRecognizer:
    """左右目图像 -> 拟合点提取 -> 三角化 -> KeyCoordinates。"""

    def __init__(self, camera_data_manager: "CameraDataManager"):
        if camera_data_manager is None:
            raise ValueError("camera_data_manager 不能为 None")
        self._camera_data_manager = camera_data_manager

        # 共享的无状态模型组件（眼部 ROI 专用）
        eye_model = EyeRoiModel()
        self._face_detector = YuNetDetector()

        # 左右眼各自独立的 ROI 跟踪与提取器
        self._roi_left = RoiStateMachine(side="left")
        self._roi_right = RoiStateMachine(side="right")
        self._extractor_left = EyeLandmarkExtractor(
            side="left", roi_state_machine=self._roi_left, eye_model=eye_model
        )
        self._extractor_right = EyeLandmarkExtractor(
            side="right", roi_state_machine=self._roi_right, eye_model=eye_model
        )

        # 调试用：保存最近一帧的 2D/ROI 信息，供可视化使用
        self._last_debug_2d: Dict[str, Any] = {}

        # 轻量级统计信息（用于观察整体健康度）
        self._stats: Dict[str, Any] = {
            "frames": 0,
            "face_left_ok": 0,
            "face_right_ok": 0,
            "roi_left_ok": 0,
            "roi_right_ok": 0,
            "ellseg_ok": 0,
            "triangulation_ok": 0,
        }

        # 人脸检测缓存
        self._last_face_info_left: Optional[Dict[str, Any]] = None
        self._last_face_info_right: Optional[Dict[str, Any]] = None
        self._last_face_bbox_left: Optional[Tuple[int, int, int, int]] = None
        self._last_face_bbox_right: Optional[Tuple[int, int, int, int]] = None

    def get_last_debug_2d(self) -> Dict[str, Any]:
        """返回最近一次 recognize 的 2D/ROI 调试数据（只读）。"""
        return self._last_debug_2d

    def recognize(
        self,
        left: BGRImage,
        right: BGRImage,
        frame_id: int = -1,
    ) -> Tuple[bool, KeyCoordinates]:
        """
        左右图分别提取拟合点，同名点三角化得到 KeyCoordinates。

        新 gate 基于当前 ROI 的 mode，并叠加 geometry_confidence。
        """
        try:
            self._stats["frames"] += 1

            face_info_left = self._face_detector.detect_with_landmarks(left.data)
            if face_info_left:
                self._last_face_info_left = face_info_left
                self._last_face_bbox_left = face_info_left.get("bbox")
            else:
                face_info_left = self._last_face_info_left
            left_bbox = face_info_left["bbox"] if face_info_left else self._last_face_bbox_left

            face_info_right = self._face_detector.detect_with_landmarks(right.data)
            if face_info_right:
                self._last_face_info_right = face_info_right
                self._last_face_bbox_right = face_info_right.get("bbox")
            else:
                face_info_right = self._last_face_info_right
            right_bbox = face_info_right["bbox"] if face_info_right else self._last_face_bbox_right

            if left_bbox is not None:
                self._stats["face_left_ok"] += 1
            if right_bbox is not None:
                self._stats["face_right_ok"] += 1

            logger.debug(
                "StereoRecognizer: face bbox left=%s right=%s",
                left_bbox,
                right_bbox,
            )

            if left_bbox is None and right_bbox is None:
                logger.warning("face bbox detection failed on both sides (no cache usable)")
                self._last_debug_2d = {"reason": "face_bbox_none_both"}
                return False, KeyCoordinates()

            ok, fitting_dict, debug = extract_fitting_dict_for_pair(
                left,
                right,
                self._extractor_left,
                self._extractor_right,
                left_bbox,
                right_bbox,
                frame_id=frame_id,
                left_face_info=face_info_left,
                right_face_info=face_info_right,
            )
            # 将人脸 bbox 一并写入调试字典，供可视化画框使用
            debug["face_bbox_left"] = left_bbox
            debug["face_bbox_right"] = right_bbox
            self._last_debug_2d = debug
            left_mode = debug.get("left", {}).get("debug", {}).get("roi_mode", "?")
            right_mode = debug.get("right", {}).get("debug_right_eye", {}).get("roi_mode", "?")
            if ok:
                logger.debug(
                    "StereoRecognizer ROI mode left=%s right=%s",
                    left_mode,
                    right_mode,
                )

            if not ok:
                reason = debug.get("reason", "unknown")
                logger.warning("stereo recognize reject: %s", reason)
                return False, KeyCoordinates()

            # 新 gate：
            # - 直接用当前帧 ROI 的 current_mode 判断是否需要 recover
            left_roi_dbg = debug.get("left", {}).get("debug", {}).get("roi_debug", {}) or {}
            right_roi_dbg = debug.get("right", {}).get("debug_right_eye", {}).get("roi_debug", {}) or {}
            left_geo = float(left_roi_dbg.get("geometry_confidence", 0.0))
            right_geo = float(right_roi_dbg.get("geometry_confidence", 0.0))
            left_mode_dbg = str(left_roi_dbg.get("current_mode", ""))
            right_mode_dbg = str(right_roi_dbg.get("current_mode", ""))
            left_need_recover = left_mode_dbg == "recover"
            right_need_recover = right_mode_dbg == "recover"

            logger.debug(
                "[CONF_GATE] L_geo=%.3f L_need_recover=%s | R_geo=%.3f R_need_recover=%s",
                left_geo,
                left_need_recover,
                right_geo,
                right_need_recover,
            )
            if left_need_recover or right_need_recover:
                logger.error(
                    "confidence gate: reject triangulation (L_need_recover=%s R_need_recover=%s L_geo=%.3f R_geo=%.3f)",
                    left_need_recover,
                    right_need_recover,
                    left_geo,
                    right_geo,
                )
                self._last_debug_2d["reason"] = "confidence_gate_need_recover"
                return False, KeyCoordinates()

            # 2D 完整性检查：核心点（pupil/iris）为强约束，可选点为软约束
            def _check_eye_pair(
                left_img_eye: Dict[str, List[Point2D]],
                right_img_eye: Dict[str, List[Point2D]],
                eye_name: str,
            ) -> Tuple[bool, str]:
                core_required = {"pupil": 1, "iris": 4}
                for k, n in core_required.items():
                    lpts = left_img_eye.get(k, [])
                    rpts = right_img_eye.get(k, [])
                    if len(lpts) != n or len(rpts) != n:
                        return False, f"{eye_name} missing core {k} (L:{len(lpts)}/{n} R:{len(rpts)}/{n})"
                return True, ""

            left_dict = fitting_dict.get("left", {})
            right_dict = fitting_dict.get("right", {})
            left_ok, left_reason = _check_eye_pair(
                left_dict.get("left", {}), right_dict.get("left", {}), "left"
            )
            right_ok, right_reason = _check_eye_pair(
                left_dict.get("right", {}), right_dict.get("right", {}), "right"
            )

            if left_ok:
                self._stats["roi_left_ok"] += 1
            if right_ok:
                self._stats["roi_right_ok"] += 1
            if left_ok and right_ok:
                self._stats["ellseg_ok"] += 1

            if not left_ok:
                logger.warning("left eye landmark core points incomplete: %s", left_reason)
                logger.warning("triangulation core points fail (left)")
                return False, KeyCoordinates()
            if not right_ok:
                logger.warning("right eye landmark core points incomplete: %s", right_reason)
                logger.warning("triangulation core points fail (right)")
                return False, KeyCoordinates()

            logger.debug("core points ready for triangulation")

            key_coordinates = self._camera_data_manager.triangulate_fitting_points(
                fitting_dict["left"], fitting_dict["right"]
            )
            self._stats["triangulation_ok"] += 1

            # 每隔 50 帧输出一次统计与 ROI 模式
            if self._stats["frames"] % 50 == 0:
                f = float(self._stats["frames"])
                logger.debug(
                    "StereoRecognizer stats: frames=%d faceL=%.2f faceR=%.2f roiL=%.2f roiR=%.2f ellseg_ok=%.2f tri_ok=%.2f mode_left=%s mode_right=%s",
                    self._stats["frames"],
                    self._stats["face_left_ok"] / f,
                    self._stats["face_right_ok"] / f,
                    self._stats["roi_left_ok"] / f,
                    self._stats["roi_right_ok"] / f,
                    self._stats["ellseg_ok"] / f,
                    self._stats["triangulation_ok"] / f,
                    left_mode,
                    right_mode,
                )
            return True, key_coordinates
        except Exception as e:
            logger.error("stereo recognize 失败: %s", e, exc_info=True)
            return False, KeyCoordinates()