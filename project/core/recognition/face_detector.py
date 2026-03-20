# 人脸 bbox 检测 - YuNet
"""
YuNet / OpenCV FaceDetectorYN 输出约定（必须遵守）：

- detect() 返回 shape [num_faces, 14]（部分版本为 15，最后一列为 score）。
- 列 0~3：bbox [x, y, w, h]
- 列 4~5：right_eye (x, y)
- 列 6~7：left_eye (x, y)
- 列 8~9：nose_tip (x, y)
- 列 10~11：right_mouth (x, y)
- 列 12~13：left_mouth (x, y)

这 5 个 landmark 是人脸检测器输出点，供 ROI 主链直接使用。
"""
import numpy as np
import logging
import cv2
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from project.config.logging_config import setup_logging
from project.config.model_config import YUNET_MODEL_FILENAME, YUNET_MODEL_SEARCH_DIRS
from project.data.model_locator import find_model_file, ModelLocateError

# 0-based case_id（case1..case4 也正好对应 idx=0..3 的均值回退口径）
YUNET_CASE_0_ALL_5_IN_BOUNDS = 0
YUNET_CASE_1_EYES_NOSE_IN_ONE_MOUTH_Y_GT_H = 1
YUNET_CASE_2_EYES_NOSE_IN_TWO_MOUTH_Y_GT_H = 2
YUNET_CASE_3_EYES_IN_NOSE_MOUTH_Y_GT_H = 3
YUNET_CASE_4_Y_GT_H_GE4 = 4
YUNET_CASE_5_EYE_X_OOB = 5
YUNET_CASE_6_ANY_Y_LT0 = 6
YUNET_CASE_7_NONE = 7
YUNET_CASE_8_UNKNOWN = 8


YUNET_CASE_NAMES = {
    YUNET_CASE_0_ALL_5_IN_BOUNDS: "case_0_all_5_in_bounds",
    YUNET_CASE_1_EYES_NOSE_IN_ONE_MOUTH_Y_GT_H: "case_1_eyes_nose_in_one_mouth_y_gt_h",
    YUNET_CASE_2_EYES_NOSE_IN_TWO_MOUTH_Y_GT_H: "case_2_eyes_nose_in_two_mouth_y_gt_h",
    YUNET_CASE_3_EYES_IN_NOSE_MOUTH_Y_GT_H: "case_3_eyes_in_nose_mouth_y_gt_h",
    YUNET_CASE_4_Y_GT_H_GE4: "case_4_y_gt_h_ge4",
    YUNET_CASE_5_EYE_X_OOB: "case_5_eye_x_oob",
    YUNET_CASE_6_ANY_Y_LT0: "case_6_any_y_lt0",
    YUNET_CASE_7_NONE: "case_7_none",
    YUNET_CASE_8_UNKNOWN: "case_8_unknown",
}

logger = setup_logging(__name__, logging.DEBUG)


def classify_yunet_case(
    face_info: Optional[Dict[str, Any]],
    image_shape: Tuple[int, ...],
) -> Tuple[int, str]:
    """
    按约定对 YuNet 5 点进行 case 分类，并返回 (case_id, case_name)。
    该输出用于上层（例如 ROI 初始化）按 case 索引不同均值/策略。
    """
    if face_info is None or image_shape is None or len(image_shape) < 2:
        return YUNET_CASE_7_NONE, YUNET_CASE_NAMES[YUNET_CASE_7_NONE]

    img_h, img_w = int(image_shape[0]), int(image_shape[1])
    keys = ["left_eye", "right_eye", "nose_tip", "left_mouth", "right_mouth"]
    pts = {k: face_info.get(k) for k in keys}

    def _in_bounds(pt: Any) -> bool:
        if pt is None:
            return False
        try:
            x, y = float(pt[0]), float(pt[1])
        except Exception:
            return False
        return 0.0 <= x <= float(img_w) and 0.0 <= y <= float(img_h)

    any_y_lt0 = False
    eye_x_oob = False
    y_gt_h_count = 0
    mouth_y_gt_h_count = 0

    for k, pt in pts.items():
        if pt is None:
            continue
        try:
            x, y = float(pt[0]), float(pt[1])
        except Exception:
            continue
        if y < 0:
            any_y_lt0 = True
        if y > img_h:
            y_gt_h_count += 1
            if k in ("left_mouth", "right_mouth"):
                mouth_y_gt_h_count += 1
        if k in ("left_eye", "right_eye") and (x < 0 or x > img_w):
            eye_x_oob = True

    # Debug: 输出 case 分类输入与关键判断量
    logger.debug(
        "[YUNET_CLASSIFY_IN] img_wh=(%d,%d) any_y_lt0=%s eye_x_oob=%s y_gt_h_count=%d mouth_y_gt_h_count=%d pts=%s",
        img_w,
        img_h,
        any_y_lt0,
        eye_x_oob,
        y_gt_h_count,
        mouth_y_gt_h_count,
        {k: pts.get(k) for k in keys},
    )

    # 规则优先级：越界提示类优先
    if any_y_lt0:
        cid = YUNET_CASE_6_ANY_Y_LT0
        logger.debug("[YUNET_CLASSIFY_CASE] cid=%d name=%s rule=any_y_lt0", cid, YUNET_CASE_NAMES[cid])
        return cid, YUNET_CASE_NAMES[cid]
    if eye_x_oob:
        cid = YUNET_CASE_5_EYE_X_OOB
        logger.debug("[YUNET_CLASSIFY_CASE] cid=%d name=%s rule=eye_x_oob", cid, YUNET_CASE_NAMES[cid])
        return cid, YUNET_CASE_NAMES[cid]
    if y_gt_h_count >= 4:
        cid = YUNET_CASE_4_Y_GT_H_GE4
        logger.debug("[YUNET_CLASSIFY_CASE] cid=%d name=%s rule=y_gt_h_count>=4", cid, YUNET_CASE_NAMES[cid])
        return cid, YUNET_CASE_NAMES[cid]

    le_in = _in_bounds(pts["left_eye"])
    re_in = _in_bounds(pts["right_eye"])
    nose_in = _in_bounds(pts["nose_tip"])
    lm_in = _in_bounds(pts["left_mouth"])
    rm_in = _in_bounds(pts["right_mouth"])
    all_in = le_in and re_in and nose_in and lm_in and rm_in

    logger.debug(
        "[YUNET_CLASSIFY_BOUNDS] le_in=%s re_in=%s nose_in=%s lm_in=%s rm_in=%s all_in=%s",
        le_in,
        re_in,
        nose_in,
        lm_in,
        rm_in,
        all_in,
    )

    if all_in:
        cid = YUNET_CASE_0_ALL_5_IN_BOUNDS
        logger.debug("[YUNET_CLASSIFY_CASE] cid=%d name=%s rule=all_in", cid, YUNET_CASE_NAMES[cid])
        return cid, YUNET_CASE_NAMES[cid]
    if le_in and re_in and nose_in and mouth_y_gt_h_count == 1:
        cid = YUNET_CASE_1_EYES_NOSE_IN_ONE_MOUTH_Y_GT_H
        logger.debug(
            "[YUNET_CLASSIFY_CASE] cid=%d name=%s rule=case2 mouth_y_gt_h_count==1",
            cid,
            YUNET_CASE_NAMES[cid],
        )
        return cid, YUNET_CASE_NAMES[cid]
    if le_in and re_in and nose_in and mouth_y_gt_h_count == 2:
        cid = YUNET_CASE_2_EYES_NOSE_IN_TWO_MOUTH_Y_GT_H
        logger.debug(
            "[YUNET_CLASSIFY_CASE] cid=%d name=%s rule=case3 mouth_y_gt_h_count==2",
            cid,
            YUNET_CASE_NAMES[cid],
        )
        return cid, YUNET_CASE_NAMES[cid]
    if le_in and re_in and (not nose_in) and mouth_y_gt_h_count == 2:
        cid = YUNET_CASE_3_EYES_IN_NOSE_MOUTH_Y_GT_H
        logger.debug(
            "[YUNET_CLASSIFY_CASE] cid=%d name=%s rule=case4 nose_in==False mouth_y_gt_h_count==2",
            cid,
            YUNET_CASE_NAMES[cid],
        )
        return cid, YUNET_CASE_NAMES[cid]

    # fallback：不满足任何规则时，按 case1 处理（上层可对 unknown 单独做防御）
    cid = YUNET_CASE_8_UNKNOWN
    logger.debug("[YUNET_CLASSIFY_CASE] cid=%d name=%s rule=fallback", cid, YUNET_CASE_NAMES[cid])
    return cid, YUNET_CASE_NAMES[cid]


def _get_yunet_model_path() -> Path:
    """
    YuNet ONNX 模型路径。

    约束：模型文件定位必须走 config + data 的统一查找接口：
    - 文件名来自 config（YUNET_MODEL_FILENAME）
    - 搜索目录来自 config（YUNET_MODEL_SEARCH_DIRS）
    - 由 data.find_model_file 返回绝对路径
    """
    try:
        return find_model_file(YUNET_MODEL_FILENAME, YUNET_MODEL_SEARCH_DIRS)
    except ModelLocateError as e:
        raise FileNotFoundError(str(e)) from None


class YuNetDetector:
    """YuNet 人脸检测，返回单张最大人脸的 bbox [x, y, w, h]。"""

    def __init__(self):
        model_path = str(_get_yunet_model_path())
        self.model = cv2.FaceDetectorYN.create(
            model_path,
            "",
            (320, 320),
            score_threshold=0.1,
        )
        self._max_detect_size = 320

    def detect(self, image: np.ndarray) -> Optional[List[int]]:
        """检测人脸，返回最大人脸的 [x, y, w, h] 或 None。图像过大时先缩小再检测并还原坐标。"""
        out = self.detect_with_landmarks(image)
        if out is None:
            return None
        x, y, w, h = out["bbox"]
        return [int(x), int(y), int(w), int(h)]

    def detect_with_landmarks(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        检测人脸并返回 bbox 与 5 个 landmark（YuNet 标准 14 列格式）。
        缩放检测时，bbox 与 landmarks 一并按 scale 还原。
        返回 None 表示未检测到人脸；否则返回：
        {
            "bbox": [x, y, w, h],
            "right_eye": (x, y),
            "left_eye": (x, y),
            "case_id": int,
        }
        """
        orig_h, orig_w = image.shape[:2]
        h, w = orig_h, orig_w
        scale = 1.0
        if w > self._max_detect_size or h > self._max_detect_size:
            scale = min(self._max_detect_size / w, self._max_detect_size / h)
            w_det, h_det = int(round(w * scale)), int(round(h * scale))
            image = cv2.resize(image, (w_det, h_det), interpolation=cv2.INTER_LINEAR)
            h, w = h_det, w_det
        self.model.setInputSize((w, h))
        _, faces = self.model.detect(image)
        if faces is None or len(faces) == 0:
            return None
        boxes = faces[:, :4]
        areas = boxes[:, 2] * boxes[:, 3]
        idx = int(np.argmax(areas))
        row = faces[idx]
        x, y, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        if scale != 1.0:
            inv = 1.0 / scale
            x, y, bw, bh = x * inv, y * inv, bw * inv, bh * inv
            right_eye = (float(row[4]) * inv, float(row[5]) * inv)
            left_eye = (float(row[6]) * inv, float(row[7]) * inv)
            nose_tip = (float(row[8]) * inv, float(row[9]) * inv)
            right_mouth = (float(row[10]) * inv, float(row[11]) * inv)
            left_mouth = (float(row[12]) * inv, float(row[13]) * inv)
        else:
            right_eye = (float(row[4]), float(row[5]))
            left_eye = (float(row[6]), float(row[7]))
            nose_tip = (float(row[8]), float(row[9]))
            right_mouth = (float(row[10]), float(row[11]))
            left_mouth = (float(row[12]), float(row[13]))
        score = float(row[14]) if len(row) > 14 else 1.0
        full_out = {
            "bbox": [int(x), int(y), int(bw), int(bh)],
            "right_eye": right_eye,
            "left_eye": left_eye,
            "nose_tip": nose_tip,
            "right_mouth": right_mouth,
            "left_mouth": left_mouth,
            "score": score,
        }
        # classify_yunet_case 的边界判断必须基于“与 out 坐标同一坐标系”的图像尺寸。
        # 此处 out 坐标已按 inv 缩放还原到原图，因此必须用原图尺寸，而不是 resized image.shape。
        case_id, case_name = classify_yunet_case(full_out, (orig_h, orig_w))
        return {
            "bbox": full_out["bbox"],
            "left_eye": full_out["left_eye"],
            "right_eye": full_out["right_eye"],
            "score": score,
            "case_name": case_name,
            "case_id": int(case_id),
        }

__all__ = [
    "YuNetDetector",
    "classify_yunet_case",
    "YUNET_CASE_0_ALL_5_IN_BOUNDS",
    "YUNET_CASE_1_EYES_NOSE_IN_ONE_MOUTH_Y_GT_H",
    "YUNET_CASE_2_EYES_NOSE_IN_TWO_MOUTH_Y_GT_H",
    "YUNET_CASE_3_EYES_IN_NOSE_MOUTH_Y_GT_H",
    "YUNET_CASE_4_Y_GT_H_GE4",
    "YUNET_CASE_5_EYE_X_OOB",
    "YUNET_CASE_7_ANY_Y_LT0",
]

