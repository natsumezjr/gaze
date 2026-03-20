"""
交互式测试工具：结合 YuNet + EyeRoiModel + ROI geometry，
用于对比“正常眼 ROI”与“眉毛 ROI”的几何/brow/final confidence 差异。

用法示例（单张图多 ROI）::

    python -m project.tools.test_yunet_pupil_inspect ^
        --image path/to/frame.png ^
        --side left ^
        --roi 200 300 220 120 ^
        --roi 200 150 220 120

日志中会输出四类记录：

- [CONF_SUMMARY] final / geometry / brow / action / reason
- [CONF_GEOMETRY] contain / center / overlap / occ / ratio / pupil-inside-iris 等
- [CONF_BROW] 眉毛相关分量 + union/pupil/iris y 分布
- [CONF_JSON] 一条结构化 json（方便离线分析）
"""

from __future__ import annotations

import json
import logging
from typing import Tuple, Dict, Any, List, Optional
import os
import argparse
import time
import cv2
import numpy as np

from project.config.logging_config import setup_logging
from project.data.data_manager import CameraDataManager
from project.core.recognition.stereo_recognizer import StereoRecognizer
from project.core.recognition.face_detector import YuNetDetector
from project.core.recognition.eye_model_ellseg import EyeRoiModel
from project.core.recognition.roi.constants import (
    UNION_MASK_BBOX_W_EXPAND_RATIO,
    UNION_MASK_BBOX_H_EXPAND_RATIO,
)
from project.core.recognition.roi.geometry import compute_roi_geometry, should_switch_to_detect, RoiGeometry
from project.core.recognition.roi.observation import build_roi_observation

logger = setup_logging(__name__, logging.DEBUG)

IMG_W = 1920
IMG_H = 1080


def _detect_full_face_info_for_debug(yunet: YuNetDetector, image: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    仅用于本测试工具显示完整 YuNet 调试信息（score + 5 点）。
    不影响主链 detect_with_landmarks 的收敛输出契约。
    """
    orig_h, orig_w = image.shape[:2]
    h, w = orig_h, orig_w
    scale = 1.0
    img_infer = image
    if w > yunet._max_detect_size or h > yunet._max_detect_size:  # noqa: SLF001
        scale = min(yunet._max_detect_size / w, yunet._max_detect_size / h)  # noqa: SLF001
        w_det, h_det = int(round(w * scale)), int(round(h * scale))
        img_infer = cv2.resize(image, (w_det, h_det), interpolation=cv2.INTER_LINEAR)
        h, w = h_det, w_det
    yunet.model.setInputSize((w, h))
    _, faces = yunet.model.detect(img_infer)
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
    base_info = yunet.detect_with_landmarks(image)
    case_id = int(base_info.get("case_id", 0)) if base_info else 0
    return {
        "bbox": [int(x), int(y), int(bw), int(bh)],
        "right_eye": right_eye,
        "left_eye": left_eye,
        "nose_tip": nose_tip,
        "right_mouth": right_mouth,
        "left_mouth": left_mouth,
        "score": score,
        "case_id": case_id,
    }


def _pt_in_bounds(pt: Any, w: int, h: int) -> bool:
    if pt is None:
        return False
    try:
        x, y = float(pt[0]), float(pt[1])
    except Exception:
        return False
    return 0.0 <= x <= float(w) and 0.0 <= y <= float(h)


def _pt_xy_int(pt: Any) -> Optional[Tuple[int, int]]:
    if pt is None:
        return None
    try:
        return int(round(float(pt[0]))), int(round(float(pt[1])))
    except Exception:
        return None


def classify_yunet_face_info(
    face_info: Optional[Dict[str, Any]],
    img_w: int = IMG_W,
    img_h: int = IMG_H,
    hint_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    将 YuNet 输出的人脸 5 点按规则分类，并预留提示接口（后续可改为事件驱动）。

    hint_callback 预留：若不为 None，可接收 (event_name:str, payload:dict)。
    """
    keys = ["left_eye", "right_eye", "nose_tip", "left_mouth", "right_mouth"]
    pts = {k: (face_info.get(k) if face_info else None) for k in keys}

    # 越界提示优先级：y<0 / 眼点 x 越界
    any_y_lt0 = False
    eye_x_oob = None  # "left" / "right"
    for k, pt in pts.items():
        if pt is None:
            continue
        try:
            x, y = float(pt[0]), float(pt[1])
        except Exception:
            continue
        if y < 0:
            any_y_lt0 = True
        if k in ("left_eye", "right_eye"):
            if x > img_w:
                eye_x_oob = "right"
            if x < 0:
                eye_x_oob = "left"

    # y>h 计数（含嘴角）
    y_gt_h_count = 0
    mouth_y_gt_h_count = 0
    for k, pt in pts.items():
        if pt is None:
            continue
        try:
            y = float(pt[1])
        except Exception:
            continue
        if y > img_h:
            y_gt_h_count += 1
            if k in ("left_mouth", "right_mouth"):
                mouth_y_gt_h_count += 1

    le_in = _pt_in_bounds(pts["left_eye"], img_w, img_h)
    re_in = _pt_in_bounds(pts["right_eye"], img_w, img_h)
    nose_in = _pt_in_bounds(pts["nose_tip"], img_w, img_h)
    lm_in = _pt_in_bounds(pts["left_mouth"], img_w, img_h)
    rm_in = _pt_in_bounds(pts["right_mouth"], img_w, img_h)
    all_in = le_in and re_in and nose_in and lm_in and rm_in

    category = "unknown"
    if any_y_lt0:
        category = "case7_any_y_lt0"
        if hint_callback is not None:
            hint_callback("hint_head_move_down", {"reason": "some_landmark_y_lt0", "pts": pts})
    elif eye_x_oob is not None:
        category = "case6_eye_x_oob"
        if hint_callback is not None:
            hint_callback(
                "hint_face_move_horizontal",
                {"reason": "eye_landmark_x_oob", "direction": eye_x_oob, "pts": pts},
            )
    elif y_gt_h_count >= 4:
        category = "case5_y_gt_h_ge4"
        if hint_callback is not None:
            hint_callback("hint_face_move_up", {"reason": "many_landmark_y_gt_h", "pts": pts})
    elif all_in:
        category = "case1_all_5_in_bounds"
    elif le_in and re_in and nose_in and (mouth_y_gt_h_count == 1):
        category = "case2_eyes_nose_in_one_mouth_y_gt_h"
    elif le_in and re_in and nose_in and (mouth_y_gt_h_count == 2):
        category = "case3_eyes_nose_in_two_mouth_y_gt_h"
    elif le_in and re_in and (not nose_in) and (mouth_y_gt_h_count == 2):
        category = "case4_eyes_in_nose_mouth_y_gt_h"

    return {
        "category": category,
        "pts": {k: _pt_xy_int(v) for k, v in pts.items()},
        "bbox": (face_info.get("bbox") if face_info else None),
        "score": (float(face_info.get("score", 0.0)) if face_info else None),
        "flags": {
            "any_y_lt0": any_y_lt0,
            "eye_x_oob": eye_x_oob,
            "y_gt_h_count": int(y_gt_h_count),
            "mouth_y_gt_h_count": int(mouth_y_gt_h_count),
        },
    }


def _mean(xs: List[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def _init_acc() -> Dict[str, Any]:
    return {
        "n": 0,
        "eye_to_roi_center_dx": [],
        "eye_to_roi_center_dy": [],
        "eye_to_roi_center_dx_r": [],  # / roi_w
        "eye_to_roi_center_dy_r": [],  # / roi_h
        "eye_to_roi_tl_x": [],
        "eye_to_roi_tl_y": [],
        "bbox_center_to_roi_center_dx": [],
        "bbox_center_to_roi_center_dy": [],
        "bbox_tl_to_roi_tl_dx": [],
        "bbox_tl_to_roi_tl_dy": [],
        "bbox_center_to_roi_center_dx_r": [],
        "bbox_center_to_roi_center_dy_r": [],
        "bbox_tl_to_roi_tl_dx_r": [],
        "bbox_tl_to_roi_tl_dy_r": [],
        # ROI 相对人脸 bbox 的归一化比例（便于对照 INIT_BAND_*）
        "roi_x_in_face_r": [],   # (roi_x - face_x) / face_w
        "roi_y_in_face_r": [],   # (roi_y - face_y) / face_h
        "roi_w_in_face_r": [],   # roi_w / face_w
        "roi_h_in_face_r": [],   # roi_h / face_h
        # 基于 union_mask_bbox 的 stable proposal 扩展后比例（便于回填 UNION_*_EXPAND_RATIO）
        "union_expand_w_over_face_w": [],
        "union_expand_h_over_face_h": [],
    }

def _category_to_case_idx(category: str) -> int:
    # case1~case4 -> idx 0~3，其余 fallback 0
    if category.startswith("case1_"):
        return 0
    if category.startswith("case2_"):
        return 1
    if category.startswith("case3_"):
        return 2
    if category.startswith("case4_"):
        return 3
    return 0


def _update_category_stats(
    acc: Dict[str, Any],
    category: str,
    yunet_cls: Dict[str, Any],
    roi_xywh: Tuple[int, int, int, int],
    ellseg_result: Any,
    eye_key: str = "right_eye",
) -> None:
    if category not in acc:
        acc[category] = _init_acc()
    st = acc[category]
    st["n"] += 1

    x, y, w, h = roi_xywh
    roi_center = (w * 0.5, h * 0.5)

    pts = yunet_cls.get("pts", {}) or {}
    # 统计：按 eye_key 选择 YuNet 眼点（用于标定 left/right）
    eye_pt = pts.get(eye_key)
    if eye_pt is not None:
        eye_center_img = (float(eye_pt[0]), float(eye_pt[1]))
        eye_in_roi = (eye_center_img[0] - x, eye_center_img[1] - y)
        st["eye_to_roi_center_dx"].append(float(eye_in_roi[0] - roi_center[0]))
        st["eye_to_roi_center_dy"].append(float(eye_in_roi[1] - roi_center[1]))
        st["eye_to_roi_center_dx_r"].append(float((eye_in_roi[0] - roi_center[0]) / max(1.0, float(w))))
        st["eye_to_roi_center_dy_r"].append(float((eye_in_roi[1] - roi_center[1]) / max(1.0, float(h))))
        st["eye_to_roi_tl_x"].append(float(eye_in_roi[0]))
        st["eye_to_roi_tl_y"].append(float(eye_in_roi[1]))

    bbox_roi = None
    if bbox_roi is None:
        try:
            seg = getattr(ellseg_result, "segmentation_mask", None)
            if seg is not None:
                ys, xs = np.where(seg.astype(np.uint8) > 0)
                if xs.size > 0 and ys.size > 0:
                    x0, y0 = int(xs.min()), int(ys.min())
                    x1, y1 = int(xs.max()), int(ys.max())
                    bbox_roi = (x0, y0, int(x1 - x0 + 1), int(y1 - y0 + 1))
        except Exception:
            bbox_roi = None

    if bbox_roi is None:
        return

    bx, by, bw, bh = bbox_roi
    bbox_center = (bx + bw * 0.5, by + bh * 0.5)
    dx_c = float(bbox_center[0] - roi_center[0])
    dy_c = float(bbox_center[1] - roi_center[1])
    st["bbox_center_to_roi_center_dx"].append(dx_c)
    st["bbox_center_to_roi_center_dy"].append(dy_c)
    st["bbox_tl_to_roi_tl_dx"].append(float(bx))
    st["bbox_tl_to_roi_tl_dy"].append(float(by))

    denom_w = float(max(1, bw))
    denom_h = float(max(1, bh))
    st["bbox_center_to_roi_center_dx_r"].append(dx_c / denom_w)
    st["bbox_center_to_roi_center_dy_r"].append(dy_c / denom_h)
    st["bbox_tl_to_roi_tl_dx_r"].append(float(bx) / denom_w)
    st["bbox_tl_to_roi_tl_dy_r"].append(float(by) / denom_h)

    # ROI 相对人脸 bbox 的比例（仅当有 YuNet face bbox）
    face_bbox = yunet_cls.get("bbox")
    if face_bbox is not None and len(face_bbox) >= 4:
        fx, fy, fw, fh = face_bbox[0], face_bbox[1], face_bbox[2], face_bbox[3]
        if fw and fh and fw > 0 and fh > 0:
            st["roi_x_in_face_r"].append(float((x - fx) / float(fw)))
            st["roi_y_in_face_r"].append(float((y - fy) / float(fh)))
            st["roi_w_in_face_r"].append(float(w / float(fw)))
            st["roi_h_in_face_r"].append(float(h / float(fh)))
            if bbox_roi is not None:
                _, _, ubw, ubh = bbox_roi
                case_idx = _category_to_case_idx(category)
                exp_w = float(ubw) * float(UNION_MASK_BBOX_W_EXPAND_RATIO[case_idx])
                exp_h = float(ubh) * float(UNION_MASK_BBOX_H_EXPAND_RATIO[case_idx])
                st["union_expand_w_over_face_w"].append(exp_w / float(fw))
                st["union_expand_h_over_face_h"].append(exp_h / float(fh))


def _log_category_stats(acc: Dict[str, Any], eye_key: str) -> None:
    for cat, st in acc.items():
        n = int(st.get("n", 0))
        if n <= 0:
            continue
        logger.info(
            "[YUNET_CASE_STATS] eye_key=%s case=%s n=%d "
            "eye_to_roi_center_mean(dx,dy)=(%.2f,%.2f) eye_to_roi_tl_mean(x,y)=(%.2f,%.2f) "
            "eye_to_roi_center_mean_r(dx,dy)=(%.3f,%.3f) "
            "bbox_center_to_roi_center_mean(dx,dy)=(%.2f,%.2f) bbox_tl_to_roi_tl_mean(dx,dy)=(%.2f,%.2f) "
            "bbox_center_to_roi_center_mean_r(dx,dy)=(%.3f,%.3f) bbox_tl_to_roi_tl_mean_r(dx,dy)=(%.3f,%.3f) "
            "roi_in_face_mean_r(x,y,w,h)=(%.3f,%.3f,%.3f,%.3f) "
            "union_expand_over_face_mean_r(w,h)=(%.3f,%.3f)",
            eye_key,
            cat,
            n,
            _mean(st["eye_to_roi_center_dx"]),
            _mean(st["eye_to_roi_center_dy"]),
            _mean(st["eye_to_roi_tl_x"]),
            _mean(st["eye_to_roi_tl_y"]),
            _mean(st["eye_to_roi_center_dx_r"]),
            _mean(st["eye_to_roi_center_dy_r"]),
            _mean(st["bbox_center_to_roi_center_dx"]),
            _mean(st["bbox_center_to_roi_center_dy"]),
            _mean(st["bbox_tl_to_roi_tl_dx"]),
            _mean(st["bbox_tl_to_roi_tl_dy"]),
            _mean(st["bbox_center_to_roi_center_dx_r"]),
            _mean(st["bbox_center_to_roi_center_dy_r"]),
            _mean(st["bbox_tl_to_roi_tl_dx_r"]),
            _mean(st["bbox_tl_to_roi_tl_dy_r"]),
            _mean(st["roi_x_in_face_r"]),
            _mean(st["roi_y_in_face_r"]),
            _mean(st["roi_w_in_face_r"]),
            _mean(st["roi_h_in_face_r"]),
            _mean(st["union_expand_w_over_face_w"]),
            _mean(st["union_expand_h_over_face_h"]),
        )


def _save_eye_ratio_calibration(acc: Dict[str, Any], eye_key: str, frame_id: int) -> None:
    """
    将各 case 的 eye_to_roi_center_mean_r 写入文件，便于回填到 ROI 初始化比例配置。
    """
    try:
        project_root = os.path.dirname(os.path.dirname(__file__))
        base_dir = os.path.join(project_root, "debug", "roi_conf_inspect")
        out_dir = os.path.join(base_dir, "calib")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"yunet_eye_ratio_calib_{eye_key}.json")
        payload = {
            "eye_key": eye_key,
            "last_frame_id": int(frame_id),
            "cases": {},
        }
        for cat, st in acc.items():
            n = int(st.get("n", 0))
            if n <= 0:
                continue
            payload["cases"][cat] = {
                "n": n,
                "eye_to_roi_center_mean_r": [
                    _mean(st.get("eye_to_roi_center_dx_r", [])),
                    _mean(st.get("eye_to_roi_center_dy_r", [])),
                ],
            }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info("saved eye ratio calibration to %s", out_path)
    except Exception as e:
        logger.error("failed to save eye ratio calibration: %s", e, exc_info=True)


def inspect_single_roi_geometry_gate(
    side: str,
    frame_id: int,
    full_bgr: np.ndarray,
    roi_xywh: Tuple[int, int, int, int],
    eye_model: EyeRoiModel,
) -> Tuple[Dict[str, Any], Any, float]:
    """
    在给定图像 full_bgr 上，对指定 ROI 计算置信度，并输出详细日志。

    Params:
        side: "left" or "right"
        frame_id: 帧编号（日志用）
        full_bgr: 整帧 BGR 图像
        roi_xywh: (x, y, w, h) 图像坐标下 ROI
        eye_model: EyeRoiModel 实例

    Returns:
        report: build_confidence_debug_dict(...) + 额外上下文字段
    """
    x, y, w, h = roi_xywh
    H, W = full_bgr.shape[:2]

    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > W or y + h > H:
        logger.error(
            "[CONF_SUMMARY] frame=%d side=%s roi=%s final=%.3f geo=%.3f brow=%.3f action=%s reason=%s",
            frame_id,
            side,
            roi_xywh,
            0.0,
            0.0,
            0.0,
            "recover",
            "invalid_roi_bounds",
        )
        return {
            "frame_id": frame_id,
            "side": side,
            "roi": roi_xywh,
            "valid": False,
            "reason": "invalid_roi_bounds",
        }

    patch_bgr = full_bgr[y: y + h, x: x + w]
    if patch_bgr.size == 0:
        logger.error(
            "[CONF_SUMMARY] frame=%d side=%s roi=%s final=%.3f geo=%.3f brow=%.3f action=%s reason=%s",
            frame_id,
            side,
            roi_xywh,
            0.0,
            0.0,
            0.0,
            "recover",
            "empty_patch",
        )
        return {
            "frame_id": frame_id,
            "side": side,
            "roi": roi_xywh,
            "valid": False,
            "reason": "empty_patch",
        }

    patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    t0_infer = time.perf_counter()
    result = eye_model.infer(patch_rgb)
    infer_ms = (time.perf_counter() - t0_infer) * 1000.0

    if result is None or not result.valid:
        logger.warning(
            "[CONF_SUMMARY] frame=%d side=%s roi=%s final=%.3f geo=%.3f brow=%.3f action=%s reason=%s",
            frame_id,
            side,
            roi_xywh,
            0.0,
            0.0,
            0.0,
            "recover",
            "ellseg_invalid",
        )
        return (
            {
            "frame_id": frame_id,
            "side": side,
            "roi": roi_xywh,
            "valid": False,
            "reason": "ellseg_invalid",
            },
            result,
            float(infer_ms),
        )

    # Build ROI observation directly from EllSegResult (no semantic dependency).
    obs = build_roi_observation(result=result, roi_xywh=roi_xywh, side=side)

    conf = compute_roi_geometry(obs)
    conf.face_bbox_is_stable = True
    conf.face_bbox_corner_max_shift = 0.0
    need_recover = should_switch_to_detect(
        geometry_confidence=conf.geometry_confidence,
        face_bbox_is_stable=conf.face_bbox_is_stable,
    )
    recover_reason = "low_geometry_confidence_recover" if need_recover else None

    # EllSeg 椭圆可视化确认窗：在 ROI patch 上叠加瞳孔/虹膜椭圆，便于人工确认
    try:
        vis = patch_bgr.copy()
        pupil_el = getattr(result, "pupil_ellipse", None)
        iris_el = getattr(result, "iris_ellipse", None)
        if pupil_el is not None:
            center = (int(round(pupil_el.cx)), int(round(pupil_el.cy)))
            axes = (
                max(1, int(round(pupil_el.major_axis * 0.5))),
                max(1, int(round(pupil_el.minor_axis * 0.5))),
            )
            cv2.ellipse(
                vis,
                center,
                axes,
                float(pupil_el.angle_deg),
                0.0,
                360.0,
                (0, 0, 255),  # red for pupil
                2,
            )
        if iris_el is not None:
            center = (int(round(iris_el.cx)), int(round(iris_el.cy)))
            axes = (
                max(1, int(round(iris_el.major_axis * 0.5))),
                max(1, int(round(iris_el.minor_axis * 0.5))),
            )
            cv2.ellipse(
                vis,
                center,
                axes,
                float(iris_el.angle_deg),
                0.0,
                360.0,
                (0, 255, 0),  # green for iris
                2,
            )
        win_preview = "EllSeg ROI Preview (press any key to continue)"
        cv2.namedWindow(win_preview, cv2.WINDOW_NORMAL)
        cv2.imshow(win_preview, vis)
        cv2.waitKey(0)
        cv2.destroyWindow(win_preview)
    except Exception as e:
        logger.error("failed to show EllSeg ROI preview: %s", e, exc_info=True)

    extra = {
        "frame_id": frame_id,
        "side": side,
        "roi": roi_xywh,
        "ellseg_infer_ms": float(infer_ms),
    }
    report = {
        **extra,
        "valid": True,
        "geometry_confidence": float(conf.geometry_confidence),
        "contain_score": float(conf.contain_score),
        "center_score": float(conf.center_score),
        "overlap_score": float(conf.overlap_score),
        "need_recover": bool(need_recover),
        "recover_reason": recover_reason,
        "obs": {
            "pupil_inside_iris_ratio": float(obs.pupil_inside_iris_ratio),
            "pupil_center_inside_iris_score": float(obs.pupil_center_inside_iris_score),
            "pupil_center_inside_iris_d2": float(obs.pupil_center_inside_iris_d2),
            "pupil_overlap_score": float(obs.pupil_overlap_score),
            "iris_overlap_score": float(obs.iris_overlap_score),
            "union_mask_bbox": obs.union_mask_bbox,
        },
    }

    # 1) SUMMARY
    logger.info(
        "[CONF_SUMMARY] frame=%d side=%s roi=%s geo=%.3f recover=%s reason=%s",
        frame_id,
        side,
        roi_xywh,
        conf.geometry_confidence,
        need_recover,
        recover_reason,
    )

    # 2) GEOMETRY 分项
    logger.info(
        "[CONF_GEOMETRY] contain=%.3f center=%.3f overlap=%.3f inside_ratio=%.3f center_inside=%.3f d2=%.3f pupil_overlap=%.3f iris_overlap=%.3f",
        conf.contain_score,
        conf.center_score,
        conf.overlap_score,
        obs.pupil_inside_iris_ratio,
        obs.pupil_center_inside_iris_score,
        obs.pupil_center_inside_iris_d2,
        obs.pupil_overlap_score,
        obs.iris_overlap_score,
    )

    # 4) 结构化 JSON（仅用于日志，人读为主）
    logger.info("[CONF_JSON] %s", json.dumps(report, ensure_ascii=False, default=str))

    # 5) 新旧主链对照决策日志（便于人工比对 ROI 与眉毛 ROI）
    logger.info(
        "[CONF_DECISION] frame=%d side=%s recover=%s geo=%.3f",
        frame_id,
        side,
        need_recover,
        conf.geometry_confidence,
    )

    # 6) 写入独立 jsonl + 彩图到 debug 目录，方便离线统计
    try:
        project_root = os.path.dirname(os.path.dirname(__file__))
        base_dir = os.path.join(project_root, "debug", "roi_conf_inspect")
        os.makedirs(base_dir, exist_ok=True)

        # jsonl：每侧一份
        jsonl_path = os.path.join(base_dir, f"roi_conf_inspect_{side}.jsonl")
        with open(jsonl_path, "a", encoding="utf-8") as f_js:
            f_js.write(json.dumps(report, ensure_ascii=False, default=str) + "\n")

        # ROI 原图：按 ROI + 帧编号命名，便于与 segmap/JSON 对齐
        roi_dir = os.path.join(base_dir, "roi")
        os.makedirs(roi_dir, exist_ok=True)
        roi_img_path = os.path.join(
            roi_dir,
            f"roi_conf_frame{frame_id}_side-{side}_x{x}_y{y}_w{w}_h{h}.png",
        )
        cv2.imwrite(roi_img_path, patch_bgr)
        logger.info("saved roi original patch to %s", roi_img_path)

        # seg map 彩图：按 ROI + 帧编号命名
        if result is not None and result.segmentation_mask is not None:
            seg_dir = os.path.join(base_dir, "segmap")
            os.makedirs(seg_dir, exist_ok=True)
            seg = result.segmentation_mask.astype("uint8")
            h_seg, w_seg = seg.shape[:2]
            seg_vis = np.zeros((h_seg, w_seg, 3), dtype=np.uint8)
            seg_vis[seg == 1] = (0, 255, 0)  # iris -> 绿
            seg_vis[seg == 2] = (0, 0, 255)  # pupil -> 红
            seg_vis = cv2.resize(seg_vis, (w, h), interpolation=cv2.INTER_NEAREST)
            out_name = os.path.join(
                seg_dir,
                f"roi_conf_seg_frame{frame_id}_side-{side}_x{x}_y{y}_w{w}_h{h}.png",
            )
            cv2.imwrite(out_name, seg_vis)
            logger.info("saved seg_map visualization to %s", out_name)
    except Exception as e:
        logger.error("failed to write roi_conf_inspect debug artifacts: %s", e, exc_info=True)

    return report, result, float(infer_ms)


def main():
    """
    取消命令行参数，改为：
    - 使用 rgbd 摄像头（idx=1）采集一帧
    - 若分辨率为 3840x1280，则取左半部分作为左眼彩图
    - 模仿 recognition_runtime.show_frame 的逻辑缩放显示（最大宽度 1280，保持宽高比）
    - 按空格键截取当前帧并运行原有 ROI 置信度评估逻辑
    其它置信度/日志逻辑保持不变。
    """
    parser = argparse.ArgumentParser(
        description="Interactive ROI confidence inspector with optional ROI arguments."
    )
    parser.add_argument(
        "--side",
        type=str,
        default="left",
        choices=["left", "right"],
        help="eye side to inspect (left/right)",
    )
    parser.add_argument(
        "--roi",
        type=int,
        nargs="*",
        metavar="X Y W H",
        help=(
            "可选：X Y W H 四个整数。"
            "若仅写 --roi 而不跟参数，则进入“只截一帧 + 交互多 ROI 选取”的模式；"
            "若提供 4 个参数，则对该 ROI 直接做一次置信度评估并仅截取一帧。"
        ),
    )
    args = parser.parse_args()

    # 与 runtime 一致：使用 CameraDataManager + StereoRecognizer 管理摄像头与左右图。
    camera_manager = CameraDataManager(rgb_d=True)
    cap = camera_manager.get_cap()
    if cap is None or not cap.isOpened():
        logger.error("无法从 CameraDataManager 获取有效摄像头句柄")
        return

    # StereoRecognizer 目前不直接用于识别，只是保持与 runtime 的初始化路径一致
    stereo_recognizer = StereoRecognizer(camera_manager)
    _ = stereo_recognizer  # 避免未使用警告

    # YuNet 人脸检测，用于在截屏后在图上叠加人脸与五点 landmark
    yunet = YuNetDetector()
    case_acc: Dict[str, Any] = {}
    perf = {
        "frames": 0,
        "yunet_ms": [],
        "ellseg_ms_per_frame": [],
    }

    def hint_callback(event_name: str, payload: Dict[str, Any]) -> None:
        # 预留接口：后续写成事件驱动（发布事件/写队列/触发 UI 提示等）
        _ = (event_name, payload)

    win_name = "ROI-Confidence-Inspect"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    display_params_logged = [False]

    eye_model = EyeRoiModel()

    frame_id = 0
    roi_cli = args.roi
    single_frame_only = roi_cli is not None
    # 标定用：当参数是 left 时统计 left_eye；否则统计 right_eye
    calib_eye_key = "left_eye" if args.side == "left" else "right_eye"
    try:
        while True:
            captured_frame = None

            # ------- 实时预览，按空格截取一帧 -------
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.error("无法从摄像头读取帧")
                    return

                camera_manager.add_frame(frame_id, frame)
                pair = camera_manager.get_stereo_pair(frame_id)
                if pair is None:
                    logger.warning("非双目帧或无法获取左右图，frame_id=%d", frame_id)
                    frame_id += 1
                    continue
                left_image, right_image = pair
                _ = right_image

                left_bgr = left_image.data
                if left_bgr is None or left_bgr.size == 0:
                    logger.warning("左图数据为空，frame_id=%d", frame_id)
                    frame_id += 1
                    continue

                src_h, src_w = left_bgr.shape[:2]
                max_display_w = 1280
                if src_w > max_display_w:
                    scale = max_display_w / float(src_w)
                    display_w, display_h = max_display_w, int(round(src_h * scale))
                    frame_display = cv2.resize(
                        left_bgr, (display_w, display_h), interpolation=cv2.INTER_LINEAR
                    )
                else:
                    display_w, display_h = src_w, src_h
                    frame_display = left_bgr

                if not display_params_logged[0]:
                    try:
                        cv2.resizeWindow(win_name, display_w, display_h)
                    except cv2.error:
                        pass
                    display_params_logged[0] = True
                cv2.imshow(win_name, frame_display)

                key = cv2.waitKey(1) & 0xFF
                if key == 32:
                    # 空格截取当前左图帧，进行一次 ROI 置信度检查
                    captured_frame = left_bgr.copy()
                    logger.info(
                        "Captured one frame from CameraDataManager for ROI confidence inspection. size=%dx%d",
                        src_w,
                        src_h,
                    )
                    break

                frame_id += 1

            if captured_frame is None:
                # 正常不会到这里，仅防御
                continue

            img = captured_frame
            H, W = img.shape[:2]

            # 侧别：来自命令行参数，默认 left
            side = args.side

            # 在截屏后的帧上叠加 YuNet 输出（人脸框 + 五点 + 置信度），并保存 YuNet 数据 + 分类
            yunet_ms = 0.0
            face_info = None
            cls = classify_yunet_face_info(None, img_w=IMG_W, img_h=IMG_H, hint_callback=hint_callback)
            try:
                t0_y = time.perf_counter()
                face_info = _detect_full_face_info_for_debug(yunet, img)
                yunet_ms = (time.perf_counter() - t0_y) * 1000.0
                cls = classify_yunet_face_info(face_info, img_w=IMG_W, img_h=IMG_H, hint_callback=hint_callback)

                if face_info is not None:
                    fx, fy, fw, fh = face_info["bbox"]
                    cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), (0, 255, 255), 2)

                    def _draw_point(pt, color, label):
                        if pt is None:
                            return None
                        x, y = int(round(float(pt[0]))), int(round(float(pt[1])))
                        cv2.circle(img, (x, y), 3, color, -1)
                        cv2.putText(
                            img,
                            f"{label}({x},{y})",
                            (x + 4, y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            color,
                            1,
                            cv2.LINE_AA,
                        )
                        return (x, y)

                    le_xy = _draw_point(face_info.get("left_eye"), (0, 255, 0), "LE")
                    re_xy = _draw_point(face_info.get("right_eye"), (0, 255, 0), "RE")
                    n_xy = _draw_point(face_info.get("nose_tip"), (255, 0, 0), "N")
                    lm_xy = _draw_point(face_info.get("left_mouth"), (0, 0, 255), "LM")
                    rm_xy = _draw_point(face_info.get("right_mouth"), (0, 0, 255), "RM")

                    score = face_info.get("score", 0.0)
                    cv2.putText(
                        img,
                        f"YuNet score={score:.3f}",
                        (fx, max(0, fy - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                else:
                    le_xy = re_xy = n_xy = lm_xy = rm_xy = None

                # 无论是否检测到人脸，都在左上角打印关键点像素坐标；缺失则为 None
                lines = [
                    f"LE: {le_xy}",
                    f"RE: {re_xy}",
                    f"N: {n_xy}",
                    f"LM: {lm_xy}",
                    f"RM: {rm_xy}",
                    f"CASE: {cls.get('category')}",
                ]
                x0, y0 = 10, 20
                for i, s in enumerate(lines):
                    cv2.putText(
                        img,
                        s,
                        (x0, y0 + 16 * i),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                # 保存 YuNet 原始输出与分类（每帧一份）
                project_root = os.path.dirname(os.path.dirname(__file__))
                base_dir = os.path.join(project_root, "debug", "roi_conf_inspect")
                os.makedirs(base_dir, exist_ok=True)
                yunet_dir = os.path.join(base_dir, "yunet")
                os.makedirs(yunet_dir, exist_ok=True)
                yunet_path = os.path.join(yunet_dir, f"yunet_frame{frame_id}_side-{side}.json")
                with open(yunet_path, "w", encoding="utf-8") as f_y:
                    json.dump(
                        {
                            "frame_id": frame_id,
                            "side": side,
                            "face_info": face_info,
                            "classification": cls,
                        },
                        f_y,
                        ensure_ascii=False,
                        default=str,
                        indent=2,
                    )
                logger.info("saved yunet info to %s", yunet_path)
            except Exception as e:
                logger.error("failed to run YuNet overlay on captured frame: %s", e, exc_info=True)

            # ROI 生成策略：
            # 1) 若命令行提供了 4 个 --roi 参数，则直接使用该 ROI；
            # 2) 若仅写 --roi（无参数），进入交互式选框模式，但只处理这一帧；
            # 3) 若未提供 --roi，则为原交互模式，可多帧多次截取。
            roi_list: List[Tuple[int, int, int, int]] = []
            if roi_cli:
                if len(roi_cli) != 4:
                    logger.error(
                        "--roi 期望 4 个整数参数 X Y W H，但收到 %d 个: %s",
                        len(roi_cli),
                        roi_cli,
                    )
                    camera_manager.release_camera()
                    cv2.destroyAllWindows()
                    return
                x, y, w, h = roi_cli
                if w > 0 and h > 0:
                    roi_list.append((int(x), int(y), int(w), int(h)))
            else:
                max_rois = 4
                sel_win = "Select ROI - ENTER/SPACE to confirm, ESC to finish"
                for _ in range(max_rois):
                    cv2.namedWindow(sel_win, cv2.WINDOW_NORMAL)
                    cv2.imshow(sel_win, img)
                    r = cv2.selectROI(sel_win, img, showCrosshair=True, fromCenter=False)
                    cv2.destroyWindow(sel_win)
                    x, y, w, h = [int(v) for v in r]
                    if w <= 0 or h <= 0:
                        break
                    roi_list.append((x, y, w, h))

            if not roi_list:
                roi_list.append((0, 0, W, H))

            logger.info(
                "Start ROI confidence inspection: source=camera idx=1 side=%s rois=%s",
                side,
                roi_list,
            )

            ellseg_ms_this_frame = 0.0
            for idx, roi_xywh in enumerate(roi_list):
                logger.info("---------- ROI #%d %s ----------", idx, roi_xywh)
                report, ellseg_result, infer_ms = inspect_single_roi_geometry_gate(
                    side=side,
                    frame_id=frame_id,
                    full_bgr=img,
                    roi_xywh=roi_xywh,
                    eye_model=eye_model,
                )
                ellseg_ms_this_frame += float(infer_ms)
                # 当前 ROI 相对 face bbox 比例，逐帧输出（供你截最小/最大）
                face_bbox = cls.get("bbox")
                if face_bbox is not None and len(face_bbox) >= 4:
                    fx, fy, fw, fh = face_bbox[0], face_bbox[1], face_bbox[2], face_bbox[3]
                    if fw > 0 and fh > 0:
                        rx = (roi_xywh[0] - fx) / float(fw)
                        ry = (roi_xywh[1] - fy) / float(fh)
                        rw = roi_xywh[2] / float(fw)
                        rh = roi_xywh[3] / float(fh)
                        logger.info(
                            "[ROI_FACE_RATIO] frame=%d side=%s case=%s roi_in_face_r(x,y,w,h)=(%.3f,%.3f,%.3f,%.3f)",
                            frame_id,
                            side,
                            cls.get("category"),
                            rx,
                            ry,
                            rw,
                            rh,
                        )
                # 基于本帧 YuNet 分类结果，统计“眼睛中心 vs ROI”“mask bbox vs ROI”的均值（info 输出）
                try:
                    _update_category_stats(
                        case_acc,
                        cls["category"],
                        cls,
                        roi_xywh,
                        ellseg_result,
                        eye_key=calib_eye_key,
                    )
                except Exception as e:
                    logger.error("failed to update case stats: %s", e, exc_info=True)

            logger.info("ROI confidence inspection done for this frame.")
            _log_category_stats(case_acc, eye_key=calib_eye_key)
            # 全量汇总：UNION 扩展后相对 face 的平均比例
            all_union_w = []
            all_union_h = []
            for st in case_acc.values():
                all_union_w.extend(st.get("union_expand_w_over_face_w", []))
                all_union_h.extend(st.get("union_expand_h_over_face_h", []))
            logger.info(
                "[UNION_EXPAND_GLOBAL_MEAN] eye_key=%s union_expand_over_face_mean_r(w,h)=(%.3f,%.3f)",
                calib_eye_key,
                _mean(all_union_w),
                _mean(all_union_h),
            )
            _save_eye_ratio_calibration(case_acc, eye_key=calib_eye_key, frame_id=frame_id)

            # 性能统计：平均每帧 YuNet 与 EllSeg 耗时（info 日志）
            perf["frames"] += 1
            perf["yunet_ms"].append(float(yunet_ms))
            perf["ellseg_ms_per_frame"].append(float(ellseg_ms_this_frame))
            logger.info(
                "[PERF_AVG] frames=%d avg_yunet_ms_per_frame=%.2f avg_ellseg_ms_per_frame=%.2f",
                perf["frames"],
                _mean(perf["yunet_ms"]),
                _mean(perf["ellseg_ms_per_frame"]),
            )

            # 若通过命令行提供了 --roi（无论是否携带坐标），则仅截取这一帧后退出；
            # 否则保持原交互行为（继续下一帧）。
            if single_frame_only:
                logger.info("single-frame ROI mode enabled by --roi, exiting after this frame.")
                return
            else:
                logger.info("按空格继续下一帧，按 Ctrl+C 退出。")
                # 处理完当前帧后暂停，避免立刻进入下一帧预览（更符合交互节奏）
                try:
                    cv2.imshow(win_name, img)
                    while True:
                        k2 = cv2.waitKey(0) & 0xFF
                        if k2 == 32:  # SPACE
                            break
                except cv2.error:
                    # 某些环境下窗口不可用，直接不阻塞
                    pass
    except KeyboardInterrupt:
        logger.info("收到 Ctrl+C，退出 test_yunet_pupil_inspect。")
    finally:
        camera_manager.release_camera()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()