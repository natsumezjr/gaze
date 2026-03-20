"""
工具：专门验证 ROI 初始化的 detect / recover 分支。

交互方式：
- 实时预览
- 按空格：截取当前帧 -> 跑一次 YuNet -> 分别走 DETECT 与 RECOVER 初始化 -> 在图上画出 ROI
- 再按空格：继续下一帧
- Ctrl+C：退出（确保释放摄像头并关闭窗口）
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np

from project.config.logging_config import setup_logging
from project.data.data_manager import CameraDataManager
from project.core.recognition.face_detector import YuNetDetector
from project.core.recognition.roi.state_machine import RoiStateMachine
from project.core.recognition.roi.constants import (
    MODE_DETECT,
    MODE_RECOVER,
    INIT_BAND_LEFT_TL_X_RATIO,
    INIT_BAND_RIGHT_TL_X_RATIO,
    INIT_BAND_TL_Y_RATIO,
    INIT_BAND_W_RATIO,
    INIT_BAND_H_RATIO,
    INIT_LEFT_EYE_DX_RATIO,
    INIT_LEFT_EYE_DY_RATIO,
    INIT_RIGHT_EYE_DX_RATIO,
    INIT_RIGHT_EYE_DY_RATIO,
    INIT_ROI_EXPAND_RATIO,
    ROI_MIN_W_FACE_RATIO,
    ROI_MIN_H_FACE_RATIO,
    ROI_MAX_W_FACE_RATIO,
    ROI_MAX_H_FACE_RATIO,
)

logger = setup_logging(__name__, logging.DEBUG)


def _draw_roi(img: np.ndarray, roi: Optional[Tuple[int, int, int, int]], color, label: str) -> None:
    if roi is None:
        cv2.putText(
            img,
            f"{label}: None",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        return
    x, y, w, h = roi
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(
        img,
        f"{label}: ({x},{y},{w},{h})",
        (x, max(0, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def _compute_init_rois(
    side: str,
    face_bbox: Tuple[int, int, int, int],
    face_info: Optional[Dict[str, Any]],
    image_shape: Tuple[int, int],
) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]]]:
    """
    复现 EyeRoiTracker._build_initial_roi_from_face_info 的三段结果：
    - fixed ROI（仅按 case 的固定比例均值）
    - eye-based ROI（按 left_eye/right_eye + dx/dy ratio）
    - final ROI（fixed 与 eye-based 融合 + expand + clip）
    """
    fx, fy, fw, fh = face_bbox
    img_h, img_w = int(image_shape[0]), int(image_shape[1])

    case_id = int(face_info.get("case_id", 0)) if face_info else 0
    # 仅使用 case1~case4 的均值；其余 case 统一回退到 case1（idx=0）
    idx = int(case_id) if 0 <= int(case_id) <= 3 else 0

    logger.debug(
        "[TEST_INIT_INPUT] side=%s case_id=%d idx=%d face_bbox=(%d,%d,%d,%d) img=(%d,%d)",
        side,
        case_id,
        idx,
        fx,
        fy,
        fw,
        fh,
        img_w,
        img_h,
    )

    if side == "left":
        band_tl_x = fx + float(INIT_BAND_LEFT_TL_X_RATIO[idx]) * fw
        dx_r = float(INIT_LEFT_EYE_DX_RATIO[idx])
        dy_r = float(INIT_LEFT_EYE_DY_RATIO[idx])
        eye_key = "left_eye"
    else:
        band_tl_x = fx + float(INIT_BAND_RIGHT_TL_X_RATIO[idx]) * fw
        dx_r = float(INIT_RIGHT_EYE_DX_RATIO[idx])
        dy_r = float(INIT_RIGHT_EYE_DY_RATIO[idx])
        eye_key = "right_eye"

    band_tl_y = fy + float(INIT_BAND_TL_Y_RATIO[idx]) * fh
    band_w = float(INIT_BAND_W_RATIO[idx]) * fw
    band_h = float(INIT_BAND_H_RATIO[idx]) * fh

    # band 使用 top-left 定义；融合时需要 center
    band_cx = band_tl_x + 0.5 * band_w
    band_cy = band_tl_y + 0.5 * band_h

    logger.debug(
        "[TEST_INIT_BAND] idx=%d band_tl=(%.3f,%.3f) band_wh=(%.3f,%.3f) band_c=(%.3f,%.3f) "
        "ratios_tl(x,y,w,h)=(%.6f,%.6f,%.6f,%.6f)",
        idx,
        band_tl_x,
        band_tl_y,
        band_w,
        band_h,
        band_cx,
        band_cy,
        float(INIT_BAND_LEFT_TL_X_RATIO[idx]) if side == "left" else float(INIT_BAND_RIGHT_TL_X_RATIO[idx]),
        float(INIT_BAND_TL_Y_RATIO[idx]),
        float(INIT_BAND_W_RATIO[idx]),
        float(INIT_BAND_H_RATIO[idx]),
    )

    # fixed ROI：显示 band 本体（最终 ROI 的 w/h 仍以 face ratio limits 为准）
    fixed_w = int(round(band_w))
    fixed_h = int(round(band_h))
    fixed_x = int(round(band_cx - 0.5 * fixed_w))
    fixed_y = int(round(band_cy - 0.5 * fixed_h))
    fixed_x = max(0, min(fixed_x, img_w - fixed_w))
    fixed_y = max(0, min(fixed_y, img_h - fixed_h))
    fixed_roi = (fixed_x, fixed_y, fixed_w, fixed_h)

    logger.debug(
        "[TEST_INIT_FIXED] fixed_roi=%s fixed_wh(clipped)=(%d,%d) fixed_center=(%.3f,%.3f)",
        fixed_roi,
        fixed_w,
        fixed_h,
        (fixed_x + 0.5 * fixed_w),
        (fixed_y + 0.5 * fixed_h),
    )

    # eye-based center
    eye_pt = face_info.get(eye_key) if face_info else None
    if eye_pt is None:
        eye_roi = None
        eye_cx, eye_cy, eye_w, eye_h = band_cx, band_cy, band_w, band_h
    else:
        ex, ey = float(eye_pt[0]), float(eye_pt[1])
        eye_cx = ex - dx_r * band_w
        eye_cy = ey - dy_r * band_h
        eye_w, eye_h = band_w, band_h

        ew = int(round(eye_w))
        eh = int(round(eye_h))
        ex0 = int(round(eye_cx - 0.5 * ew))
        ey0 = int(round(eye_cy - 0.5 * eh))
        ex0 = max(0, min(ex0, img_w - ew))
        ey0 = max(0, min(ey0, img_h - eh))
        eye_roi = (ex0, ey0, ew, eh)

    logger.debug(
        "[TEST_INIT_EYE] eye_key=%s eye_pt=%s dx_r=%.6f dy_r=%.6f eye_c=(%.3f,%.3f) eye_wh=(%.3f,%.3f) eye_roi=%s",
        eye_key,
        (None if eye_pt is None else (float(eye_pt[0]), float(eye_pt[1]))),
        dx_r,
        dy_r,
        eye_cx,
        eye_cy,
        eye_w,
        eye_h,
        eye_roi,
    )

    # final fused + expand + clip
    final_cx = 0.5 * (band_cx + eye_cx)
    final_cy = 0.5 * (band_cy + eye_cy)
    final_w = 0.5 * (band_w + eye_w) + float(INIT_ROI_EXPAND_RATIO) * band_w
    final_h = 0.5 * (band_h + eye_h) + float(INIT_ROI_EXPAND_RATIO) * band_h

    logger.debug(
        "[TEST_INIT_FINAL_PRECLIP] final_c=(%.3f,%.3f) final_wh_preexpand=(%.3f,%.3f) final_wh_after_expand=(%.3f,%.3f)",
        final_cx,
        final_cy,
        0.5 * (band_w + eye_w),
        0.5 * (band_h + eye_h),
        final_w,
        final_h,
    )

    # final ROI 的 w/h 约束来自 face bbox ratio（与 EyeRoiTracker._get_roi_size_limits 一致）
    min_w = max(1, int(round(fw * ROI_MIN_W_FACE_RATIO)))
    min_h = max(1, int(round(fh * ROI_MIN_H_FACE_RATIO)))
    max_w = max(min_w, int(round(fw * ROI_MAX_W_FACE_RATIO)))
    max_h = max(min_h, int(round(fh * ROI_MAX_H_FACE_RATIO)))
    min_w = min(min_w, img_w)
    min_h = min(min_h, img_h)
    max_w = min(max_w, img_w)
    max_h = min(max_h, img_h)
    if min_w > max_w or min_h > max_h:
        return fixed_roi, eye_roi, None

    fw_i = int(np.clip(final_w, min_w, max_w))
    fh_i = int(np.clip(final_h, min_h, max_h))
    fx0 = int(round(final_cx - 0.5 * fw_i))
    fy0 = int(round(final_cy - 0.5 * fh_i))
    fx0 = max(0, min(fx0, img_w - fw_i))
    fy0 = max(0, min(fy0, img_h - fh_i))
    final_roi = (fx0, fy0, fw_i, fh_i)

    logger.debug("[TEST_INIT_FINAL_CLIPPED] final_roi=%s", final_roi)

    return fixed_roi, eye_roi, final_roi


def _draw_face_info(img: np.ndarray, face_info: Optional[Dict[str, Any]]) -> None:
    if not face_info:
        cv2.putText(
            img,
            "YuNet: face_info=None",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return
    fx, fy, fw, fh = face_info.get("bbox", [0, 0, 0, 0])
    cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), (0, 255, 255), 2)
    score = float(face_info.get("score", 0.0) or 0.0)
    case_name = str(face_info.get("case_name", ""))
    cv2.putText(
        img,
        f"YuNet score={score:.3f} {case_name}",
        (fx, max(0, fy - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    for k, color in (
        ("left_eye", (0, 255, 0)),
        ("right_eye", (0, 255, 0)),
        ("nose_tip", (255, 0, 0)),
        ("left_mouth", (0, 0, 255)),
        ("right_mouth", (0, 0, 255)),
    ):
        pt = face_info.get(k)
        if pt is None:
            continue
        x, y = int(round(float(pt[0]))), int(round(float(pt[1])))
        cv2.circle(img, (x, y), 3, color, -1)
        cv2.putText(
            img,
            f"{k}({x},{y})",
            (x + 4, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )


def run_once_on_frame(
    frame_bgr: np.ndarray,
    side: str,
    face_info: Optional[Dict[str, Any]],
) -> np.ndarray:
    vis = frame_bgr.copy()

    face_bbox = face_info.get("bbox") if face_info else None
    if face_bbox is not None:
        face_bbox = (int(face_bbox[0]), int(face_bbox[1]), int(face_bbox[2]), int(face_bbox[3]))

    _draw_face_info(vis, face_info)

    if face_bbox is not None:
        fixed_roi, eye_roi, final_roi = _compute_init_rois(
            side=side,
            face_bbox=face_bbox,
            face_info=face_info,
            image_shape=vis.shape[:2],
        )
        # 红/绿/蓝：fixed / eye-based / final
        _draw_roi(vis, fixed_roi, (0, 0, 255), "FIXED_ROI")
        _draw_roi(vis, eye_roi, (0, 255, 0), "EYE_ROI")
        _draw_roi(vis, final_roi, (255, 0, 0), "FINAL_ROI")
    else:
        _draw_roi(vis, None, (0, 0, 255), "FIXED_ROI")
        _draw_roi(vis, None, (0, 255, 0), "EYE_ROI")
        _draw_roi(vis, None, (255, 0, 0), "FINAL_ROI")

    # 仍保留：用 tracker 走 DETECT/RECOVER 分支（与 FINAL_ROI 应一致）
    tracker_detect = RoiStateMachine(side=side)
    tracker_detect._state.mode = MODE_DETECT  # 测试工具：直接设置模式（内部字段）
    roi_detect, _ = tracker_detect.predict(vis, face_bbox, face_info)
    tracker_recover = RoiStateMachine(side=side)
    tracker_recover._state.mode = MODE_RECOVER  # 测试工具：直接设置模式（内部字段）
    roi_recover, _ = tracker_recover.predict(vis, face_bbox, face_info)
    _draw_roi(vis, roi_detect.as_tuple() if roi_detect else None, (0, 255, 255), "DETECT_ROI")
    _draw_roi(vis, roi_recover.as_tuple() if roi_recover else None, (255, 255, 0), "RECOVER_ROI")

    return vis


def main() -> None:
    parser = argparse.ArgumentParser(description="Test DETECT/RECOVER ROI initialization on one frame.")
    parser.add_argument("--side", choices=["left", "right"], default="left")
    args = parser.parse_args()

    camera_manager = CameraDataManager(rgb_d=True)
    cap = camera_manager.get_cap()
    if cap is None or not cap.isOpened():
        logger.error("无法从 CameraDataManager 获取有效摄像头句柄")
        return

    yunet = YuNetDetector()

    win = "Detect/Recover ROI Init (SPACE=capture/continue, Ctrl+C=exit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    display_params_logged = [False]

    frame_id = 0
    try:
        while True:
            # 预览循环：按空格截帧
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.error("无法从摄像头读取帧")
                    return

                camera_manager.add_frame(frame_id, frame)
                pair = camera_manager.get_stereo_pair(frame_id)
                if pair is None:
                    frame_id += 1
                    continue
                left_img, right_img = pair
                view = left_img.data if args.side == "left" else right_img.data
                if view is None or view.size == 0:
                    frame_id += 1
                    continue

                src_h, src_w = view.shape[:2]
                max_display_w = 1280
                if src_w > max_display_w:
                    scale = max_display_w / float(src_w)
                    display_w, display_h = max_display_w, int(round(src_h * scale))
                    frame_display = cv2.resize(
                        view, (display_w, display_h), interpolation=cv2.INTER_LINEAR
                    )
                else:
                    display_w, display_h = src_w, src_h
                    frame_display = view

                if not display_params_logged[0]:
                    try:
                        cv2.resizeWindow(win, display_w, display_h)
                    except cv2.error:
                        pass
                    display_params_logged[0] = True

                cv2.imshow(win, frame_display)
                k = cv2.waitKey(1) & 0xFF
                if k == 32:  # SPACE
                    # 注意：截取原始未缩放帧用于后续处理
                    captured = view.copy()
                    break
                frame_id += 1

            face_info = yunet.detect_with_landmarks(captured)
            if face_info:
                try:
                    score = float(face_info.get("score", 0.0) or 0.0)
                except Exception:
                    score = 0.0
                case_name = str(face_info.get("case_name", "") or "")
                case_id = face_info.get("case_id", None)
                logger.info(
                    "[DETECT_FACE] side=%s score=%.4f case_id=%s case_name=%s bbox=%s",
                    args.side,
                    score,
                    case_id,
                    case_name,
                    face_info.get("bbox"),
                )
            else:
                logger.info("[DETECT_FACE] side=%s face_info=None", args.side)
            vis = run_once_on_frame(captured, side=args.side, face_info=face_info)

            cv2.imshow(win, vis)
            # 暂停等待空格继续
            while True:
                k2 = cv2.waitKey(0) & 0xFF
                if k2 == 32:
                    break
    except KeyboardInterrupt:
        logger.info("收到 Ctrl+C，退出 test_detect_recover_roi。")
    finally:
        camera_manager.release_camera()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

