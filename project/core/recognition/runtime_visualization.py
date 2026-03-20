"""
Recognition runtime visualization helpers.

从 `recognition_runtime.py` 抽离出的纯显示/绘制逻辑，避免 runtime 主文件承担过多职责。
行为保持不变：只是移动代码位置与调用方式。
"""

from __future__ import annotations

from typing import Optional, Tuple, Any, List, Dict

import cv2
import numpy as np

from project.config.logging_config import setup_logging
from project.data.data_models import Ellipse2D, Landmark, Point2D


logger = setup_logging(__name__)


def ellseg_segmentation_mask_to_bgr(
    segmentation_mask: Optional[np.ndarray],
    dst_h: int,
    dst_w: int,
) -> Optional[np.ndarray]:
    """
    将 EllSeg `segmentation_mask` 转为 BGR 彩图，供 runtime debug 落盘。

    约定（与 `eye_model_ellseg` / evaluate_ellseg 一致）：
    - 离散标签：0=背景，1=虹膜，2=瞳孔。
    - legacy_onnx 等路径可能为 0/1 浮点合并图：整体标成一种前景色。
    - 其它连续值：归一化后 JET 伪彩色。
    """
    if segmentation_mask is None or dst_h <= 0 or dst_w <= 0:
        return None
    m = np.asarray(segmentation_mask)
    if m.ndim != 2:
        if m.ndim == 3 and m.shape[2] >= 1:
            m = m[:, :, 0]
        else:
            return None
    if m.size == 0:
        return None

    h0, w0 = int(m.shape[0]), int(m.shape[1])
    if h0 != dst_h or w0 != dst_w:
        m = cv2.resize(m.astype(np.float32), (dst_w, dst_h), interpolation=cv2.INTER_NEAREST)

    m_f = m.astype(np.float32, copy=False)
    mx = float(np.nanmax(m_f)) if m_f.size else 0.0
    mn = float(np.nanmin(m_f)) if m_f.size else 0.0

    # 离散 0/1/2 类图（含以 float 存储的整类值）
    if mx <= 2.5 and mn >= 0.0:
        m_int = np.rint(m_f).astype(np.int32)
        if int(m_int.max()) <= 2 and int(m_int.min()) >= 0:
            out = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
            # BGR：虹膜偏青橙、瞳孔绿色（与 draw_ellseg 椭圆色可区分）
            out[m_int == 1] = (255, 160, 0)
            out[m_int == 2] = (0, 255, 0)
            return out
        if int(m_int.max()) == 1 and int(m_int.min()) == 0:
            out = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
            out[m_int >= 1] = (0, 255, 255)
            return out

    # 连续图 / 未知标量场
    if mx <= mn:
        return np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    g = ((m_f - mn) / (mx - mn) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(g, cv2.COLORMAP_JET)


def native_geometry_for_roi_crop(
    native_geometry: Optional[Dict[str, Any]],
    roi_x: int,
    roi_y: int,
) -> Dict[str, Any]:
    """
    将 `EyeLandmarkExtractor` 里的 native_geometry 从「整图像素坐标」转为「ROI crop 子图坐标」。

    landmark 里椭圆/瞳孔中心已加上 ROI 原点；落盘 crop 时须减去 (roi_x, roi_y)，
    几何才会与屏幕上在全图绘制的相对位置一致。
    `segmentation_mask` 本身为 ROI 内尺寸，原样保留。
    """
    if not native_geometry:
        return {}
    rx, ry = float(roi_x), float(roi_y)
    out: Dict[str, Any] = {}
    m = native_geometry.get("segmentation_mask")
    if m is not None:
        out["segmentation_mask"] = m

    pc = native_geometry.get("pupil_center")
    if pc is not None:
        out["pupil_center"] = Point2D(x=float(pc.x) - rx, y=float(pc.y) - ry)

    for key in ("pupil_ellipse", "iris_ellipse"):
        el = native_geometry.get(key)
        if el is None:
            continue
        out[key] = Ellipse2D(
            cx=float(el.cx) - rx,
            cy=float(el.cy) - ry,
            major_axis=float(el.major_axis),
            minor_axis=float(el.minor_axis),
            angle_deg=float(el.angle_deg),
            confidence=el.confidence,
        )
    return out


def scale_landmarks_for_display(
    landmarks: List[Landmark],
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
) -> List[Landmark]:
    """将原始图像坐标下的关键点缩放到显示尺寸，用于与 resize 后的画面对齐。"""
    if not landmarks or src_w <= 0 or src_h <= 0:
        return landmarks
    sx, sy = dst_w / src_w, dst_h / src_h
    return [Landmark(x=lm.x * sx, y=lm.y * sy, z=lm.z, visibility=lm.visibility) for lm in landmarks]


def draw_bbox_and_rois(
    frame_display: np.ndarray,
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
    bbox_rois: Optional[Tuple[Any, Any, Any]] = None,
) -> np.ndarray:
    """若有则在人脸 bbox、左右眼 ROI 上画框。bbox_rois=(bbox, left_roi, right_roi) 时用该元组（与显示图一致）。"""
    if bbox_rois is None:
        return frame_display
    bbox, left_roi, right_roi = bbox_rois
    if src_w <= 0 or src_h <= 0:
        return frame_display
    sx, sy = dst_w / src_w, dst_h / src_h
    out = frame_display.copy()
    if bbox is not None and len(bbox) >= 4:
        x, y, bw, bh = bbox[0], bbox[1], bbox[2], bbox[3]
        pt1 = (int(x * sx), int(y * sy))
        pt2 = (int((x + bw) * sx), int((y + bh) * sy))
        cv2.rectangle(out, pt1, pt2, (0, 255, 0), 2)  # 人脸 bbox 绿色
    if left_roi is not None and len(left_roi) >= 4:
        x, y, rw, rh = left_roi[0], left_roi[1], left_roi[2], left_roi[3]
        pt1 = (int(x * sx), int(y * sy))
        pt2 = (int((x + rw) * sx), int((y + rh) * sy))
        cv2.rectangle(out, pt1, pt2, (255, 0, 0), 1)  # 左眼 ROI 蓝色
    if right_roi is not None and len(right_roi) >= 4:
        x, y, rw, rh = right_roi[0], right_roi[1], right_roi[2], right_roi[3]
        pt1 = (int(x * sx), int(y * sy))
        pt2 = (int((x + rw) * sx), int((y + rh) * sy))
        cv2.rectangle(out, pt1, pt2, (255, 255, 0), 1)  # 右眼 ROI 青色
    return out


def draw_ellseg_native_geometry(
    frame: np.ndarray,
    debug_primary: dict,
    debug_secondary: Optional[dict] = None,
    src_w: int = 0,
    src_h: int = 0,
    dst_w: int = 0,
    dst_h: int = 0,
) -> np.ndarray:
    """
    绘制 EllSeg 原生输出几何（pupil/iris ellipse + pupil center）。

    说明：
    - 输入 debug_primary/debug_secondary 为 `EyeLandmarkExtractor` 的 debug 字典（含 native_geometry）。
    - 坐标缩放与 `show_frame` 的显示尺寸保持一致。
    """
    if frame is None or frame.size == 0:
        return frame
    debug_list = [d for d in (debug_primary, debug_secondary) if d]
    if not debug_list:
        return frame
    sx = dst_w / src_w if src_w else 1.0
    sy = dst_h / src_h if src_h else 1.0

    for debug in debug_list:
        ng = debug.get("native_geometry", {})
        for el_key, color in [("pupil_ellipse", (0, 255, 0)), ("iris_ellipse", (0, 200, 255))]:
            el = ng.get(el_key)
            if el is not None:
                cx = int(el.cx * sx)
                cy = int(el.cy * sy)
                ax1 = int(el.major_axis * 0.5 * sx)
                ax2 = int(el.minor_axis * 0.5 * sy)
                if ax1 > 0 and ax2 > 0:
                    cv2.ellipse(frame, (cx, cy), (ax1, ax2), el.angle_deg, 0, 360, color, 1)
        pc = ng.get("pupil_center")
        if pc is not None:
            px, py = int(pc.x * sx), int(pc.y * sy)
            cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)
    return frame


def draw_roi_debug_overlay(
    frame: np.ndarray,
    debug_primary: dict,
    debug_secondary: Optional[dict] = None,
    src_w: int = 0,
    src_h: int = 0,
    dst_w: int = 0,
    dst_h: int = 0,
) -> np.ndarray:
    """绘制 ROI 模式/文本，并调用 `draw_ellseg_native_geometry` 画出 EllSeg 原生几何。"""
    if frame is None or frame.size == 0:
        return frame

    # 先画 EllSeg 原生输出（几何信息）
    frame = draw_ellseg_native_geometry(
        frame,
        debug_primary=debug_primary,
        debug_secondary=debug_secondary,
        src_w=src_w,
        src_h=src_h,
        dst_w=dst_w,
        dst_h=dst_h,
    )

    rd = debug_primary.get("roi_debug") or {}
    mode = str(rd.get("current_mode") or debug_primary.get("roi_mode") or "?")
    cg = float(rd.get("geometry_confidence", 0.0))
    face_stable = bool(rd.get("face_bbox_is_stable", False))
    shift = float(rd.get("face_bbox_corner_max_shift", 0.0))
    track_from_previous_bbox_applied = bool(rd.get("track_from_previous_bbox_applied", False))

    lines = [
        f"mode={mode} cg={cg:.2f}",
        f"face_stable={face_stable} shift={shift:.4f} track_from_previous_bbox_applied={track_from_previous_bbox_applied}",
    ]
    for i, text in enumerate(lines):
        if not text:
            continue
        cv2.putText(
            frame,
            text,
            (8, 24 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return frame


def show_frame(
    win_name: str,
    image,
    display_params_logged: list,
    frame_to_show=None,
    max_display_w: int = 1280,
    show: bool = True,
):
    """
    按固定比例得到显示帧与尺寸，可选地执行一次日志、resizeWindow、imshow、waitKey(1)。
    frame_to_show 不为 None 时显示该帧（如叠加关键点后的图），否则显示由 image 生成的帧。
    show=False 时只计算并返回 (frame_display, display_w, display_h, w, h)，不 imshow。
    返回 (frame_display, display_w, display_h, w, h)。
    """
    frame_for_draw = image.data if hasattr(image, "data") else image
    h, w = frame_for_draw.shape[:2]
    if w > max_display_w:
        scale = max_display_w / w
        display_w, display_h = max_display_w, int(round(h * scale))
        frame_display = cv2.resize(frame_for_draw, (display_w, display_h), interpolation=cv2.INTER_LINEAR)
    else:
        display_w, display_h = w, h
        frame_display = frame_for_draw
    if show:
        if not display_params_logged[0]:
            logger.debug(
                "[显示参数] 图像尺寸 h=%d w=%d 显示尺寸 %dx%d 宽高比=%.3f (窗口将固定为该尺寸)",
                h,
                w,
                display_w,
                display_h,
                display_w / display_h if display_h else 0,
            )
            display_params_logged[0] = True
        try:
            cv2.resizeWindow(win_name, display_w, display_h)
        except cv2.error:
            pass
        cv2.imshow(win_name, frame_to_show if frame_to_show is not None else frame_display)
        cv2.waitKey(1)
    return frame_display, display_w, display_h, w, h


__all__ = [
    "scale_landmarks_for_display",
    "draw_bbox_and_rois",
    "ellseg_segmentation_mask_to_bgr",
    "native_geometry_for_roi_crop",
    "draw_ellseg_native_geometry",
    "draw_roi_debug_overlay",
    "show_frame",
]