"""
使用 RGB-D 摄像头（idx=1，并排双目）实时采集，
将 3840x1080 图像按宽度一分为二作为左右图，
在“左图”和“右图”上分别画出 YuNet 输出的：
- 人脸 bbox
- 4-5: 右眼中心
- 6-7: 左眼中心

注意：
- 这里只使用 OpenCV 摄像头 API + YuNet 模型加载，不依赖 CameraDataManager 等其它封装；
- 拆分逻辑参考 `CameraDataManager.get_stereo_pair`：按宽度对半切。

运行方式（在项目根目录）：

    python -m project.tools.test_yunet_landmarks
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from project.config.model_config import YUNET_MODEL_FILENAME, YUNET_MODEL_SEARCH_DIRS
from project.data.model_locator import find_model_file, ModelLocateError


# 与 face_detector.YuNetDetector 一致：检测时最长边缩到该尺寸，再按比例还原坐标
_MAX_DETECT_SIZE = 320


def _create_yunet_detector(input_size: Tuple[int, int] = (320, 320)) -> cv2.FaceDetectorYN:
    """
    创建 YuNet 检测器，模型路径和搜索目录与正式代码保持一致。
    """
    try:
        model_path = str(find_model_file(YUNET_MODEL_FILENAME, YUNET_MODEL_SEARCH_DIRS))
    except ModelLocateError as e:
        raise FileNotFoundError(f"YuNet 模型未找到: {e}") from None

    detector = cv2.FaceDetectorYN.create(
        model_path,
        "",
        input_size,
        score_threshold=0.6,
    )
    return detector


def _detect_one_face_yunet(
    yunet: cv2.FaceDetectorYN,
    image: np.ndarray,
) -> np.ndarray | None:
    """
    与 face_detector.YuNetDetector 一致：图过大时先缩到最长边 320 再检测，
    只取面积最大的一张脸，bbox 与 landmark 坐标按 scale 还原到原图尺寸。
    返回一行 shape (15,) 的数组：
    [x, y, w, h,
     right_eye_x, right_eye_y,
     left_eye_x,  left_eye_y,
     nose_x, nose_y,
     mouth_right_x, mouth_right_y,
     mouth_left_x,  mouth_left_y,
     score]，
    为原图坐标；无人脸则 None。
    """
    h, w = image.shape[:2]
    scale = 1.0
    img_det = image
    if w > _MAX_DETECT_SIZE or h > _MAX_DETECT_SIZE:
        scale = min(_MAX_DETECT_SIZE / w, _MAX_DETECT_SIZE / h)
        w_det = int(round(w * scale))
        h_det = int(round(h * scale))
        img_det = cv2.resize(image, (w_det, h_det), interpolation=cv2.INTER_LINEAR)
        w, h = w_det, h_det
    yunet.setInputSize((w, h))
    _, faces = yunet.detect(img_det)
    if faces is None or len(faces) == 0:
        return None
    boxes = faces[:, :4]
    areas = boxes[:, 2] * boxes[:, 3]
    idx = int(np.argmax(areas))
    face = faces[idx].copy()  # shape 通常为 (15,)
    if scale != 1.0:
        inv = 1.0 / scale
        # bbox 与 5 个 landmark（索引 0-13）按比例还原到原图坐标；
        # 最后一个索引 14 为 score，不参与缩放。
        face[:14] *= inv
    return face


def _draw_yunet_output(
    frame: np.ndarray,
    face: np.ndarray | None,
    label_prefix: str = "",
) -> None:
    """
    在 frame 上画出单张人脸的 bbox 和全部 YuNet landmark（原图坐标）：
    - 右眼中心
    - 左眼中心
    - 鼻尖
    - 右嘴角
    - 左嘴角
    face 为 _detect_one_face_yunet 返回的一行，或 None 不画。
    """
    if face is None or face.size < 15:
        return

    x, y, w, h = face[:4].astype(int)
    cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2,
    )
    # 解析 YuNet landmark 坐标（参考 OpenCV YuNet 文档）
    re_x, re_y = face[4:6]    # right eye
    le_x, le_y = face[6:8]    # left eye
    nose_x, nose_y = face[8:10]
    mr_x, mr_y = face[10:12]  # mouth right
    ml_x, ml_y = face[12:14]  # mouth left
    score = float(face[14])

    pts = {
        "RE": ((int(round(re_x)), int(round(re_y))), (0, 0, 255)),      # 红：右眼
        "LE": ((int(round(le_x)), int(round(le_y))), (255, 0, 0)),      # 蓝：左眼
        "N":  ((int(round(nose_x)), int(round(nose_y))), (0, 255, 255)),# 黄：鼻尖
        "MR": ((int(round(mr_x)), int(round(mr_y))), (255, 0, 255)),    # 品红：右嘴角
        "ML": ((int(round(ml_x)), int(round(ml_y))), (0, 255, 0)),      # 绿：左嘴角
    }

    for tag, (pt, color) in pts.items():
        cv2.circle(frame, pt, 3, color, -1)
        cv2.putText(
            frame,
            f"{label_prefix}{tag}",
            (pt[0] + 3, pt[1] - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )


def main() -> None:
    # 创建输出目录（可选，用于保存截图）
    out_dir = Path(__file__).resolve().parents[1] / "debug" / "yunet_landmarks"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 直接打开 idx=1 的 RGB-D 摄像头（参考 CameraDataManager._open_rgb_d）
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("无法打开 RGB-D 摄像头（idx=1）")

    # 尝试设置为 3840x1080；若不支持，再退回 1920x1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if actual_w < 2000:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"RGB-D camera opened, resolution = {actual_w}x{actual_h}")

    # YuNet 单实例，左右图共用
    yunet = _create_yunet_detector()

    # 左右两个窗口，固定显示分辨率（保持 16:9 比例，例如 960x540）
    win_left = "YuNet-Left"
    win_right = "YuNet-Right"
    cv2.namedWindow(win_left, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_right, cv2.WINDOW_NORMAL)
    display_w, display_h = 960, 540
    cv2.resizeWindow(win_left, display_w, display_h)
    cv2.resizeWindow(win_right, display_w, display_h)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("读取 RGB-D 摄像头帧失败，结束。")
                break

            h, w = frame.shape[:2]
            if w < 2:
                print(f"帧宽度异常 w={w}，跳过。")
                continue

            # 按宽度对半切成左右图（参考 CameraDataManager.get_stereo_pair）
            half = w // 2
            left_bgr = frame[:, :half].copy()
            right_bgr = frame[:, half:].copy()

            # 左图检测（与 YuNetDetector 一致：缩图检测 + 只取最大脸 + 坐标还原到原图）
            face_left = _detect_one_face_yunet(yunet, left_bgr)
            left_vis = left_bgr.copy()
            _draw_yunet_output(left_vis, face_left, label_prefix="L-")

            # 右图检测
            face_right = _detect_one_face_yunet(yunet, right_bgr)
            right_vis = right_bgr.copy()
            _draw_yunet_output(right_vis, face_right, label_prefix="R-")

            # 固定窗口显示尺寸：缩放到 display_w x display_h
            left_show = cv2.resize(left_vis, (display_w, display_h), interpolation=cv2.INTER_LINEAR)
            right_show = cv2.resize(right_vis, (display_w, display_h), interpolation=cv2.INTER_LINEAR)

            cv2.imshow(win_left, left_show)
            cv2.imshow(win_right, right_show)

            # 等待 1ms，允许按 `q` 提前退出；否则 Ctrl+C 终止脚本
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    except KeyboardInterrupt:
        print("捕获到 Ctrl+C，安全退出。")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

