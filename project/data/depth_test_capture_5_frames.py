"""
单功能：打开双目摄像头(3840x1080)，预热后采集 5 帧。
每帧：
  - 从整图切左右(1920x1080)
  - 用 camera_params.json 的 stereo/left/right 做校正(remap)【不再 resize 到 1280】
  - 跑 SGBM 得到 disparity（OpenCV 输出是 Q16 定点，真实视差=raw/16）
  - 打印中心视差 raw 和 /16
  - 保存 rect_left / rect_right 到 depth_debug/

不依赖项目其它文件，仅依赖：opencv-python, numpy
运行：
  python -m project.data.depth_test_capture_5_frames
"""

from __future__ import annotations

import json
import os
import platform
import sys

import cv2
import numpy as np

# ===== 可调参数 =====
OUT_DIR = "depth_debug"
NUM_WARMUP = 15
NUM_FRAMES = 5
CAMERA_INDEX = 1

# SGBM 参数（先给一套稳妥的起点；你可后续再调）
SGBM_MIN_DISPARITY = 0
SGBM_NUM_DISPARITIES = 16 * 12  # 192（必须 16 的倍数）
SGBM_BLOCK_SIZE = 5
SGBM_P1 = 8 * 3 * (SGBM_BLOCK_SIZE ** 2)
SGBM_P2 = 32 * 3 * (SGBM_BLOCK_SIZE ** 2)
SGBM_DISP12_MAX_DIFF = 1
SGBM_UNIQUENESS_RATIO = 8
SGBM_SPECKLE_WINDOW_SIZE = 80
SGBM_SPECKLE_RANGE = 2
SGBM_PREFILTER_CAP = 63

# 期望的双目输出分辨率（整图）
STEREO_W, STEREO_H = 3840, 1080


def _repo_root() -> str:
    """尽量找到项目根目录（包含 project/ 目录）"""
    here = os.path.abspath(__file__)
    # .../project/data/depth_test_capture_5_frames.py -> 往上三层到仓库根
    root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    return root


def _load_camera_params() -> dict:
    """加载 project/config/camera_params.json（与你现在的标定脚本输出一致）"""
    root = _repo_root()
    path = os.path.join(root, "project", "config", "camera_params.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"未找到 camera_params.json: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_stereo_processor(cfg: dict):
    """
    从 camera_params.json 构建：
      - 左右 remap maps（在 stereo.image_size 上工作）
      - StereoSGBM
    重要：你已重新标定到 1920×1080，因此 stereo.image_size 应为 [1920,1080]
    """
    if "left" not in cfg or "right" not in cfg or "stereo" not in cfg:
        raise ValueError("camera_params.json 缺少 left/right/stereo 字段")

    left = cfg["left"]
    right = cfg["right"]
    stereo = cfg["stereo"]

    image_size = tuple(stereo["image_size"])  # (w,h)
    if image_size != (STEREO_W // 2, STEREO_H):
        print(f"[提示] stereo.image_size={image_size}，与你期望的单眼 {(STEREO_W//2, STEREO_H)} 不一致。")
        print("      若你刚完成 3840 重新标定，这里应该是 (1920,1080)。请确认 camera_params.json 是否是新文件。")

    K1 = np.array(left["K"], dtype=np.float64)
    dist1 = np.array(left["dist"], dtype=np.float64).reshape(-1, 1)
    K2 = np.array(right["K"], dtype=np.float64)
    dist2 = np.array(right["dist"], dtype=np.float64).reshape(-1, 1)

    R1 = np.array(stereo["R1"], dtype=np.float64)
    R2 = np.array(stereo["R2"], dtype=np.float64)
    P1 = np.array(stereo["P1"], dtype=np.float64)
    P2 = np.array(stereo["P2"], dtype=np.float64)
    Q = np.array(stereo["Q"], dtype=np.float64)

    map1x, map1y = cv2.initUndistortRectifyMap(K1, dist1, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, dist2, R2, P2, image_size, cv2.CV_32FC1)

    sgbm = cv2.StereoSGBM_create(
        minDisparity=SGBM_MIN_DISPARITY,
        numDisparities=SGBM_NUM_DISPARITIES,
        blockSize=SGBM_BLOCK_SIZE,
        P1=SGBM_P1,
        P2=SGBM_P2,
        disp12MaxDiff=SGBM_DISP12_MAX_DIFF,
        uniquenessRatio=SGBM_UNIQUENESS_RATIO,
        speckleWindowSize=SGBM_SPECKLE_WINDOW_SIZE,
        speckleRange=SGBM_SPECKLE_RANGE,
        preFilterCap=SGBM_PREFILTER_CAP,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    return map1x, map1y, map2x, map2y, sgbm, Q, image_size


def _compute_rectified_and_disparity(frame_bgr, map1x, map1y, map2x, map2y, sgbm, image_size):
    """整图 -> 切左右 -> remap 校正 -> 灰度 -> SGBM disparity"""
    h, w = frame_bgr.shape[:2]
    if (w, h) != (STEREO_W, STEREO_H):
        return None

    half = w // 2
    left = frame_bgr[:, :half]
    right = frame_bgr[:, half:]

    # 关键：不再 resize 到 1280。这里要求 left/right 已经是 image_size（应为 1920×1080）
    if (left.shape[1], left.shape[0]) != image_size or (right.shape[1], right.shape[0]) != image_size:
        # 如果驱动给的不是 1920×1080 单眼，直接失败，避免“又引入缩放”
        print(f"[错误] 单眼尺寸 left={(left.shape[1], left.shape[0])}, 但标定 image_size={image_size}。")
        print("      请确保：1) 摄像头输出 3840×1080；2) camera_params.json 是 1920×1080 标定。")
        return None

    rect_l = cv2.remap(left, map1x, map1y, cv2.INTER_LINEAR)
    rect_r = cv2.remap(right, map2x, map2y, cv2.INTER_LINEAR)

    gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)
    # SGBM 输出 int16 Q16：真实视差 = raw/16；深度计算须 disp=raw/16，disp≤0 无效，再 reprojectImageTo3D(disp, Q)
    disp_raw = sgbm.compute(gray_l, gray_r)

    return rect_l, rect_r, disp_raw


def main():
    cfg = _load_camera_params()
    map1x, map1y, map2x, map2y, sgbm, Q, image_size = _build_stereo_processor(cfg)

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW if platform.system() == "Windows" else CAMERA_INDEX)
    if not cap.isOpened():
        print("无法打开摄像头")
        return 1

    # 请求 3840×1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, STEREO_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STEREO_H)

    # 用实际读到的帧尺寸为准
    ret, f0 = cap.read()
    if not ret or f0 is None:
        print("读帧失败")
        cap.release()
        return 1

    h0, w0 = f0.shape[:2]
    if (w0, h0) != (STEREO_W, STEREO_H):
        print(f"当前摄像头输出为 {w0}x{h0}，不是 {STEREO_W}x{STEREO_H}，无法按双目处理。")
        cap.release()
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)

    cv2.namedWindow("Stereo Rectified (Left | Right)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Stereo Rectified (Left | Right)", 1280, 540)

    print(f"标定工作分辨率 stereo.image_size={image_size}（应为 1920x1080）")
    print(f"预热 {NUM_WARMUP} 帧...")
    for _ in range(NUM_WARMUP):
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        half = frame.shape[1] // 2
        vis = np.hstack([frame[:, :half], frame[:, half:]])
        if vis.shape[1] > 1280:
            vis = cv2.resize(vis, (1280, vis.shape[0] * 1280 // vis.shape[1]))
        cv2.imshow("Stereo Rectified (Left | Right)", vis)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            return 0

    print(f"采集 {NUM_FRAMES} 帧并保存到 {OUT_DIR}/")
    for n in range(1, NUM_FRAMES + 1):
        ret, frame = cap.read()
        if not ret or frame is None:
            print("读帧失败")
            break

        out = _compute_rectified_and_disparity(frame, map1x, map1y, map2x, map2y, sgbm, image_size)
        if out is None:
            continue

        rect_l, rect_r, disp_raw = out

        # disparity 诊断：raw 是 Q16 定点（真实视差 = raw/16）
        cy, cx = disp_raw.shape[0] // 2, disp_raw.shape[1] // 2
        center_raw = int(disp_raw[cy, cx])
        dmin, dmax = int(np.min(disp_raw)), int(np.max(disp_raw))
        center_disp = center_raw / 16.0 if center_raw > 0 else float("nan")
        print(f"[Frame {n}] center disparity raw={center_raw}  disp={center_disp:.2f}  dtype={disp_raw.dtype}  min/max={dmin}/{dmax}")

        # 保存左右校正图
        cv2.imwrite(os.path.join(OUT_DIR, f"rect_left_{n}.png"), rect_l)
        cv2.imwrite(os.path.join(OUT_DIR, f"rect_right_{n}.png"), rect_r)

        # 预览
        vis = np.hstack([rect_l, rect_r])
        if vis.shape[1] > 1280:
            vis = cv2.resize(vis, (1280, vis.shape[0] * 1280 // vis.shape[1]))
        cv2.imshow("Stereo Rectified (Left | Right)", vis)
        cv2.waitKey(1)
        print(f"  第 {n} 帧已保存")

    cap.release()
    cv2.destroyAllWindows()
    print(f"完成: {OUT_DIR}/rect_left_1..{NUM_FRAMES}.png, rect_right_1..{NUM_FRAMES}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())