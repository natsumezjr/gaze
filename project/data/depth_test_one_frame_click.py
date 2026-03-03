"""
单帧立体深度交互测试脚本（只测 1 帧）：
- 打开双目相机（默认 3840x1080 并排）
- 用 camera_params.json 的 stereo 标定在 1920x1080 上做 remap + SGBM
- 计算 1 帧 disparity 与 depth(Z, mm)
- 显示 rectified 左图；鼠标单击任意像素，打印/叠加显示该点深度（mm / m）与视差

运行：
  poetry run python -m project.data.depth_test_one_frame_click
或直接：
  python depth_test_one_frame_click.py

依赖：opencv-python, numpy
"""

from __future__ import annotations

import os
import json
import platform
import cv2
import numpy as np

# ========= 可配置项 =========
CAMERA_INDEX = 1
REQ_W, REQ_H = 3840, 1080  # 双目并排输出
OUT_DIR = "depth_debug_one"

# SGBM 参数（你现在的配置可直接填这里；必要时调 numDisparities）
SGBM_MIN_DISPARITY = 0
SGBM_NUM_DISPARITIES = 16 * 12  # 192（建议先用 192；若仍无效可 256）
SGBM_BLOCK_SIZE = 5
SGBM_P1 = 8 * 3 * (SGBM_BLOCK_SIZE ** 2)
SGBM_P2 = 32 * 3 * (SGBM_BLOCK_SIZE ** 2)
SGBM_DISP12_MAX_DIFF = 1
SGBM_UNIQUENESS_RATIO = 8
SGBM_SPECKLE_WINDOW_SIZE = 80
SGBM_SPECKLE_RANGE = 32
SGBM_PREFILTER_CAP = 63

# 深度单位：reprojectImageTo3D 输出 Z 的单位与标定 T 一致（你这里是 mm）
DEPTH_SCALE_TO_M = 0.001  # mm -> m


def _find_camera_params_json() -> str:
    """
    尝试在以下位置找到 camera_params.json：
    - 当前文件同级/上级常见路径
    - 项目：project/config/camera_params.json
    """
    here = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.join(here, "camera_params.json"),
        os.path.join(here, "..", "config", "camera_params.json"),
        os.path.join(here, "..", "..", "config", "camera_params.json"),
        os.path.join(here, "..", "..", "..", "project", "config", "camera_params.json"),
        os.path.join(here, "..", "..", "project", "config", "camera_params.json"),
        os.path.join(os.getcwd(), "project", "config", "camera_params.json"),
        os.path.join(os.getcwd(), "camera_params.json"),
    ]
    for p in candidates:
        p = os.path.normpath(p)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError("未找到 camera_params.json（请把路径写死或把文件放在 project/config/ 下）")


def _load_calib(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "left" not in cfg or "right" not in cfg or "stereo" not in cfg:
        raise ValueError("camera_params.json 缺少 left/right/stereo 字段")
    return cfg


def _build_rectify_and_sgbm(cfg: dict):
    left = cfg["left"]
    right = cfg["right"]
    stereo = cfg["stereo"]

    image_size = tuple(stereo["image_size"])  # (w,h)，应为 (1920,1080)

    K1 = np.array(left["K"], dtype=np.float64)
    dist1 = np.array(left["dist"], dtype=np.float64).reshape(-1, 1)
    K2 = np.array(right["K"], dtype=np.float64)
    dist2 = np.array(right["dist"], dtype=np.float64).reshape(-1, 1)

    R1 = np.array(stereo["R1"], dtype=np.float64)
    R2 = np.array(stereo["R2"], dtype=np.float64)
    P1 = np.array(stereo["P1"], dtype=np.float64)
    P2 = np.array(stereo["P2"], dtype=np.float64)
    Q = np.array(stereo["Q"], dtype=np.float64)

    map1x, map1y = cv2.initUndistortRectifyMap(
        K1, dist1, R1, P1, image_size, cv2.CV_32FC1
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        K2, dist2, R2, P2, image_size, cv2.CV_32FC1
    )

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

    # 由 P1/P2 推导 baseline（mm）方便你核对 5.2cm/6.5cm
    fx = float(P1[0, 0])
    B_mm = float(abs(P2[0, 3]) / fx) if fx != 0 else float("nan")

    return map1x, map1y, map2x, map2y, sgbm, Q, image_size, B_mm


def _open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW if platform.system() == "Windows" else CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头，请检查 CAMERA_INDEX")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQ_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQ_H)
    # 有些驱动要 read 一帧才生效
    ret, f = cap.read()
    if not ret or f is None:
        cap.release()
        raise RuntimeError("读帧失败")
    h, w = f.shape[:2]
    if (w, h) != (REQ_W, REQ_H):
        print(f"[警告] 实际分辨率 {w}x{h} != 期望 {REQ_W}x{REQ_H}，仍继续（只要是左右并排即可）")
    return cap


def _compute_one_frame_depth(frame_bgr: np.ndarray, map1x, map1y, map2x, map2y, sgbm, Q, image_size):
    """split(半幅) -> 若与 stereo.image_size 一致则不 resize -> remap -> SGBM -> disp_raw/16 -> reproject -> depth(mm)。
    标定与运行一致(3840×1080, image_size=(1920,1080))时：直接 split(1920) -> remap(1920) -> SGBM(1920)。"""
    h, w = frame_bgr.shape[:2]
    half = w // 2
    left = frame_bgr[:, :half]
    right = frame_bgr[:, half:]

    w_cal, h_cal = image_size
    if left.shape[1] != w_cal or left.shape[0] != h_cal:
        left = cv2.resize(left, (w_cal, h_cal), interpolation=cv2.INTER_LINEAR)
    if right.shape[1] != w_cal or right.shape[0] != h_cal:
        right = cv2.resize(right, (w_cal, h_cal), interpolation=cv2.INTER_LINEAR)

    rect_left = cv2.remap(left, map1x, map1y, cv2.INTER_LINEAR)
    rect_right = cv2.remap(right, map2x, map2y, cv2.INTER_LINEAR)

    gray_l = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)

    disp_raw = sgbm.compute(gray_l, gray_r)  # int16, Q16 定点：真实视差 = raw/16，≤0 视为无效
    disp = disp_raw.astype(np.float32) / 16.0
    invalid_disp = disp <= 0
    disp_safe = disp.copy()
    disp_safe[invalid_disp] = 0.0

    pts3d = cv2.reprojectImageTo3D(disp_safe, Q)  # 用 float disparity 更稳
    depth_mm = pts3d[:, :, 2].astype(np.float32)

    # 无效深度置 nan
    depth_mm[invalid_disp] = np.nan
    depth_mm[~np.isfinite(depth_mm) | (depth_mm <= 0)] = np.nan

    return rect_left, rect_right, disp_raw, disp, depth_mm


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    calib_path = _find_camera_params_json()
    cfg = _load_calib(calib_path)
    map1x, map1y, map2x, map2y, sgbm, Q, image_size, B_mm = _build_rectify_and_sgbm(cfg)

    print(f"camera_params.json: {calib_path}")
    print(f"标定工作分辨率 stereo.image_size={image_size}")
    print(f"由 P1/P2 推导 baseline≈{B_mm:.2f} mm ({B_mm/10.0:.2f} cm)  <- 用于核对尺子测量")
    print("按任意键采集 1 帧并计算深度；之后鼠标左键点击像素查看深度。按 q 退出。")

    cap = _open_camera()

    # 先预览，等待用户按键抓一帧
    win_preview = "Preview (Press any key to capture)"
    cv2.namedWindow(win_preview, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_preview, 1280, 540)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("读帧失败")
            cap.release()
            return 1
        half = frame.shape[1] // 2
        vis = np.hstack([frame[:, :half], frame[:, half:]])
        if vis.shape[1] > 1280:
            vis = cv2.resize(vis, (1280, int(vis.shape[0] * 1280 / vis.shape[1])))
        cv2.imshow(win_preview, vis)
        k = cv2.waitKey(1) & 0xFF
        if k != 255:  # 有按键
            break

    # 采集并计算 1 帧
    ret, frame = cap.read()
    cap.release()
    cv2.destroyWindow(win_preview)
    if not ret or frame is None:
        print("抓取帧失败")
        return 1

    rect_left, rect_right, disp_raw, disp, depth_mm = _compute_one_frame_depth(
        frame, map1x, map1y, map2x, map2y, sgbm, Q, image_size
    )

    # 打印统计
    cy, cx = depth_mm.shape[0] // 2, depth_mm.shape[1] // 2
    center_disp = float(disp[cy, cx]) if disp[cy, cx] > 0 else float("nan")
    center_z = float(depth_mm[cy, cx]) if np.isfinite(depth_mm[cy, cx]) else float("nan")
    nan_pct = 100.0 * np.mean(~np.isfinite(depth_mm))

    print(f"[单帧] disparity raw dtype={disp_raw.dtype} min={int(disp_raw.min())} max={int(disp_raw.max())}")
    print(f"[单帧] center disp={center_disp:.2f} px, center Z={center_z:.1f} mm ({center_z*DEPTH_SCALE_TO_M:.3f} m), nan%={nan_pct:.1f}%")

    # 保存文件，方便你复盘
    cv2.imwrite(os.path.join(OUT_DIR, "rect_left.png"), rect_left)
    cv2.imwrite(os.path.join(OUT_DIR, "rect_right.png"), rect_right)

    # 深度可视化（仅用于显示，不影响点击读数）
    d = depth_mm.copy()
    valid = np.isfinite(d)
    if np.any(valid):
        lo, hi = np.percentile(d[valid], [2, 98])
        d_show = np.clip(np.nan_to_num(d, nan=lo), lo, hi)
        if hi > lo:
            d_show = ((d_show - lo) / (hi - lo) * 255).astype(np.uint8)
        else:
            d_show = np.zeros_like(d_show, dtype=np.uint8)
    else:
        d_show = np.zeros_like(d, dtype=np.uint8)
    depth_color = cv2.applyColorMap(d_show, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(OUT_DIR, "depth_vis.png"), depth_color)

    # 点击查询：显示在 rect_left 上
    state = {
        "rect_left": rect_left,
        "disp": disp,
        "depth_mm": depth_mm,
        "last": None,  # (x,y,disp,zmm)
    }

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        disp_v = float(state["disp"][y, x]) if state["disp"][y, x] > 0 else float("nan")
        zmm = float(state["depth_mm"][y, x]) if np.isfinite(state["depth_mm"][y, x]) else float("nan")
        zm = zmm * DEPTH_SCALE_TO_M if np.isfinite(zmm) else float("nan")
        state["last"] = (x, y, disp_v, zmm)
        print(f"[点击] (x={x}, y={y}) disp={disp_v:.2f} px, Z={zmm:.1f} mm ({zm:.3f} m)")

    win = "RectLeft Click to Query Depth (press q to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        canvas = state["rect_left"].copy()
        if state["last"] is not None:
            x, y, disp_v, zmm = state["last"]
            zm = zmm * DEPTH_SCALE_TO_M if np.isfinite(zmm) else float("nan")
            cv2.circle(canvas, (x, y), 4, (0, 255, 0), -1)
            text = f"({x},{y}) disp={disp_v:.2f}  Z={zmm:.1f}mm ({zm:.3f}m)"
            cv2.putText(canvas, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(win, canvas)
        k = cv2.waitKey(10) & 0xFF
        if k == ord("q") or k == 27:
            break

    cv2.destroyAllWindows()
    print(f"已保存：{OUT_DIR}/rect_left.png rect_right.png depth_vis.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())