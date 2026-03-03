"""
双目重新标定：从“生成棋盘格图片 -> 采集新的标定图片(3840x1080) -> 单目标定 -> 双目标定+校正验收”
核心思路不变：仍然是 split 左右 -> 单目内参 -> stereoCalibrate(FIX_INTRINSIC) -> stereoRectify -> 验收 Δy RMS。
区别：本脚本会【强制以运行时 3840×1080（单眼 1920×1080）】为标定基准，解决你之前 2560/3840 不一致的问题。

依赖：
  pip install opencv-python numpy

用法（建议放在 project/config/calibration_recalibrate_3840.py 之类）：
  1) 生成棋盘格打印图：
       python -m project.config.calibration_recalibrate_3840 --gen-board
  2) 打开摄像头采集标定图（保存为 3840x1080 原始拼接图）：
       python -m project.config.calibration_recalibrate_3840 --capture
     操作说明：
       - 按 Space 保存一张（建议 30~60 对，覆盖不同距离/角度/位置）
       - 按 q 退出
  3) 用采集到的图片进行完整标定，并写入 camera_params.json：
       python -m project.config.calibration_recalibrate_3840 --calibrate

可选：
  - 先统计 blur 分布，辅助选 blur_var_min：
       python -m project.config.calibration_recalibrate_3840 --analyze-blur
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from dataclasses import dataclass

import cv2
import numpy as np

# ===================== 用户需要改的配置区 =====================

# 棋盘格“内角点数量”（不是格子数！）
# 你给的新配置是 (9, 6) —— 这意味着棋盘格格子数是 (10, 7)
PATTERN_SIZE = (9, 6)

# 每个格子的真实边长（毫米）
SQUARE_SIZE_MM = 23.5

# 运行时真实双目输出尺寸（强制用于重新标定）
STEREO_FULL_SIZE = (3840, 1080)  # 整图：左+右
LEFT_RIGHT_SIZE = (STEREO_FULL_SIZE[0] // 2, STEREO_FULL_SIZE[1])  # 单眼：1920x1080

# 摄像头索引（按你项目里习惯：双目一般是 1）
CAMERA_INDEX = 1

# 标定图片保存目录（建议放到 project/config/calibration_3840/）
DEFAULT_CALIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration")

# 输出 JSON（建议覆盖原 camera_params.json，也可以另存一个）
DEFAULT_OUTPUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_params.json")

# 采集标定图的建议数量（只是提示，不是硬限制）
SUGGEST_NUM_IMAGES = 45

# ============================================================


@dataclass
class PairDebug:
    path: str
    area_ratio_l: float
    area_ratio_r: float
    blur_l: float
    blur_r: float


# ===================== 工具函数 =====================

def generate_checkerboard_image(
    pattern_size_param=PATTERN_SIZE,
    square_size_mm_param=SQUARE_SIZE_MM,
    pixels_per_mm=12,
    margin_mm=15,
    save_path="checkerboard.png",
):
    """
    生成棋盘格标定图，可打印后用于相机标定。
    说明：
      - pattern_size 是内角点数 (cols, rows)
      - 棋盘格格子数 = (cols+1, rows+1)
    """
    cols, rows = pattern_size_param
    square_px = int(square_size_mm_param * pixels_per_mm)
    margin_px = int(margin_mm * pixels_per_mm)

    board_cols = cols + 1
    board_rows = rows + 1

    w = board_cols * square_px + 2 * margin_px
    h = board_rows * square_px + 2 * margin_px
    img = np.ones((h, w), dtype=np.uint8) * 255

    for i in range(board_rows):
        for j in range(board_cols):
            color = 0 if (i + j) % 2 == 0 else 255
            y1 = margin_px + i * square_px
            y2 = margin_px + (i + 1) * square_px
            x1 = margin_px + j * square_px
            x2 = margin_px + (j + 1) * square_px
            img[y1:y2, x1:x2] = color

    cv2.imwrite(save_path, img)
    print(f"[OK] 已生成棋盘格图片: {os.path.abspath(save_path)}")
    print("     建议：按真实尺寸打印（每格 %.1f mm），贴平整硬板，避免翘曲。" % square_size_mm_param)
    return img


def _prepare_object_points(pattern_size_param, square_size_mm_param):
    """构建 3D 物理坐标（棋盘格平面 Z=0），单位为 mm。"""
    cols, rows = pattern_size_param
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size_mm_param)
    return objp


def _detect_corners(gray, pattern_size_param):
    """棋盘格角点检测 + 亚像素优化。"""
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, pattern_size_param, flags)
    if not ret:
        return False, None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners_refined


def _compute_reprojection_error(objpoints, imgpoints, K, dist, rvecs, tvecs):
    """平均重投影误差（像素）。"""
    total_error = 0.0
    total_points = 0
    for objp, imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        error = cv2.norm(imgp, projected, cv2.NORM_L2) ** 2
        total_error += error
        total_points += len(objp)
    if total_points == 0:
        return float("inf")
    return float(np.sqrt(total_error / total_points))


def _split_train_test(objpoints, imgpoints, train_ratio=0.8, seed=42):
    """随机切分 train/test，用于输出泛化误差（solvePnP 评估）。"""
    n = len(objpoints)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_train = int(n * train_ratio)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    obj_train = [objpoints[i] for i in train_idx]
    img_train = [imgpoints[i] for i in train_idx]
    obj_test = [objpoints[i] for i in test_idx]
    img_test = [imgpoints[i] for i in test_idx]
    return obj_train, img_train, obj_test, img_test


def _compute_reprojection_error_pnp(objpoints, imgpoints, K, dist):
    """用 solvePnP + projectPoints 在 test 集评估 RMS（像素）。"""
    total_error = 0.0
    total_points = 0
    for objp, imgp in zip(objpoints, imgpoints):
        ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        projected, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        error = cv2.norm(imgp, projected, cv2.NORM_L2) ** 2
        total_error += error
        total_points += len(objp)
    if total_points == 0:
        return float("inf")
    return float(np.sqrt(total_error / total_points))


def _laplacian_var(gray: np.ndarray) -> float:
    """清晰度指标：Laplacian 方差，越大越清晰。"""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _corners_area_ratio(corners: np.ndarray, image_size: tuple[int, int]) -> float:
    """棋盘角点外接矩形面积 / 图像面积。"""
    w, h = image_size
    pts = corners.reshape(-1, 2)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    area = max(0.0, (x_max - x_min)) * max(0.0, (y_max - y_min))
    return float(area / (w * h + 1e-9))


def _build_rectify_maps(K1, dist1, K2, dist2, R, T, image_size, alpha=0.0):
    """stereoRectify + initUndistortRectifyMap，得到 R1/R2/P1/P2/Q 以及 remap maps。"""
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, dist1, K2, dist2, image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=float(alpha),
    )
    map1x, map1y = cv2.initUndistortRectifyMap(K1, dist1, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, dist2, R2, P2, image_size, cv2.CV_32FC1)
    return (R1, R2, P1, P2, Q, roi1, roi2), (map1x, map1y, map2x, map2y)


def _stack_with_horizontal_lines(img_l: np.ndarray, img_r: np.ndarray, step: int = 40) -> np.ndarray:
    """拼接左右校正图并画水平线，便于肉眼检查极线是否对齐。"""
    vis_l = cv2.cvtColor(img_l, cv2.COLOR_GRAY2BGR) if img_l.ndim == 2 else img_l.copy()
    vis_r = cv2.cvtColor(img_r, cv2.COLOR_GRAY2BGR) if img_r.ndim == 2 else img_r.copy()
    h = min(vis_l.shape[0], vis_r.shape[0])
    vis_l = vis_l[:h, :]
    vis_r = vis_r[:h, :]
    canvas = np.hstack([vis_l, vis_r])
    H, W = canvas.shape[:2]
    for y in range(0, H, step):
        cv2.line(canvas, (0, y), (W - 1, y), (0, 255, 0), 1)
    return canvas


def _evaluate_rectified_epiline_alignment(
    image_paths,
    pattern_size_param,
    stereo_full_size,
    maps,
    image_size,
    debug_dir,
    max_debug_save=30,
):
    """
    在校正后的左右图上重新检测角点并计算 Δy = yL - yR 的 RMS。
    Δy RMS 越小，极线对齐越好（通常 < 0.7px 算不错）。
    """
    map1x, map1y, map2x, map2y = maps
    dy_all = []
    used = 0
    saved = 0
    os.makedirs(debug_dir, exist_ok=True)

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        if (w, h) != stereo_full_size:
            # 本流程必须保证标定图就是 3840×1080，否则直接跳过
            continue

        half = w // 2
        left_img = img[:, :half]
        right_img = img[:, half:]

        rect_l = cv2.remap(left_img, map1x, map1y, interpolation=cv2.INTER_LINEAR)
        rect_r = cv2.remap(right_img, map2x, map2y, interpolation=cv2.INTER_LINEAR)

        gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)

        ok_l, corners_l = _detect_corners(gray_l, pattern_size_param)
        ok_r, corners_r = _detect_corners(gray_r, pattern_size_param)
        if not (ok_l and ok_r):
            continue

        used += 1
        y_l = corners_l.reshape(-1, 2)[:, 1]
        y_r = corners_r.reshape(-1, 2)[:, 1]
        dy_all.extend((y_l - y_r).tolist())

        if saved < max_debug_save:
            vis = _stack_with_horizontal_lines(rect_l, rect_r, step=40)
            wl = image_size[0]
            vis_l = vis[:, :wl]
            vis_r = vis[:, wl:]
            cv2.drawChessboardCorners(vis_l, pattern_size_param, corners_l, True)
            cv2.drawChessboardCorners(vis_r, pattern_size_param, corners_r, True)
            base = os.path.splitext(os.path.basename(path))[0]
            cv2.imwrite(os.path.join(debug_dir, f"rectified_{base}.jpg"), vis)
            saved += 1

    if len(dy_all) == 0:
        return {"used_pairs_for_dy": 0, "dy_rms": float("inf"), "dy_max_abs": float("inf")}

    dy_all = np.array(dy_all, dtype=np.float64)
    dy_rms = float(np.sqrt(np.mean(dy_all ** 2)))
    dy_max_abs = float(np.max(np.abs(dy_all)))
    return {"used_pairs_for_dy": used, "dy_rms": dy_rms, "dy_max_abs": dy_max_abs}


# ===================== 采集标定图片（关键：用 3840x1080） =====================

def capture_stereo_calibration_images(
    out_dir: str,
    camera_index: int = CAMERA_INDEX,
    stereo_full_size: tuple[int, int] = STEREO_FULL_SIZE,
):
    """
    打开双目摄像头，以 3840×1080 实时预览，并按 Space 保存原始拼接图。
    保存的每一张图就是“标定输入”，后续标定脚本直接从磁盘读取它们并一分为二。
    """
    import platform
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if platform.system() == "Windows" else camera_index)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头，请检查 CAMERA_INDEX。")

    # 请求分辨率（驱动可能要 read 一帧后才生效）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, stereo_full_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, stereo_full_size[1])

    # 预热
    print("[采集] 预热 15 帧，让曝光/白平衡稳定...")
    for _ in range(15):
        ret, frame = cap.read()
        if not ret:
            break

    # 读取一帧确认实际分辨率
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        raise RuntimeError("读帧失败。")

    h, w = frame.shape[:2]
    if (w, h) != stereo_full_size:
        cap.release()
        raise RuntimeError(
            f"当前摄像头输出为 {w}x{h}，不是期望的 {stereo_full_size[0]}x{stereo_full_size[1]}。\n"
            "请检查相机驱动是否支持 3840×1080；或在 Windows 相机/驱动面板里选择正确模式。"
        )

    print("[采集] 已确认摄像头输出: %dx%d" % (w, h))
    print("[采集] 操作：按 Space 保存一张；按 q 退出。建议采集 %d 张左右。" % SUGGEST_NUM_IMAGES)
    print("       建议覆盖：远/近(30~100cm)、左/右/上/下、旋转、倾斜、不同占比(0.1~0.4)。")
    print("")

    cv2.namedWindow("Stereo Capture (Left | Right)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Stereo Capture (Left | Right)", 1280, 540)

    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[采集] 读帧失败，结束。")
                break

            # 预览缩小显示
            vis = frame
            if vis.shape[1] > 1280:
                vis = cv2.resize(vis, (1280, vis.shape[0] * 1280 // vis.shape[1]))
            cv2.imshow("Stereo Capture (Left | Right)", vis)

            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                break
            if key == 32:  # Space
                idx += 1
                save_path = os.path.join(out_dir, f"stereo_{idx:04d}.png")
                cv2.imwrite(save_path, frame)  # 保存原始 3840×1080，不做任何裁剪/缩放
                print(f"[采集] 已采样 #{idx}，已保存 {save_path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"[采集] 完成，共保存 {idx} 张到 {os.path.abspath(out_dir)}")
    return idx


# ===================== 标定流程（强制以 1920x1080 单眼为基准） =====================

def _calibrate_one_eye_from_stereo_images(
    images_dir: str,
    eye: str,
    pattern_size_param=PATTERN_SIZE,
    square_size_mm_param=SQUARE_SIZE_MM,
    stereo_full_size: tuple[int, int] = STEREO_FULL_SIZE,
):
    """
    从“3840×1080 拼接图”中切左右半幅(1920×1080)来做单目标定。
    重要：这里不允许混入 2560×1080 的旧图（会直接跳过），避免再次出现不一致问题。
    """
    assert eye in ("left", "right")
    image_paths = sorted(
        glob.glob(os.path.join(images_dir, "*.jpg"))
        + glob.glob(os.path.join(images_dir, "*.png"))
        + glob.glob(os.path.join(images_dir, "*.jpeg"))
    )
    if not image_paths:
        raise RuntimeError(f"目录 {images_dir} 没有找到标定图片（jpg/png/jpeg）。")

    objp_template = _prepare_object_points(pattern_size_param, square_size_mm_param)
    objpoints, imgpoints = [], []
    used = 0

    debug_dir = os.path.join(images_dir, "debug_single")
    os.makedirs(debug_dir, exist_ok=True)

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        if (w, h) != stereo_full_size:
            # 严格要求：重新标定只用 3840×1080
            continue

        half = w // 2
        roi = img[:, :half] if eye == "left" else img[:, half:]

        # 单眼尺寸应为 1920×1080
        if roi.shape[1] != LEFT_RIGHT_SIZE[0] or roi.shape[0] != LEFT_RIGHT_SIZE[1]:
            continue

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ok, corners = _detect_corners(gray, pattern_size_param)
        if not ok:
            continue

        objpoints.append(objp_template.copy())
        imgpoints.append(corners)
        used += 1

        # debug：保存角点图，方便快速筛选
        if used <= 40:
            vis = roi.copy()
            cv2.drawChessboardCorners(vis, pattern_size_param, corners, True)
            base = os.path.splitext(os.path.basename(path))[0]
            cv2.imwrite(os.path.join(debug_dir, f"{eye}_{base}.jpg"), vis)

    if used < 8:
        raise RuntimeError(f"{eye} 眼有效图片太少（{used} 张）。建议至少 15~25 张，姿态要丰富。")

    image_size = (LEFT_RIGHT_SIZE[0], LEFT_RIGHT_SIZE[1])  # (w,h) = (1920,1080)

    # 切 train/test
    obj_train, img_train, obj_test, img_test = _split_train_test(objpoints, imgpoints, train_ratio=0.8)

    # 单目标定（求 K, dist）
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_train, img_train, image_size, None, None)

    train_rms = _compute_reprojection_error(obj_train, img_train, K, dist, rvecs, tvecs)
    test_rms = _compute_reprojection_error_pnp(obj_test, img_test, K, dist)

    return {
        "success": bool(ret),
        "image_size": [image_size[0], image_size[1]],
        "K": K.tolist(),
        "dist": dist.reshape(-1).tolist(),
        "reprojection_error_train": float(train_rms),
        "reprojection_error_test": float(test_rms),
        "used_images": int(used),
        "used_train": int(len(obj_train)),
        "used_test": int(len(obj_test)),
    }


def calibrate_stereo_from_3840_images(
    images_dir: str,
    output_json_path: str,
    pattern_size_param=PATTERN_SIZE,
    square_size_mm_param=SQUARE_SIZE_MM,
    stereo_full_size: tuple[int, int] = STEREO_FULL_SIZE,
    alpha: float = 0.3,
    area_ratio_min: float = 0.10,
    blur_var_min: float = 80.0,
    enable_outlier_pruning: bool = True,
    prune_drop_top_k: int = 3,
    prune_keep_min_pairs: int = 12,
):
    """
    重新标定的主入口：
      1) 单目标定 left/right（基于 1920×1080）
      2) 双目标定 stereoCalibrate(FIX_INTRINSIC) 求 R/T/E/F
      3) stereoRectify 得到 R1/R2/P1/P2/Q + remap
      4) rectify 验收（Δy RMS）
      5) 写入 camera_params.json
    """
    # ---------- 1) 单目标定 ----------
    left = _calibrate_one_eye_from_stereo_images(
        images_dir, "left", pattern_size_param, square_size_mm_param, stereo_full_size
    )
    right = _calibrate_one_eye_from_stereo_images(
        images_dir, "right", pattern_size_param, square_size_mm_param, stereo_full_size
    )

    print(f"[单目] 左眼 Train RMS={left['reprojection_error_train']:.4f}px, Test RMS={left['reprojection_error_test']:.4f}px, Images={left['used_train']}/{left['used_test']}")
    print(f"[单目] 右眼 Train RMS={right['reprojection_error_train']:.4f}px, Test RMS={right['reprojection_error_test']:.4f}px, Images={right['used_train']}/{right['used_test']}")
    print("")

    K1 = np.array(left["K"], dtype=np.float64)
    dist1 = np.array(left["dist"], dtype=np.float64).reshape(-1, 1)
    K2 = np.array(right["K"], dtype=np.float64)
    dist2 = np.array(right["dist"], dtype=np.float64).reshape(-1, 1)
    image_size = tuple(left["image_size"])  # (1920,1080)

    # ---------- 2) 收集双目点对 ----------
    objp_template = _prepare_object_points(pattern_size_param, square_size_mm_param)
    objpoints, imgpoints_l, imgpoints_r, used_paths = [], [], [], []

    debug_pair_dir = os.path.join(images_dir, "debug_stereo")
    debug_rect_dir = os.path.join(images_dir, "debug_rectified")
    os.makedirs(debug_pair_dir, exist_ok=True)
    os.makedirs(debug_rect_dir, exist_ok=True)

    rejected_no_corners = 0
    rejected_area = 0
    rejected_blur = 0

    image_paths = sorted(
        glob.glob(os.path.join(images_dir, "*.jpg"))
        + glob.glob(os.path.join(images_dir, "*.png"))
        + glob.glob(os.path.join(images_dir, "*.jpeg"))
    )

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        if (w, h) != stereo_full_size:
            continue

        half = w // 2
        left_img = img[:, :half]
        right_img = img[:, half:]

        # 必须是 1920×1080 单眼
        if (left_img.shape[1], left_img.shape[0]) != image_size:
            continue
        if (right_img.shape[1], right_img.shape[0]) != image_size:
            continue

        gray_l = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        ok_l, corners_l = _detect_corners(gray_l, pattern_size_param)
        ok_r, corners_r = _detect_corners(gray_r, pattern_size_param)
        if not (ok_l and ok_r):
            rejected_no_corners += 1
            continue

        # 面积占比筛选
        area_l = _corners_area_ratio(corners_l, image_size)
        area_r = _corners_area_ratio(corners_r, image_size)
        if area_l < area_ratio_min or area_r < area_ratio_min:
            rejected_area += 1
            continue

        # 清晰度筛选
        blur_l = _laplacian_var(gray_l)
        blur_r = _laplacian_var(gray_r)
        if blur_l < blur_var_min or blur_r < blur_var_min:
            rejected_blur += 1
            continue

        objpoints.append(objp_template.copy())
        imgpoints_l.append(corners_l)
        imgpoints_r.append(corners_r)
        used_paths.append(path)

        # debug：保存前 30 对角点可视化
        if len(objpoints) <= 30:
            vis = np.hstack([left_img.copy(), right_img.copy()])
            cv2.drawChessboardCorners(vis[:, :image_size[0]], pattern_size_param, corners_l, True)
            cv2.drawChessboardCorners(vis[:, image_size[0]:], pattern_size_param, corners_r, True)
            base = os.path.splitext(os.path.basename(path))[0]
            cv2.imwrite(os.path.join(debug_pair_dir, f"pair_{base}.jpg"), vis)

    if len(objpoints) < 10:
        raise RuntimeError(
            f"[双目] 有效样本太少（{len(objpoints)} 对）。\n"
            f"统计：no_corners={rejected_no_corners}, area={rejected_area}, blur={rejected_blur}\n"
            "建议：多采集一些姿态丰富、棋盘更大、更清晰的样本，或适当降低 area_ratio_min/blur_var_min。"
        )

    print(f"[双目] 收集到有效样本对：{len(objpoints)} / 总文件={len(image_paths)}")
    print(f"[双目] 剔除统计：no_corners={rejected_no_corners}, area={rejected_area}, blur={rejected_blur}  (阈值 area_ratio_min={area_ratio_min}, blur_var_min={blur_var_min})")
    print("")

    # ---------- 3) stereoCalibrate：固定内参，只求外参 ----------
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    flags = cv2.CALIB_FIX_INTRINSIC

    stereo_rms, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r,
        K1, dist1, K2, dist2,
        image_size,
        criteria=criteria,
        flags=flags,
    )

    # ---------- 4) rectify + 验收 ----------
    (R1, R2, P1, P2, Q, roi1, roi2), maps = _build_rectify_maps(
        K1, dist1, K2, dist2, R, T, image_size, alpha=alpha
    )

    dy_stats = _evaluate_rectified_epiline_alignment(
        used_paths,
        pattern_size_param,
        stereo_full_size,
        maps,
        image_size,
        debug_rect_dir,
        max_debug_save=30,
    )

    # ---------- 5) （可选）离群剔除：用 rectified Δy RMS 思路做二次标定 ----------
    pruning_info = {"enabled": bool(enable_outlier_pruning)}
    if enable_outlier_pruning and len(used_paths) >= (prune_keep_min_pairs + 2):
        # 这里复用“每对样本做 rectified 后角点 Δy RMS”的做法，但简化：只丢最差 K 对
        # （保持脚本短一些；你也可以把你原先那套完整 pruning 函数搬过来）
        per_pair = []
        map1x, map1y, map2x, map2y = maps
        for p in used_paths:
            img = cv2.imread(p)
            if img is None:
                continue
            left_img = img[:, :stereo_full_size[0] // 2]
            right_img = img[:, stereo_full_size[0] // 2:]
            rect_l = cv2.remap(left_img, map1x, map1y, interpolation=cv2.INTER_LINEAR)
            rect_r = cv2.remap(right_img, map2x, map2y, interpolation=cv2.INTER_LINEAR)
            ok_l, cl = _detect_corners(cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY), pattern_size_param)
            ok_r, cr = _detect_corners(cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY), pattern_size_param)
            if not (ok_l and ok_r):
                dy_rms = float("inf")
                dy_max = float("inf")
            else:
                dy = (cl.reshape(-1, 2)[:, 1] - cr.reshape(-1, 2)[:, 1]).astype(np.float64)
                dy_rms = float(np.sqrt(np.mean(dy ** 2)))
                dy_max = float(np.max(np.abs(dy)))
            per_pair.append((p, dy_rms, dy_max))

        per_pair.sort(key=lambda x: x[1], reverse=True)  # dy_rms 从大到小
        drop_k = int(prune_drop_top_k)
        if len(per_pair) - drop_k < prune_keep_min_pairs:
            drop_k = max(0, len(per_pair) - prune_keep_min_pairs)

        dropped = per_pair[:drop_k]
        kept_set = set(p for p, _, _ in per_pair[drop_k:])

        pruning_info.update({
            "drop_top_k": int(prune_drop_top_k),
            "keep_min_pairs": int(prune_keep_min_pairs),
            "before_pairs": int(len(used_paths)),
            "after_pairs": int(len(kept_set)),
            "dropped": [{"file": os.path.basename(p), "dy_rms": float(r), "dy_max_abs": float(m)} for p, r, m in dropped],
        })

        if drop_k > 0:
            obj2, l2, r2, p2 = [], [], [], []
            for objp, cl, cr, p in zip(objpoints, imgpoints_l, imgpoints_r, used_paths):
                if p in kept_set:
                    obj2.append(objp)
                    l2.append(cl)
                    r2.append(cr)
                    p2.append(p)

            stereo_rms2, _, _, _, _, R2b, T2b, E2b, F2b = cv2.stereoCalibrate(
                obj2, l2, r2,
                K1, dist1, K2, dist2,
                image_size,
                criteria=criteria,
                flags=flags,
            )
            (RR1, RR2, PP1, PP2, QQ, rroi1, rroi2), maps2 = _build_rectify_maps(
                K1, dist1, K2, dist2, R2b, T2b, image_size, alpha=alpha
            )
            dy_stats2 = _evaluate_rectified_epiline_alignment(
                p2, pattern_size_param, stereo_full_size, maps2, image_size, debug_rect_dir, max_debug_save=30
            )
            pruning_info["second_pass"] = {
                "rms": float(stereo_rms2),
                "dy_rms": float(dy_stats2["dy_rms"]),
                "dy_max_abs": float(dy_stats2["dy_max_abs"]),
                "used_pairs_for_dy": int(dy_stats2["used_pairs_for_dy"]),
            }

            # 用 dy_rms 更小的结果
            if dy_stats2["dy_rms"] < dy_stats["dy_rms"]:
                stereo_rms, R, T, E, F = stereo_rms2, R2b, T2b, E2b, F2b
                R1, R2, P1, P2, Q, roi1, roi2 = RR1, RR2, PP1, PP2, QQ, rroi1, rroi2
                dy_stats = dy_stats2
                used_paths = p2
                pruning_info["accepted_second_pass"] = True
            else:
                pruning_info["accepted_second_pass"] = False
        else:
            pruning_info["note"] = "No pairs dropped (keep_min_pairs constraint)."
    else:
        pruning_info["note"] = "Pruning disabled or not enough pairs."

    # ---------- 6) 写 JSON ----------
    data = {
        "pattern_size": list(pattern_size_param),
        "square_size_mm": float(square_size_mm_param),
        "left": left,
        "right": right,
        "stereo": {
            "rms": float(stereo_rms),
            "R": R.tolist(),
            "T": T.reshape(-1).tolist(),  # 单位：mm（因为 objpoints 是 mm）
            "E": E.tolist(),
            "F": F.tolist(),
            "R1": R1.tolist(),
            "R2": R2.tolist(),
            "P1": P1.tolist(),
            "P2": P2.tolist(),
            "Q": Q.tolist(),
            "roi1": list(roi1),
            "roi2": list(roi2),
            "image_size": list(image_size),  # 关键：这里应为 1920×1080
            "num_pairs_total": int(len(image_paths)),
            "num_pairs_accepted": int(len(objpoints)),
            "rejected": {
                "no_corners": int(rejected_no_corners),
                "area": int(rejected_area),
                "blur": int(rejected_blur),
                "area_ratio_min": float(area_ratio_min),
                "blur_var_min": float(blur_var_min),
            },
            "rectify": {
                "alpha": float(alpha),
                "dy_rms": float(dy_stats["dy_rms"]),
                "dy_max_abs": float(dy_stats["dy_max_abs"]),
                "used_pairs_for_dy": int(dy_stats["used_pairs_for_dy"]),
            },
            "used_pair_paths": used_paths,
            "debug_dirs": {
                "pairs": debug_pair_dir,
                "rectified": debug_rect_dir,
            },
            "pruning": pruning_info,
        },
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("[写入] camera_params.json 已更新:", os.path.abspath(output_json_path))
    print("[结果] stereoCalibrate RMS=%.4f px" % float(stereo_rms))
    print("[验收] Rectify Δy RMS=%.4f px, max|Δy|=%.4f px, used_pairs=%d" % (
        float(dy_stats["dy_rms"]), float(dy_stats["dy_max_abs"]), int(dy_stats["used_pairs_for_dy"])
    ))
    print("[Debug] pair角点=%s  rectify对齐=%s" % (debug_pair_dir, debug_rect_dir))
    print("")
    print("下一步建议：把你的立体深度管线改为【不再 resize 到 1280】，直接在 1920×1080 上 remap + SGBM。")
    return data


# ===================== blur 统计（可选） =====================

def analyze_blur_distribution_for_stereo(
    images_dir: str,
    pattern_size_param=PATTERN_SIZE,
    stereo_full_size: tuple[int, int] = STEREO_FULL_SIZE,
):
    """
    扫一遍所有“角点检测成功”的样本对，统计 blur_l / blur_r 的分布，
    辅助选择 blur_var_min。
    """
    image_paths = sorted(
        glob.glob(os.path.join(images_dir, "*.jpg"))
        + glob.glob(os.path.join(images_dir, "*.png"))
        + glob.glob(os.path.join(images_dir, "*.jpeg"))
    )
    if not image_paths:
        print(f"[analyze_blur] 在 {images_dir} 没有找到标定图片。")
        return

    blur_all = []
    infos: list[PairDebug] = []

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        if (w, h) != stereo_full_size:
            continue

        half = w // 2
        left_img = img[:, :half]
        right_img = img[:, half:]

        gray_l = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        ok_l, corners_l = _detect_corners(gray_l, pattern_size_param)
        ok_r, corners_r = _detect_corners(gray_r, pattern_size_param)
        if not (ok_l and ok_r):
            continue

        area_l = _corners_area_ratio(corners_l, LEFT_RIGHT_SIZE)
        area_r = _corners_area_ratio(corners_r, LEFT_RIGHT_SIZE)
        b_l = _laplacian_var(gray_l)
        b_r = _laplacian_var(gray_r)

        blur_all.extend([b_l, b_r])
        infos.append(PairDebug(path=path, area_ratio_l=area_l, area_ratio_r=area_r, blur_l=b_l, blur_r=b_r))

    if not blur_all:
        print("[analyze_blur] 没有任何一对样本左右同时检测到角点。")
        return

    arr = np.array(blur_all, dtype=np.float64)
    ps = [5, 10, 25, 50, 75, 90, 95]
    print(f"[analyze_blur] 有效样本对数量: {len(infos)}")
    print(f"[analyze_blur] blur(min/max) = {arr.min():.2f} / {arr.max():.2f}")
    print("[analyze_blur] 百分位：")
    for p in ps:
        print("  P%02d: %.2f" % (p, float(np.percentile(arr, p))))

    csv_path = os.path.join(images_dir, "blur_stats.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("path,area_l,area_r,blur_l,blur_r\n")
        for it in infos:
            f.write("%s,%.6f,%.6f,%.3f,%.3f\n" % (
                os.path.basename(it.path), it.area_ratio_l, it.area_ratio_r, it.blur_l, it.blur_r
            ))
    print(f"[analyze_blur] 详细统计已写入: {os.path.abspath(csv_path)}")
    print("建议：blur_var_min 可从 P10~P25 之间选一个起点。")


# ===================== main =====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-board", action="store_true", help="生成棋盘格打印图 checkerboard.png")
    parser.add_argument("--capture", action="store_true", help="打开摄像头采集新的 3840×1080 标定图")
    parser.add_argument("--calibrate", action="store_true", help="用采集到的标定图进行完整重新标定并写 JSON")
    parser.add_argument("--analyze-blur", action="store_true", help="统计 blur 分布辅助选 blur_var_min")

    parser.add_argument("--dir", type=str, default=DEFAULT_CALIB_DIR, help="标定图目录（默认 project/config/calibration_3840/）")
    parser.add_argument("--out-json", type=str, default=DEFAULT_OUTPUT_JSON, help="输出 JSON 路径（默认 project/config/camera_params.json）")

    # 下面这些参数你可以按实际情况调
    parser.add_argument("--alpha", type=float, default=0.3, help="stereoRectify alpha（0裁剪多，0.3~0.5更保视野）")
    parser.add_argument("--area-ratio-min", type=float, default=0.05, help="棋盘面积占比阈值")
    parser.add_argument("--blur-var-min", type=float, default=70.0, help="清晰度阈值（Laplacian 方差）")
    parser.add_argument("--no-prune", action="store_true", help="关闭离群剔除")
    parser.add_argument("--prune-drop-k", type=int, default=3, help="离群剔除：丢掉最差K对")
    parser.add_argument("--prune-keep-min", type=int, default=12, help="离群剔除：至少保留这么多对")
    args = parser.parse_args()

    if args.gen_board:
        generate_checkerboard_image(
            pattern_size_param=PATTERN_SIZE,
            square_size_mm_param=SQUARE_SIZE_MM,
            save_path="checkerboard.png",
        )

    if args.capture:
        capture_stereo_calibration_images(out_dir=args.dir)

    if args.analyze_blur:
        analyze_blur_distribution_for_stereo(images_dir=args.dir)

    if args.calibrate:
        calibrate_stereo_from_3840_images(
            images_dir=args.dir,
            output_json_path=args.out_json,
            alpha=float(args.alpha),
            area_ratio_min=float(args.area_ratio_min),
            blur_var_min=float(args.blur_var_min),
            enable_outlier_pruning=not bool(args.no_prune),
            prune_drop_top_k=int(args.prune_drop_k),
            prune_keep_min_pairs=int(args.prune_keep_min),
        )

    if not (args.gen_board or args.capture or args.analyze_blur or args.calibrate):
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())