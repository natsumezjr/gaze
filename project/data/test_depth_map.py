"""双目相机深度图测试脚本

使用 CameraDataManager（camera_params.json 标定）计算校正左/右图与深度图；
若无双目标定或帧尺寸不匹配，则回退为未标定 StereoSGBM 视差显示。

用法: poetry run python -m project.data.test_depth_map
按 q 退出
"""
import os
import sys
import platform

import cv2
import numpy as np

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from project.config.stereo_config import STEREO_FRAME_SIZES
from project.data.data_manager import CameraDataManager


def create_stereo_matcher():
    """无双目标定时使用的未标定 SGBM。"""
    from project.config.stereo_config import (
        SGBM_BLOCK_SIZE,
        SGBM_DISP12_MAX_DIFF,
        SGBM_MIN_DISPARITY,
        SGBM_NUM_DISPARITIES,
        SGBM_P1,
        SGBM_P2,
        SGBM_PREFILTER_CAP,
        SGBM_SPECKLE_RANGE,
        SGBM_SPECKLE_WINDOW_SIZE,
        SGBM_UNIQUENESS_RATIO,
    )
    return cv2.StereoSGBM_create(
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


def main():
    print("=" * 50)
    print("双目相机 - 深度图测试（CameraDataManager + camera_params）")
    print("=" * 50)

    manager = CameraDataManager(rgb_d=True)
    cap = manager.get_cap()
    if cap is None or not cap.isOpened():
        print("[失败] 无法打开摄像头")
        return 1

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[成功] 分辨率: {w}x{h}")
    print("左半幅 -> 左目, 右半幅 -> 右目")
    print("按 q 退出\n")

    stereo_fallback = create_stereo_matcher()

    cv2.namedWindow("Left", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Right", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Disparity / Depth", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Left", 640, 360)
    cv2.resizeWindow("Right", 640, 360)
    cv2.resizeWindow("Disparity / Depth", 640, 360)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[警告] 读取帧失败")
                break

            result = manager.compute_stereo_depth(frame)
            if result is not None and (frame.shape[1], frame.shape[0]) in STEREO_FRAME_SIZES:
                rect_left, rect_right, depth_mm = result
                disp_show = np.nan_to_num(depth_mm, nan=0.0, posinf=0.0, neginf=0.0)
                valid = np.isfinite(depth_mm) & (depth_mm > 0)
                if np.any(valid):
                    lo, hi = np.percentile(depth_mm[valid], [2, 98])
                    disp_show = np.clip(disp_show, lo, hi)
                    if hi > lo:
                        disp_show = ((disp_show - lo) / (hi - lo) * 255).astype(np.uint8)
                    else:
                        disp_show = np.zeros_like(disp_show, dtype=np.uint8)
                else:
                    disp_show = np.zeros_like(disp_show, dtype=np.uint8)
                disp_colormap = cv2.applyColorMap(disp_show, cv2.COLORMAP_JET)
                left_show = rect_left
                right_show = rect_right
            else:
                mid = frame.shape[1] // 2
                left_show = frame[:, 0:mid]
                right_show = frame[:, mid:]
                gray_l = cv2.cvtColor(left_show, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(right_show, cv2.COLOR_BGR2GRAY)
                disparity = stereo_fallback.compute(gray_l, gray_r)
                disp_display = np.clip(disparity, 0, disparity.max())
                if disp_display.max() > disp_display.min():
                    disp_display = (
                        (disp_display - disp_display.min())
                        / (disp_display.max() - disp_display.min())
                        * 255
                    ).astype(np.uint8)
                else:
                    disp_display = np.zeros_like(disparity, dtype=np.uint8)
                disp_colormap = cv2.applyColorMap(disp_display, cv2.COLORMAP_JET)

            cv2.imshow("Left", left_show)
            cv2.imshow("Right", right_show)
            cv2.imshow("Disparity / Depth", disp_colormap)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在退出...")
    finally:
        manager.release_camera()
        cv2.destroyAllWindows()
        print("摄像头资源已释放")
    return 0


if __name__ == "__main__":
    exit(main())
