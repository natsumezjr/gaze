#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
眼动追踪系统主程序
"""

from __future__ import annotations

import sys
import os
import cv2
import numpy as np
import logging
import time

from datetime import datetime

# 配置日志
def setup_logging():
    """配置日志系统"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"recognition_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"日志文件创建: {log_file}")
    return log_file

# 设置日志
log_file = setup_logging()

# 抑制TensorFlow警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
workspace_root = os.path.dirname(project_dir)
if workspace_root not in sys.path:
    sys.path.insert(1, workspace_root)

# 导入项目模块
try:
    from project.recognition.core.detector import FaceDetector
    from project.recognition.utils.camera_calibration import CameraCalibrator
    from project.recognition.utils.camera_data_manager import add_frame, get_image, get_depth, get_resolution
    
    # 导入校准模块
    from project.fitting.app.api import start_session, push_sample, fit_kappa, compute_gaze
    from project.fitting.app.state import SESSION_MANAGER
    from project.fitting.main import main as fit_main
    from project.fitting.core.kappa_calibrator_pro import evaluate_fit, build_samples_from_arrays
except ImportError as e:
    logging.error(f"导入错误: {e}")
    logging.error("请确保已安装项目包: pip install -e .")
    sys.exit(1)

def nine_point_grid(resolution: tuple[int, int] | None) -> list[tuple[int, int]]:
    """生成9点校准网格的像素坐标"""
    if resolution is None:
        w, h = 1920, 1080
    else:
        w, h = resolution
    xs = [int(w * r) for r in (0.1, 0.5, 0.9)]
    ys = [int(h * r) for r in (0.1, 0.5, 0.9)]
    points = [(x, y) for y in ys for x in xs]
    return points

def main():
    frame_id = 0
    rgb_d = True
    camera_calibrator = CameraCalibrator(rgb_d=rgb_d)
    cap = camera_calibrator.get_cap()
    
    if cap is None:
        logging.error("无法初始化摄像头")
        return
    
    camera_params = camera_calibrator.load_camera_params()
    face_detector = FaceDetector(camera_params, rgb_d=rgb_d)

    # 校准状态
    calibration_started = False
    calibration_finished = False
    target_points = []
    current_point_index = 0
    
    # 检测稳态控制（去抖与滞回）
    face_present = False
    miss_count = 0
    detect_count = 0
    miss_threshold = 3      # 连续丢失阈值
    detect_threshold = 2    # 连续检出阈值
    warmup_frames = 15      # 启动预热帧数（不报丢失）

    def enhance_for_detection(img: np.ndarray) -> np.ndarray:
        """轻量增强：对亮度通道做CLAHE，提升弱光鲁棒性"""
        try:
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            y_eq = clahe.apply(y)
            ycrcb_eq = cv2.merge((y_eq, cr, cb))
            return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
        except Exception:
            return img
    
    try:
        while not calibration_finished:
            ret, frame = cap.read()
            if not ret:
                logging.error("无法读取摄像头数据")
                break
            
            # 使用固定深度图
            h, w = frame.shape[:2]
            fixed_depth_meters = 0.6
            depth_map = np.ones((h, w), dtype=np.float32) * (fixed_depth_meters / camera_params['depth_scale'])
            
            # 添加帧数据
            add_frame(frame_id, frame, depth_map)
            image = get_image(frame_id)
            depth = get_depth(frame_id)
            
            if image is not None and depth is not None:
                # 预处理增强后送入检测
                enhanced = enhance_for_detection(image)
                detected = face_detector.detect_face(enhanced, depth)
                
                # 去抖与滞回状态机
                if detected:
                    detect_count += 1
                    miss_count = 0
                    if not face_present and detect_count >= detect_threshold:
                        face_present = True
                else:
                    miss_count += 1
                    detect_count = 0
                    if face_present and (frame_id > warmup_frames) and miss_count >= miss_threshold:
                        face_present = False

                if detected:
                    face_detector.update_fitting_data()
                    
                    if not calibration_started:
                        logging.info("检测到人脸，开始校准流程。请看向屏幕上的红点并按 's' 键。")
                        start_session("integrated-calibration")
                        res = get_resolution()
                        target_points = nine_point_grid(res)
                        calibration_started = True
                    
                    if calibration_started:
                        if current_point_index < len(target_points):
                            # 在屏幕上绘制当前目标点
                            current_target = target_points[current_point_index]
                            cv2.circle(frame, current_target, 10, (0, 0, 255), -1)
                            cv2.putText(frame, 
                                        f"Point {current_point_index + 1}/{len(target_points)}", 
                                        (current_target[0] + 20, current_target[1] + 20), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        
                else:
                    # 按滞回状态输出更稳定的告警
                    if frame_id > warmup_frames:
                        if not face_present:
                            if not calibration_started:
                                logging.warning(f"帧 {frame_id}: 未检测到人脸")
                            else:
                                logging.warning(f"帧 {frame_id}: 人脸丢失")
            else:
                logging.error(f"帧 {frame_id}: 图像或深度图获取失败")
                break

            # 覆盖层：状态/计数展示
            status_text = "Face: OK" if face_present else "Face: LOST"
            color = (0, 200, 0) if face_present else (0, 0, 200)
            cv2.putText(frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"det:{detect_count} miss:{miss_count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 0), 2)

            cv2.imshow("Gaze Calibration", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s') and calibration_started and current_point_index < len(target_points):
                # 收集当前点数据
                pupil_data = face_detector.recg_fit_data_manager.get_coordinate_point("left", "pupil")
                if pupil_data:
                    current_target = target_points[current_point_index]
                    push_sample({"timestamp": time.time(), "target_pixel": current_target, "eye": "left"})
                    logging.info(f"已收集第 {current_point_index + 1} 个点的数据：{current_target}")
                    current_point_index += 1
                else:
                    logging.warning("未检测到瞳孔数据，无法采集样本。")
            
            # 完成所有点后进行校准
            if calibration_started and current_point_index == len(target_points):
                print("== Step 1: Initial fitting ==")
                fit_main()

                print("== Step 2: Fit kappa ==")
                out = fit_kappa()
                print("Kappa model:", out["kappa_model"])

                print("== Step 3: Compute compensated gaze ==")
                center_uv = target_points[4]
                gaze = compute_gaze("left", center_uv)
                print("Gaze (compensated):", gaze["gaze"])
                
                print("== Step 4: Evaluate fit on current samples ==")
                session = SESSION_MANAGER.get_session()
                eyes = np.vstack([s.eye_center.reshape(1, 3) for s in session.samples])
                pupils = np.vstack([s.pupil_center.reshape(1, 3) for s in session.samples])
                target_pixels = np.vstack([np.array(s.target_pixel).reshape(1, 2) for s in session.samples])
                K = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]], dtype=float)
                samples_pro = build_samples_from_arrays(eyes, pupils, target_pixels=target_pixels, K=K)
                angle_rad = np.radians(out["kappa_model"]["angle_deg"]) if isinstance(out, dict) and "kappa_model" in out else 0.0
                axis = np.array(out["kappa_model"]["axis"], dtype=float) if isinstance(out, dict) and "kappa_model" in out else np.array([0, 0, 1], dtype=float)
                kappa_vec = axis * angle_rad
                metrics = evaluate_fit(samples_pro, kappa_vec, K=K, return_pixel_err=True)
                print("Evaluation:", metrics)
                
                calibration_finished = True
            
            # 检查退出键
            if key == ord('q'):
                logging.info("用户按q退出")
                break
                
            frame_id += 1
                
    except KeyboardInterrupt:
        logging.info("用户中断")
    except Exception as e:
        logging.error(f"发生错误: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()