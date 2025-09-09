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
    from project.fitting.core.kappa_calibrator_pro import (
        estimate_kappa as estimate_kappa_pro,
        apply_kappa,
        evaluate_fit,
        incremental_update,
        build_samples_from_arrays
    )
    from project.fitting.core.gaze_estimator import compute_theoretical_gaze
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

def get_coarse_fitting_results():
    """获取粗拟合结果，供前端调用"""
    return coarse_fitting_results.copy()

def send_to_frontend(data):
    """发送数据给前端（示例函数，需要根据实际前端接口实现）"""
    # 这里可以添加实际的发送逻辑，比如：
    # - WebSocket发送
    # - HTTP POST请求
    # - 写入共享文件
    # - 通过消息队列发送
    logging.info(f"发送给前端: {data}")
    print(f"[前端通信] 发送数据: {data}")

def run_kappa_calibration():
    """运行kappa校准"""
    logging.info("\n" + "="*80)
    logging.info("开始KAPPA校准")
    logging.info("="*80)
    
    try:
        # 获取会话数据
        logging.info("检查会话管理器状态...")
        session = SESSION_MANAGER.get_session()
        if not session:
            logging.error("会话管理器中没有会话数据")
            return None, None, None
        elif not session.samples:
            logging.error(f"会话中有 {len(session.samples)} 个样本，需要至少1个样本进行kappa校准")
            return None, None, None
        
        logging.info(f"会话数据检查通过，共有 {len(session.samples)} 个样本")
        
        # 提取数据
        logging.info("提取校准数据...")
        try:
            eyes = np.vstack([s.eye_center.reshape(1, 3) for s in session.samples])
            pupils = np.vstack([s.pupil_center.reshape(1, 3) for s in session.samples])
            target_pixels = np.vstack([np.array(s.target_pixel).reshape(1, 2) for s in session.samples])
            logging.info("数据提取成功")
        except Exception as e:
            logging.error(f"数据提取失败: {e}")
            return None, None, None
        
        # 相机内参矩阵
        K = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]], dtype=float)
        logging.info(f"相机内参矩阵: {K}")
        
        logging.info(f"使用 {len(session.samples)} 个校准样本进行kappa校准")
        logging.info(f"眼球中心范围: {np.min(eyes, axis=0)} 到 {np.max(eyes, axis=0)}")
        logging.info(f"瞳孔中心范围: {np.min(pupils, axis=0)} 到 {np.max(pupils, axis=0)}")
        logging.info(f"目标像素范围: {np.min(target_pixels, axis=0)} 到 {np.max(target_pixels, axis=0)}")
        
        # 构建样本
        logging.info("构建kappa校准样本...")
        try:
            samples_pro = build_samples_from_arrays(
                eyes, pupils, 
                target_pixels=target_pixels, 
                K=K
            )
            logging.info(f"样本构建成功，共 {len(samples_pro)} 个样本")
        except Exception as e:
            logging.error(f"样本构建失败: {e}")
            return None, None, None
        
        # 估计kappa
        logging.info("开始估计kappa参数...")
        try:
            kappa_pro, info_pro = estimate_kappa_pro(samples_pro, lock_roll=True)
            logging.info(f"Kappa校准结果 - Kappa: {np.degrees(kappa_pro)}°")
            logging.info(f"校准信息: {info_pro}")
        except Exception as e:
            logging.error(f"Kappa估计失败: {e}")
            return None, None, None
        
        # 应用kappa补偿
        logging.info("应用kappa补偿...")
        theoretical_gaze = compute_theoretical_gaze(eyes[0], pupils[0])
        compensated_gaze = apply_kappa(theoretical_gaze.reshape(1, 3), kappa_pro)[0]
        
        logging.info(f"理论视线: {theoretical_gaze}")
        logging.info(f"补偿后视线: {compensated_gaze}")
        
        # 评估拟合质量
        logging.info("评估拟合质量...")
        fit_quality = evaluate_fit(samples_pro, kappa_pro, K, return_pixel_err=True)
        logging.info(f"拟合质量: {fit_quality}")
        
        # 演示增量更新
        logging.info("演示增量kappa更新...")
        kappa_prev = kappa_pro.copy()
        v = theoretical_gaze
        d = compensated_gaze
        
        for i in range(3):
            kappa_new = incremental_update(kappa_prev, v, d, beta=0.1)
            angle_change = np.degrees(np.linalg.norm(kappa_new - kappa_prev))
            logging.info(f"第{i+1}次更新: Kappa变化 {angle_change:.4f}°")
            kappa_prev = kappa_new.copy()
        
        logging.info("Kappa校准完成！")
        return kappa_pro, info_pro, fit_quality
        
    except Exception as e:
        logging.error(f"Kappa校准失败: {e}")
        import traceback
        error_trace = traceback.format_exc()
        logging.error(f"详细错误信息: {error_trace}")
        return None, None, None

# 校准状态
calibration_started = False
calibration_finished = False
target_points = []
current_point_index = 0

# 两段式拟合状态
coarse_fitting_done = False

# 存储拟合结果用于发送给前端
coarse_fitting_results = {
    'kappa_angle': None,
    'fit_quality': None,
    'status': 'pending',  # pending, success, failed
    'message': ''
}

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
    
    # 两段式拟合状态
    coarse_fitting_done = False
    
    # 简化：不做去抖/滞回与图像增强
    
    try:
        while not calibration_finished:
            ret, frame = cap.read()
            if not ret:
                logging.error("无法读取摄像头数据")
                break
            
            # 使用可配置的深度图
            h, w = frame.shape[:2]
            
            # 加载深度配置
            try:
                import json
                depth_config_path = os.path.join(current_dir, "config", "depth_settings.json")
                with open(depth_config_path, 'r', encoding='utf-8') as f:
                    depth_config = json.load(f)
                depth_mode = depth_config.get('depth_mode', 'auto')
                fixed_depths = depth_config.get('fixed_depth_options', {})
                auto_settings = depth_config.get('auto_depth_settings', {})
            except Exception as e:
                logging.warning(f"加载深度配置失败，使用默认设置: {e}")
                depth_mode = 'auto'
                fixed_depths = {'near': 0.4, 'medium': 0.8, 'far': 1.2, 'default': 0.6}
                auto_settings = {'min_depth': 0.3, 'max_depth': 2.0, 'real_pupil_distance': 0.065, 'fallback_depth': 0.6}
            
            if depth_mode == 'auto':
                # 基于人脸大小的动态深度估计
                try:
                    # 检测人脸以估计距离
                    face_detected = face_detector.detect_face(frame, None)
                    if face_detected:
                        # 获取人脸关键点
                        landmarks = face_detector.recg_fit_data_manager.get_coordinate_point("left", "pupil")
                        if landmarks and len(landmarks) > 0:
                            # 基于瞳孔间距估计深度
                            # 这里简化处理，实际应该计算左右瞳孔的像素距离
                            pupil_distance_pixels = 50  # 简化：假设瞳孔间距为50像素
                            # 深度 = (真实瞳孔间距 * 焦距) / 像素瞳孔间距
                            fx = camera_params['intrinsic_params']['fx']
                            real_pupil_distance = auto_settings.get('real_pupil_distance', 0.065)
                            estimated_depth = (real_pupil_distance * fx) / pupil_distance_pixels
                            
                            # 限制在合理范围内
                            min_depth = auto_settings.get('min_depth', 0.3)
                            max_depth = auto_settings.get('max_depth', 2.0)
                            estimated_depth = max(min_depth, min(max_depth, estimated_depth))
                            
                            depth_map = np.ones((h, w), dtype=np.float32) * (estimated_depth / camera_params['depth_scale'])
                            logging.info(f"动态估计深度: {estimated_depth:.2f}m")
                        else:
                            # 使用默认深度
                            fallback_depth = auto_settings.get('fallback_depth', 0.6)
                            depth_map = np.ones((h, w), dtype=np.float32) * (fallback_depth / camera_params['depth_scale'])
                            logging.info(f"使用默认深度: {fallback_depth}m")
                    else:
                        # 未检测到人脸，使用默认深度
                        fallback_depth = auto_settings.get('fallback_depth', 0.6)
                        depth_map = np.ones((h, w), dtype=np.float32) * (fallback_depth / camera_params['depth_scale'])
                        logging.info(f"未检测到人脸，使用默认深度: {fallback_depth}m")
                except Exception as e:
                    # 出错时使用默认深度
                    fallback_depth = auto_settings.get('fallback_depth', 0.6)
                    depth_map = np.ones((h, w), dtype=np.float32) * (fallback_depth / camera_params['depth_scale'])
                    logging.warning(f"深度估计失败，使用默认深度: {e}")
            else:
                # 使用配置的固定深度
                fixed_depth_meters = fixed_depths.get(depth_mode, fixed_depths.get('default', 0.6))
                depth_map = np.ones((h, w), dtype=np.float32) * (fixed_depth_meters / camera_params['depth_scale'])
                logging.debug(f"使用固定深度: {fixed_depth_meters}m")
            
            # 添加帧数据
            add_frame(frame_id, frame, depth_map)
            image = get_image(frame_id)
            depth = get_depth(frame_id)
            
            if image is not None and depth is not None:
                detected = face_detector.detect_face(image, depth)

                if detected:
                    face_detector.update_fitting_data()
                    
                    # 第一步：粗拟合（在检测到人脸后立即进行）
                    if not coarse_fitting_done:
                        logging.info("="*80)
                        logging.info("== 第一步：粗拟合（MediaPipe） ==")
                        logging.info("="*80)
                        print("== 第一步：粗拟合（MediaPipe） ==")
                        
                        # Step 1: 眼球形状拟合
                        logging.info("Step 1: 粗拟合 - 眼球形状拟合")
                        print("Step 1: 粗拟合 - 眼球形状拟合")
                        
                        # 检查数据管理器状态
                        logging.info("检查数据管理器状态...")
                        data_manager = face_detector.recg_fit_data_manager
                        left_points = data_manager.get_coordinate_point("left", "pupil")
                        right_points = data_manager.get_coordinate_point("right", "pupil")
                        logging.info(f"左眼瞳孔点数: {len(left_points) if left_points else 0}")
                        logging.info(f"右眼瞳孔点数: {len(right_points) if right_points else 0}")
                        
                        # 执行眼球形状拟合
                        try:
                            logging.info("开始执行眼球形状拟合...")
                            fit_main()
                            logging.info("眼球形状拟合完成")
                        except Exception as e:
                            logging.error(f"眼球形状拟合失败: {e}")
                            print(f"眼球形状拟合失败: {e}")

                        # Step 2: 初始化会话（为后续收集校准样本做准备）
                        logging.info("Step 2: 初始化校准会话")
                        print("Step 2: 初始化校准会话")
                        
                        # 开始校准会话
                        try:
                            session_info = start_session("integrated-calibration")
                            logging.info(f"会话初始化成功: {session_info}")
                            print(f"会话初始化成功: {session_info}")
                        except Exception as e:
                            logging.error(f"会话初始化失败: {e}")
                            print(f"会话初始化失败: {e}")
                        
                        # 注意：kappa校准需要先收集校准样本，所以这里不进行kappa校准
                        # kappa校准将在收集完9个校准点后进行
                        logging.info("眼球形状拟合完成，等待收集校准样本进行kappa校准")
                        print("眼球形状拟合完成，等待收集校准样本进行kappa校准")
                        
                        # 设置粗拟合完成状态，但不进行kappa校准
                        kappa_pro_coarse, info_pro_coarse, fit_quality_coarse = None, None, None
                        
                        # 更新粗拟合状态（眼球形状拟合完成）
                        coarse_fitting_results.update({
                            'status': 'success',
                            'message': '眼球形状拟合完成，等待收集校准样本'
                        })
                        
                        logging.info("眼球形状拟合完成！等待收集校准样本")
                        print("眼球形状拟合完成！等待收集校准样本")
                        print("请按 's' 键开始收集9个校准点数据")
                        
                        # 发送给前端
                        send_to_frontend(coarse_fitting_results)
                        
                        coarse_fitting_done = True
                    
                    # 第二步：开始校准流程（收集9个点）
                    if not calibration_started and coarse_fitting_done:
                        logging.info("开始校准流程。请看向屏幕上的红点并按 's' 键。")
                        print("开始校准流程。请看向屏幕上的红点并按 's' 键。")
                        # 会话已在第一步初始化，这里只需要设置目标点
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
                    if not calibration_started:
                        logging.warning(f"帧 {frame_id}: 未检测到人脸")
                    else:
                        logging.warning(f"帧 {frame_id}: 人脸丢失")
            else:
                logging.error(f"帧 {frame_id}: 图像或深度图获取失败")
                break

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
            
            # 第四步：精细拟合（使用9个校准点数据）
            if calibration_started and current_point_index == len(target_points):
                logging.info("== 第四步：精细拟合（9点校准数据） ==")
                print("== 第四步：精细拟合（9点校准数据） ==")
                
                logging.info("Step 1: 精细拟合 - 眼球形状拟合")
                print("Step 1: 精细拟合 - 眼球形状拟合")
                fit_main()

                logging.info("Step 2: 精细拟合 - Kappa校准")
                print("Step 2: 精细拟合 - Kappa校准")
                kappa_pro_fine, info_pro_fine, fit_quality_fine = run_kappa_calibration()
                
                if kappa_pro_fine is not None:
                    logging.info("== Step 3: 计算补偿后的视线 ==")
                    print("== Step 3: 计算补偿后的视线 ==")
                    center_uv = target_points[4]
                    gaze = compute_gaze("left", center_uv)
                    print("Gaze (compensated):", gaze["gaze"])
                    
                    logging.info("== Step 4: 评估拟合质量 ==")
                    print("== Step 4: 评估拟合质量 ==")
                    print("Evaluation:", fit_quality_fine)
                    
                    logging.info("精细拟合完成！校准流程结束！")
                    print("精细拟合完成！校准流程结束！")
                else:
                    logging.error("精细拟合失败，校准流程终止")
                    print("精细拟合失败，校准流程终止")
                
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