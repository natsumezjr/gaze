#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
眼动追踪系统主程序
"""

import sys
import os
import cv2
import numpy as np
import time
import logging
import json
from typing import Optional, Dict, Tuple
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
import os
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
    from project.recognition.utils.camera_data_manager import add_frame, get_image, get_depth
except ImportError as e:
    logging.error(f"导入错误: {e}")
    logging.error("请确保已安装项目包: pip install -e .")
    sys.exit(1)

def main():
    frame_id = 0
    rgb_d = False
    camera_calibrator = CameraCalibrator(rgb_d=rgb_d)
    cap = camera_calibrator.get_cap()
    
    if cap is None:
        logging.error("无法初始化摄像头")
        return
    
    camera_params = camera_calibrator.load_camera_params()
    
    # 初始化人脸检测器
    face_detector = FaceDetector(camera_params, rgb_d=rgb_d)

    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("无法读取摄像头数据")
                break
                
            cv2.imshow("frame", frame)
            
            # 使用固定深度图（0.6米）
            h, w = frame.shape[:2]
            fixed_depth_meters = 0.6
            # 转换为相机原始单位
            depth_map = np.ones((h, w), dtype=np.float32) * (fixed_depth_meters / camera_params['depth_scale'])
            
            # 添加帧数据
            add_frame(frame_id, frame, depth_map)
            image = get_image(frame_id)
            depth = get_depth(frame_id)
            
            if image is not None and depth is not None:
                # 检测人脸
                if face_detector.detect_face(image, depth):
                    # 获取RecgFitDataManager实例
                    face_detector.update_fitting_data()
                    logging.info(f"帧 {frame_id}: 更新拟合数据")
                    try:
                        from project.fitting.main import main as fitting
                        fitting()
                    except Exception as e:
                        logging.error(f"拟合失败: {e}")
                    break  # 用于测试一帧拟合
                        
                else:
                    logging.warning(f"帧 {frame_id}: 未检测到人脸")
            else:
                logging.error(f"帧 {frame_id}: 图像或深度图获取失败")
                break

            frame_id += 1
            
            # 检查退出键
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("用户按q退出")
                break
                
    except KeyboardInterrupt:
        logging.info("用户中断")
    except Exception as e:
        logging.error(f"发生错误: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()