#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
眼动追踪系统主程序

按照README要求实现主循环，包含：
1. 正确的模块导包
2. 相机参数加载
3. 人脸检测器初始化
4. 实时处理循环
5. 结果输出和状态监控
"""

import sys
import os
import cv2
import numpy as np
import time
import logging
import json
from typing import Optional, Dict, Tuple

# 抑制TensorFlow警告 - 必须在导入其他模块之前设置
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=全部, 1=无INFO, 2=无WARNING, 3=无ERROR
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 禁用GPU以避免CUDA相关警告

# 添加项目根目录到Python路径（使得可用绝对导入 `project.*`）
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)  # .../project
workspace_root = os.path.dirname(project_dir)  # 父目录，包含 `project/`
if workspace_root not in sys.path:
    sys.path.insert(1, workspace_root)

# 导入项目模块
try:
    from project.recognition.core.detector import FaceDetector
    from project.recognition.utils.camera_calibration import CameraCalibrator
    from project.recognition.utils.data_manager import add_frame, get_image, get_depth
    from project.recognition.config.settings import DATA_PATH, CAMERA_PARAMS_PATH
    from project.recognition.config.constants import STATUS_SUCCESS, STATUS_NO_FACE_DETECTED
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装项目包: pip install -e .")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    frame_id = 0
    
    camera_calibrator = CameraCalibrator(rgb_d=True)
    cap = camera_calibrator.get_cap()
    
    if cap is None:
        logger.error("无法初始化摄像头")
        return
    
    camera_params = camera_calibrator.load_camera_params()
    

    # 初始化人脸检测器（使用调整后的相机参数）
    face_detector = FaceDetector(camera_params)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("无法读取摄像头数据")
                break
                
            cv2.imshow("frame", frame)
            
            # 使用固定深度图（0.6米）
            h, w = frame.shape[:2]
            fixed_depth_meters = 0.6  # 固定深度0.6米
            depth_map = np.ones((h, w), dtype=np.float32) * (fixed_depth_meters / camera_params['depth_scale'])
            
            # 添加帧数据
            add_frame(frame_id, frame, depth_map)
            image = get_image(frame_id)
            depth = get_depth(frame_id)
            
            if image is not None and depth is not None:
                if face_detector.detect_face(image, depth):
                    # 获取瞳孔中心
                    pupil_centers = face_detector.get_pupil_centers()
                    
                    # 获取虹膜边界
                    iris_boundaries = face_detector.get_iris_boundaries()
                    
                    # 获取眼轮廓
                    eyes_contours = face_detector.get_eyes_contours()
                    
                    # 处理关键点坐标
                    key_coordinates = {
                        'pupil_centers': pupil_centers,
                        'iris_boundaries': iris_boundaries,
                        'eyes_contours': eyes_contours
                    }
                    
                    from project.fitting.main import main as fitting
                    fitting(key_coordinates)
                    
                    # TODO: 添加后续处理逻辑
                    
                else:
                    logger.warning(f"id:{frame_id}，未检测到人脸")
                    # 继续循环，不退出
            else:
                logger.error(f"id:{frame_id}，图像或深度图获取失败")
                break

            frame_id += 1
            
            # 检查退出键
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("用户按q退出")
                break
                
    except KeyboardInterrupt:
        logger.info("用户中断")
    except Exception as e:
        logger.error(f"发生错误: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()