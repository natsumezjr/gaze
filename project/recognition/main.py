
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
from typing import Optional, Dict, Tuple

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入项目模块
try:
    from recognition.core.detector import FaceDetector
    from recognition.utils.camera_calibration import load_camera_params
    from recognition.utils.data_manager import add_frame, get_image, get_depth
    from recognition.config.settings import DATA_PATH, CAMERA_PARAMS_PATH
    from recognition.config.constants import STATUS_SUCCESS, STATUS_NO_FACE_DETECTED
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装项目包: pip install -e .")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    frame_id = 0
    
    def initialize_camera():
        """初始化摄像头"""
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            return cap
        return None
    
    # 初始化摄像头
    cap = initialize_camera()
    if cap is None:
        logger.error("无法初始化摄像头")
        return
    
    # 初始化人脸检测器（只创建一次）
    face_detector = FaceDetector(load_camera_params(CAMERA_PARAMS_PATH))
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("无法读取摄像头数据")
                break
                
            cv2.imshow("frame", frame)
            
            # 模拟深度图
            h, w = frame.shape[:2]
            depth_map = ((np.random.rand(h, w).astype(np.float32) - 0.5) * 0.1 + 0.7) / load_camera_params(CAMERA_PARAMS_PATH)['depth_scale']
            
            # 添加帧数据
            add_frame(frame_id, frame, depth_map)
            image = get_image(frame_id)
            depth = get_depth(frame_id)
            
            if image is not None and depth is not None:
                logger.info(f"id:{frame_id}，图像或深度图获取成功")
                
                if face_detector.detect_face(image, depth):
                    logger.info(f"id:{frame_id}，检测到人脸")
                    pupil_centers = face_detector.get_pupil_centers()
                    iris_boundaries = face_detector.get_iris_boundaries()
                    logger.info(f"id:{frame_id}，瞳孔中心：{pupil_centers}")
                    logger.info(f"id:{frame_id}，虹膜边界：{iris_boundaries}")
                    
                    # 处理关键点坐标
                    key_coordinates = {
                        'pupil_centers': pupil_centers,
                        'iris_boundaries': iris_boundaries
                    }
                    
                    from fitting.main import main as fitting
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
    