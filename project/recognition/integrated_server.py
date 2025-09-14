#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
眼动追踪系统集成服务器
整合了识别模块和拟合模块的前后端功能
按照 structure.md 的流程组织
"""

import sys
import os
import json
import time
import threading
import logging
import numpy as np
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory, render_template
from flask_cors import CORS

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
workspace_root = os.path.dirname(project_dir)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

# 导入模块
from project.recognition.core.detector import FaceDetector
from project.recognition.utils.camera_calibration import CameraCalibrator
from project.recognition.utils.camera_data_manager import add_frame, get_image, get_depth, get_resolution
from project.fitting.app.state import SESSION_MANAGER
from project.fitting.app.api import start_session, push_sample, fit_kappa, compute_gaze
from project.fitting.main import initialize_calibration_session, collect_calibration_sample

# 配置日志
def setup_logging():
    """配置日志系统"""
    log_dir = os.path.join(current_dir, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"integrated_server_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"集成服务器日志文件: {log_file}")
    return log_file

# 设置日志
log_file = setup_logging()

app = Flask(__name__, 
           template_folder='frontend/templates',
           static_folder='frontend/static')
CORS(app)

# 全局状态管理
class IntegratedSystemState:
    def __init__(self):
        # 识别模块状态
        self.is_camera_active = False
        self.face_detector = None
        self.camera_calibrator = None
        self.cap = None
        self.camera_thread = None
        self.camera_running = False
        
        # 拟合模块状态
        self.is_coarse_fitting_done = False
        self.is_calibrating = False
        self.current_point_index = 0
        self.total_points = 6  # 6点校准
        self.collected_samples = []
        self.coarse_fitting_results = {}
        
        # 追踪模块状态（LSTM相关）
        self.lstm_model_loaded = False
        self.gaze_prediction_active = False
        
        # 交互模块状态
        self.roi_regions = []
        self.current_gaze_point = None
        
    def reset_calibration(self):
        """重置校准状态"""
        self.is_calibrating = False
        self.current_point_index = 0
        self.collected_samples = []
        
    def reset_system(self):
        """重置整个系统"""
        self.reset_calibration()
        self.is_coarse_fitting_done = False
        self.lstm_model_loaded = False
        self.gaze_prediction_active = False

system_state = IntegratedSystemState()

def camera_processing_thread():
    """摄像头数据处理线程 - 识别模块核心功能"""
    logger = logging.getLogger(__name__)
    logger.info("摄像头处理线程启动")
    frame_id = 0
    
    try:
        while system_state.camera_running:
            if system_state.cap is None:
                time.sleep(0.1)
                continue
                
            ret, frame = system_state.cap.read()
            if not ret:
                logger.warning("无法读取摄像头数据")
                time.sleep(0.1)
                continue
            
            # 生成深度图（固定深度0.6m）
            h, w = frame.shape[:2]
            depth_scale = system_state.camera_calibrator.get_depth_scale()
            depth_map = np.ones((h, w), dtype=np.float32) * (0.6 / depth_scale)
            
            # 添加帧数据
            add_frame(frame_id, frame, depth_map)
            image = get_image(frame_id)
            depth = get_depth(frame_id)
            
            if image is not None and depth is not None:
                # 检测人脸 - 识别模块核心功能
                detected = system_state.face_detector.detect_face(image, depth)
                
                if detected:
                    system_state.face_detector.update_fitting_data()
                    
                    # 如果还没有完成粗拟合，进行粗拟合 - 拟合模块功能
                    if not system_state.is_coarse_fitting_done:
                        logger.info("执行粗拟合...")
                        try:
                            # 执行眼球形状拟合
                            from project.fitting.main import main as fit_main
                            fit_main()

                            # 初始化校准会话
                            initialize_calibration_session()

                            # 更新状态
                            system_state.is_coarse_fitting_done = True
                            system_state.coarse_fitting_results = {
                                'status': 'success',
                                'message': '眼球形状拟合完成，等待开始校准',
                                'timestamp': time.time()
                            }
                            logger.info("粗拟合完成！")

                        except Exception as e:
                            logger.error(f"粗拟合失败: {e}")
                            system_state.coarse_fitting_results = {
                                'status': 'failed',
                                'message': f'粗拟合失败: {str(e)}',
                                'timestamp': time.time()
                            }
                else:
                    logger.debug(f"人脸检测失败 - 帧ID: {frame_id}")
            
            frame_id += 1
            time.sleep(1/30)  # 30fps
            
    except Exception as e:
        logger.error(f"摄像头处理线程错误: {e}")
    finally:
        logger.info("摄像头处理线程结束")

# ===========================================
# 前端路由
# ===========================================

@app.route('/')
def index():
    """主页面 - 眼动校准界面"""
    return render_template('calibration_integrated.html')

@app.route('/test')
def gaze_test():
    """注视点测试页面"""
    return render_template('gaze_test.html')

@app.route('/debug')
def gaze_debug():
    """实时注视点调试页面"""
    return render_template('gaze_debug_live.html')

# ===========================================
# 识别模块 API
# ===========================================

@app.route('/api/recognition/camera/start', methods=['POST'])
def start_camera():
    """启动摄像头 - 识别模块功能"""
    try:
        if system_state.is_camera_active:
            return jsonify({
                'success': True, 
                'message': '摄像头已激活',
                'status': 'already_active'
            })
        
        logging.info("初始化摄像头...")
        
        # 初始化摄像头
        system_state.camera_calibrator = CameraCalibrator(rgb_d=True)
        system_state.cap = system_state.camera_calibrator.get_cap()
        
        if system_state.cap is None:
            raise Exception("无法初始化摄像头")
        
        # 加载相机参数并初始化检测器
        camera_params = system_state.camera_calibrator.load_camera_params()
        system_state.face_detector = FaceDetector(camera_params, rgb_d=True)
        
        # 启动摄像头处理线程
        system_state.camera_running = True
        system_state.camera_thread = threading.Thread(target=camera_processing_thread, daemon=True)
        system_state.camera_thread.start()
        
        system_state.is_camera_active = True
        
        return jsonify({
            'success': True,
            'message': '摄像头启动成功',
            'status': 'started'
        })
        
    except Exception as e:
        logging.error(f"启动摄像头失败: {e}")
        return jsonify({
            'success': False,
            'message': f'启动摄像头失败: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/api/recognition/camera/stop', methods=['POST'])
def stop_camera():
    """停止摄像头"""
    try:
        if not system_state.is_camera_active:
            return jsonify({
                'success': True,
                'message': '摄像头未激活',
                'status': 'not_active'
            })
        
        # 停止摄像头处理线程
        system_state.camera_running = False
        if system_state.camera_thread:
            system_state.camera_thread.join(timeout=2)
        
        # 释放摄像头资源
        if system_state.cap:
            system_state.cap.release()
            system_state.cap = None
        
        system_state.is_camera_active = False
        
        return jsonify({
            'success': True,
            'message': '摄像头已停止',
            'status': 'stopped'
        })
        
    except Exception as e:
        logging.error(f"停止摄像头失败: {e}")
        return jsonify({
            'success': False,
            'message': f'停止摄像头失败: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/api/recognition/gaze/current', methods=['GET'])
def get_current_gaze():
    """获取当前注视点坐标 - 识别模块核心功能"""
    try:
        if not system_state.is_camera_active or not system_state.face_detector:
            return jsonify({
                'success': False,
                'message': '摄像头未激活或检测器未初始化',
                'gaze': {
                    'gaze_x': 960,
                    'gaze_y': 540,
                    'confidence': 0.0
                },
                'accuracy': 'poor'
            })
        
        # 获取最新的面部检测数据
        data_manager = system_state.face_detector.get_data_manager()
        left_pupil_data = data_manager.get_coordinate_point("left", "pupil")

        if not left_pupil_data or len(left_pupil_data) == 0:
            return jsonify({
                'success': False,
                'message': '未检测到瞳孔数据',
                'gaze': {
                    'gaze_x': 960,
                    'gaze_y': 540,
                    'confidence': 0.0
                },
                'accuracy': 'poor'
            })

        # 获取屏幕分辨率
        screen_width = 1920
        screen_height = 1080

        # 获取最新的瞳孔数据点进行平滑处理
        recent_pupils = left_pupil_data[-min(5, len(left_pupil_data)):]
        if not recent_pupils:
            return jsonify({
                'success': False,
                'message': '瞳孔数据为空',
                'gaze': {
                    'gaze_x': screen_width // 2,
                    'gaze_y': screen_height // 2,
                    'confidence': 0.0
                },
                'accuracy': 'poor'
            })

        # 平滑处理：使用最近几个点的加权平均
        weights = [0.4, 0.3, 0.2, 0.1] if len(recent_pupils) >= 4 else [1.0/len(recent_pupils)] * len(recent_pupils)
        avg_pupil_x = sum(float(p[0]) * w for p, w in zip(recent_pupils[-len(weights):], weights[:len(recent_pupils)]))
        avg_pupil_y = sum(float(p[1]) * w for p, w in zip(recent_pupils[-len(weights):], weights[:len(recent_pupils)]))

        latest_pupil = [avg_pupil_x, avg_pupil_y]
        
        # 获取相机分辨率
        try:
            camera_params = system_state.camera_calibrator.load_camera_params()
            camera_width = camera_params.get('image_resolution', {}).get('width', 640)
            camera_height = camera_params.get('image_resolution', {}).get('height', 480)
        except:
            camera_width = 640
            camera_height = 480

        # 改进的注视点计算算法
        pupil_x = float(latest_pupil[0]) if len(latest_pupil) > 0 else 320
        pupil_y = float(latest_pupil[1]) if len(latest_pupil) > 1 else 240

        # 获取粗拟合结果中的眼球中心数据
        eye_center_3d = None
        has_coarse_fitting = False

        try:
            session = SESSION_MANAGER.get_session()
            if session is not None and hasattr(session, 'last_fitting') and "left" in session.last_fitting:
                fitting_result = session.last_fitting["left"]
                if hasattr(fitting_result, 'center'):
                    eye_center_3d = fitting_result.center
                    has_coarse_fitting = True
        except Exception as e:
            logging.error(f"获取粗拟合结果时发生错误: {e}")

        # 使用粗拟合结果进行注视点计算
        if eye_center_3d is not None:
            # 获取瞳孔的3D坐标
            pupil_3d = [pupil_x, pupil_y, float(latest_pupil[2]) if len(latest_pupil) > 2 else 0.0]

            # 计算瞳孔相对于眼球中心的3D偏移向量
            offset_3d = [
                pupil_3d[0] - eye_center_3d[0],
                pupil_3d[1] - eye_center_3d[1],
                pupil_3d[2] - eye_center_3d[2]
            ]

            # 将3D偏移向量投影到屏幕平面
            norm_offset_x = offset_3d[0] / 50.0  # 假设50mm为最大偏移范围
            norm_offset_y = offset_3d[1] / 50.0

            # 限制在合理范围内
            norm_offset_x = max(-1.0, min(1.0, norm_offset_x))
            norm_offset_y = max(-1.0, min(1.0, norm_offset_y))

            # 改进的映射到屏幕坐标算法
            scale_x = screen_width * 0.45
            scale_y = screen_height * 0.35

            # 应用非线性变换以改善边缘区域的精度
            enhanced_offset_x = norm_offset_x * (1.2 - 0.2 * abs(norm_offset_x))
            enhanced_offset_y = norm_offset_y * (1.2 - 0.2 * abs(norm_offset_y))

            gaze_x = screen_width / 2 + enhanced_offset_x * scale_x
            gaze_y = screen_height / 2 - enhanced_offset_y * scale_y  # Y轴反转

        else:
            # 检查瞳孔坐标是否为已处理的相对坐标
            if abs(pupil_x) < 200 and abs(pupil_y) < 200:
                # 相对坐标映射
                max_coord_range = 100.0
                norm_x = max(-1.0, min(1.0, pupil_x / max_coord_range))
                norm_y = max(-1.0, min(1.0, pupil_y / max_coord_range))

                scale_x = screen_width * 0.42
                scale_y = screen_height * 0.32

                enhanced_x = norm_x * (1.3 - 0.3 * abs(norm_x))
                enhanced_y = norm_y * (1.3 - 0.3 * abs(norm_y))

                gaze_x = screen_width/2 + enhanced_x * scale_x
                gaze_y = screen_height/2 - enhanced_y * scale_y
            else:
                # 绝对坐标映射
                norm_x = (pupil_x - camera_width/2) / (camera_width/2)
                norm_y = (pupil_y - camera_height/2) / (camera_height/2)

                smooth_x = norm_x * (1.4 - 0.4 * abs(norm_x))
                smooth_y = norm_y * (1.35 - 0.35 * abs(norm_y))

                gaze_x = screen_width/2 + smooth_x * screen_width * 0.38
                gaze_y = screen_height/2 - smooth_y * screen_height * 0.32
        
        # 确保坐标在屏幕范围内
        gaze_x = max(0, min(screen_width, gaze_x))
        gaze_y = max(0, min(screen_height, gaze_y))
        
        # 置信度计算
        base_confidence = 0.6
        data_confidence = min(0.2, len(left_pupil_data) / 50.0)
        
        if len(recent_pupils) >= 3:
            x_coords = [float(p[0]) for p in recent_pupils]
            y_coords = [float(p[1]) for p in recent_pupils]
            x_std = np.std(x_coords) if len(x_coords) > 1 else 0
            y_std = np.std(y_coords) if len(y_coords) > 1 else 0
            consistency_confidence = max(0, 0.2 - (x_std + y_std) / 200.0)
        else:
            consistency_confidence = 0

        fitting_confidence = 0.1 if has_coarse_fitting else 0
        confidence = base_confidence + data_confidence + consistency_confidence + fitting_confidence
        confidence = min(1.0, confidence)
        
        # 准确度等级判断
        if confidence > 0.85:
            accuracy = 'excellent'
        elif confidence > 0.7:
            accuracy = 'good'
        elif confidence > 0.5:
            accuracy = 'fair'
        else:
            accuracy = 'poor'

        if has_coarse_fitting and accuracy == 'poor':
            accuracy = 'fair'
        
        return jsonify({
            'success': True,
            'gaze': {
                'gaze_x': round(gaze_x, 1),
                'gaze_y': round(gaze_y, 1),
                'confidence': round(confidence, 3)
            },
            'accuracy': accuracy,
            'message': '注视点数据获取成功'
        })
        
    except Exception as e:
        logging.error(f"获取注视点数据失败: {e}")
        return jsonify({
            'success': False,
            'message': f'获取注视点数据失败: {str(e)}',
            'gaze_x': 0,
            'gaze_y': 0,
            'confidence': 0.0,
            'accuracy': 'poor'
        }), 500

# ===========================================
# 拟合模块 API
# ===========================================

@app.route('/api/fitting/calibration/status', methods=['GET'])
def get_calibration_status():
    """获取校准状态 - 拟合模块功能"""
    return jsonify({
        'is_camera_active': system_state.is_camera_active,
        'is_coarse_fitting_done': system_state.is_coarse_fitting_done,
        'is_calibrating': system_state.is_calibrating,
        'current_point_index': system_state.current_point_index,
        'total_points': system_state.total_points,
        'collected_samples': len(system_state.collected_samples),
        'coarse_fitting_results': system_state.coarse_fitting_results
    })

@app.route('/api/fitting/calibration/start', methods=['POST'])
def start_calibration():
    """开始校准流程 - 拟合模块功能"""
    try:
        # 检查前置条件
        if not system_state.is_camera_active:
            return jsonify({
                'success': False,
                'message': '请先启动摄像头',
                'status': 'camera_not_active'
            }), 400
        
        if not system_state.is_coarse_fitting_done:
            return jsonify({
                'success': False,
                'message': '请等待粗拟合完成',
                'status': 'coarse_fitting_pending'
            }), 400
        
        if system_state.is_calibrating:
            return jsonify({
                'success': True,
                'message': '校准已在进行中',
                'status': 'already_calibrating'
            })
        
        # 开始校准
        system_state.reset_calibration()
        system_state.is_calibrating = True
        
        logging.info("开始6点校准流程")
        
        return jsonify({
            'success': True,
            'message': '校准已开始',
            'status': 'started',
            'total_points': system_state.total_points
        })
        
    except Exception as e:
        logging.error(f"开始校准失败: {e}")
        return jsonify({
            'success': False,
            'message': f'开始校准失败: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/api/fitting/calibration/collect', methods=['POST'])
def collect_calibration_point():
    """收集校准点数据 - 拟合模块功能"""
    try:
        if not system_state.is_calibrating:
            return jsonify({
                'success': False,
                'message': '校准未开始',
                'status': 'not_calibrating'
            }), 400
        
        data = request.get_json()
        target_pixel = data.get('target_pixel', [0, 0])
        point_index = data.get('point_index', system_state.current_point_index)
        
        # 检查是否有瞳孔数据
        if system_state.face_detector:
            pupil_data = system_state.face_detector.get_data_manager().get_coordinate_point("left", "pupil")
            if not pupil_data:
                return jsonify({
                    'success': False,
                    'message': '未检测到瞳孔数据，请确保人脸在摄像头视野内',
                    'status': 'no_pupil_data'
                }), 400
        
        # 收集校准样本
        success = collect_calibration_sample(target_pixel, "left")
        
        if success:
            system_state.collected_samples.append({
                'point_index': point_index,
                'target_pixel': target_pixel,
                'timestamp': time.time()
            })
            
            system_state.current_point_index += 1
            
            logging.info(f"已收集第 {point_index + 1} 个校准点: {target_pixel}")
            
            # 检查是否完成所有点
            is_complete = system_state.current_point_index >= system_state.total_points
            
            return jsonify({
                'success': True,
                'message': f'第 {point_index + 1} 个校准点收集成功',
                'status': 'collected',
                'current_point_index': system_state.current_point_index,
                'total_points': system_state.total_points,
                'is_complete': is_complete
            })
        else:
            return jsonify({
                'success': False,
                'message': '校准点数据收集失败',
                'status': 'collection_failed'
            }), 500
            
    except Exception as e:
        logging.error(f"收集校准点失败: {e}")
        return jsonify({
            'success': False,
            'message': f'收集校准点失败: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/api/fitting/calibration/complete', methods=['POST'])
def complete_calibration():
    """完成校准 - 拟合模块功能"""
    try:
        if system_state.current_point_index < system_state.total_points:
            return jsonify({
                'success': False,
                'message': f'校准未完成，还需收集 {system_state.total_points - system_state.current_point_index} 个点',
                'status': 'incomplete'
            }), 400
        
        # 重置校准状态
        system_state.is_calibrating = False
        
        logging.info("6点校准数据收集完成")
        
        return jsonify({
            'success': True,
            'message': '校准数据收集完成，已准备好进行kappa校准',
            'status': 'completed',
            'collected_samples': len(system_state.collected_samples)
        })
        
    except Exception as e:
        logging.error(f"完成校准失败: {e}")
        return jsonify({
            'success': False,
            'message': f'完成校准失败: {str(e)}',
            'status': 'error'
        }), 500

# ===========================================
# 追踪模块 API (LSTM相关)
# ===========================================

@app.route('/api/tracking/lstm/status', methods=['GET'])
def get_lstm_status():
    """获取LSTM模型状态 - 追踪模块功能"""
    return jsonify({
        'model_loaded': system_state.lstm_model_loaded,
        'prediction_active': system_state.gaze_prediction_active,
        'message': 'LSTM模型状态'
    })

@app.route('/api/tracking/lstm/load', methods=['POST'])
def load_lstm_model():
    """加载LSTM模型 - 追踪模块功能"""
    try:
        # 这里应该实现LSTM模型加载逻辑
        # 目前只是模拟
        system_state.lstm_model_loaded = True
        system_state.gaze_prediction_active = True
        
        return jsonify({
            'success': True,
            'message': 'LSTM模型加载成功',
            'status': 'loaded'
        })
        
    except Exception as e:
        logging.error(f"加载LSTM模型失败: {e}")
        return jsonify({
            'success': False,
            'message': f'加载LSTM模型失败: {str(e)}',
            'status': 'error'
        }), 500

# ===========================================
# 交互模块 API
# ===========================================

@app.route('/api/interaction/roi/regions', methods=['GET'])
def get_roi_regions():
    """获取ROI区域 - 交互模块功能"""
    return jsonify({
        'success': True,
        'regions': system_state.roi_regions,
        'message': 'ROI区域列表'
    })

@app.route('/api/interaction/roi/add', methods=['POST'])
def add_roi_region():
    """添加ROI区域 - 交互模块功能"""
    try:
        data = request.get_json()
        region = {
            'id': len(system_state.roi_regions) + 1,
            'name': data.get('name', f'ROI_{len(system_state.roi_regions) + 1}'),
            'center': data.get('center', [0, 0]),
            'radius': data.get('radius', 50),
            'priority': data.get('priority', 1),
            'timestamp': time.time()
        }
        
        system_state.roi_regions.append(region)
        
        return jsonify({
            'success': True,
            'message': 'ROI区域添加成功',
            'region': region
        })
        
    except Exception as e:
        logging.error(f"添加ROI区域失败: {e}")
        return jsonify({
            'success': False,
            'message': f'添加ROI区域失败: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/api/interaction/gaze/action', methods=['POST'])
def process_gaze_action():
    """处理注视交互 - 交互模块功能"""
    try:
        data = request.get_json()
        gaze_point = data.get('gaze_point', [0, 0])
        
        # 检查注视点是否在ROI区域内
        active_region = None
        for region in system_state.roi_regions:
            center = region['center']
            radius = region['radius']
            distance = ((gaze_point[0] - center[0]) ** 2 + (gaze_point[1] - center[1]) ** 2) ** 0.5
            
            if distance <= radius:
                if active_region is None or region['priority'] > active_region['priority']:
                    active_region = region
        
        system_state.current_gaze_point = gaze_point
        
        return jsonify({
            'success': True,
            'gaze_point': gaze_point,
            'active_region': active_region,
            'message': '注视交互处理完成'
        })
        
    except Exception as e:
        logging.error(f"处理注视交互失败: {e}")
        return jsonify({
            'success': False,
            'message': f'处理注视交互失败: {str(e)}',
            'status': 'error'
        }), 500

# ===========================================
# 系统状态 API
# ===========================================

@app.route('/api/system/status', methods=['GET'])
def get_system_status():
    """获取系统整体状态"""
    return jsonify({
        'recognition': {
            'camera_active': system_state.is_camera_active,
            'face_detector_ready': system_state.face_detector is not None
        },
        'fitting': {
            'coarse_fitting_done': system_state.is_coarse_fitting_done,
            'calibrating': system_state.is_calibrating,
            'collected_points': len(system_state.collected_samples)
        },
        'tracking': {
            'lstm_loaded': system_state.lstm_model_loaded,
            'prediction_active': system_state.gaze_prediction_active
        },
        'interaction': {
            'roi_regions_count': len(system_state.roi_regions),
            'current_gaze_point': system_state.current_gaze_point
        }
    })

@app.route('/api/system/reset', methods=['POST'])
def reset_system():
    """重置整个系统"""
    try:
        system_state.reset_system()
        
        return jsonify({
            'success': True,
            'message': '系统重置成功',
            'status': 'reset'
        })
        
    except Exception as e:
        logging.error(f"重置系统失败: {e}")
        return jsonify({
            'success': False,
            'message': f'重置系统失败: {str(e)}',
            'status': 'error'
        }), 500

def open_browser(url, delay=2):
    """延迟打开浏览器"""
    import webbrowser
    import threading
    
    def open_url():
        time.sleep(delay)
        try:
            webbrowser.open(url)
            print(f"✓ 已自动打开浏览器: {url}")
        except Exception as e:
            print(f"⚠ 无法自动打开浏览器: {e}")
            print(f"请手动访问: {url}")
    
    # 在后台线程中打开浏览器
    browser_thread = threading.Thread(target=open_url, daemon=True)
    browser_thread.start()

if __name__ == '__main__':
    logging.info("启动眼动追踪集成服务器...")
    print("="*80)
    print("眼动追踪集成服务器")
    print("="*80)
    print("前端地址: http://localhost:5000")
    print("API文档: http://localhost:5000/api/system/status")
    print("按 Ctrl+C 退出")
    print("="*80)
    
    # 自动打开浏览器
    open_browser("http://localhost:5000")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        logging.info("服务器已停止")
        # 清理资源
        if system_state.is_camera_active:
            system_state.camera_running = False
            if system_state.cap:
                system_state.cap.release()
