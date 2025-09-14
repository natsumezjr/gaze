#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
眼动校准前后端集成服务器
提供REST API接口连接前端6点校准界面和后端摄像头数据
"""

import sys
import os
import json
import time
import threading
import logging
import numpy as np
import random
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
workspace_root = os.path.dirname(project_dir)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

# 导入后端模块
from project.recognition.main import main as recognition_main, get_coarse_fitting_results
from project.fitting.app.state import SESSION_MANAGER
from project.fitting.app.api import start_session, push_sample, fit_kappa, compute_gaze
from project.fitting.main import initialize_calibration_session, collect_calibration_sample
from project.recognition.core.detector import FaceDetector
from project.recognition.utils.camera_calibration import CameraCalibrator
from project.recognition.utils.camera_data_manager import add_frame, get_image, get_depth, get_resolution

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局状态管理
class CalibrationState:
    def __init__(self):
        self.is_camera_active = False
        self.is_coarse_fitting_done = False
        self.is_calibrating = False
        self.current_point_index = 0
        self.total_points = 6  # 6点校准
        self.collected_samples = []
        self.coarse_fitting_results = {}
        self.face_detector = None
        self.camera_calibrator = None
        self.cap = None
        self.camera_thread = None
        self.camera_running = False
        
    def reset(self):
        """重置校准状态"""
        self.is_calibrating = False
        self.current_point_index = 0
        self.collected_samples = []
        
calibration_state = CalibrationState()

def camera_processing_thread():
    """摄像头数据处理线程"""
    logger.info("摄像头处理线程启动")
    frame_id = 0
    
    try:
        while calibration_state.camera_running:
            if calibration_state.cap is None:
                time.sleep(0.1)
                continue
                
            ret, frame = calibration_state.cap.read()
            if not ret:
                logger.warning("无法读取摄像头数据")
                time.sleep(0.1)
                continue
            
            # 生成深度图（固定深度0.6m）
            h, w = frame.shape[:2]
            depth_scale = calibration_state.camera_calibrator.get_depth_scale()
            depth_map = np.ones((h, w), dtype=np.float32) * (0.6 / depth_scale)
            
            # 添加帧数据
            add_frame(frame_id, frame, depth_map)
            image = get_image(frame_id)
            depth = get_depth(frame_id)
            
            if image is not None and depth is not None:
                # 检测人脸
                detected = calibration_state.face_detector.detect_face(image, depth)
                
                if detected:
                    calibration_state.face_detector.update_fitting_data()
                    
                    # 记录检测状态（用于调试）
                    logger.debug(f"人脸检测成功 - 帧ID: {frame_id}")

                    # 如果还没有完成粗拟合，进行粗拟合
                    if not calibration_state.is_coarse_fitting_done:
                        logger.info("执行粗拟合...")
                        try:
                            # 执行眼球形状拟合
                            from project.fitting.main import main as fit_main
                            fit_main()

                            # 初始化校准会话
                            from project.fitting.main import initialize_calibration_session
                            initialize_calibration_session()

                            # 更新状态
                            calibration_state.is_coarse_fitting_done = True
                            calibration_state.coarse_fitting_results = {
                                'status': 'success',
                                'message': '眼球形状拟合完成，等待开始校准',
                                'timestamp': time.time()
                            }
                            logger.info("粗拟合完成！")

                        except Exception as e:
                            logger.error(f"粗拟合失败: {e}")
                            calibration_state.coarse_fitting_results = {
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

@app.route('/')
def index():
    """提供静态HTML文件"""
    return send_from_directory('.', 'calibration_integrated.html')

@app.route('/test')
def gaze_test():
    """提供注视点测试页面"""
    return send_from_directory('.', 'gaze_test.html')

@app.route('/debug')
def gaze_debug():
    """提供实时注视点调试页面"""
    return send_from_directory('.', 'gaze_debug_live.html')

@app.route('/api/debug/fitting', methods=['GET'])
def debug_fitting_status():
    """调试粗拟合状态"""
    try:
        from project.fitting.app.state import SESSION_MANAGER

        session = SESSION_MANAGER.get_session()
        debug_info = {
            'has_session': session is not None,
            'session_id': session.session_id if session else None,
            'has_last_fitting': False,
            'fitting_keys': [],
            'left_eye_center': None,
            'error': None
        }

        if session:
            debug_info['has_last_fitting'] = hasattr(session, 'last_fitting') and len(session.last_fitting) > 0
            if hasattr(session, 'last_fitting'):
                debug_info['fitting_keys'] = list(session.last_fitting.keys())

                if 'left' in session.last_fitting:
                    fitting_result = session.last_fitting['left']
                    if hasattr(fitting_result, 'center'):
                        debug_info['left_eye_center'] = [float(x) for x in fitting_result.center]

        return jsonify({
            'success': True,
            'debug': debug_info
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        })

@app.route('/<path:filename>')
def static_files(filename):
    """提供静态文件"""
    return send_from_directory('.', filename)

@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    """启动摄像头"""
    try:
        if calibration_state.is_camera_active:
            return jsonify({
                'success': True, 
                'message': '摄像头已激活',
                'status': 'already_active'
            })
        
        logger.info("初始化摄像头...")
        
        # 初始化摄像头
        calibration_state.camera_calibrator = CameraCalibrator(rgb_d=True)
        calibration_state.cap = calibration_state.camera_calibrator.get_cap()
        
        if calibration_state.cap is None:
            raise Exception("无法初始化摄像头")
        
        # 加载相机参数并初始化检测器
        camera_params = calibration_state.camera_calibrator.load_camera_params()
        calibration_state.face_detector = FaceDetector(camera_params, rgb_d=True)
        
        # 启动摄像头处理线程
        calibration_state.camera_running = True
        calibration_state.camera_thread = threading.Thread(target=camera_processing_thread, daemon=True)
        calibration_state.camera_thread.start()
        
        calibration_state.is_camera_active = True
        
        return jsonify({
            'success': True,
            'message': '摄像头启动成功',
            'status': 'started'
        })
        
    except Exception as e:
        logger.error(f"启动摄像头失败: {e}")
        return jsonify({
            'success': False,
            'message': f'启动摄像头失败: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    """停止摄像头"""
    try:
        if not calibration_state.is_camera_active:
            return jsonify({
                'success': True,
                'message': '摄像头未激活',
                'status': 'not_active'
            })
        
        # 停止摄像头处理线程
        calibration_state.camera_running = False
        if calibration_state.camera_thread:
            calibration_state.camera_thread.join(timeout=2)
        
        # 释放摄像头资源
        if calibration_state.cap:
            calibration_state.cap.release()
            calibration_state.cap = None
        
        calibration_state.is_camera_active = False
        
        return jsonify({
            'success': True,
            'message': '摄像头已停止',
            'status': 'stopped'
        })
        
    except Exception as e:
        logger.error(f"停止摄像头失败: {e}")
        return jsonify({
            'success': False,
            'message': f'停止摄像头失败: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/api/calibration/status', methods=['GET'])
def get_calibration_status():
    """获取校准状态"""
    return jsonify({
        'is_camera_active': calibration_state.is_camera_active,
        'is_coarse_fitting_done': calibration_state.is_coarse_fitting_done,
        'is_calibrating': calibration_state.is_calibrating,
        'current_point_index': calibration_state.current_point_index,
        'total_points': calibration_state.total_points,
        'collected_samples': len(calibration_state.collected_samples),
        'coarse_fitting_results': calibration_state.coarse_fitting_results
    })

@app.route('/api/calibration/start', methods=['POST'])
def start_calibration():
    """开始校准流程"""
    try:
        # 检查前置条件
        if not calibration_state.is_camera_active:
            return jsonify({
                'success': False,
                'message': '请先启动摄像头',
                'status': 'camera_not_active'
            }), 400
        
        if not calibration_state.is_coarse_fitting_done:
            return jsonify({
                'success': False,
                'message': '请等待粗拟合完成',
                'status': 'coarse_fitting_pending'
            }), 400
        
        if calibration_state.is_calibrating:
            return jsonify({
                'success': True,
                'message': '校准已在进行中',
                'status': 'already_calibrating'
            })
        
        # 开始校准
        calibration_state.reset()
        calibration_state.is_calibrating = True
        
        logger.info("开始6点校准流程")
        
        return jsonify({
            'success': True,
            'message': '校准已开始',
            'status': 'started',
            'total_points': calibration_state.total_points
        })
        
    except Exception as e:
        logger.error(f"开始校准失败: {e}")
        return jsonify({
            'success': False,
            'message': f'开始校准失败: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/api/calibration/collect', methods=['POST'])
def collect_calibration_point():
    """收集校准点数据"""
    try:
        if not calibration_state.is_calibrating:
            return jsonify({
                'success': False,
                'message': '校准未开始',
                'status': 'not_calibrating'
            }), 400
        
        data = request.get_json()
        target_pixel = data.get('target_pixel', [0, 0])
        point_index = data.get('point_index', calibration_state.current_point_index)
        
        # 检查是否有瞳孔数据
        if calibration_state.face_detector:
            pupil_data = calibration_state.face_detector.get_data_manager().get_coordinate_point("left", "pupil")
            if not pupil_data:
                return jsonify({
                    'success': False,
                    'message': '未检测到瞳孔数据，请确保人脸在摄像头视野内',
                    'status': 'no_pupil_data'
                }), 400
        
        # 收集校准样本
        success = collect_calibration_sample(target_pixel, "left")
        
        if success:
            calibration_state.collected_samples.append({
                'point_index': point_index,
                'target_pixel': target_pixel,
                'timestamp': time.time()
            })
            
            calibration_state.current_point_index += 1
            
            logger.info(f"已收集第 {point_index + 1} 个校准点: {target_pixel}")
            
            # 检查是否完成所有点
            is_complete = calibration_state.current_point_index >= calibration_state.total_points
            
            return jsonify({
                'success': True,
                'message': f'第 {point_index + 1} 个校准点收集成功',
                'status': 'collected',
                'current_point_index': calibration_state.current_point_index,
                'total_points': calibration_state.total_points,
                'is_complete': is_complete
            })
        else:
            return jsonify({
                'success': False,
                'message': '校准点数据收集失败',
                'status': 'collection_failed'
            }), 500
            
    except Exception as e:
        logger.error(f"收集校准点失败: {e}")
        return jsonify({
            'success': False,
            'message': f'收集校准点失败: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/api/calibration/complete', methods=['POST'])
def complete_calibration():
    """完成校准（为后续kappa校准做准备）"""
    try:
        if calibration_state.current_point_index < calibration_state.total_points:
            return jsonify({
                'success': False,
                'message': f'校准未完成，还需收集 {calibration_state.total_points - calibration_state.current_point_index} 个点',
                'status': 'incomplete'
            }), 400
        
        # 重置校准状态
        calibration_state.is_calibrating = False
        
        logger.info("6点校准数据收集完成")
        
        return jsonify({
            'success': True,
            'message': '校准数据收集完成，已准备好进行kappa校准',
            'status': 'completed',
            'collected_samples': len(calibration_state.collected_samples)
        })
        
    except Exception as e:
        logger.error(f"完成校准失败: {e}")
        return jsonify({
            'success': False,
            'message': f'完成校准失败: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/api/gaze/current', methods=['GET'])
def get_current_gaze():
    """获取当前注视点坐标"""
    try:
        # 添加详细的状态检查日志
        logger.info(f"注视点API调用 - 摄像头激活: {calibration_state.is_camera_active}, "
                    f"检测器存在: {calibration_state.face_detector is not None}")

        if not calibration_state.is_camera_active:
            logger.warning("摄像头未激活")
            return jsonify({
                'success': False,
                'message': '摄像头未激活',
                'gaze': {
                    'gaze_x': 960,
                    'gaze_y': 540,
                    'confidence': 0.0
                },
                'accuracy': 'poor'
            })

        if not calibration_state.face_detector:
            logger.warning("面部检测器未初始化")
            return jsonify({
                'success': False,
                'message': '面部检测器未初始化',
                'gaze': {
                    'gaze_x': 960,
                    'gaze_y': 540,
                    'confidence': 0.0
                },
                'accuracy': 'poor'
            })
        
        # 获取最新的面部检测数据
        data_manager = calibration_state.face_detector.get_data_manager()
        
        # 获取左眼瞳孔坐标
        left_pupil_data = data_manager.get_coordinate_point("left", "pupil")

        # 检查数据质量和可用性
        if not left_pupil_data or len(left_pupil_data) == 0:
            return jsonify({
                'success': False,
                'message': '未检测到瞳孔数据',
                'gaze': {
                    'gaze_x': 960,  # 屏幕中心
                    'gaze_y': 540,
                    'confidence': 0.0
                },
                'accuracy': 'poor'
            })

        # 获取屏幕分辨率
        screen_width = 1920
        screen_height = 1080

        # 获取最新的多个瞳孔数据点进行平滑处理
        recent_pupils = left_pupil_data[-min(5, len(left_pupil_data)):]  # 取最近5个数据点
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
        
        # 获取相机分辨率（从相机参数中获取）
        try:
            from project.recognition.utils.camera_calibration import CameraCalibrator
            camera_calibrator = CameraCalibrator()
            camera_params = camera_calibrator.load_camera_params()
            camera_width = camera_params.get('image_resolution', {}).get('width', 640)
            camera_height = camera_params.get('image_resolution', {}).get('height', 480)
        except:
            # 默认摄像头分辨率
            camera_width = 640
            camera_height = 480

        # 改进的注视点计算算法
        pupil_x = float(latest_pupil[0]) if len(latest_pupil) > 0 else 320
        pupil_y = float(latest_pupil[1]) if len(latest_pupil) > 1 else 240

        # 获取粗拟合结果中的眼球中心数据
        eye_center_3d = None
        has_coarse_fitting = False

        try:
            # 导入SESSION_MANAGER
            from project.fitting.app.state import SESSION_MANAGER

            # 安全地获取会话信息
            session = SESSION_MANAGER.get_session()
            logger.debug(f"获取到会话: {session is not None}")

            if session is not None:
                logger.debug(f"会话中的last_fitting键: {list(session.last_fitting.keys()) if hasattr(session, 'last_fitting') else '无last_fitting属性'}")

                if hasattr(session, 'last_fitting') and "left" in session.last_fitting:
                    fitting_result = session.last_fitting["left"]
                    logger.debug(f"找到左眼拟合结果: {type(fitting_result)}")

                    # 检查拟合结果是否有center属性
                    if hasattr(fitting_result, 'center'):
                        eye_center_3d = fitting_result.center
                        has_coarse_fitting = True
                        logger.debug(f"成功获取眼球中心: {eye_center_3d}")
                    else:
                        logger.warning(f"拟合结果缺少center属性: {dir(fitting_result)}")
                else:
                    logger.debug("未找到左眼拟合结果")
            else:
                logger.debug("未找到活动会话")

        except ImportError as e:
            logger.error(f"无法导入SESSION_MANAGER: {e}")
        except Exception as e:
            logger.error(f"获取粗拟合结果时发生错误: {e}")
            logger.error(f"错误类型: {type(e).__name__}")
            logger.error(f"错误详情: {str(e)}")

        # 如果没有粗拟合结果，使用备选方案
        if not has_coarse_fitting:
            try:
                logger.debug("使用iris数据作为备选方案")
                iris_data = data_manager.get_coordinate_point("left", "iris")
                if iris_data and len(iris_data) > 0:
                    iris_point = iris_data[-1]
                    eye_center_3d = [float(iris_point[0]), float(iris_point[1]), float(iris_point[2])]
                    logger.debug(f"使用iris中心: {eye_center_3d}")
                else:
                    logger.debug("iris数据也不可用")
            except Exception as e:
                logger.warning(f"获取iris数据失败: {e}")
                eye_center_3d = None

        # 使用粗拟合结果进行注视点计算
        if eye_center_3d is not None:
            # 获取瞳孔的3D坐标 [x, y, z, visibility]
            pupil_3d = [pupil_x, pupil_y, float(latest_pupil[2]) if len(latest_pupil) > 2 else 0.0]

            # 计算瞳孔相对于眼球中心的3D偏移向量（单位：毫米）
            offset_3d = [
                pupil_3d[0] - eye_center_3d[0],
                pupil_3d[1] - eye_center_3d[1],
                pupil_3d[2] - eye_center_3d[2]
            ]

            logger.debug(f"3D偏移向量: [{offset_3d[0]:.2f}, {offset_3d[1]:.2f}, {offset_3d[2]:.2f}] mm")

            # 将3D偏移向量投影到屏幕平面
            # 使用X和Y分量（忽略Z深度信息）
            norm_offset_x = offset_3d[0] / 50.0  # 假设50mm为最大偏移范围
            norm_offset_y = offset_3d[1] / 50.0

            # 限制在合理范围内
            norm_offset_x = max(-1.0, min(1.0, norm_offset_x))
            norm_offset_y = max(-1.0, min(1.0, norm_offset_y))

            # 改进的映射到屏幕坐标算法
            # 使用更精确的映射比例和非线性变换
            scale_x = screen_width * 0.45   # 稍微增加水平敏感度
            scale_y = screen_height * 0.35  # 稍微增加垂直敏感度

            # 应用非线性变换以改善边缘区域的精度
            enhanced_offset_x = norm_offset_x * (1.2 - 0.2 * abs(norm_offset_x))
            enhanced_offset_y = norm_offset_y * (1.2 - 0.2 * abs(norm_offset_y))

            gaze_x = screen_width / 2 + enhanced_offset_x * scale_x
            gaze_y = screen_height / 2 - enhanced_offset_y * scale_y  # Y轴反转

            logger.debug(f"使用粗拟合结果计算注视点: 3D偏移({offset_3d[0]:.1f}, {offset_3d[1]:.1f}, {offset_3d[2]:.1f})mm -> 注视点({gaze_x:.1f}, {gaze_y:.1f})")

        else:
            # 检查瞳孔坐标是否为已处理的相对坐标（通常在-100到100范围内）
            if abs(pupil_x) < 200 and abs(pupil_y) < 200:
                # 这些可能是已经处理过的相对坐标
                logger.debug(f"检测到相对坐标，使用相对映射算法")

                # 直接将相对坐标映射到屏幕坐标
                # 假设瞳孔坐标范围在-100到100之间
                max_coord_range = 100.0
                norm_x = max(-1.0, min(1.0, pupil_x / max_coord_range))
                norm_y = max(-1.0, min(1.0, pupil_y / max_coord_range))

                # 改进的映射敏感度和非线性变换
                scale_x = screen_width * 0.42   # 优化的水平敏感度
                scale_y = screen_height * 0.32  # 优化的垂直敏感度

                # 应用平滑非线性变换改善响应特性
                enhanced_x = norm_x * (1.3 - 0.3 * abs(norm_x))
                enhanced_y = norm_y * (1.3 - 0.3 * abs(norm_y))

                # 映射到屏幕坐标，注意Y轴方向可能需要反转
                gaze_x = screen_width/2 + enhanced_x * scale_x
                gaze_y = screen_height/2 - enhanced_y * scale_y  # Y轴反转

                logger.debug(f"相对坐标映射: 瞳孔({pupil_x:.1f}, {pupil_y:.1f}) -> 归一化({norm_x:.2f}, {norm_y:.2f}) -> 注视点({gaze_x:.1f}, {gaze_y:.1f})")

            else:
                # 传统的绝对坐标映射
                logger.debug(f"使用绝对坐标映射算法")
                norm_x = (pupil_x - camera_width/2) / (camera_width/2)
                norm_y = (pupil_y - camera_height/2) / (camera_height/2)

                # 改进的平滑非线性变换
                smooth_x = norm_x * (1.4 - 0.4 * abs(norm_x))  # 优化中心区域敏感度
                smooth_y = norm_y * (1.35 - 0.35 * abs(norm_y))

                # 改进的映射到屏幕坐标
                gaze_x = screen_width/2 + smooth_x * screen_width * 0.38  # 增加映射范围
                gaze_y = screen_height/2 - smooth_y * screen_height * 0.32  # Y轴反转，增加映射范围

                logger.debug(f"绝对坐标映射: 瞳孔({pupil_x:.1f}, {pupil_y:.1f}) -> 注视点({gaze_x:.1f}, {gaze_y:.1f})")
        
        # 暂时移除随机扰动以便调试
        # gaze_x += random.uniform(-5, 5)
        # gaze_y += random.uniform(-5, 5)
        
        # 确保坐标在屏幕范围内
        gaze_x = max(0, min(screen_width, gaze_x))
        gaze_y = max(0, min(screen_height, gaze_y))
        
        # 改进的置信度计算
        # 基础置信度基于检测稳定性
        base_confidence = 0.6

        # 基于数据量的置信度（更多数据 = 更高置信度）
        data_confidence = min(0.2, len(left_pupil_data) / 50.0)

        # 基于数据一致性的置信度（最近几个点的变化幅度）
        if len(recent_pupils) >= 3:
            x_coords = [float(p[0]) for p in recent_pupils]
            y_coords = [float(p[1]) for p in recent_pupils]
            x_std = np.std(x_coords) if len(x_coords) > 1 else 0
            y_std = np.std(y_coords) if len(y_coords) > 1 else 0

            # 标准差越小，一致性越好，置信度越高
            consistency_confidence = max(0, 0.2 - (x_std + y_std) / 200.0)
        else:
            consistency_confidence = 0

        # 基于是否有粗拟合结果的置信度加成
        fitting_confidence = 0.1 if has_coarse_fitting else 0

        # 总置信度
        confidence = base_confidence + data_confidence + consistency_confidence + fitting_confidence
        confidence = min(1.0, confidence)  # 确保不超过1.0
        
        # 改进的准确度等级判断，考虑更多因素
        if confidence > 0.85:
            accuracy = 'excellent'
        elif confidence > 0.7:
            accuracy = 'good'
        elif confidence > 0.5:
            accuracy = 'fair'
        else:
            accuracy = 'poor'

        # 如果有粗拟合结果，准确度至少为'fair'
        if has_coarse_fitting and accuracy == 'poor':
            accuracy = 'fair'
        
        # 添加详细的调试信息
        debug_info = {
            'raw_pupil_data_count': len(left_pupil_data),
            'latest_pupil_raw': [float(latest_pupil[0]), float(latest_pupil[1])],
            'camera_resolution': f"{camera_width}x{camera_height}",
            'screen_resolution': f"{screen_width}x{screen_height}",
            'pupil_coordinate_range': {
                'min_x': min(float(p[0]) for p in recent_pupils),
                'max_x': max(float(p[0]) for p in recent_pupils),
                'min_y': min(float(p[1]) for p in recent_pupils),
                'max_y': max(float(p[1]) for p in recent_pupils)
            },
            'mapping_algorithm': 'coarse_fitting_3d' if has_coarse_fitting else 'pupil_relative',
            'has_coarse_fitting': has_coarse_fitting,
            'calculated_gaze_before_noise': [round(gaze_x, 1), round(gaze_y, 1)]
        }

        if eye_center_3d is not None:
            debug_info['eyeball_center_3d'] = [round(float(c), 2) for c in eye_center_3d]
            debug_info['pupil_3d_offset'] = {
                'x': round(float(latest_pupil[0]) - eye_center_3d[0], 2),
                'y': round(float(latest_pupil[1]) - eye_center_3d[1], 2),
                'z': round((float(latest_pupil[2]) if len(latest_pupil) > 2 else 0.0) - eye_center_3d[2], 2)
            }
            debug_info['has_coarse_fitting'] = has_coarse_fitting
        else:
            debug_info['eyeball_center_3d'] = None
            debug_info['pupil_3d_offset'] = None

        return jsonify({
            'success': True,
            'gaze': {
                'gaze_x': round(gaze_x, 1),
                'gaze_y': round(gaze_y, 1),
                'confidence': round(confidence, 3)
            },
            'accuracy': accuracy,
            'message': '注视点数据获取成功',
            'debug': debug_info
        })
        
    except Exception as e:
        logger.error(f"获取注视点数据失败: {e}")
        return jsonify({
            'success': False,
            'message': f'获取注视点数据失败: {str(e)}',
            'gaze_x': 0,
            'gaze_y': 0,
            'confidence': 0.0,
            'accuracy': 'poor'
        }), 500

@app.route('/api/gaze/accuracy', methods=['POST'])
def check_gaze_accuracy():
    """检查注视准确度"""
    try:
        if not calibration_state.is_camera_active or not calibration_state.face_detector:
            return jsonify({
                'success': False,
                'message': '摄像头未激活',
                'accuracy': None
            })
        
        data = request.get_json()
        target_x = data.get('target_x', 0)
        target_y = data.get('target_y', 0)
        
        # 获取当前注视点数据
        gaze_response = get_current_gaze()
        if isinstance(gaze_response, tuple):
            # 如果返回的是元组（错误情况），取第一个元素
            gaze_data = gaze_response[0].get_json()
        else:
            # 正常情况，直接获取JSON数据
            gaze_data = gaze_response.get_json()
        
        if not gaze_data.get('success', False):
            return jsonify({
                'success': False,
                'message': '无法获取注视点数据',
                'accuracy': None
            })
        
        gaze_info = gaze_data.get('gaze', {})
        current_gaze_x = gaze_info.get('gaze_x', 0)
        current_gaze_y = gaze_info.get('gaze_y', 0)
        confidence = gaze_info.get('confidence', 0)
        
        # 计算注视点与目标点的距离
        dx = current_gaze_x - target_x
        dy = current_gaze_y - target_y
        distance = (dx * dx + dy * dy) ** 0.5
        
        # 基于距离和置信度确定准确度等级
        # 调整阈值为更宽松的值，便于测试和调试
        excellent_threshold = 150  # 150像素内为excellent
        good_threshold = 300       # 300像素内为good  
        fair_threshold = 500       # 500像素内为fair
        
        # 使用更宽松的条件来触发校准动画
        if distance <= excellent_threshold:
            level = 'excellent'
        elif distance <= good_threshold:
            level = 'good'
        elif distance <= fair_threshold:
            level = 'fair'
        else:
            level = 'poor'
            
        # 如果置信度太低，降级准确度
        if confidence < 0.3 and level != 'poor':
            if level == 'excellent':
                level = 'good'
            elif level == 'good':
                level = 'fair'
        
        accuracy_data = {
            'level': level,
            'distance': round(distance, 2),
            'confidence': confidence,
            'deviation': {
                'x': round(dx, 2),
                'y': round(dy, 2),
                'total': round(distance, 2)
            },
            'target': {
                'x': target_x,
                'y': target_y
            },
            'current': {
                'x': current_gaze_x,
                'y': current_gaze_y
            }
        }
        
        return jsonify({
            'success': True,
            'accuracy': accuracy_data,
            'message': f'准确度检查完成: {level}'
        })
        
    except Exception as e:
        logger.error(f"检查注视准确度失败: {e}")
        return jsonify({
            'success': False,
            'message': f'检查注视准确度失败: {str(e)}',
            'accuracy': None
        }), 500

if __name__ == '__main__':
    logger.info("启动眼动校准集成服务器...")
    print("="*80)
    print("眼动校准集成服务器")
    print("="*80)
    print("前端地址: http://localhost:5000")
    print("API文档: http://localhost:5000/api/calibration/status")
    print("按 Ctrl+C 退出")
    print("="*80)
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        logger.info("服务器已停止")
        # 清理资源
        if calibration_state.is_camera_active:
            calibration_state.camera_running = False
            if calibration_state.cap:
                calibration_state.cap.release()