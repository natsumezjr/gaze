#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
眼动追踪系统主程序
实现完整的眼动追踪校准流程，包含前端界面
按照 workflow_summary.md 的流程组织
"""

from __future__ import annotations

import sys
import os
import cv2
import numpy as np
import logging
import time
import threading
import webbrowser
import signal
import atexit
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

# 配置日志
def setup_logging():
    """配置日志系统 - 所有输出都记录到日志文件"""
    # 获取项目根目录的logs文件夹
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    workspace_root = os.path.dirname(project_dir)
    log_dir = os.path.join(workspace_root, "logs")
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"gaze_tracking_{timestamp}.log")
    
    # 创建自定义日志格式
    class ColoredFormatter(logging.Formatter):
        def format(self, record):
            # 添加颜色代码到日志级别
            if record.levelno == logging.INFO:
                record.levelname = f"\033[32m{record.levelname}\033[0m"  # 绿色
            elif record.levelno == logging.WARNING:
                record.levelname = f"\033[33m{record.levelname}\033[0m"  # 黄色
            elif record.levelno == logging.ERROR:
                record.levelname = f"\033[31m{record.levelname}\033[0m"  # 红色
            elif record.levelno == logging.DEBUG:
                record.levelname = f"\033[36m{record.levelname}\033[0m"  # 青色
            return super().format(record)
    
    # 文件处理器（无颜色）
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # 控制台处理器（有颜色）
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )
    
    logging.info(f"眼动追踪系统日志: {log_file}")
    logging.info("="*80)
    logging.info("🎯 眼动追踪系统启动")
    logging.info("="*80)
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
    from project.fitting.app.state import SESSION_MANAGER
    from project.fitting.main import main as fit_main, run_kappa_calibration, initialize_calibration_session, collect_calibration_sample, compute_calibrated_gaze
except ImportError as e:
    logging.error(f"导入错误: {e}")
    logging.error("请确保已安装项目包: pip install -e .")
    sys.exit(1)

# 创建Flask应用
app = Flask(__name__, 
           template_folder=os.path.join(current_dir, 'frontend', 'templates'),
           static_folder=os.path.join(current_dir, 'frontend', 'static'))
CORS(app)

# 全局变量用于端口管理
_port_process = None

def cleanup_port():
    """清理端口占用"""
    global _port_process
    try:
        if _port_process:
            _port_process.terminate()
            _port_process = None
        logging.info("✅ 端口已释放")
    except Exception as e:
        logging.warning(f"⚠ 端口清理时出现警告: {e}")

def close_browser_tabs():
    """关闭浏览器标签页"""
    try:
        import subprocess
        import platform
        
        system = platform.system()
        
        if system == "Darwin":  # macOS
            # 关闭包含 localhost:2233 的标签页
            subprocess.run([
                "osascript", "-e", 
                'tell application "Safari" to close (every tab whose URL contains "localhost:2233")'
            ], check=False, capture_output=True)
            
            subprocess.run([
                "osascript", "-e", 
                'tell application "Google Chrome" to close (every tab whose URL contains "localhost:2233")'
            ], check=False, capture_output=True)
            
            subprocess.run([
                "osascript", "-e", 
                'tell application "Firefox" to close (every tab whose URL contains "localhost:2233")'
            ], check=False, capture_output=True)
            
        elif system == "Windows":
            # Windows 下关闭浏览器标签页
            subprocess.run([
                "taskkill", "/f", "/im", "chrome.exe"
            ], check=False, capture_output=True)
            
        elif system == "Linux":
            # Linux 下关闭浏览器进程
            subprocess.run([
                "pkill", "-f", "chrome.*localhost:2233"
            ], check=False, capture_output=True)
            
        logging.info("🌐 浏览器标签页已关闭")
        
    except Exception as e:
        logging.warning(f"⚠ 关闭浏览器时出现警告: {e}")

def signal_handler(signum, frame):
    """信号处理器"""
    logging.info("🛑 正在停止系统...")
    cleanup_port()
    close_browser_tabs()
    sys.exit(0)

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(cleanup_port)
atexit.register(close_browser_tabs)

# 全局状态管理
class GazeTrackingSystem:
    def __init__(self):
        # 系统状态
        self.is_initialized = False
        self.is_camera_active = False
        self.is_coarse_fitting_done = False
        self.is_calibrating = False
        self.is_calibration_complete = False
        
        # 摄像头相关
        self.camera_calibrator = None
        self.face_detector = None
        self.cap = None
        self.camera_thread = None
        self.camera_running = False
        
        # 校准相关
        self.current_point_index = 0
        self.total_points = 9  # 9点校准
        self.target_points = []
        self.collected_samples = []
        
        # 拟合结果
        self.coarse_fitting_results = {
            'status': 'pending',
            'message': '等待初始化...'
        }
        self.final_results = {
            'kappa_angle': None,
            'fit_quality': None,
            'status': 'pending'
        }
        
        # 当前注视点
        self.current_gaze = {
            'gaze_x': 960,
            'gaze_y': 540,
            'confidence': 0.0
        }
        
    def reset(self):
        """重置系统状态"""
        self.is_calibrating = False
        self.is_calibration_complete = False
        self.current_point_index = 0
        self.collected_samples = []
        self.coarse_fitting_results = {
            'status': 'pending',
            'message': '等待初始化...'
        }
        self.final_results = {
            'kappa_angle': None,
            'fit_quality': None,
            'status': 'pending'
        }

# 创建系统实例
gaze_system = GazeTrackingSystem()

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

def camera_processing_thread():
    """摄像头数据处理线程 - 核心工作流程"""
    logging.info("摄像头处理线程启动")
    frame_id = 0
    
    try:
        while gaze_system.camera_running:
            if gaze_system.cap is None:
                time.sleep(0.1)
                continue
                
            ret, frame = gaze_system.cap.read()
            if not ret:
                logging.warning("无法读取摄像头数据")
                time.sleep(0.1)
                continue
            
            # 生成深度图（固定深度0.6m）
            h, w = frame.shape[:2]
            depth_scale = gaze_system.camera_calibrator.get_depth_scale()
            depth_map = np.ones((h, w), dtype=np.float32) * (0.6 / depth_scale)
            
            # 添加帧数据
            add_frame(frame_id, frame, depth_map)
            image = get_image(frame_id)
            depth = get_depth(frame_id)
            
            if image is not None and depth is not None:
                # 人脸检测
                detected = gaze_system.face_detector.detect_face(image, depth)
                
                if detected:
                    gaze_system.face_detector.update_fitting_data()
                    
                    # 第一步：粗拟合（检测到人脸后立即进行）
                    if not gaze_system.is_coarse_fitting_done:
                        logging.info("="*80)
                        logging.info("== 第一步：粗拟合（眼球形状拟合） ==")
                        logging.info("="*80)
                        
                        try:
                            # 执行眼球形状拟合
                            logging.info("开始执行眼球形状拟合...")
                            fit_main()
                            
                            # 初始化校准会话
                            initialize_calibration_session()
                            
                            # 更新状态
                            gaze_system.is_coarse_fitting_done = True
                            gaze_system.coarse_fitting_results = {
                                'status': 'success',
                                'message': '眼球形状拟合完成，等待开始校准',
                                'timestamp': time.time()
                            }
                            
                            logging.info("眼球形状拟合完成！")
                            logging.info("请访问 http://localhost:2233 开始校准")
                            
                        except Exception as e:
                            logging.error(f"眼球形状拟合失败: {e}")
                            gaze_system.coarse_fitting_results = {
                                'status': 'failed',
                                'message': f'眼球形状拟合失败: {str(e)}',
                                'timestamp': time.time()
                            }
                    
                    # 实时注视点计算
                    if gaze_system.is_coarse_fitting_done:
                        try:
                            # 获取瞳孔数据
                            data_manager = gaze_system.face_detector.get_data_manager()
                            left_pupil_data = data_manager.get_coordinate_point("left", "pupil")
                            
                            if left_pupil_data and len(left_pupil_data) > 0:
                                # 计算注视点
                                latest_pupil = left_pupil_data[-1]
                                pupil_x = float(latest_pupil[0])
                                pupil_y = float(latest_pupil[1])
                                
                                # 简单的注视点映射
                                screen_width = 1920
                                screen_height = 1080
                                camera_width = 640
                                camera_height = 480
                                
                                # 归一化坐标
                                norm_x = (pupil_x - camera_width/2) / (camera_width/2)
                                norm_y = (pupil_y - camera_height/2) / (camera_height/2)
                                
                                # 映射到屏幕坐标
                                gaze_x = screen_width/2 + norm_x * screen_width * 0.4
                                gaze_y = screen_height/2 - norm_y * screen_height * 0.3
                                
                                # 确保在屏幕范围内
                                gaze_x = max(0, min(screen_width, gaze_x))
                                gaze_y = max(0, min(screen_height, gaze_y))
                                
                                # 更新当前注视点
                                gaze_system.current_gaze = {
                                    'gaze_x': gaze_x,
                                    'gaze_y': gaze_y,
                                    'confidence': 0.8
                                }
                                
                        except Exception as e:
                            logging.debug(f"注视点计算失败: {e}")
                
                else:
                    logging.debug(f"人脸检测失败 - 帧ID: {frame_id}")
            
            frame_id += 1
            time.sleep(1/30)  # 30fps
            
    except Exception as e:
        logging.error(f"摄像头处理线程错误: {e}")
    finally:
        logging.info("摄像头处理线程结束")

# ===========================================
# Flask 路由
# ===========================================

@app.route('/')
def index():
    """主页面"""
    return render_template('calibration_integrated.html')

@app.route('/api/status', methods=['GET'])
def get_system_status():
    """获取系统状态"""
    return jsonify({
        'is_initialized': gaze_system.is_initialized,
        'is_camera_active': gaze_system.is_camera_active,
        'is_coarse_fitting_done': gaze_system.is_coarse_fitting_done,
        'is_calibrating': gaze_system.is_calibrating,
        'is_calibration_complete': gaze_system.is_calibration_complete,
        'current_point_index': gaze_system.current_point_index,
        'total_points': gaze_system.total_points,
        'coarse_fitting_results': gaze_system.coarse_fitting_results,
        'final_results': gaze_system.final_results
    })

@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    """启动摄像头"""
    try:
        if gaze_system.is_camera_active:
            return jsonify({
                'success': True,
                'message': '摄像头已激活',
                'status': 'already_active'
            })
        
        logging.info("初始化摄像头...")
        
        # 初始化摄像头
        gaze_system.camera_calibrator = CameraCalibrator(rgb_d=True)
        gaze_system.cap = gaze_system.camera_calibrator.get_cap()
        
        if gaze_system.cap is None:
            raise Exception("无法初始化摄像头")
        
        # 加载相机参数并初始化检测器
        camera_params = gaze_system.camera_calibrator.load_camera_params()
        gaze_system.face_detector = FaceDetector(camera_params, rgb_d=True)
        
        # 启动摄像头处理线程
        gaze_system.camera_running = True
        gaze_system.camera_thread = threading.Thread(target=camera_processing_thread, daemon=True)
        gaze_system.camera_thread.start()
        
        gaze_system.is_camera_active = True
        gaze_system.is_initialized = True
        
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

@app.route('/api/calibration/start', methods=['POST'])
def start_calibration():
    """开始校准流程"""
    try:
        if not gaze_system.is_camera_active:
            return jsonify({
                'success': False,
                'message': '请先启动摄像头',
                'status': 'camera_not_active'
            }), 400
        
        if not gaze_system.is_coarse_fitting_done:
            return jsonify({
                'success': False,
                'message': '请等待粗拟合完成',
                'status': 'coarse_fitting_pending'
            }), 400
        
        if gaze_system.is_calibrating:
            return jsonify({
                'success': True,
                'message': '校准已在进行中',
                'status': 'already_calibrating'
            })
        
        # 开始校准
        gaze_system.reset()
        gaze_system.is_calibrating = True
        
        # 生成9点校准网格
        res = get_resolution()
        gaze_system.target_points = nine_point_grid(res)
        
        logging.info("开始9点校准流程")
        print("开始9点校准流程")
        
        return jsonify({
            'success': True,
            'message': '校准已开始',
            'status': 'started',
            'total_points': gaze_system.total_points,
            'target_points': gaze_system.target_points
        })
        
    except Exception as e:
        logging.error(f"开始校准失败: {e}")
        return jsonify({
            'success': False,
            'message': f'开始校准失败: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/api/calibration/collect', methods=['POST'])
def collect_calibration_point():
    """收集校准点数据"""
    try:
        if not gaze_system.is_calibrating:
            return jsonify({
                'success': False,
                'message': '校准未开始',
                'status': 'not_calibrating'
            }), 400
        
        data = request.get_json()
        target_pixel = data.get('target_pixel', [0, 0])
        point_index = data.get('point_index', gaze_system.current_point_index)
        
        # 检查是否有瞳孔数据
        if gaze_system.face_detector:
            pupil_data = gaze_system.face_detector.get_data_manager().get_coordinate_point("left", "pupil")
            if not pupil_data:
                return jsonify({
                    'success': False,
                    'message': '未检测到瞳孔数据，请确保人脸在摄像头视野内',
                    'status': 'no_pupil_data'
                }), 400
        
        # 收集校准样本
        success = collect_calibration_sample(target_pixel, "left")
        
        if success:
            gaze_system.collected_samples.append({
                'point_index': point_index,
                'target_pixel': target_pixel,
                'timestamp': time.time()
            })
            
            gaze_system.current_point_index += 1
            
            logging.info(f"已收集第 {point_index + 1} 个校准点: {target_pixel}")
            
            # 检查是否完成所有点
            is_complete = gaze_system.current_point_index >= gaze_system.total_points
            
            if is_complete:
                # 开始精细拟合
                logging.info("== 第四步：精细拟合（9点校准数据） ==")
                
                try:
                    # 重新进行眼球形状拟合
                    fit_main()
                    
                    # Kappa校准
                    kappa_pro, info_pro, fit_quality = run_kappa_calibration()
                    
                    if kappa_pro is not None:
                        gaze_system.final_results = {
                            'kappa_angle': float(kappa_pro),
                            'fit_quality': fit_quality,
                            'status': 'success'
                        }
                        
                        logging.info("精细拟合完成！校准流程结束！")
                        logging.info(f"Kappa角度: {kappa_pro:.2f}°")
                        logging.info(f"拟合质量: {fit_quality}")
                        
                        gaze_system.is_calibration_complete = True
                        gaze_system.is_calibrating = False
                        
                    else:
                        gaze_system.final_results = {
                            'status': 'failed',
                            'message': 'Kappa校准失败'
                        }
                        logging.error("精细拟合失败")
                        
                except Exception as e:
                    logging.error(f"精细拟合失败: {e}")
                    gaze_system.final_results = {
                        'status': 'failed',
                        'message': f'精细拟合失败: {str(e)}'
                    }
            
            return jsonify({
                'success': True,
                'message': f'第 {point_index + 1} 个校准点收集成功',
                'status': 'collected',
                'current_point_index': gaze_system.current_point_index,
                'total_points': gaze_system.total_points,
                'is_complete': is_complete,
                'final_results': gaze_system.final_results if is_complete else None
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

@app.route('/api/gaze/current', methods=['GET'])
def get_current_gaze():
    """获取当前注视点坐标"""
    return jsonify({
        'success': True,
        'gaze': gaze_system.current_gaze,
        'message': '注视点数据获取成功'
    })

@app.route('/api/gaze/accuracy', methods=['POST'])
def check_gaze_accuracy():
    """检查注视准确度"""
    try:
        data = request.get_json()
        target_x = data.get('target_x', 0)
        target_y = data.get('target_y', 0)
        
        current_gaze = gaze_system.current_gaze
        current_x = current_gaze['gaze_x']
        current_y = current_gaze['gaze_y']
        
        # 计算距离
        dx = current_x - target_x
        dy = current_y - target_y
        distance = (dx * dx + dy * dy) ** 0.5
        
        # 确定准确度等级
        if distance <= 50:
            level = 'excellent'
        elif distance <= 100:
            level = 'good'
        elif distance <= 200:
            level = 'fair'
        else:
            level = 'poor'
        
        return jsonify({
            'success': True,
            'accuracy': {
                'level': level,
                'distance': round(distance, 2),
                'deviation': {
                    'x': round(dx, 2),
                    'y': round(dy, 2)
                }
            },
            'message': f'准确度检查完成: {level}'
        })
        
    except Exception as e:
        logging.error(f"检查注视准确度失败: {e}")
        return jsonify({
            'success': False,
            'message': f'检查注视准确度失败: {str(e)}',
            'accuracy': None
        }), 500

def open_browser(url="http://localhost:2233", delay=3):
    """延迟打开浏览器并全屏显示"""
    def open_url():
        time.sleep(delay)
        try:
            # 使用JavaScript实现全屏
            fullscreen_url = f"{url}#fullscreen"
            webbrowser.open(fullscreen_url)
            logging.info(f"✓ 已自动打开浏览器并全屏: {url}")
        except Exception as e:
            logging.error(f"⚠ 无法自动打开浏览器: {e}")
            logging.info(f"请手动访问: {url}")
    
    # 在后台线程中打开浏览器
    browser_thread = threading.Thread(target=open_url, daemon=True)
    browser_thread.start()

def main():
    """主函数"""
    logging.info("="*80)
    logging.info("🎯 眼动追踪系统 - 完整工作流程")
    logging.info("="*80)
    logging.info("按照 workflow_summary.md 的流程实现")
    logging.info("="*80)
    
    # 自动打开浏览器
    open_browser()
    
    try:
        # 启动Flask服务器
        logging.info("启动眼动追踪系统...")
        logging.info("前端地址: http://localhost:2233")
        logging.info("按 Ctrl+C 退出")
        logging.info("="*80)
        
        # 使用更安全的服务器启动方式
        app.run(host='0.0.0.0', port=2233, debug=False, threaded=True, use_reloader=False)
        
    except KeyboardInterrupt:
        logging.info("🛑 正在停止系统...")
        
    except Exception as e:
        logging.error(f"❌ 系统运行出错: {e}")
        
    finally:
        # 清理资源
        logging.info("🧹 清理系统资源...")
        
        # 停止摄像头
        if gaze_system.is_camera_active:
            gaze_system.camera_running = False
            if gaze_system.cap:
                gaze_system.cap.release()
                logging.info("📹 摄像头已释放")
        
        # 清理端口
        cleanup_port()
        
        # 关闭浏览器标签页
        close_browser_tabs()
        
        logging.info("✅ 系统已停止")
        logging.info("👋 再见！")

if __name__ == "__main__":
    main()