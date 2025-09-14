#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
眼动追踪系统集成启动脚本
按照 structure.md 的流程启动完整的眼动追踪系统
"""

import sys
import os
import argparse
import logging
from datetime import datetime

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
workspace_root = os.path.dirname(project_dir)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

def setup_logging(level=logging.INFO):
    """配置日志系统"""
    log_dir = os.path.join(current_dir, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"integrated_system_{timestamp}.log")
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"集成系统日志文件: {log_file}")
    return log_file

def check_dependencies():
    """检查依赖项"""
    required_modules = [
        'flask',
        'flask_cors',
        'numpy',
        'opencv-python',
        'mediapipe',
        'scipy'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module.replace('-', '_'))
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        logging.error(f"缺少以下依赖模块: {', '.join(missing_modules)}")
        logging.error("请运行: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='眼动追踪系统集成启动器')
    parser.add_argument('--mode', choices=['full', 'recognition', 'fitting', 'tracking', 'interaction'], 
                       default='full', help='运行模式')
    parser.add_argument('--port', type=int, default=5000, help='服务器端口')
    parser.add_argument('--host', default='0.0.0.0', help='服务器主机')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='日志级别')
    parser.add_argument('--no-browser', action='store_true', help='不自动打开浏览器')
    
    args = parser.parse_args()
    
    # 设置日志
    log_level = getattr(logging, args.log_level.upper())
    setup_logging(log_level)
    
    logging.info("="*80)
    logging.info("眼动追踪系统集成启动器")
    logging.info("="*80)
    logging.info(f"运行模式: {args.mode}")
    logging.info(f"服务器地址: {args.host}:{args.port}")
    logging.info(f"调试模式: {args.debug}")
    logging.info(f"日志级别: {args.log_level}")
    logging.info("="*80)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 根据模式启动相应的服务
    if args.mode == 'full':
        # 启动完整的集成服务器
        from integrated_server import app, system_state, open_browser
        logging.info("启动完整集成服务器...")
        
        # 自动打开浏览器（除非用户指定不打开）
        if not args.no_browser:
            url = f"http://{args.host}:{args.port}"
            open_browser(url)
        
        try:
            app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
        except KeyboardInterrupt:
            logging.info("服务器已停止")
            # 清理资源
            if system_state.is_camera_active:
                system_state.camera_running = False
                if system_state.cap:
                    system_state.cap.release()
    
    elif args.mode == 'recognition':
        # 只启动识别模块
        logging.info("启动识别模块...")
        from main import main as recognition_main
        recognition_main()
    
    elif args.mode == 'fitting':
        # 只启动拟合模块
        logging.info("启动拟合模块...")
        from project.fitting.main import main as fitting_main
        fitting_main()
    
    elif args.mode == 'tracking':
        # 启动追踪模块（LSTM）
        logging.info("启动追踪模块...")
        logging.warning("追踪模块（LSTM）功能尚未完全实现")
    
    elif args.mode == 'interaction':
        # 启动交互模块
        logging.info("启动交互模块...")
        logging.warning("交互模块功能尚未完全实现")

if __name__ == '__main__':
    main()
