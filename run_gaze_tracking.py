#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
眼动追踪系统快速启动脚本
直接运行 recognition/main.py 实现完整工作流程
"""

import sys
import os
import subprocess
import time
import webbrowser
import threading
import requests
import signal
import atexit
from datetime import datetime

def print_banner():
    """打印启动横幅"""
    print("="*80)
    print("🎯 眼动追踪系统 - 完整工作流程")
    print("="*80)
    print("按照 workflow_summary.md 的流程实现")
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

def check_port_available(port=2233):
    """检查端口是否可用"""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex(('localhost', port))
            return result != 0
    except:
        return True

def kill_port_process(port=2233):
    """终止占用端口的进程"""
    try:
        import subprocess
        # 查找占用端口的进程
        result = subprocess.run(['lsof', '-ti', f':{port}'], 
                              capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    subprocess.run(['kill', '-9', pid], check=False)
                    print(f"🔧 已终止占用端口 {port} 的进程 (PID: {pid})")
    except Exception as e:
        print(f"⚠ 清理端口时出现警告: {e}")

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
            
        print("🌐 浏览器标签页已关闭")
        
    except Exception as e:
        print(f"⚠ 关闭浏览器时出现警告: {e}")

def cleanup_on_exit():
    """程序退出时的清理函数"""
    print("\n🛑 正在停止系统...")
    kill_port_process()
    close_browser_tabs()
    print("✅ 系统已停止")
    print("👋 再见！")

# 注册退出清理函数
atexit.register(cleanup_on_exit)
signal.signal(signal.SIGINT, lambda s, f: cleanup_on_exit() or sys.exit(0))
signal.signal(signal.SIGTERM, lambda s, f: cleanup_on_exit() or sys.exit(0))

def check_dependencies():
    """检查依赖"""
    print("🔍 检查系统依赖...")
    
    required_modules = ['flask', 'flask_cors', 'numpy', 'cv2', 'mediapipe', 'scipy']
    missing = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError:
            missing.append(module)
            print(f"  ✗ {module} (缺失)")
    
    if missing:
        print(f"\n❌ 缺少依赖: {', '.join(missing)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖检查通过")
    return True

def open_browser(url="http://localhost:2233", delay=3):
    """延迟打开浏览器"""
    def open_url():
        time.sleep(delay)
        try:
            webbrowser.open(url)
            print(f"✓ 已自动打开浏览器: {url}")
        except Exception as e:
            print(f"⚠ 无法打开浏览器: {e}")
            print(f"请手动访问: {url}")
    
    # 在后台线程中打开浏览器
    browser_thread = threading.Thread(target=open_url, daemon=True)
    browser_thread.start()

def wait_for_server(port=2233, timeout=30):
    """等待服务器启动"""
    print(f"⏳ 等待服务器启动 (端口 {port})...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://localhost:{port}/api/status", timeout=2)
            if response.status_code == 200:
                print("✅ 服务器启动成功")
                return True
        except:
            pass
        
        time.sleep(1)
        print(".", end="", flush=True)
    
    print(f"\n❌ 服务器启动超时 ({timeout}秒)")
    return False

def main():
    """主函数"""
    print_banner()
    
    # 检查端口是否被占用
    if not check_port_available():
        print(f"⚠ 端口 2233 被占用，正在清理...")
        kill_port_process()
        time.sleep(2)  # 等待端口释放
        
        if not check_port_available():
            print("❌ 端口清理失败，请手动检查端口占用情况")
            sys.exit(1)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 进入项目目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    recognition_dir = os.path.join(current_dir, "project", "recognition")
    
    if not os.path.exists(recognition_dir):
        print("❌ 找不到项目目录")
        sys.exit(1)
    
    # 自动打开浏览器
    open_browser()
    
    print("\n🚀 启动眼动追踪系统...")
    print("📋 工作流程:")
    print("  1️⃣ 系统启动阶段 - 初始化摄像头和人脸检测器")
    print("  2️⃣ 粗拟合阶段 - 眼球形状拟合")
    print("  3️⃣ 校准点收集阶段 - 9点校准网格")
    print("  4️⃣ 精细拟合阶段 - Kappa校准和视线计算")
    print("\n🌐 前端地址: http://localhost:2233")
    print("按 Ctrl+C 停止系统")
    print("-" * 80)
    
    try:
        # 启动主程序
        os.chdir(recognition_dir)
        subprocess.run([sys.executable, "main.py"])
        
    except KeyboardInterrupt:
        print("\n\n🛑 正在停止系统...")
        print("✅ 系统已停止")
        print("👋 再见！")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
