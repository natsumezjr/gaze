#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
眼动追踪系统快速启动脚本
自动启动系统并打开浏览器
"""

import sys
import os
import subprocess
import time
import webbrowser
import threading
import requests
from datetime import datetime

def print_banner():
    """打印启动横幅"""
    print("="*60)
    print("🎯 眼动追踪集成系统 - 快速启动")
    print("="*60)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

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

def start_server():
    """启动服务器"""
    print("\n🚀 启动眼动追踪系统...")
    
    # 进入项目目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    recognition_dir = os.path.join(current_dir, "project", "recognition")
    
    if not os.path.exists(recognition_dir):
        print("❌ 找不到项目目录")
        return None
    
    # 构建启动命令
    cmd = [
        sys.executable, "run_integrated.py", 
        "--mode", "full", 
        "--host", "0.0.0.0", 
        "--port", "5000"
    ]
    
    try:
        # 启动服务器进程
        process = subprocess.Popen(
            cmd,
            cwd=recognition_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        print("⏳ 系统启动中，请稍候...")
        return process
        
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return None

def wait_for_server(port=5000, timeout=30):
    """等待服务器启动"""
    print(f"⏳ 等待服务器启动 (端口 {port})...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://localhost:{port}/api/system/status", timeout=2)
            if response.status_code == 200:
                print("✅ 服务器启动成功")
                return True
        except:
            pass
        
        time.sleep(1)
        print(".", end="", flush=True)
    
    print(f"\n❌ 服务器启动超时 ({timeout}秒)")
    return False

def open_browser(url="http://localhost:5000"):
    """打开浏览器"""
    print(f"\n🌐 打开浏览器: {url}")
    
    try:
        webbrowser.open(url)
        print("✅ 浏览器已打开")
        return True
    except Exception as e:
        print(f"❌ 无法打开浏览器: {e}")
        print(f"请手动访问: {url}")
        return False

def monitor_server(process):
    """监控服务器进程"""
    print("\n📊 系统运行中...")
    print("按 Ctrl+C 停止系统")
    print("-" * 60)
    
    try:
        # 读取服务器输出
        for line in iter(process.stdout.readline, ''):
            if line:
                print(f"[服务器] {line.strip()}")
    except KeyboardInterrupt:
        print("\n\n🛑 正在停止系统...")
        process.terminate()
        try:
            process.wait(timeout=5)
            print("✅ 系统已停止")
        except subprocess.TimeoutExpired:
            process.kill()
            print("✅ 系统已强制停止")

def main():
    """主函数"""
    print_banner()
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 启动服务器
    server_process = start_server()
    if not server_process:
        sys.exit(1)
    
    # 等待服务器启动
    if not wait_for_server():
        print("❌ 服务器启动失败")
        server_process.terminate()
        sys.exit(1)
    
    # 打开浏览器
    open_browser()
    
    # 监控服务器
    try:
        monitor_server(server_process)
    except KeyboardInterrupt:
        print("\n👋 再见！")
    finally:
        if server_process.poll() is None:
            server_process.terminate()

if __name__ == '__main__':
    main()
