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
from datetime import datetime

def print_banner():
    """打印启动横幅"""
    print("="*80)
    print("🎯 眼动追踪系统 - 完整工作流程")
    print("="*80)
    print("按照 workflow_summary.md 的流程实现")
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

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
