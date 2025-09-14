#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试浏览器自动关闭功能
"""

import subprocess
import platform
import time

def close_browser_tabs():
    """关闭浏览器标签页"""
    try:
        system = platform.system()
        
        if system == "Darwin":  # macOS
            print("🍎 检测到 macOS 系统")
            
            # 关闭包含 localhost:2233 的标签页
            print("🔍 正在关闭 Safari 中的相关标签页...")
            result = subprocess.run([
                "osascript", "-e", 
                'tell application "Safari" to close (every tab whose URL contains "localhost:2233")'
            ], check=False, capture_output=True, text=True)
            print(f"Safari 结果: {result.returncode}")
            
            print("🔍 正在关闭 Chrome 中的相关标签页...")
            result = subprocess.run([
                "osascript", "-e", 
                'tell application "Google Chrome" to close (every tab whose URL contains "localhost:2233")'
            ], check=False, capture_output=True, text=True)
            print(f"Chrome 结果: {result.returncode}")
            
            print("🔍 正在关闭 Firefox 中的相关标签页...")
            result = subprocess.run([
                "osascript", "-e", 
                'tell application "Firefox" to close (every tab whose URL contains "localhost:2233")'
            ], check=False, capture_output=True, text=True)
            print(f"Firefox 结果: {result.returncode}")
            
        elif system == "Windows":
            print("🪟 检测到 Windows 系统")
            subprocess.run([
                "taskkill", "/f", "/im", "chrome.exe"
            ], check=False, capture_output=True)
            
        elif system == "Linux":
            print("🐧 检测到 Linux 系统")
            subprocess.run([
                "pkill", "-f", "chrome.*localhost:2233"
            ], check=False, capture_output=True)
            
        print("✅ 浏览器标签页关闭完成")
        
    except Exception as e:
        print(f"❌ 关闭浏览器时出现错误: {e}")

if __name__ == "__main__":
    print("🧪 测试浏览器自动关闭功能")
    print("="*50)
    
    # 先打开一个测试页面
    print("🌐 正在打开测试页面...")
    import webbrowser
    webbrowser.open("http://localhost:2233")
    
    print("⏳ 等待 3 秒...")
    time.sleep(3)
    
    # 测试关闭功能
    print("🔄 测试关闭浏览器标签页...")
    close_browser_tabs()
    
    print("✅ 测试完成！")
