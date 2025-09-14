#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
眼动追踪系统图形化启动器
提供友好的GUI界面来启动和管理系统
"""

import sys
import os
import subprocess
import threading
import time
import webbrowser
from datetime import datetime

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
workspace_root = os.path.dirname(project_dir)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, scrolledtext
    import requests
except ImportError as e:
    print(f"缺少GUI依赖: {e}")
    print("请安装: pip install tkinter")
    sys.exit(1)

class GazeTrackingLauncher:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("眼动追踪系统启动器")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # 系统状态
        self.server_process = None
        self.server_running = False
        self.server_port = 5000
        
        self.setup_ui()
        self.check_system_status()
        
    def setup_ui(self):
        """设置用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # 标题
        title_label = ttk.Label(main_frame, text="眼动追踪集成系统", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # 系统状态区域
        status_frame = ttk.LabelFrame(main_frame, text="系统状态", padding="10")
        status_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        status_frame.columnconfigure(1, weight=1)
        
        # 服务器状态
        ttk.Label(status_frame, text="服务器状态:").grid(row=0, column=0, sticky=tk.W)
        self.status_label = ttk.Label(status_frame, text="未启动", foreground="red")
        self.status_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # 端口状态
        ttk.Label(status_frame, text="端口:").grid(row=1, column=0, sticky=tk.W)
        self.port_label = ttk.Label(status_frame, text=f"localhost:{self.server_port}")
        self.port_label.grid(row=1, column=1, sticky=tk.W, padx=(10, 0))
        
        # 摄像头状态
        ttk.Label(status_frame, text="摄像头:").grid(row=2, column=0, sticky=tk.W)
        self.camera_label = ttk.Label(status_frame, text="未检测")
        self.camera_label.grid(row=2, column=1, sticky=tk.W, padx=(10, 0))
        
        # 控制按钮区域
        control_frame = ttk.LabelFrame(main_frame, text="系统控制", padding="10")
        control_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 启动按钮
        self.start_button = ttk.Button(control_frame, text="启动系统", 
                                     command=self.start_server, style="Accent.TButton")
        self.start_button.grid(row=0, column=0, padx=(0, 10))
        
        # 停止按钮
        self.stop_button = ttk.Button(control_frame, text="停止系统", 
                                    command=self.stop_server, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=(0, 10))
        
        # 打开浏览器按钮
        self.browser_button = ttk.Button(control_frame, text="打开界面", 
                                       command=self.open_browser, state="disabled")
        self.browser_button.grid(row=0, column=2, padx=(0, 10))
        
        # 重启按钮
        self.restart_button = ttk.Button(control_frame, text="重启系统", 
                                       command=self.restart_server, state="disabled")
        self.restart_button.grid(row=0, column=3)
        
        # 配置区域
        config_frame = ttk.LabelFrame(main_frame, text="配置选项", padding="10")
        config_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        config_frame.columnconfigure(1, weight=1)
        
        # 端口配置
        ttk.Label(config_frame, text="端口:").grid(row=0, column=0, sticky=tk.W)
        self.port_var = tk.StringVar(value=str(self.server_port))
        port_entry = ttk.Entry(config_frame, textvariable=self.port_var, width=10)
        port_entry.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # 调试模式
        self.debug_var = tk.BooleanVar()
        debug_check = ttk.Checkbutton(config_frame, text="调试模式", 
                                     variable=self.debug_var)
        debug_check.grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        
        # 自动打开浏览器
        self.auto_browser_var = tk.BooleanVar(value=True)
        auto_browser_check = ttk.Checkbutton(config_frame, text="自动打开浏览器", 
                                           variable=self.auto_browser_var)
        auto_browser_check.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=(5, 0))
        
        # 日志区域
        log_frame = ttk.LabelFrame(main_frame, text="系统日志", padding="10")
        log_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        # 日志文本框
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, width=80)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 清空日志按钮
        clear_log_button = ttk.Button(log_frame, text="清空日志", 
                                    command=self.clear_log)
        clear_log_button.grid(row=1, column=0, sticky=tk.E, pady=(5, 0))
        
        # 状态栏
        self.status_bar = ttk.Label(main_frame, text="就绪", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # 启动状态检查定时器
        self.status_timer = None
        self.start_status_check()
        
    def log_message(self, message):
        """添加日志消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def clear_log(self):
        """清空日志"""
        self.log_text.delete(1.0, tk.END)
        
    def check_system_status(self):
        """检查系统状态"""
        try:
            response = requests.get(f"http://localhost:{self.server_port}/api/system/status", 
                                  timeout=2)
            if response.status_code == 200:
                self.server_running = True
                self.status_label.config(text="运行中", foreground="green")
                self.start_button.config(state="disabled")
                self.stop_button.config(state="normal")
                self.browser_button.config(state="normal")
                self.restart_button.config(state="normal")
                
                # 更新摄像头状态
                data = response.json()
                if data.get('recognition', {}).get('camera_active', False):
                    self.camera_label.config(text="已激活", foreground="green")
                else:
                    self.camera_label.config(text="未激活", foreground="orange")
            else:
                self.server_running = False
                self.status_label.config(text="未启动", foreground="red")
                self.start_button.config(state="normal")
                self.stop_button.config(state="disabled")
                self.browser_button.config(state="disabled")
                self.restart_button.config(state="disabled")
                self.camera_label.config(text="未检测", foreground="gray")
        except:
            self.server_running = False
            self.status_label.config(text="未启动", foreground="red")
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.browser_button.config(state="disabled")
            self.restart_button.config(state="disabled")
            self.camera_label.config(text="未检测", foreground="gray")
    
    def start_status_check(self):
        """启动状态检查定时器"""
        self.check_system_status()
        self.status_timer = self.root.after(3000, self.start_status_check)  # 每3秒检查一次
    
    def start_server(self):
        """启动服务器"""
        try:
            self.server_port = int(self.port_var.get())
        except ValueError:
            messagebox.showerror("错误", "请输入有效的端口号")
            return
        
        if self.server_running:
            messagebox.showwarning("警告", "服务器已在运行中")
            return
        
        self.log_message("正在启动眼动追踪系统...")
        self.status_bar.config(text="正在启动...")
        
        # 构建启动命令
        cmd = [sys.executable, "run_integrated.py", "--mode", "full", 
               "--host", "0.0.0.0", "--port", str(self.server_port)]
        
        if self.debug_var.get():
            cmd.append("--debug")
        
        if not self.auto_browser_var.get():
            cmd.append("--no-browser")
        
        # 在后台启动服务器
        def run_server():
            try:
                self.server_process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    cwd=current_dir
                )
                
                # 读取输出
                for line in iter(self.server_process.stdout.readline, ''):
                    if line:
                        self.log_message(line.strip())
                
                self.server_process.wait()
                
            except Exception as e:
                self.log_message(f"启动失败: {e}")
                self.status_bar.config(text="启动失败")
        
        # 启动服务器线程
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # 延迟检查启动状态
        self.root.after(2000, self.check_startup)
        
    def check_startup(self):
        """检查启动状态"""
        self.check_system_status()
        if self.server_running:
            self.log_message("✓ 系统启动成功")
            self.status_bar.config(text="系统运行中")
            if self.auto_browser_var.get():
                self.open_browser()
        else:
            self.log_message("⚠ 系统启动中，请稍候...")
            self.root.after(2000, self.check_startup)
    
    def stop_server(self):
        """停止服务器"""
        if not self.server_running:
            messagebox.showwarning("警告", "服务器未运行")
            return
        
        self.log_message("正在停止系统...")
        self.status_bar.config(text="正在停止...")
        
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
                self.log_message("✓ 系统已停止")
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.log_message("✓ 系统已强制停止")
            except Exception as e:
                self.log_message(f"停止失败: {e}")
        
        self.check_system_status()
        self.status_bar.config(text="已停止")
    
    def restart_server(self):
        """重启服务器"""
        if self.server_running:
            self.stop_server()
            self.root.after(1000, self.start_server)
        else:
            self.start_server()
    
    def open_browser(self):
        """打开浏览器"""
        if not self.server_running:
            messagebox.showwarning("警告", "服务器未运行")
            return
        
        url = f"http://localhost:{self.server_port}"
        try:
            webbrowser.open(url)
            self.log_message(f"✓ 已打开浏览器: {url}")
        except Exception as e:
            self.log_message(f"⚠ 无法打开浏览器: {e}")
            messagebox.showerror("错误", f"无法打开浏览器: {e}")
    
    def on_closing(self):
        """关闭窗口时的处理"""
        if self.server_running:
            if messagebox.askokcancel("退出", "系统正在运行，是否要停止并退出？"):
                self.stop_server()
                self.root.destroy()
        else:
            self.root.destroy()
    
    def run(self):
        """运行启动器"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

def main():
    """主函数"""
    print("启动眼动追踪系统图形化启动器...")
    
    # 检查依赖
    try:
        import requests
    except ImportError:
        print("缺少requests库，正在安装...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    
    # 启动GUI
    launcher = GazeTrackingLauncher()
    launcher.run()

if __name__ == '__main__':
    main()
