# 端口和浏览器自动管理功能

## 功能概述

为了提供更好的用户体验，系统现在支持：

1. **端口自动管理** - 程序启动时检查端口占用，结束时自动释放
2. **浏览器自动关闭** - 程序结束时自动关闭相关的浏览器标签页

## 新增功能

### 1. 端口自动管理

#### 启动时检查
- 自动检查端口 2233 是否被占用
- 如果被占用，自动清理占用进程
- 确保系统能够正常启动

#### 结束时清理
- 程序退出时自动释放端口
- 支持多种退出方式：Ctrl+C、正常结束、异常退出
- 使用信号处理器确保资源清理

### 2. 浏览器自动关闭

#### 跨平台支持
- **macOS**: 使用 AppleScript 关闭 Safari、Chrome、Firefox 中的相关标签页
- **Windows**: 使用 taskkill 关闭 Chrome 进程
- **Linux**: 使用 pkill 关闭相关浏览器进程

#### 智能识别
- 只关闭包含 `localhost:2233` 的标签页
- 不影响其他正在使用的浏览器标签页
- 支持多种主流浏览器

## 技术实现

### 信号处理
```python
import signal
import atexit

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # 终止信号
atexit.register(cleanup_port)                  # 程序退出
atexit.register(close_browser_tabs)            # 关闭浏览器
```

### 端口检查
```python
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
```

### 浏览器关闭 (macOS)
```python
def close_browser_tabs():
    """关闭浏览器标签页"""
    # Safari
    subprocess.run([
        "osascript", "-e", 
        'tell application "Safari" to close (every tab whose URL contains "localhost:2233")'
    ], check=False, capture_output=True)
    
    # Chrome
    subprocess.run([
        "osascript", "-e", 
        'tell application "Google Chrome" to close (every tab whose URL contains "localhost:2233")'
    ], check=False, capture_output=True)
    
    # Firefox
    subprocess.run([
        "osascript", "-e", 
        'tell application "Firefox" to close (every tab whose URL contains "localhost:2233")'
    ], check=False, capture_output=True)
```

## 使用方法

### 正常启动
```bash
# 使用启动脚本（推荐）
python run_gaze_tracking.py

# 或直接运行主程序
python project/recognition/main.py
```

### 程序退出
- **正常退出**: 程序完成工作后自动清理
- **手动退出**: 按 `Ctrl+C` 触发清理
- **异常退出**: 系统自动捕获并清理资源

## 日志输出

程序运行时会显示详细的清理信息：

```
🛑 正在停止系统...
✅ 端口已释放
🌐 浏览器标签页已关闭
🧹 清理系统资源...
📹 摄像头已释放
✅ 系统已停止
👋 再见！
```

## 测试功能

可以使用测试脚本验证浏览器关闭功能：

```bash
python test_browser_close.py
```

## 注意事项

1. **权限要求**: macOS 下需要授权 AppleScript 控制浏览器
2. **浏览器支持**: 目前支持 Safari、Chrome、Firefox
3. **安全考虑**: 只关闭包含特定 URL 的标签页，不会影响其他页面
4. **错误处理**: 如果关闭失败会显示警告，但不会影响程序退出

## 故障排除

### 端口仍被占用
```bash
# 手动检查端口占用
lsof -i :2233

# 手动清理
kill -9 <PID>
```

### 浏览器未关闭
- 检查是否有权限控制浏览器
- 确认浏览器名称正确
- 查看日志中的警告信息

## 更新记录

- **2025-09-14**: 添加端口自动管理和浏览器自动关闭功能
- 支持跨平台浏览器关闭
- 改进错误处理和日志输出
- 添加测试脚本和文档
