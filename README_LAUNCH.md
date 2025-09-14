# 眼动追踪系统启动指南

## 🚀 快速启动

### 方式1: 一键启动 (推荐)
```bash
python quick_start.py
```
- ✅ 自动检查依赖
- ✅ 自动启动系统
- ✅ 自动打开浏览器
- ✅ 实时显示日志

### 方式2: 图形化启动器
```bash
./start_integrated_system.sh
# 选择 "1) 图形化启动器"
```
- 🖥️ 友好的GUI界面
- 📊 实时系统状态监控
- ⚙️ 可视化配置选项
- 📝 实时日志显示

### 方式3: 命令行启动
```bash
./start_integrated_system.sh
# 选择 "2) 命令行启动"
```
- 💻 传统命令行界面
- 🔧 适合开发者使用

### 方式4: 后台启动
```bash
./start_integrated_system.sh
# 选择 "3) 后台启动"
```
- 🔄 后台运行
- 📄 日志保存到文件
- 🖥️ 不占用终端

## 📋 系统要求

### 必需依赖
- Python 3.8+
- OpenCV (cv2)
- MediaPipe
- Flask
- NumPy
- SciPy

### 可选依赖
- tkinter (图形化启动器)
- requests (状态检查)

## 🎯 访问地址

启动成功后，访问以下地址：

- **主界面**: http://localhost:5000
- **测试页面**: http://localhost:5000/test
- **调试页面**: http://localhost:5000/debug
- **API状态**: http://localhost:5000/api/system/status

## ⚙️ 配置选项

### 端口配置
默认端口: 5000
```bash
python run_integrated.py --port 8080
```

### 调试模式
```bash
python run_integrated.py --debug
```

### 不自动打开浏览器
```bash
python run_integrated.py --no-browser
```

## 🔧 故障排除

### 1. 端口被占用
```bash
# 查看端口占用
lsof -i :5000

# 杀死占用进程
kill -9 <PID>
```

### 2. 摄像头权限问题
- macOS: 系统偏好设置 → 安全性与隐私 → 摄像头
- Windows: 设置 → 隐私 → 摄像头
- Linux: 检查用户组权限

### 3. 依赖缺失
```bash
pip install -r requirements.txt
```

### 4. 虚拟环境问题
```bash
# 激活虚拟环境
source gazing/bin/activate

# 或创建新的虚拟环境
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 📊 系统监控

### 检查系统状态
```bash
curl http://localhost:5000/api/system/status
```

### 查看日志
```bash
# 实时日志
tail -f project/recognition/logs/integrated_server_*.log

# 系统日志 (后台模式)
tail -f project/recognition/system.log
```

### 停止系统
```bash
# 前台运行
Ctrl+C

# 后台运行
pkill -f run_integrated.py
```

## 🎮 使用流程

1. **启动系统** - 选择任一启动方式
2. **等待初始化** - 系统自动检测摄像头和人脸
3. **开始校准** - 点击"开始校准"按钮
4. **跟随指示** - 注视屏幕上的校准点
5. **完成校准** - 系统自动完成Kappa角校准
6. **实时追踪** - 享受精确的眼动追踪体验

## 🔄 更新日志

### v1.0.0
- ✅ 集成四大模块 (识别、拟合、追踪、交互)
- ✅ 多种启动方式
- ✅ 自动浏览器打开
- ✅ 图形化启动器
- ✅ 实时状态监控
- ✅ 完善的错误处理

## 💡 使用技巧

1. **首次使用**: 建议使用图形化启动器，可以实时查看系统状态
2. **开发调试**: 使用命令行启动，便于查看详细日志
3. **生产环境**: 使用后台启动，系统稳定运行
4. **快速测试**: 使用一键启动，最快速度体验系统

## 🆘 获取帮助

如果遇到问题，请：

1. 查看日志文件
2. 检查系统状态API
3. 确认所有依赖已安装
4. 检查摄像头权限
5. 尝试重启系统

---

**享受您的眼动追踪体验！** 🎯✨
