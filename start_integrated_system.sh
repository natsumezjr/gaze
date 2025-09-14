#!/bin/bash
# 眼动追踪集成系统启动脚本

echo "=========================================="
echo "眼动追踪集成系统启动器"
echo "=========================================="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python3"
    exit 1
fi

# 检查虚拟环境
if [ -d "gazing" ]; then
    echo "激活虚拟环境..."
    source gazing/bin/activate
fi

# 进入项目目录
cd project/recognition

# 检查依赖
echo "检查依赖..."
python3 -c "
import sys
required_modules = ['flask', 'flask_cors', 'numpy', 'cv2', 'mediapipe', 'scipy']
missing = []
for module in required_modules:
    try:
        __import__(module)
    except ImportError:
        missing.append(module)
if missing:
    print(f'缺少依赖: {missing}')
    print('请运行: pip install -r requirements.txt')
    sys.exit(1)
else:
    print('依赖检查通过')
"

if [ $? -ne 0 ]; then
    exit 1
fi

# 选择启动方式
echo ""
echo "请选择启动方式:"
echo "1) 图形化启动器 (推荐)"
echo "2) 命令行启动"
echo "3) 后台启动"
echo ""
read -p "请输入选择 (1-3): " choice

case $choice in
    1)
        echo "启动图形化启动器..."
        python3 launch_gui.py
        ;;
    2)
        echo "启动眼动追踪集成系统..."
        echo "前端地址: http://localhost:5000"
        echo "按 Ctrl+C 停止系统"
        echo "=========================================="
        python3 run_integrated.py --mode full --host 0.0.0.0 --port 5000
        ;;
    3)
        echo "后台启动系统..."
        nohup python3 run_integrated.py --mode full --host 0.0.0.0 --port 5000 > system.log 2>&1 &
        echo "系统已在后台启动，日志文件: system.log"
        echo "前端地址: http://localhost:5000"
        echo "停止系统: pkill -f run_integrated.py"
        ;;
    *)
        echo "无效选择，使用默认方式启动..."
        python3 run_integrated.py --mode full --host 0.0.0.0 --port 5000
        ;;
esac
