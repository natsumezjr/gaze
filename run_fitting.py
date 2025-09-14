
#!/usr/bin/env python"""
运行fitting模块的启动脚本
直接运行: python run_fitting.py
"""
import sys
import os

# 将当前目录添加到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 导入并运行main
from project.fitting.main import main

if __name__ == "__main__":
    main()