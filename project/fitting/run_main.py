#!/usr/bin/env python3
"""
运行main.py的脚本，确保正确的Python路径
"""
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 现在可以导入项目模块
from project.fitting.main import main

if __name__ == "__main__":
    main()
