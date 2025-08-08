#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
识别模块安装配置文件
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gaze_recognition",
    version="1.0.0",
    author="Gaze Team",
    author_email="gaze@example.com",
    description="眼动追踪系统识别模块",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gaze/recognition",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "opencv-python>=4.5.0",
        "mediapipe>=0.10.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pillow>=8.0.0",
        "matplotlib>=3.3.0",
        "pandas>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
