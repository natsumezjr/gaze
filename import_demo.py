#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
landmark_extractor模块导入演示
按照README.md文件中的描述进行导入
"""

print("=== landmark_extractor模块导入演示 ===")
print("根据README.md文件的说明，有以下几种导入方式：\n")

# 方式1：直接导入核心功能（推荐）
print("方式1：直接导入核心功能（推荐）")
print("```python")
print("from recognition.core.landmark_extractor import (")
print("    extract_landmarks,")
print("    get_landmark_indices,")
print("    validate_landmarks,")
print("    get_eye_landmarks,")
print("    get_pupil_landmarks,")
print("    get_iris_landmarks,")
print("    calculate_center")
print(")")
print("```")

# 方式2：导入整个模块
print("\n方式2：导入整个模块")
print("```python")
print("import recognition.core.landmark_extractor as landmark_extractor")
print("```")

# 方式3：从包中导入
print("\n方式3：从包中导入")
print("```python")
print("from recognition.core import landmark_extractor")
print("```")

print("\n=== 使用示例 ===")
print("```python")
print("# 1. 获取关键点索引")
print("indices = get_landmark_indices()")
print("print(f'获取到 {len(indices)} 个关键点分类')")
print("")
print("# 2. 从图像中提取关键点")
print("landmarks = extract_landmarks(bgr_image)")
print("")
print("# 3. 验证关键点数据")
print("is_valid = validate_landmarks(landmarks)")
print("")
print("# 4. 提取眼睛关键点")
print("left_eye = get_eye_landmarks(landmarks, 'left')")
print("right_eye = get_eye_landmarks(landmarks, 'right')")
print("")
print("# 5. 获取瞳孔中心")
print("pupil_centers = get_pupil_landmarks(landmarks)")
print("")
print("# 6. 获取虹膜边界")
print("iris_boundaries = get_iris_landmarks(landmarks)")
print("```")

print("\n=== 安装说明 ===")
print("根据README.md，需要先安装项目包：")
print("```bash")
print("cd project/recognition")
print("pip install -e .")
print("```")

print("\n=== 主要功能 ===")
print("1. extract_landmarks(): 从BGR图像中提取468个关键点")
print("2. get_landmark_indices(): 获取关键点分类索引")
print("3. validate_landmarks(): 验证关键点数据质量")
print("4. get_eye_landmarks(): 提取指定眼睛的关键点")
print("5. get_pupil_landmarks(): 计算瞳孔中心位置")
print("6. get_iris_landmarks(): 提取虹膜边界关键点")
print("7. calculate_center(): 计算多个点的几何中心")

print("\n=== 输入输出格式 ===")
print("输入：")
print("- bgr_image: BGR格式的numpy数组，形状为(H, W, 3)")
print("")
print("输出：")
print("- landmarks: 468个关键点的列表，格式为[[x, y, z], ...]")
print("- 坐标系统：像素坐标(x,y) + 相对深度(z)")

print("\n=== 技术特点 ===")
print("1. 基于MediaPipe Face Mesh模型")
print("2. 支持468个精确的3D关键点")
print("3. 实时处理能力（< 100ms）")
print("4. 高精度检测（瞳孔中心精度 ±2像素）")
print("5. 鲁棒性设计（支持不同光照、角度）")

print("\n✅ 演示完成！")
print("请参考README.md文件获取更详细的使用说明。") 