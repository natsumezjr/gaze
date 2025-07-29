#!/usr/bin/env python3
"""
测试所有人脸关键点提取相关函数
"""

import cv2
import numpy as np
import sys
import os

# 添加项目路径到sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.landmark_extractor import (
    extract_landmarks, 
    validate_landmarks, 
    get_eye_landmarks, 
    get_pupil_landmarks, 
    get_iris_landmarks,
    get_landmark_indices
)

def test_validate_landmarks():
    """测试关键点验证函数"""
    print("=== 测试 validate_landmarks 函数 ===")
    
    # 测试1: 空列表
    result = validate_landmarks([])
    print(f"空列表测试: {result} (期望: False)")
    
    # 测试2: 长度不对的列表
    result = validate_landmarks([(1, 2, 3), (4, 5, 6)])
    print(f"长度不对测试: {result} (期望: False)")
    
    # 测试3: 有效的关键点（模拟）
    valid_landmarks = []
    for i in range(468):
        x = 100 + i * 0.5  # 合理的x坐标
        y = 100 + i * 0.3  # 合理的y坐标
        z = 0.1 + i * 0.001  # 合理的z坐标
        valid_landmarks.append((x, y, z))
    
    result = validate_landmarks(valid_landmarks)
    print(f"有效关键点测试: {result} (期望: True)")
    
    # 测试4: 包含无效值的关键点
    invalid_landmarks = valid_landmarks.copy()
    invalid_landmarks[0] = (float('inf'), 100, 0.1)  # 无穷大值
    result = validate_landmarks(invalid_landmarks)
    print(f"无效值测试: {result} (期望: False)")

def test_get_eye_landmarks():
    """测试眼部关键点提取函数"""
    print("\n=== 测试 get_eye_landmarks 函数 ===")
    
    # 创建模拟的关键点数据
    landmarks = []
    for i in range(468):
        x = 100 + i * 0.5
        y = 100 + i * 0.3
        z = 0.1 + i * 0.001
        landmarks.append((x, y, z))
    
    # 测试左眼关键点
    left_eye = get_eye_landmarks(landmarks, 'left')
    print(f"左眼关键点数量: {len(left_eye)} (期望: 22)")
    print(f"左眼关键点示例: {left_eye[:3]}")
    
    # 测试右眼关键点
    right_eye = get_eye_landmarks(landmarks, 'right')
    print(f"右眼关键点数量: {len(right_eye)} (期望: 22)")
    print(f"右眼关键点示例: {right_eye[:3]}")
    
    # 测试错误输入
    try:
        result = get_eye_landmarks(landmarks, 'invalid')
        print(f"错误输入测试: {result}")
    except ValueError as e:
        print(f"错误输入测试: 正确抛出异常 - {e}")

def test_get_pupil_landmarks():
    """测试瞳孔中心提取函数"""
    print("\n=== 测试 get_pupil_landmarks 函数 ===")
    
    # 创建模拟的关键点数据
    landmarks = []
    for i in range(468):
        x = 100 + i * 0.5
        y = 100 + i * 0.3
        z = 0.1 + i * 0.001
        landmarks.append((x, y, z))
    
    # 测试瞳孔中心提取
    pupil_landmarks = get_pupil_landmarks(landmarks)
    print(f"瞳孔中心结果: {pupil_landmarks}")
    
    if pupil_landmarks['left']:
        print(f"左瞳孔中心: {pupil_landmarks['left']}")
    if pupil_landmarks['right']:
        print(f"右瞳孔中心: {pupil_landmarks['right']}")
    
    # 测试空输入
    empty_result = get_pupil_landmarks([])
    print(f"空输入测试: {empty_result}")

def test_get_iris_landmarks():
    """测试虹膜边界提取函数"""
    print("\n=== 测试 get_iris_landmarks 函数 ===")
    
    # 创建模拟的关键点数据
    landmarks = []
    for i in range(468):
        x = 100 + i * 0.5
        y = 100 + i * 0.3
        z = 0.1 + i * 0.001
        landmarks.append((x, y, z))
    
    # 测试虹膜边界提取
    iris_landmarks = get_iris_landmarks(landmarks)
    print(f"左眼虹膜关键点数量: {len(iris_landmarks['left'])} (期望: 12)")
    print(f"右眼虹膜关键点数量: {len(iris_landmarks['right'])} (期望: 12)")
    
    print(f"左眼虹膜关键点示例: {iris_landmarks['left'][:3]}")
    print(f"右眼虹膜关键点示例: {iris_landmarks['right'][:3]}")
    
    # 测试空输入
    empty_result = get_iris_landmarks([])
    print(f"空输入测试: {empty_result}")

def test_with_real_image():
    """使用真实图像测试所有函数"""
    print("\n=== 使用真实图像测试 ===")
    
    # 尝试从摄像头获取图像
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("无法从摄像头获取图像，跳过真实图像测试")
        return
    
    print("从摄像头获取图像成功")
    
    # 提取关键点
    landmarks = extract_landmarks(frame)
    
    if landmarks:
        print(f"成功提取 {len(landmarks)} 个关键点")
        
        # 测试验证函数
        is_valid = validate_landmarks(landmarks)
        print(f"关键点验证结果: {is_valid}")
        
        if is_valid:
            # 测试眼部关键点提取
            left_eye = get_eye_landmarks(landmarks, 'left')
            right_eye = get_eye_landmarks(landmarks, 'right')
            print(f"左眼关键点数量: {len(left_eye)}")
            print(f"右眼关键点数量: {len(right_eye)}")
            
            # 测试瞳孔中心提取
            pupil_centers = get_pupil_landmarks(landmarks)
            print(f"瞳孔中心: {pupil_centers}")
            
            # 测试虹膜边界提取
            iris_boundaries = get_iris_landmarks(landmarks)
            print(f"左眼虹膜关键点数量: {len(iris_boundaries['left'])}")
            print(f"右眼虹膜关键点数量: {len(iris_boundaries['right'])}")
            
            # 在图像上绘制结果
            frame_with_landmarks = frame.copy()
            
            # 绘制所有关键点
            for x, y, z in landmarks:
                cv2.circle(frame_with_landmarks, (int(x), int(y)), 1, (0, 255, 0), -1)
            
            # 绘制眼部关键点（用不同颜色）
            for x, y, z in left_eye:
                cv2.circle(frame_with_landmarks, (int(x), int(y)), 3, (255, 0, 0), -1)
            for x, y, z in right_eye:
                cv2.circle(frame_with_landmarks, (int(x), int(y)), 3, (0, 0, 255), -1)
            
            # 绘制瞳孔中心
            if pupil_centers['left']:
                x, y, z = pupil_centers['left']
                cv2.circle(frame_with_landmarks, (int(x), int(y)), 5, (255, 255, 0), -1)
            if pupil_centers['right']:
                x, y, z = pupil_centers['right']
                cv2.circle(frame_with_landmarks, (int(x), int(y)), 5, (255, 255, 0), -1)
            
            # 显示结果
            cv2.imshow('All Landmarks', frame_with_landmarks)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # 保存结果
            cv2.imwrite('all_landmarks_result.jpg', frame_with_landmarks)
            print("结果已保存为 'all_landmarks_result.jpg'")
        else:
            print("关键点验证失败，跳过后续测试")
    else:
        print("未检测到人脸")

def test_landmark_indices():
    """测试关键点索引函数"""
    print("\n=== 测试 get_landmark_indices 函数 ===")
    
    indices = get_landmark_indices()
    print(f"关键点索引数量: {len(indices)}")
    
    # 显示一些重要的索引
    important_parts = ['right_eye_contour', 'left_eye_contour', 'right_iris', 'left_iris', 'nose']
    for part in important_parts:
        if part in indices:
            print(f"{part}: {indices[part]}")

def main():
    """主测试函数"""
    print("人脸关键点提取函数完整测试")
    print("=" * 50)
    
    # 测试各个函数
    test_validate_landmarks()
    test_get_eye_landmarks()
    test_get_pupil_landmarks()
    test_get_iris_landmarks()
    test_landmark_indices()
    
    # 询问是否进行真实图像测试
    choice = input("\n是否进行真实图像测试？(y/n): ").strip().lower()
    if choice == 'y':
        test_with_real_image()
    
    print("\n所有测试完成！")

if __name__ == "__main__":
    main() 