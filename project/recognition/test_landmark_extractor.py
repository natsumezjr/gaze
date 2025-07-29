#!/usr/bin/env python3
"""
测试人脸关键点提取功能
演示如何使用extract_landmarks函数
"""

import cv2
import numpy as np
import sys
import os

# 添加项目路径到sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.landmark_extractor import extract_landmarks, get_landmark_indices

def test_with_camera():
    """使用摄像头测试关键点提取"""
    print("=== 摄像头测试 ===")
    print("按 'q' 退出，按 's' 保存当前帧并提取关键点")
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头")
            break
            
        # 显示原始图像
        cv2.imshow('Camera Feed - Press q to quit, s to save', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # 提取关键点
            landmarks = extract_landmarks(frame)
            
            if landmarks:
                print(f"✓ 成功检测到人脸，提取了 {len(landmarks)} 个关键点")
                
                # 在图像上绘制关键点
                frame_with_landmarks = frame.copy()
                for i, (x, y, z) in enumerate(landmarks):
                    cv2.circle(frame_with_landmarks, (int(x), int(y)), 2, (0, 255, 0), -1)
                
                # 显示带关键点的图像
                cv2.imshow('Face Landmarks', frame_with_landmarks)
                cv2.waitKey(0)
                
                # 保存结果
                cv2.imwrite('face_landmarks_result.jpg', frame_with_landmarks)
                print("✓ 结果已保存为 'face_landmarks_result.jpg'")
                
                # 显示一些关键点的坐标
                indices = get_landmark_indices()
                print("\n关键点坐标示例：")
                print(f"右眼中心 (索引 159): {landmarks[159]}")
                print(f"左眼中心 (索引 386): {landmarks[386]}")
                print(f"鼻子尖 (索引 4): {landmarks[4]}")
                print(f"嘴巴中心 (索引 13): {landmarks[13]}")
                
            else:
                print("✗ 未检测到人脸")
    
    cap.release()
    cv2.destroyAllWindows()

def test_with_image_file(image_path):
    """使用图像文件测试关键点提取"""
    print(f"=== 图像文件测试: {image_path} ===")
    
    # 读取图像
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"✗ 无法读取图像文件: {image_path}")
        return
    
    print(f"图像尺寸: {frame.shape}")
    
    # 提取关键点
    landmarks = extract_landmarks(frame)
    
    if landmarks:
        print(f"✓ 成功检测到人脸，提取了 {len(landmarks)} 个关键点")
        
        # 在图像上绘制关键点
        frame_with_landmarks = frame.copy()
        for i, (x, y, z) in enumerate(landmarks):
            cv2.circle(frame_with_landmarks, (int(x), int(y)), 2, (0, 255, 0), -1)
        
        # 显示结果
        cv2.imshow('Original Image', frame)
        cv2.imshow('Face Landmarks', frame_with_landmarks)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 保存结果
        output_path = f"landmarks_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, frame_with_landmarks)
        print(f"✓ 结果已保存为 '{output_path}'")
        
        # 分析关键点数据
        analyze_landmarks(landmarks)
        
    else:
        print("✗ 未检测到人脸")

def analyze_landmarks(landmarks):
    """分析关键点数据"""
    print("\n=== 关键点数据分析 ===")
    
    # 转换为numpy数组便于分析
    landmarks_array = np.array(landmarks)
    
    print(f"关键点数量: {len(landmarks)}")
    print(f"坐标范围:")
    print(f"  X: {landmarks_array[:, 0].min():.2f} - {landmarks_array[:, 0].max():.2f}")
    print(f"  Y: {landmarks_array[:, 1].min():.2f} - {landmarks_array[:, 1].max():.2f}")
    print(f"  Z: {landmarks_array[:, 2].min():.2f} - {landmarks_array[:, 2].max():.2f}")
    
    # 显示重要关键点的索引
    indices = get_landmark_indices()
    print("\n重要关键点索引:")
    for name, idx_list in indices.items():
        if len(idx_list) <= 5:  # 只显示短列表
            print(f"  {name}: {idx_list}")
        else:
            print(f"  {name}: {idx_list[0]}-{idx_list[-1]} ({len(idx_list)}个点)")

def create_sample_image():
    """创建一个简单的测试图像（如果没有真实图像）"""
    print("=== 创建测试图像 ===")
    
    # 创建一个简单的测试图像
    img = np.ones((480, 640, 3), dtype=np.uint8) * 128  # 灰色背景
    
    # 添加一个简单的人脸轮廓（非常简化）
    cv2.circle(img, (320, 240), 100, (255, 255, 255), -1)  # 白色圆形作为人脸
    cv2.circle(img, (290, 220), 15, (0, 0, 0), -1)  # 左眼
    cv2.circle(img, (350, 220), 15, (0, 0, 0), -1)  # 右眼
    cv2.ellipse(img, (320, 270), (30, 20), 0, 0, 180, (0, 0, 0), 2)  # 嘴巴
    
    cv2.imwrite('sample_face.jpg', img)
    print("✓ 测试图像已保存为 'sample_face.jpg'")
    return 'sample_face.jpg'

def main():
    """主函数"""
    print("人脸关键点提取测试程序")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        # 如果提供了图像文件路径
        image_path = sys.argv[1]
        test_with_image_file(image_path)
    else:
        # 交互式选择
        print("请选择测试方式:")
        print("1. 使用摄像头测试")
        print("2. 使用图像文件测试")
        print("3. 创建测试图像")
        
        choice = input("请输入选择 (1/2/3): ").strip()
        
        if choice == '1':
            test_with_camera()
        elif choice == '2':
            image_path = input("请输入图像文件路径: ").strip()
            if os.path.exists(image_path):
                test_with_image_file(image_path)
            else:
                print(f"✗ 文件不存在: {image_path}")
        elif choice == '3':
            sample_path = create_sample_image()
            test_with_image_file(sample_path)
        else:
            print("无效选择")

if __name__ == "__main__":
    main() 