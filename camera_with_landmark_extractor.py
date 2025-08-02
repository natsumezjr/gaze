#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用landmark_extractor模块的实时摄像头面部特征提取
基于项目中的landmark_extractor.py模块进行面部关键点检测
"""

import sys
import os
import cv2
import numpy as np
import time
import json

# 添加项目路径
project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'project', 'recognition')
sys.path.insert(0, project_path)

def initialize_camera():
    """初始化摄像头"""
    print("📹 正在初始化摄像头...")
    
    # 尝试不同的摄像头索引
    for camera_index in [0, 1, 2]:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            print(f"✅ 成功连接到摄像头 {camera_index}")
            
            # 设置摄像头参数
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            return cap
    
    print("❌ 无法找到可用的摄像头")
    return None

def draw_landmarks_on_image(image, landmarks, indices=None):
    """在图像上绘制关键点"""
    if not landmarks:
        return image
    
    # 绘制所有关键点（绿色小点）
    for i, point in enumerate(landmarks):
        x, y = int(point[0]), int(point[1])
        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
    
    # 如果有索引信息，绘制特殊区域
    if indices:
        try:
            # type: ignore - 动态路径导入
            from core.landmark_extractor import get_landmark_indices
            landmark_indices = get_landmark_indices()
            
            # 绘制眼睛区域
            for eye_type in ['left', 'right']:
                if eye_type == 'left':
                    eye_contour_indices = landmark_indices['left_eye_contour']
                    eye_iris_indices = landmark_indices['left_iris']
                    color = (255, 0, 0)  # 蓝色
                else:
                    eye_contour_indices = landmark_indices['right_eye_contour']
                    eye_iris_indices = landmark_indices['right_iris']
                    color = (0, 0, 255)  # 红色
                
                # 绘制眼睛轮廓
                for idx in eye_contour_indices:
                    if idx < len(landmarks):
                        x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
                        cv2.circle(image, (x, y), 3, color, -1)
                
                # 绘制虹膜
                for idx in eye_iris_indices:
                    if idx < len(landmarks):
                        x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
                        cv2.circle(image, (x, y), 2, (255, 255, 0), -1)  # 黄色
            
            # 绘制鼻子区域
            nose_indices = landmark_indices['nose']
            for idx in nose_indices:
                if idx < len(landmarks):
                    x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
                    cv2.circle(image, (x, y), 2, (0, 255, 255), -1)  # 青色
            
            # 绘制嘴巴区域
            mouth_indices = landmark_indices['mouth_outer']
            for idx in mouth_indices:
                if idx < len(landmarks):
                    x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
                    cv2.circle(image, (x, y), 2, (255, 0, 255), -1)  # 紫色
            
            # 绘制眉毛
            eyebrow_indices = landmark_indices['left_eyebrow'] + landmark_indices['right_eyebrow']
            for idx in eyebrow_indices:
                if idx < len(landmarks):
                    x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # 绿色
                    
        except ImportError as e:
            print(f"⚠️ 无法导入landmark_extractor模块: {e}")
    
    return image

def extract_landmarks_from_image(image):
    """从图像中提取关键点"""
    try:
        # type: ignore - 动态路径导入
        from core.landmark_extractor import extract_landmarks
        
        # 提取关键点
        landmarks = extract_landmarks(image)
        
        if landmarks:
            # 转换为元组格式
            landmarks_tuple = [(point[0], point[1], point[2]) for point in landmarks]
            return landmarks_tuple
        else:
            return None
            
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return None
    except Exception as e:
        print(f"❌ 提取关键点失败: {e}")
        return None

def analyze_landmarks(landmarks):
    """分析关键点数据"""
    if not landmarks:
        return {}
    
    try:
        # type: ignore - 动态路径导入
        from core.landmark_extractor import (
            validate_landmarks,
            get_eye_landmarks,
            get_pupil_landmarks,
            get_iris_landmarks,
            calculate_center
        )
        
        analysis = {}
        
        # 验证关键点质量
        analysis['is_valid'] = validate_landmarks(landmarks)
        
        # 获取眼睛关键点
        left_eye = get_eye_landmarks(landmarks, "left")
        right_eye = get_eye_landmarks(landmarks, "right")
        analysis['left_eye_count'] = len(left_eye)
        analysis['right_eye_count'] = len(right_eye)
        
        # 计算眼睛中心
        if left_eye:
            left_eye_center = calculate_center(left_eye)
            analysis['left_eye_center'] = left_eye_center
        else:
            analysis['left_eye_center'] = None
            
        if right_eye:
            right_eye_center = calculate_center(right_eye)
            analysis['right_eye_center'] = right_eye_center
        else:
            analysis['right_eye_center'] = None
        
        # 获取瞳孔中心
        pupil_centers = get_pupil_landmarks(landmarks)
        analysis['pupil_centers'] = pupil_centers
        
        # 获取虹膜边界
        iris_boundaries = get_iris_landmarks(landmarks)
        analysis['iris_boundaries'] = {
            'left_count': len(iris_boundaries['left']),
            'right_count': len(iris_boundaries['right'])
        }
        
        return analysis
        
    except ImportError as e:
        print(f"❌ 分析模块导入失败: {e}")
        return {}
    except Exception as e:
        print(f"❌ 分析关键点失败: {e}")
        return {}

def display_info(image, landmarks, analysis, fps, frame_count):
    """在图像上显示信息"""
    # 显示帧率和帧数
    cv2.putText(image, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f"Frame: {frame_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 显示关键点数量
    if landmarks:
        cv2.putText(image, f"Landmarks: {len(landmarks)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 显示分析结果
        if analysis:
            y_offset = 120
            if 'is_valid' in analysis:
                status = "VALID" if analysis['is_valid'] else "INVALID"
                color = (0, 255, 0) if analysis['is_valid'] else (0, 0, 255)
                cv2.putText(image, f"Quality: {status}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                y_offset += 30
            
            if 'left_eye_count' in analysis and 'right_eye_count' in analysis:
                cv2.putText(image, f"Left Eye: {analysis['left_eye_count']} pts", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                y_offset += 25
                cv2.putText(image, f"Right Eye: {analysis['right_eye_count']} pts", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                y_offset += 25
            
            if 'iris_boundaries' in analysis:
                cv2.putText(image, f"Iris L/R: {analysis['iris_boundaries']['left_count']}/{analysis['iris_boundaries']['right_count']}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    else:
        cv2.putText(image, "Landmarks: 0", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(image, "Status: NO FACE", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # 显示使用说明
    cv2.putText(image, "Press 'q' to quit", (10, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(image, "Press 's' to save frame", (10, image.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(image, "Press 'a' to analyze", (10, image.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image

def save_frame_with_analysis(image, landmarks, analysis, frame_count):
    """保存带有关键点和分析的帧"""
    try:
        # 创建保存目录
        save_dir = "saved_frames"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 保存图像
        filename = f"{save_dir}/frame_{frame_count:04d}.jpg"
        cv2.imwrite(filename, image)
        
        # 保存关键点数据
        if landmarks:
            landmark_filename = f"{save_dir}/landmarks_{frame_count:04d}.txt"
            with open(landmark_filename, 'w') as f:
                f.write(f"Frame {frame_count}\n")
                f.write(f"Total landmarks: {len(landmarks)}\n")
                f.write("Format: x, y, z\n")
                for i, point in enumerate(landmarks):
                    f.write(f"{i}: {point[0]:.2f}, {point[1]:.2f}, {point[2]:.3f}\n")
        
        # 保存分析结果
        if analysis:
            analysis_filename = f"{save_dir}/analysis_{frame_count:04d}.json"
            with open(analysis_filename, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
        
        print(f"✅ 已保存帧 {frame_count} 到 {filename}")
        
    except Exception as e:
        print(f"❌ 保存失败: {e}")

def run_camera_detection():
    """运行摄像头检测"""
    print("🚀 启动实时面部特征点检测（使用landmark_extractor模块）")
    print("=" * 60)
    
    # 初始化摄像头
    cap = initialize_camera()
    if cap is None:
        return
    
    print("\n📋 使用说明:")
    print("- 按 'q' 键退出程序")
    print("- 按 's' 键保存当前帧和分析数据")
    print("- 按 'a' 键显示详细分析信息")
    print("- 按 'h' 键显示帮助信息")
    
    # 初始化变量
    frame_count = 0
    start_time = time.time()
    show_analysis = False
    
    try:
        while True:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print("❌ 无法读取摄像头帧")
                break
            
            frame_count += 1
            
            # 计算FPS
            current_time = time.time()
            fps = frame_count / (current_time - start_time)
            
            # 提取关键点
            landmarks = extract_landmarks_from_image(frame)
            
            # 分析关键点
            analysis = {}
            if landmarks:
                analysis = analyze_landmarks(landmarks)
            
            # 绘制关键点
            if landmarks:
                frame = draw_landmarks_on_image(frame, landmarks, indices=True)
            
            # 显示信息
            frame = display_info(frame, landmarks, analysis, fps, frame_count)
            
            # 显示图像
            cv2.imshow('Face Landmark Detection (landmark_extractor)', frame)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("👋 用户退出程序")
                break
            elif key == ord('s'):
                save_frame_with_analysis(frame, landmarks, analysis, frame_count)
            elif key == ord('a'):
                show_analysis = not show_analysis
                if show_analysis and analysis:
                    print("\n📊 详细分析信息:")
                    print(f"  数据质量: {'✅ 有效' if analysis.get('is_valid') else '❌ 无效'}")
                    print(f"  左眼关键点: {analysis.get('left_eye_count', 0)} 个")
                    print(f"  右眼关键点: {analysis.get('right_eye_count', 0)} 个")
                    print(f"  左眼虹膜: {analysis.get('iris_boundaries', {}).get('left_count', 0)} 个点")
                    print(f"  右眼虹膜: {analysis.get('iris_boundaries', {}).get('right_count', 0)} 个点")
                    
                    if analysis.get('pupil_centers'):
                        left_pupil = analysis['pupil_centers'].get('left')
                        right_pupil = analysis['pupil_centers'].get('right')
                        if left_pupil:
                            print(f"  左眼瞳孔中心: ({left_pupil[0]:.1f}, {left_pupil[1]:.1f})")
                        if right_pupil:
                            print(f"  右眼瞳孔中心: ({right_pupil[0]:.1f}, {right_pupil[1]:.1f})")
                else:
                    print("📊 分析信息已隐藏")
            elif key == ord('h'):
                print("\n📋 帮助信息:")
                print("- q: 退出程序")
                print("- s: 保存当前帧和分析数据")
                print("- a: 切换详细分析信息显示")
                print("- h: 显示帮助")
            
    except KeyboardInterrupt:
        print("\n👋 程序被用户中断")
    except Exception as e:
        print(f"❌ 程序运行错误: {e}")
    finally:
        # 清理资源
        cap.release()
        cv2.destroyAllWindows()
        print("✅ 摄像头已释放")

def test_landmark_extractor():
    """测试landmark_extractor模块"""
    print("🔍 测试landmark_extractor模块...")
    
    try:
        # type: ignore - 动态路径导入
        from core.landmark_extractor import (
            get_landmark_indices,
            validate_landmarks,
            get_eye_landmarks,
            get_pupil_landmarks,
            get_iris_landmarks,
            calculate_center
        )
        
        # 测试获取索引
        indices = get_landmark_indices()
        print(f"✅ 成功获取 {len(indices)} 个关键点分类")
        
        # 测试计算中心点
        test_points = [(1.0, 2.0, 0.1), (3.0, 4.0, 0.2), (5.0, 6.0, 0.3)]
        center = calculate_center(test_points)
        print(f"✅ 中心点计算: {center}")
        
        print("✅ landmark_extractor模块测试通过")
        return True
        
    except ImportError as e:
        print(f"❌ landmark_extractor模块导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ landmark_extractor模块测试失败: {e}")
        return False

def check_dependencies():
    """检查依赖"""
    print("🔍 检查依赖...")
    
    # 检查OpenCV
    try:
        import cv2
        print(f"✅ OpenCV版本: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV未安装，请运行: pip install opencv-python")
        return False
    
    # 检查MediaPipe
    try:
        import mediapipe as mp
        print("✅ MediaPipe已安装")
    except ImportError:
        print("❌ MediaPipe未安装，请运行: pip install mediapipe")
        return False
    
    # 检查NumPy
    try:
        import numpy as np
        print("✅ NumPy已安装")
    except ImportError:
        print("❌ NumPy未安装，请运行: pip install numpy")
        return False
    
    return True

def test_camera_connection():
    """测试摄像头连接"""
    print("🔍 测试摄像头连接...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return False
    
    # 读取一帧测试
    ret, frame = cap.read()
    if not ret:
        print("❌ 无法读取摄像头帧")
        cap.release()
        return False
    
    print(f"✅ 摄像头测试成功，图像尺寸: {frame.shape}")
    cap.release()
    return True

if __name__ == "__main__":
    print("🎯 基于landmark_extractor的实时面部特征点检测程序")
    print("=" * 60)
    
    # 检查依赖
    if not check_dependencies():
        print("❌ 依赖检查失败，请安装必要的库")
        exit(1)
    
    # 测试landmark_extractor模块
    if not test_landmark_extractor():
        print("❌ landmark_extractor模块测试失败")
        exit(1)
    
    # 测试摄像头
    if not test_camera_connection():
        print("❌ 摄像头测试失败")
        exit(1)
    
    # 运行检测
    run_camera_detection() 