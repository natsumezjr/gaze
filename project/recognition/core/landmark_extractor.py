import numpy as np
import mediapipe as mp
import cv2
from typing import List, Dict, Tuple, Optional, Union
try:
    from ..config.constants import *
    from ..config.settings import *
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    from config.constants import *
    from config.settings import *

def extract_landmarks(bgr_image: np.ndarray) -> List[List[float]]:
    """
    输入：一张包含人脸的BGR格式numpy数组图像
    处理：使用MediaPipe Face Mesh模型分析图片
    输出：478个精确的2D/3D关键点，格式为[[x, y, z], ...]

    :param bgr_image: BGR格式的numpy数组，形状为(H, W, 3)
    :return: 478个关键点的嵌套列表
    """
    # 初始化MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,  # 启用478个关键点
        min_detection_confidence=0.5
    )
    
    # 将BGR转换为RGB（MediaPipe需要RGB格式）
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    
    # 检测人脸关键点
    results = face_mesh.process(rgb_image)
    
    # 释放资源
    face_mesh.close()
    
    # 如果没有检测到人脸，返回空列表
    if not results.multi_face_landmarks:
        return []
    
    # 获取第一个检测到的人脸关键点
    face_landmarks = results.multi_face_landmarks[0]
    
    # 获取图像尺寸
    height, width = bgr_image.shape[:2]
    
    # 提取478个关键点坐标
    landmarks = []
    for landmark in face_landmarks.landmark:
        # 将相对坐标转换为像素坐标
        x = landmark.x * width
        y = landmark.y * height
        z = landmark.z  # z坐标保持相对值
        
        landmarks.append([x, y, z])
    
    return landmarks

def get_landmark_indices() -> Dict[str, List[int]]:
    """
    返回MediaPipe Face Mesh 478个关键点的索引映射
    
    :return: 关键点索引字典
    """
    return {
        # 眼睛轮廓关键点
        "right_eye_contour": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
        "left_eye_contour": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
        
        # 虹膜关键点（478个关键点模式）
        "right_iris": [468, 469, 470, 471, 472],
        "left_iris": [473, 474, 475, 476, 477],
        
        # 瞳孔中心关键点
        "right_pupil": [468],
        "left_pupil": [473],
        
        # 其他面部特征
        "nose": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
        "mouth_outer": [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
        "mouth_inner": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],
        "right_eyebrow": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
        "left_eyebrow": [336, 296, 334, 293, 300, 276, 283, 282, 295, 285],
        "right_cheek": [132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323],
        "left_cheek": [361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132],
        "chin": [152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 297, 338, 10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148],
        "forehead": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
        "right_ear": [234, 127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93],
        "left_ear": [93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 297, 338, 10, 109, 67, 103, 54, 21, 162, 127, 234]
    }

def validate_landmarks(landmarks: List[List[float]]) -> bool:
    """
    验证关键点数据的有效性
    
    :param landmarks: 关键点列表
    :return: 是否有效
    """
    if not landmarks or len(landmarks) != 478:
        return False
    
    # 转换为numpy数组便于计算
    landmarks_array = np.array(landmarks)
    
    # 检查1: 确保所有坐标都是有效的数值
    if not np.all(np.isfinite(landmarks_array)):
        return False
    
    # 检查2: 检查坐标范围是否合理
    x_coords = landmarks_array[:, 0]
    y_coords = landmarks_array[:, 1]
    z_coords = landmarks_array[:, 2]
    
    # X和Y坐标应该在合理范围内
    if np.any(x_coords < -1000) or np.any(x_coords > 10000) or \
       np.any(y_coords < -1000) or np.any(y_coords > 10000):
        return False
    
    # Z坐标应该在合理范围内
    if np.any(z_coords < -10) or np.any(z_coords > 10):
        return False
    
    # 检查3: 检查关键点之间的距离是否合理
    distances = []
    for i in range(len(landmarks) - 1):
        dist = np.linalg.norm(landmarks_array[i+1] - landmarks_array[i])
        distances.append(dist)
    
    distances = np.array(distances)
    
    # 如果距离过小或过大，认为质量差
    if np.any(distances < 0.1) or np.any(distances > 1000):
        return False
    
    # 检查4: 检查关键点的分布是否合理
    x_std = np.std(x_coords)
    y_std = np.std(y_coords)
    
    if x_std < 1.0 or y_std < 1.0:
        return False
    
    # 检查5: 检查是否有重复的关键点
    unique_points = set()
    for point in landmarks:
        rounded_point = (round(point[0], 2), round(point[1], 2), round(point[2], 2))
        if rounded_point in unique_points:
            return False
        unique_points.add(rounded_point)
    
    return True

def get_pupil_centers_landmarks(landmarks: List[List[float]]) -> Dict[str, Optional[List[float]]]:
    """
    提取左右瞳孔中心关键点
    
    :param landmarks: 关键点列表
    :return: 左右瞳孔中心字典
    """
    if not landmarks or len(landmarks) != 478:
        return {'left': None, 'right': None}
    
    # MediaPipe Face Mesh中瞳孔中心的索引
    LEFT_PUPIL_INDEX = 473  # 左瞳孔中心
    RIGHT_PUPIL_INDEX = 468  # 右瞳孔中心
    
    left_pupil = landmarks[LEFT_PUPIL_INDEX] if LEFT_PUPIL_INDEX < len(landmarks) else None
    right_pupil = landmarks[RIGHT_PUPIL_INDEX] if RIGHT_PUPIL_INDEX < len(landmarks) else None
    
    return {
        'left': left_pupil,
        'right': right_pupil
    }

def get_eye_contours_landmarks(landmarks: List[List[float]]) -> Dict[str, List[List[float]]]:
    """
    提取左右眼睑关键点
    
    :param landmarks: 关键点列表
    :return: 左右眼睑关键点字典
    """
    if not landmarks or len(landmarks) != 478:
        return {'left': [], 'right': []}
    
    # 获取关键点索引
    indices = get_landmark_indices()
    
    # 提取左眼睑关键点
    left_eyelid_indices = indices["left_eye_contour"]
    left_eyelid_landmarks = [landmarks[i] for i in left_eyelid_indices if i < len(landmarks)]
    
    # 提取右眼睑关键点
    right_eyelid_indices = indices["right_eye_contour"]
    right_eyelid_landmarks = [landmarks[i] for i in right_eyelid_indices if i < len(landmarks)]
    
    return {
        'left': left_eyelid_landmarks,
        'right': right_eyelid_landmarks
    }

def get_iris_boundaries_landmarks(landmarks: List[List[float]]) -> Dict[str, List[List[float]]]:
    """
    获取左右虹膜边界关键点
    
    :param landmarks: 关键点列表
    :return: 左右虹膜边界关键点字典
    """
    if not landmarks or len(landmarks) != 478:
        return {'left': [], 'right': []}
    
    # 获取关键点索引
    indices = get_landmark_indices()
    
    # 提取左眼虹膜关键点
    left_iris_indices = indices["left_iris"]
    left_iris_landmarks = [landmarks[i] for i in left_iris_indices if i < len(landmarks)]
    
    # 提取右眼虹膜关键点
    right_iris_indices = indices["right_iris"]
    right_iris_landmarks = [landmarks[i] for i in right_iris_indices if i < len(landmarks)]
    
    return {
        'left': left_iris_landmarks,
        'right': right_iris_landmarks
    }

if __name__ == "__main__":
    print("�� Landmark Extractor 模块测试")
    print("=" * 50)
    
    def initialize_camera():
        """初始化摄像头"""
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            return cap
        return None
    
    def draw_landmarks_on_image(image, landmarks):
        """在图像上绘制关键点"""
        if not landmarks:
            return image
        
        try:
            # 获取瞳孔中心
            pupil_centers = get_pupil_centers_landmarks(landmarks)
            
            # 获取眼睑
            eyelids = get_eye_contours_landmarks(landmarks)
            
            # 获取虹膜边界
            iris_boundaries = get_iris_boundaries_landmarks(landmarks)
            
            # 绘制瞳孔中心点（左右眼不同颜色）
            if pupil_centers['left']:
                x, y = int(pupil_centers['left'][0]), int(pupil_centers['left'][1])
                cv2.circle(image, (x, y), 4, (255, 0, 0), -1)  # 蓝色 - 左瞳孔中心
            
            if pupil_centers['right']:
                x, y = int(pupil_centers['right'][0]), int(pupil_centers['right'][1])
                cv2.circle(image, (x, y), 4, (0, 0, 255), -1)  # 红色 - 右瞳孔中心
            
            # 绘制眼睑点（黄色）
            for point in eyelids['left']:
                x, y = int(point[0]), int(point[1])
                cv2.circle(image, (x, y), 2, (0, 255, 255), -1)  # 黄色 - 左眼睑
            
            for point in eyelids['right']:
                x, y = int(point[0]), int(point[1])
                cv2.circle(image, (x, y), 2, (0, 255, 255), -1)  # 黄色 - 右眼睑
            
            # 绘制虹膜边界点（绿色）
            for point in iris_boundaries['left']:
                x, y = int(point[0]), int(point[1])
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # 绿色 - 左虹膜边界
            
            for point in iris_boundaries['right']:
                x, y = int(point[0]), int(point[1])
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # 绿色 - 右虹膜边界
                    
        except Exception as e:
            print(f"绘制关键点失败: {e}")
        
        return image
    
    def analyze_and_display_results(landmarks, frame_count):
        """分析并显示结果"""
        if not landmarks:
            print(f"帧 {frame_count}: ❌ 未检测到人脸")
            return
        
        try:
            # 验证数据质量
            is_valid = validate_landmarks(landmarks)
            
            # 获取瞳孔中心
            pupil_centers = get_pupil_centers_landmarks(landmarks)
            
            # 获取眼睑
            eyelids = get_eye_contours_landmarks(landmarks)
            
            # 获取虹膜边界
            iris_boundaries = get_iris_boundaries_landmarks(landmarks)
            
            # 计算z值统计信息
            z_values = [point[2] for point in landmarks]
            z_min = min(z_values)
            z_max = max(z_values)
            z_avg = sum(z_values) / len(z_values)
            
            # 显示结果
            status = "✅ 有效" if is_valid else "❌ 无效"
            print(f"帧 {frame_count}: {status} | 关键点: {len(landmarks)} | 左眼睑: {len(eyelids['left'])} | 右眼睑: {len(eyelids['right'])} | 虹膜L/R: {len(iris_boundaries['left'])}/{len(iris_boundaries['right'])}")
            print(f"   Z值范围: {z_min:.3f} ~ {z_max:.3f} (平均: {z_avg:.3f})")
            
            if pupil_centers['left']:
                print(f"   左眼瞳孔中心: ({pupil_centers['left'][0]:.1f}, {pupil_centers['left'][1]:.1f}, z={pupil_centers['left'][2]:.3f})")
            if pupil_centers['right']:
                print(f"   右眼瞳孔中心: ({pupil_centers['right'][0]:.1f}, {pupil_centers['right'][1]:.1f}, z={pupil_centers['right'][2]:.3f})")
                
        except Exception as e:
            print(f"帧 {frame_count}: ❌ 分析失败 - {e}")
    
    def run_real_time_test():
        """运行实时测试"""
        print("🚀 启动实时测试...")
        
        cap = initialize_camera()
        if cap is None:
            print("❌ 无法初始化摄像头")
            return
        
        import time
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 提取关键点
                landmarks = extract_landmarks(frame)
                
                # 分析并显示结果
                analyze_and_display_results(landmarks, frame_count)
                
                # 绘制关键点
                if landmarks:
                    frame = draw_landmarks_on_image(frame, landmarks)
                
                # 显示图像
                cv2.imshow('Landmark Extractor Test', frame)
                
                # 检查退出条件
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # 每10帧显示一次统计
                if frame_count % 10 == 0:
                    elapsed_time = time.time() - start_time
                    fps = frame_count / elapsed_time
                    print(f"📊 统计: 帧数={frame_count}, FPS={fps:.1f}")
                
        except KeyboardInterrupt:
            print("\n👋 用户中断")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # 显示最终统计
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            print(f"\n📊 最终统计:")
            print(f"   处理帧数: {frame_count}")
            print(f"   平均FPS: {fps:.1f}")
            print("✅ 测试完成")
    
    # 运行测试
    try:
        run_real_time_test()
    except Exception as e:
        print(f"❌ 测试错误: {e}")