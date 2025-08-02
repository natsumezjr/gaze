import numpy as np
import mediapipe as mp
import cv2
from typing import List, Dict, Tuple, Optional, Union
from config.constants import *
from config.settings import *

def extract_landmarks(bgr_image: np.ndarray) -> List[List[float]]:
    """
    输入：一张包含人脸的BGR格式numpy数组图像
    处理：使用MediaPipe Face Mesh模型分析图片
    输出：468个精确的2D/3D关键点，格式为[[x, y, z], ...]

    :param bgr_image: BGR格式的numpy数组，形状为(H, W, 3)
    示例：
    bgr_image = np.ndarray(shape=(720, 1280, 3), dtype=np.uint8)  # BGR图像

    :return: 468个关键点的嵌套列表
    """
    # 初始化MediaPipe Face Mesh
    # type: ignore - MediaPipe API可能因版本而异
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(  # type: ignore[reportAttributeAccessIssue]
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,  # 改为False以保持468个关键点
        min_detection_confidence=DETECTION_CONFIDENCE_THRESHOLD
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
    
    # 提取468个关键点坐标
    landmarks = []
    for landmark in face_landmarks.landmark:
        # 将相对坐标转换为像素坐标
        x = landmark.x * width
        y = landmark.y * height
        z = landmark.z  # z坐标保持相对值
        
        landmarks.append([x, y, z])
    
    return landmarks

# 关键点索引说明（MediaPipe Face Mesh的468个关键点）
def get_landmark_indices() -> Dict[str, List[int]]:
    """
    返回重要关键点的索引，后续可以用于直接字符串键值对获取值。
    
    返回：关键点索引字典
    示例
    indices = {
        "right_eye_contour": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],      # 右眼轮廓索引
        "left_eye_contour": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],  # 左眼轮廓索引
        "right_iris": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],  # 右眼虹膜索引
        "left_iris": [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43],   # 左眼虹膜索引
        "nose": [44, 45, 46, ..., 67],                           # 鼻子索引
        "mouth_outer": [68, 69, 70, ..., 83],                    # 嘴巴外轮廓索引
        "mouth_inner": [84, 85, 86, ..., 107],                   # 嘴巴内轮廓索引
        "right_eyebrow": [108, 109, 110, ..., 127],              # 右眉毛索引
        "left_eyebrow": [128, 129, 130, ..., 147],               # 左眉毛索引
        "right_cheek": [148, 149, 150, ..., 167],                # 右脸颊索引
        "left_cheek": [168, 169, 170, ..., 187],                 # 左脸颊索引
        "chin": [188, 189, 190, ..., 207],                       # 下巴索引
        "forehead": [208, 209, 210, ..., 227],                   # 额头索引
        "right_temple": [228, 229, 230, ..., 247],               # 右太阳穴索引
        "left_temple": [248, 249, 250, ..., 267],                # 左太阳穴索引
        "right_ear": [268, 269, 270, ..., 287],                  # 右耳索引
        "left_ear": [288, 289, 290, ..., 307],                   # 左耳索引
        "right_earlobe": [308, 309, 310, ..., 327],              # 右耳垂索引
        "left_earlobe": [328, 329, 330, ..., 347],               # 左耳垂索引
        "right_ear_helix": [348, 349, 350, ..., 367],            # 右耳轮索引
        "left_ear_helix": [368, 369, 370, ..., 387],             # 左耳轮索引
        "right_ear_tragus": [388, 389, 390, ..., 407],           # 右耳屏索引
        "left_ear_tragus": [408, 409, 410, ..., 427],            # 左耳屏索引
        "right_earlobe_bottom": [428, 429, 430, ..., 447],       # 右耳垂底部索引
        "left_earlobe_bottom": [448, 449, 450, ..., 467]         # 左耳垂底部索引
    }
    
    调用关系：
    被调用：
    - get_eye_landmarks() 调用此函数
    - get_pupil_landmarks() 调用此函数
    - get_iris_landmarks() 调用此函数
    
    调用：
    - 无直接调用其他函数
    
    可能用到的库函数：无
    """
    return {
        "right_eye_contour": list(range(0, 10)),
        "left_eye_contour": list(range(10, 20)),
        "right_iris": list(range(20, 32)),
        "left_iris": list(range(32, 44)),
        "nose": list(range(44, 68)),
        "mouth_outer": list(range(68, 84)),
        "mouth_inner": list(range(84, 108)),
        "right_eyebrow": list(range(108, 128)),
        "left_eyebrow": list(range(128, 148)),
        "right_cheek": list(range(148, 168)),
        "left_cheek": list(range(168, 188)),
        "chin": list(range(188, 208)),
        "forehead": list(range(208, 228)),
        "right_temple": list(range(228, 248)),
        "left_temple": list(range(248, 268)),
        "right_ear": list(range(268, 288)),
        "left_ear": list(range(288, 308)),
        "right_earlobe": list(range(308, 328)),
        "left_earlobe": list(range(328, 348)),
        "right_ear_helix": list(range(348, 368)),
        "left_ear_helix": list(range(368, 388)),
        "right_ear_tragus": list(range(388, 408)),
        "left_ear_tragus": list(range(408, 428)),
        "right_earlobe_bottom": list(range(428, 448)),
        "left_earlobe_bottom": list(range(448, 468))
    }

def validate_landmarks(landmarks: List[Tuple[float, float, float]]) -> bool:
    """
    输入：landmarks的
    
    :param landmarks: 关键点列表
    示例
    landmarks = [
        (x1, y1, z1),  # 关键点1坐标 (像素x, 像素y, 相对深度z)
        (x2, y2, z2),  # 关键点2坐标
        ...,
        (x468, y468, z468)  # 关键点468坐标
    ]
    
    调用关系：
    被调用：
    - detector.py 中的 detect_face() 调用此函数
    
    调用：
    - 无直接调用其他函数
    
    可能用到的库函数：numpy
    """
    if not landmarks or len(landmarks) != 468:
        return False
    
    # 转换为numpy数组便于计算
    landmarks_array = np.array(landmarks)
    
    # 检查1: 确保所有坐标都是有效的数值
    if not np.all(np.isfinite(landmarks_array)):
        return False
    
    # 检查2: 检查坐标范围是否合理（防止异常值）
    x_coords = landmarks_array[:, 0]
    y_coords = landmarks_array[:, 1]
    z_coords = landmarks_array[:, 2]
    
    # X和Y坐标应该在合理范围内（假设图像尺寸不会超过10000像素）
    if np.any(x_coords < -1000) or np.any(x_coords > 10000) or \
       np.any(y_coords < -1000) or np.any(y_coords > 10000):
        return False
    
    # Z坐标应该在合理范围内（MediaPipe的z值通常在-1到1之间）
    if np.any(z_coords < -10) or np.any(z_coords > 10):
        return False
    
    # 检查3: 检查关键点之间的距离是否合理
    # 计算相邻关键点之间的距离，防止关键点过于密集或稀疏
    distances = []
    for i in range(len(landmarks) - 1):
        dist = np.linalg.norm(landmarks_array[i+1] - landmarks_array[i])
        distances.append(dist)
    
    distances = np.array(distances)
    
    # 如果距离过小（关键点重叠）或过大（关键点缺失），认为质量差
    if np.any(distances < 0.1) or np.any(distances > 1000):
        return False
    
    # 检查4: 检查关键点的分布是否合理
    # 计算关键点的标准差，如果标准差过小说明关键点可能有问题
    x_std = np.std(x_coords)
    y_std = np.std(y_coords)
    
    if x_std < 1.0 or y_std < 1.0:  # 关键点分布过于集中
        return False
    
    # 检查5: 检查是否有重复的关键点
    unique_points = set()
    for point in landmarks:
        # 将坐标四舍五入到小数点后2位，避免浮点数精度问题
        rounded_point = (round(point[0], 2), round(point[1], 2), round(point[2], 2))
        if rounded_point in unique_points:
            return False  # 发现重复点
        unique_points.add(rounded_point)
    
    return True

def get_eye_landmarks(landmarks: List[Tuple[float, float, float]], eye_type: str) -> List[Tuple[float, float, float]]:
    """
    输入：468个关键点(x,y,z)
    输出：眼睛的关键点
    
    :param landmarks: 关键点列表
    示例
    landmarks = [
        (x1, y1, z1),  # 关键点1坐标
        (x2, y2, z2),  # 关键点2坐标
        ...,
        (x468, y468, z468)  # 关键点468坐标
    ]
    
    :param eye_type: 眼睛类型
    示例
    eye_type = "left"   # 左眼
    eye_type = "right"  # 右眼
    
    调用关系：
    被调用：
    - detector.py 中的 get_eye_centers() 调用此函数
    
    调用：
    - get_landmark_indices() (内部函数)
    
    可能用到的库函数：无
    """
    if not landmarks or len(landmarks) != 468:
        return []
    
    if eye_type.lower() not in ['left', 'right']:
        raise ValueError("eye_type 必须是 'left' 或 'right'")
    
    # 获取关键点索引
    indices = get_landmark_indices()
    
    if eye_type.lower() == 'left':
        # 左眼关键点：轮廓(10-19) + 虹膜(32-43)
        left_eye_contour = indices["left_eye_contour"]  # [10, 11, 12, ..., 19]
        left_iris = indices["left_iris"]                # [32, 33, 34, ..., 43]
        eye_indices = left_eye_contour + left_iris
    else:  # right
        # 右眼关键点：轮廓(0-9) + 虹膜(20-31)
        right_eye_contour = indices["right_eye_contour"]  # [0, 1, 2, ..., 9]
        right_iris = indices["right_iris"]                # [20, 21, 22, ..., 31]
        eye_indices = right_eye_contour + right_iris
    
    # 提取指定眼睛的关键点
    eye_landmarks = [landmarks[i] for i in eye_indices]
    
    return eye_landmarks

def get_pupil_landmarks(landmarks: List[Tuple[float, float, float]]) -> Dict[str, Union[Tuple[float, float, float], None]]:
    """
    提取左右瞳孔中心关键点
    
    :param landmarks: 关键点列表
    示例
    landmarks = [
        (x1, y1, z1),  # 关键点1坐标
        (x2, y2, z2),  # 关键点2坐标
        ...,
        (x468, y468, z468)  # 关键点468坐标
    ]
    
    调用关系：
    被调用：
    - detector.py 中的 get_pupil_centers() 调用此函数
    
    调用：
    - calculate_center() (内部函数)
    
    可能用到的库函数：无
    """
    if not landmarks or len(landmarks) != 468:
        # 返回None值，但类型注解允许None
        return {'left': None, 'right': None}
    
    # MediaPipe Face Mesh中瞳孔中心的索引
    # 这些索引是基于MediaPipe的468个关键点中瞳孔中心的位置
    LEFT_PUPIL_INDEX = 468  # 左瞳孔中心（如果使用refine_landmarks=True）
    RIGHT_PUPIL_INDEX = 473  # 右瞳孔中心（如果使用refine_landmarks=True）
    
    # 如果没有refine_landmarks，则使用虹膜中心作为近似
    # 左眼虹膜中心：索引32-43的平均值
    # 右眼虹膜中心：索引20-31的平均值
    left_iris_indices = list(range(32, 44))   # 左眼虹膜
    right_iris_indices = list(range(20, 32))  # 右眼虹膜
    
    # 计算左眼虹膜中心
    left_iris_points = [landmarks[i] for i in left_iris_indices]
    left_pupil = calculate_center(left_iris_points)
    
    # 计算右眼虹膜中心
    right_iris_points = [landmarks[i] for i in right_iris_indices]
    right_pupil = calculate_center(right_iris_points)
    
    return {
        'left': left_pupil,
        'right': right_pupil
    }

def calculate_center(points: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    """
    计算多个点的中心点
    
    :param points: 点列表
    示例
    points = [
        (x1, y1, z1),  # 点1坐标
        (x2, y2, z2),  # 点2坐标
        ...,
        (xn, yn, zn)   # 点n坐标
    ]
    
    调用关系：
    被调用：
    - get_pupil_landmarks() 调用此函数
    - get_iris_landmarks() 调用此函数
    
    调用：
    - 无直接调用其他函数
    
    可能用到的库函数：无
    """
    if not points:
        return (0.0, 0.0, 0.0)
    
    # 分别计算x, y, z的平均值
    x_sum = sum(point[0] for point in points)
    y_sum = sum(point[1] for point in points)
    z_sum = sum(point[2] for point in points)
    
    center_x = x_sum / len(points)
    center_y = y_sum / len(points)
    center_z = z_sum / len(points)
    
    return (center_x, center_y, center_z)

def get_iris_landmarks(landmarks: List[Tuple[float, float, float]]) -> Dict[str, List[Tuple[float, float, float]]]:
    """
    提取左右虹膜边界关键点
    
    :param landmarks: 关键点列表
    示例
    landmarks = [
        (x1, y1, z1),  # 关键点1坐标
        (x2, y2, z2),  # 关键点2坐标
        ...,
        (x468, y468, z468)  # 关键点468坐标
    ]
    
    调用关系：
    被调用：
    - detector.py 中的 get_iris_boundaries() 调用此函数
    
    调用：
    - get_landmark_indices() (内部函数)
    
    可能用到的库函数：无
    """
    if not landmarks or len(landmarks) != 468:
        return {'left': [], 'right': []}
    
    # 获取关键点索引
    indices = get_landmark_indices()
    
    # 提取左眼虹膜关键点（索引32-43）
    left_iris_indices = indices["left_iris"]  # [32, 33, 34, ..., 43]
    left_iris_landmarks = [landmarks[i] for i in left_iris_indices]
    
    # 提取右眼虹膜关键点（索引20-31）
    right_iris_indices = indices["right_iris"]  # [20, 21, 22, ..., 31]
    right_iris_landmarks = [landmarks[i] for i in right_iris_indices]
    
    return {
        'left': left_iris_landmarks,
        'right': right_iris_landmarks
    }

if __name__ == "__main__":
    print(" landmark_extractor模块实时测试")
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
        
        # 绘制所有关键点（绿色小点）
        for point in landmarks:
            x, y = int(point[0]), int(point[1])
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
        
        # 绘制眼睛区域
        try:
            landmark_indices = get_landmark_indices()
            
            # 左眼（蓝色）
            left_eye_indices = landmark_indices['left_eye_contour'] + landmark_indices['left_iris']
            for idx in left_eye_indices:
                if idx < len(landmarks):
                    x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
                    cv2.circle(image, (x, y), 2, (255, 0, 0), -1)
            
            # 右眼（红色）
            right_eye_indices = landmark_indices['right_eye_contour'] + landmark_indices['right_iris']
            for idx in right_eye_indices:
                if idx < len(landmarks):
                    x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
                    cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
                    
        except Exception:
            pass
        
        return image
    
    def analyze_and_display_results(landmarks, frame_count):
        """分析并显示结果"""
        if not landmarks:
            print(f"帧 {frame_count}: ❌ 未检测到人脸")
            return
        
        try:
            landmarks_tuple = [(point[0], point[1], point[2]) for point in landmarks]
            
            # 验证数据质量
            is_valid = validate_landmarks(landmarks_tuple)
            
            # 获取眼睛关键点
            left_eye = get_eye_landmarks(landmarks_tuple, "left")
            right_eye = get_eye_landmarks(landmarks_tuple, "right")
            
            # 获取瞳孔中心
            pupil_centers = get_pupil_landmarks(landmarks_tuple)
            
            # 获取虹膜边界
            iris_boundaries = get_iris_landmarks(landmarks_tuple)
            
            # 计算z值统计信息
            z_values = [point[2] for point in landmarks]
            z_min = min(z_values)
            z_max = max(z_values)
            z_avg = sum(z_values) / len(z_values)
            
            # 显示结果
            status = "✅ 有效" if is_valid else "❌ 无效"
            print(f"帧 {frame_count}: {status} | 关键点: {len(landmarks)} | 左眼: {len(left_eye)} | 右眼: {len(right_eye)} | 虹膜L/R: {len(iris_boundaries['left'])}/{len(iris_boundaries['right'])}")
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
                
                # 使用data_manager存储图像
                from utils.data_manager import add_frame, get_image
                
                # 模拟深度图
                h, w = frame.shape[:2]
                depth_map = np.random.rand(h, w).astype(np.float32) * 2.0
                
                # 存储到data_manager
                add_frame(frame_count, frame, depth_map)
                
                # 从data_manager获取图像进行测试
                stored_image = get_image(frame_count)
                if stored_image is not None:
                    # 提取关键点
                    landmarks = extract_landmarks(stored_image)
                    
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
            from utils.data_manager import get_frame_count
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            print(f"\n📊 最终统计:")
            print(f"   处理帧数: {frame_count}")
            print(f"   平均FPS: {fps:.1f}")
            print(f"   存储帧数: {get_frame_count()}")
            print("✅ 测试完成")
    
    # 运行测试
    try:
        run_real_time_test()
    except Exception as e:
        print(f"❌ 测试错误: {e}")