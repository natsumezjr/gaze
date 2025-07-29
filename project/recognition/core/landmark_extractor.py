import numpy as np
import mediapipe as mp
import cv2
from typing import List, Dict, Tuple

def extract_landmarks(rgb_image: np.ndarray) -> List[Tuple[float, float, float]]:
    """
    提取468个人脸关键点
    
    :param rgb_image: RGB图像数据
    示例
    rgb_image = np.ndarray(
        shape=(720, 1280, 3),  # 图像尺寸 (高度, 宽度, 3通道)
        dtype=np.uint8,        # 数据类型 (0-255的整数)
        data=[[[R,G,B], ...], ...]  # RGB像素值数组
    )
    
    传入参数的函数：
    cv2.imread(file_path: str) -> np.ndarray
        传入参数：file_path (图像文件路径)
        返回：RGB图像数组
    
    cv2.VideoCapture.read() -> (bool, np.ndarray)
        传入参数：无 (从摄像头读取)
        返回：(成功标志, RGB图像数组)
    
    可能用到的库函数：mediapipe, opencv
    """
    # 初始化MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    
    # 检查输入图像格式（MediaPipe需要RGB格式）
    if len(rgb_image.shape) == 3 and rgb_image.shape[2] == 3:
        # 输入图像已经是RGB格式（由detector统一处理）
        rgb_image_rgb = rgb_image
    else:
        raise ValueError("输入图像必须是3通道RGB图像")
    
    # 获取图像尺寸
    height, width = rgb_image_rgb.shape[:2]
    
    # 检测人脸关键点
    results = face_mesh.process(rgb_image_rgb)
    
    if results.multi_face_landmarks:
        # 获取第一个检测到的人脸的关键点
        face_landmarks = results.multi_face_landmarks[0]
        
        # 转换关键点格式
        landmarks = []
        for landmark in face_landmarks.landmark:
            # 将相对坐标转换为像素坐标
            x = landmark.x * width
            y = landmark.y * height
            z = landmark.z  # 保持相对深度值
            landmarks.append((x, y, z))
        
        return landmarks
    else:
        # 如果没有检测到人脸，返回空列表
        return []

# 示例函数：展示如何使用extract_landmarks
def example_usage():
    """
    使用示例：
    1. 从摄像头读取图像
    2. 提取人脸关键点
    3. 在图像上绘制关键点
    """
    # 示例1：从摄像头读取图像
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # 提取关键点
        landmarks = extract_landmarks(frame)
        
        if landmarks:
            print(f"成功检测到人脸，提取了 {len(landmarks)} 个关键点")
            
            # 示例：获取第一个关键点的坐标
            first_landmark = landmarks[0]
            print(f"第一个关键点坐标: x={first_landmark[0]:.2f}, y={first_landmark[1]:.2f}, z={first_landmark[2]:.2f}")
            
            # 示例：在图像上绘制关键点
            for i, (x, y, z) in enumerate(landmarks):
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
                if i == 0:  # 只标注第一个关键点
                    cv2.putText(frame, f"Point 0", (int(x)+5, int(y)-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # 显示结果
            cv2.imshow('Face Landmarks', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("未检测到人脸")
    
    # 示例2：从图像文件读取
    # image_path = "path/to/your/image.jpg"
    # frame = cv2.imread(image_path)
    # landmarks = extract_landmarks(frame)

# 关键点索引说明（MediaPipe Face Mesh的468个关键点）
def get_landmark_indices():
    """
    返回重要关键点的索引
    
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
    
    传入参数的函数：无
    
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
    验证关键点质量
    
    :param landmarks: 关键点列表
    示例
    landmarks = [
        (x1, y1, z1),  # 关键点1坐标 (像素x, 像素y, 相对深度z)
        (x2, y2, z2),  # 关键点2坐标
        ...,
        (x468, y468, z468)  # 关键点468坐标
    ]
    
    传入参数的函数：
    extract_landmarks(rgb_image: np.ndarray) -> List[Tuple[float, float, float]]
        传入参数：rgb_image (RGB图像数组)
        返回：468个关键点列表
    
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
    提取指定眼睛的关键点
    
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
    
    传入参数的函数：
    extract_landmarks(rgb_image: np.ndarray) -> List[Tuple[float, float, float]]
        传入参数：rgb_image (RGB图像数组)
        返回：468个关键点列表
    
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

def get_pupil_landmarks(landmarks: List[Tuple[float, float, float]]) -> Dict[str, Tuple[float, float, float]]:
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
    
    传入参数的函数：
    extract_landmarks(rgb_image: np.ndarray) -> List[Tuple[float, float, float]]
        传入参数：rgb_image (RGB图像数组)
        返回：468个关键点列表
    
    可能用到的库函数：无
    """
    if not landmarks or len(landmarks) != 468:
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
    
    传入参数的函数：
    get_iris_landmarks(landmarks: List) -> Dict[str, List[Tuple[float, float, float]]]
        传入参数：landmarks (关键点列表)
        返回：虹膜关键点字典
    
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
    
    传入参数的函数：
    extract_landmarks(rgb_image: np.ndarray) -> List[Tuple[float, float, float]]
        传入参数：rgb_image (RGB图像数组)
        返回：468个关键点列表
    
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
