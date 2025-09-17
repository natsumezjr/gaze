# 关键点提取模块
import numpy as np
import mediapipe as mp
import cv2
from typing import List, Dict, Optional
import logging
from project.data.data_models import (
    Landmark, KeyCoordinates, FITTING_TYPE, 
    EYE_TYPE, BGRImage, Point3DWithVisibility
)

# 配置日志
from project.config.logging_config import setup_logging, get_logger
setup_logging()
logger = get_logger(__name__)

def calculate_visibility(landmark: Landmark, image: BGRImage) -> float:
    """计算关键点的可见性分数"""
    try:
        x, y, z = landmark.x, landmark.y, landmark.z
        height, width = image.height, image.width
        
        # 检查坐标是否在图像范围内
        if x < 0 or x >= width or y < 0 or y >= height:
            return 0.0
        
        # 检查坐标是否合理（不是NaN或无穷大）
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
            return 0.0
        
        # 基于坐标位置的可见性计算
        # 中心区域可见性更高
        center_x, center_y = width // 2, height // 2
        distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # 距离中心越近，可见性越高
        visibility = max(0.1, 1.0 - (distance_from_center / max_distance) * 0.5)
        
        # 检查周围像素的亮度（简单的光照检测）
        try:
            x_int, y_int = int(x), int(y)
            if 0 <= x_int < width-1 and 0 <= y_int < height-1:
                # 检查周围3x3区域的亮度
                region = image.data[y_int-1:y_int+2, x_int-1:x_int+2]
                if region.size > 0:
                    brightness = np.mean(region)
                    # 亮度适中时可见性更高
                    if 50 < brightness < 200:
                        visibility *= 1.2
                    elif brightness < 30 or brightness > 220:
                        visibility *= 0.7
        except:
            pass
        
        return min(1.0, max(0.1, visibility))
        
    except Exception as e:
        logger.warning(f"计算可见性时出错: {e}")
        return 0.5

def extract_landmarks(bgr_image: BGRImage) -> List[Landmark]:
    """
    从BGR图像中提取关键点
    
    Args:
        bgr_image: BGR图像对象
        
    Returns:
        landmarks: 关键点列表，每个点包含[x, y, z, visibility]
    """
    try:
        # 创建MediaPipe FaceMesh实例
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # 转换为RGB格式
        rgb_image = bgr_image.to_rgb()
        results = face_mesh.process(rgb_image)
        face_mesh.close()
        
        if not results.multi_face_landmarks:
            return []
        
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = []
        
        for i, landmark in enumerate(face_landmarks.landmark):
            # 转换为像素坐标，添加边界检查
            x = landmark.x * bgr_image.width
            y = landmark.y * bgr_image.height
            z = landmark.z
            
            # 边界检查：确保坐标在有效范围内
            x = max(0, min(x, bgr_image.width - 1))
            y = max(0, min(y, bgr_image.height - 1))
            
            # 调试信息：检查MediaPipe原始坐标和转换后的坐标
            if i < 10 or landmark.y > 1.0:  # 打印前10个关键点或y坐标超过1.0的关键点
                logger.debug(f"MediaPipe关键点 {i}: 原始=({landmark.x:.3f}, {landmark.y:.3f}), 转换后=({x:.2f}, {y:.2f})")
                if landmark.y > 1.0:
                    logger.warning(f"关键点 {i} 原始y坐标超出范围: {landmark.y:.3f}，已限制到边界")
            
            # 创建Landmark对象
            landmark_obj = Landmark(x=x, y=y, z=z, visibility=1.0)
            
            # 计算可见性
            visibility = calculate_visibility(landmark_obj, bgr_image)
            landmark_obj.visibility = visibility
            
            landmarks.append(landmark_obj)
        
        logger.debug(f"提取到 {len(landmarks)} 个关键点")
        return landmarks
        
    except Exception as e:
        logger.error(f"提取关键点失败: {e}")
        return []

def get_landmark_indices() -> Dict[str, Dict[str, List[int]]]:
    """返回MediaPipe Face Mesh 拟合所需的关键点索引映射"""
    return {
        "left": {
            "pupil": [468],
            "iris": [469, 470, 471, 472],
            "inner_canthus": [133],
            "upper_eyelid": [157, 158, 159, 160, 173],
            "lower_eyelid": [145, 153, 154, 155, 161],
            "outer_canthus": [246]
        },
        "right": {
            "pupil": [473],
            "iris": [474, 475, 476, 477],
            "inner_canthus": [362],
            "upper_eyelid": [384, 385, 386, 387, 398],
            "lower_eyelid": [374, 380, 381, 382, 390],
            "outer_canthus": [466]
        }
    }

def get_fitting_landmarks(landmarks: List[Landmark]) -> KeyCoordinates:
    """
    获取所有眼部关键点，整合为fitting模块所需的格式
    
    Args:
        landmarks: 关键点列表
        
    Returns:
        key_coordinates: 整合后的眼部关键点
    """
    key_coordinates = KeyCoordinates()
    
    try:
        indices = get_landmark_indices()
        
        # 填充数据
        for eye in EYE_TYPE:
            for fitting_type in FITTING_TYPE:
                # 获取该类型的关键点索引
                landmark_indices = indices[eye][fitting_type]
                points = []
                
                for idx in landmark_indices:
                    if idx < len(landmarks):
                        landmark = landmarks[idx]
                        # 创建Point3DWithVisibility对象
                        point = Point3DWithVisibility(
                            x=landmark.x, y=landmark.y, z=landmark.z, 
                            visibility=landmark.visibility
                        )
                        points.append(point)
                
                # 设置坐标
                key_coordinates.set_points(eye, fitting_type, points)
        
        logger.debug(f"整合完成：左眼 {len(key_coordinates.left_eye)} 种类型，右眼 {len(key_coordinates.right_eye)} 种类型")
        return key_coordinates
        
    except Exception as e:
        logger.error(f"整合眼部关键点失败: {e}")
        return KeyCoordinates()

def validate_landmarks(landmarks: List[Landmark]) -> bool:
    """验证关键点数据的有效性"""
    if not landmarks:
        logger.warning("验证失败: 关键点列表为空")
        return False
        
    if len(landmarks) != 478:
        logger.warning(f"验证失败: 关键点数量不匹配，期望478个，实际{len(landmarks)}个")
        return False
    
    try:
        # 检查所有关键点是否有效
        for i, landmark in enumerate(landmarks):
            if not (np.isfinite(landmark.x) and np.isfinite(landmark.y) and 
                   np.isfinite(landmark.z) and np.isfinite(landmark.visibility)):
                logger.warning(f"验证失败: 关键点{i}包含无效值 - x:{landmark.x}, y:{landmark.y}, z:{landmark.z}, v:{landmark.visibility}")
                return False
            
            # 检查坐标范围
            if (landmark.x < -1000 or landmark.x > 10000 or 
                landmark.y < -1000 or landmark.y > 10000 or
                landmark.z < -1 or landmark.z > 1):
                logger.warning(f"验证失败: 关键点{i}坐标超出范围 - x:{landmark.x}, y:{landmark.y}, z:{landmark.z}")
                return False
        
        logger.info("关键点验证通过")
        return True
    except Exception as e:
        logger.warning(f"验证失败: 异常 {e}")
        return False

def calculate_data_quality(landmarks: List[Landmark], image: BGRImage) -> Dict[str, float]:
    """计算数据质量指标"""
    try:
        if not landmarks:
            return {'quality': 0.0, 'visibility': 0.0, 'stability': 0.0}
        
        # 计算平均可见性
        visibility_values = [landmark.visibility for landmark in landmarks]
        avg_visibility = np.mean(visibility_values)
        
        # 计算稳定性（坐标变化的标准差）
        x_coords = [landmark.x for landmark in landmarks]
        y_coords = [landmark.y for landmark in landmarks]
        
        x_std = np.std(x_coords)
        y_std = np.std(y_coords)
        
        # 稳定性分数（标准差越小越稳定）
        stability = max(0.0, 1.0 - (x_std + y_std) * 0.001)
        
        # 综合质量分数
        quality = (avg_visibility * 0.6 + stability * 0.4)
        
        return {
            'quality': quality,
            'visibility': avg_visibility,
            'stability': stability,
            'point_count': len(landmarks),
            'valid_points': len([l for l in landmarks if l.is_visible()])
        }
        
    except Exception as e:
        logger.warning(f"计算数据质量时出错: {e}")
        return {'quality': 0.0, 'visibility': 0.0, 'stability': 0.0}

# 测试函数
def test_landmark_extractor():
    """测试关键点提取器功能"""
    print("=== 关键点提取器测试 ===")
    
    # 创建测试图像
    test_image = BGRImage(data=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    
    # 提取关键点
    landmarks = extract_landmarks(test_image)
    print(f"提取到 {len(landmarks)} 个关键点")
    
    # 验证关键点
    is_valid = validate_landmarks(landmarks)
    print(f"关键点验证: {'通过' if is_valid else '失败'}")
    
    # 获取拟合数据
    key_coordinates = get_fitting_landmarks(landmarks)
    print(f"拟合数据: {key_coordinates}")
    
    # 计算数据质量
    quality = calculate_data_quality(landmarks, test_image)
    print(f"数据质量: {quality}")

if __name__ == "__main__":
    test_landmark_extractor()