import numpy as np
from typing import List, Dict

def extract_landmarks(rgb_image: np.ndarray) -> List:
    """
    提取468个人脸关键点
    :param rgb_image: RGB图像
    :return: 关键点列表
    可能用到的库函数：mediapipe, opencv
    """
    pass

def validate_landmarks(landmarks: List) -> bool:
    """
    验证关键点质量
    :param landmarks: 关键点列表
    :return: 是否有效
    可能用到的库函数：numpy
    """
    pass

def get_eye_landmarks(landmarks: List, eye_type: str) -> List:
    """
    提取指定眼睛的关键点
    :param landmarks: 关键点列表
    :param eye_type: 'left' 或 'right'
    :return: 眼部关键点列表
    可能用到的库函数：无
    """
    pass

def get_pupil_landmarks(landmarks: List) -> Dict[str, object]:
    """
    提取左右瞳孔中心关键点
    :param landmarks: 关键点列表
    :return: {'left': landmark, 'right': landmark}
    可能用到的库函数：无
    """
    pass

def get_iris_landmarks(landmarks: List) -> Dict[str, List]:
    """
    提取左右虹膜边界关键点
    :param landmarks: 关键点列表
    :return: {'left': [landmark...], 'right': [landmark...]}
    可能用到的库函数：无
    """
    pass
