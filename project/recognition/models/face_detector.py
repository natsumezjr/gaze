import numpy as np
from typing import List, Tuple

class FaceMeshModel:
    def __init__(self):
        """
        初始化人脸检测模型
        可能用到的库函数：mediapipe
        """
        pass

    def detect(self, rgb_image: np.ndarray) -> List[Tuple[float, float, float]]:
        """
        检测人脸关键点
        
        :param rgb_image: RGB图像，numpy数组格式，形状为(H, W, 3)
        注意：输入图像必须是RGB格式
        
        :return: 关键点列表
        返回格式：List[Tuple[float, float, float]]
        示例：[(x1, y1, z1), (x2, y2, z2), ..., (x468, y468, z468)]
        
        可能用到的库函数：mediapipe
        """
        pass

    def get_face_mesh(self) -> List[Tuple[float, float, float]]:
        """
        获取人脸网格
        
        :return: 网格点列表
        返回格式：List[Tuple[float, float, float]]
        示例：[(x1, y1, z1), (x2, y2, z2), ..., (xn, yn, zn)]
        
        可能用到的库函数：mediapipe
        """
        pass

    def is_face_detected(self) -> bool:
        """
        判断是否检测到人脸
        :return: 是否检测到
        可能用到的库函数：无
        """
        pass

    def get_face_confidence(self) -> float:
        """
        获取人脸检测置信度
        :return: 置信度
        可能用到的库函数：无
        """
        pass

    def reset_detector(self):
        """
        重置检测器状态
        可能用到的库函数：无
        """
        pass
