import numpy as np
from typing import List, Dict

class FaceDetector:
    def __init__(self, camera_params: dict):
        """
        初始化检测器
        :param camera_params: 相机内参字典
        可能用到的库函数：无
        """
        pass

    def detect_face(self, rgb_image: np.ndarray, depth_map: np.ndarray) -> bool:
        """
        检测人脸和关键点
        :param rgb_image: RGB图像
        :param depth_map: 深度图
        :return: 检测是否成功
        可能用到的库函数：mediapipe, opencv
        """
        pass

    def get_eye_centers(self) -> Dict[str, np.ndarray]:
        """
        获取左右眼球中心三维坐标
        :return: {'left': np.ndarray, 'right': np.ndarray}
        可能用到的库函数：numpy
        """
        pass

    def get_pupil_centers(self) -> Dict[str, np.ndarray]:
        """
        获取左右瞳孔中心三维坐标
        :return: {'left': np.ndarray, 'right': np.ndarray}
        可能用到的库函数：numpy
        """
        pass

    def get_iris_boundaries(self) -> Dict[str, List[np.ndarray]]:
        """
        获取左右虹膜边界点三维坐标
        :return: {'left': List[np.ndarray], 'right': List[np.ndarray]}
        可能用到的库函数：numpy
        """
        pass

    def get_detection_confidence(self) -> float:
        """
        获取检测置信度
        :return: 置信度分数
        可能用到的库函数：无
        """
        pass

    def get_detection_status(self) -> str:
        """
        获取检测状态信息
        :return: 状态字符串
        可能用到的库函数：无
        """
        pass
