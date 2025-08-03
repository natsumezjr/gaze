import numpy as np
import json
from typing import Dict

def load_camera_params(file_path: str) -> Dict:
    """
    加载相机参数
    :param file_path: 参数文件路径
    :return: 相机参数字典
    可能用到的库函数：json
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        params = json.load(f)
    return params

def validate_camera_params(params: Dict) -> bool:
    """
    验证相机参数有效性
    :param params: 相机参数字典
    :return: 是否有效
    可能用到的库函数：无
    """
    required_keys = ['fx', 'fy', 'cx', 'cy']
    return all(key in params and isinstance(params[key], (int, float)) for key in required_keys)

def get_intrinsic_matrix(params: Dict) -> np.ndarray:
    """
    获取内参矩阵
    :param params: 相机参数字典
    :return: 3x3内参矩阵
    可能用到的库函数：numpy
    """
    fx = params['fx']
    fy = params['fy']
    cx = params['cx']
    cy = params['cy']
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]])

def save_camera_params(params: Dict, file_path: str):
    """
    保存相机参数
    :param params: 相机参数字典
    :param file_path: 保存路径
    可能用到的库函数：json
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=4, ensure_ascii=False)

def get_camera_info(params: Dict) -> str:
    """
    获取相机信息字符串
    :param params: 相机参数字典
    :return: 信息字符串
    可能用到的库函数：无
    """
    info = (
        f"相机内参：\n"
        f"fx = {params.get('fx', 'N/A')}, fy = {params.get('fy', 'N/A')}\n"
        f"cx = {params.get('cx', 'N/A')}, cy = {params.get('cy', 'N/A')}"
    )
    return info
