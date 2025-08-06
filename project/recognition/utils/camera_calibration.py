import numpy as np
import json
from typing import Dict
import json

def load_camera_params(file_path: str) -> Dict:
    """
    输入：相机参数文件路径
    处理：从JSON文件加载相机参数
    输出：相机参数字典
    
    :param file_path: 参数文件路径
    示例
    file_path = "config/camera_params.json"  # 相机参数文件路径
    
    :return: 相机参数字典
    返回值示例：
    {
        "intrinsic_params": {
            "fx": 925.0,  # 焦距x
            "fy": 925.0,  # 焦距y
            "cx": 640.0,  # 主点x坐标
            "cy": 360.0   # 主点y坐标
        },
        "image_resolution": {
            "width": 1280,   # 图像宽度
            "height": 720    # 图像高度
        },
        "depth_scale": 0.001  # 深度值缩放因子
    }
    
    调用关系：
    被调用：
    - main()
    
    调用：
    - 无直接调用其他函数
    
    可能用到的库函数：json
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        params = json.load(f)
    return params

def validate_camera_params(params: Dict) -> bool:
    """
    输入：相机参数字典
    处理：验证相机参数的有效性（必要字段、数值范围等）
    输出：是否有效的布尔值
    
    :param params: 相机参数字典
    示例
    params = {
        "intrinsic_params": {
            "fx": 925.0,  # 焦距x
            "fy": 925.0,  # 焦距y
            "cx": 640.0,  # 主点x坐标
            "cy": 360.0   # 主点y坐标
        },
        "image_resolution": {
            "width": 1280,   # 图像宽度
            "height": 720    # 图像高度
        }
    }
    
    :return: 是否有效
    返回值示例：
    True  # 相机参数有效
    False  # 相机参数无效
    
    调用关系：
    被调用：
    - main()
    
    调用：
    - 无直接调用其他函数
    
    可能用到的库函数：无
    """
    required_keys = ['fx', 'fy', 'cx', 'cy']
    return all(key in params and isinstance(params[key], (int, float)) for key in required_keys)

def get_intrinsic_matrix(params: Dict) -> np.ndarray:
    """
    输入：相机参数字典
    处理：从相机参数中提取内参矩阵
    输出：3x3内参矩阵
    
    :param params: 相机参数字典
    示例
    params = {
        "intrinsic_params": {
            "fx": 925.0,  # 焦距x
            "fy": 925.0,  # 焦距y
            "cx": 640.0,  # 主点x坐标
            "cy": 360.0   # 主点y坐标
        }
    }
    
    :return: 3x3内参矩阵
    返回值示例：
    np.array([
        [925.0, 0.0, 640.0],   # 第一行
        [0.0, 925.0, 360.0],   # 第二行
        [0.0, 0.0, 1.0]        # 第三行
    ])
    
    调用关系：
    被调用：
    - ?
    
    调用：
    - 无直接调用其他函数
    
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
    输入：相机参数字典和保存路径
    处理：将相机参数保存到JSON文件
    输出：无返回值
    
    :param params: 相机参数字典
    示例
    params = {
        "intrinsic_params": {
            "fx": 925.0,  # 焦距x
            "fy": 925.0,  # 焦距y
            "cx": 640.0,  # 主点x坐标
            "cy": 360.0   # 主点y坐标
        },
        "image_resolution": {
            "width": 1280,   # 图像宽度
            "height": 720    # 图像高度
        }
    }
    
    :param file_path: 保存路径
    示例
    file_path = "config/camera_params.json"  # 保存文件路径
    
    调用关系：
    被调用：
    - ?
    
    调用：
    - 无直接调用其他函数
    
    可能用到的库函数：json
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=4, ensure_ascii=False)

def get_camera_info(params: Dict) -> str:
    """
    输入：相机参数字典
    处理：生成相机信息的可读字符串
    输出：相机信息字符串
    
    :param params: 相机参数字典
    示例
    params = {
        "camera_name": "Intel RealSense D435i",
        "intrinsic_params": {
            "fx": 925.0,  # 焦距x
            "fy": 925.0,  # 焦距y
            "cx": 640.0,  # 主点x坐标
            "cy": 360.0   # 主点y坐标
        },
        "image_resolution": {
            "width": 1280,   # 图像宽度
            "height": 720    # 图像高度
        }
    }
    
    :return: 信息字符串
    返回值示例：
    "相机型号：Intel RealSense D435i
    图像分辨率：1280x720
    焦距：fx=925.0, fy=925.0
    主点：cx=640.0, cy=360.0"
    
    调用关系：
    被调用：
    - ?
    
    调用：
    - 无直接调用其他函数
    
    可能用到的库函数：无
    """
    info = (
        f"相机内参：\n"
        f"fx = {params.get('fx', 'N/A')}, fy = {params.get('fy', 'N/A')}\n"
        f"cx = {params.get('cx', 'N/A')}, cy = {params.get('cy', 'N/A')}"
    )
    return info
