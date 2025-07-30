import numpy as np
import cv2
import base64
import json
from typing import Dict, Tuple, Optional, Union
from pathlib import Path

def decode_base64_image(b64_string: str) -> np.ndarray:
    """
    输入：base64编码的jpg/png图像字符串
    处理：解码base64字符串为numpy数组（BGR格式，OpenCV默认）
    输出：BGR格式的numpy数组
    
    :param b64_string: base64编码的图像字符串
    示例
    b64_string = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDA..."  # base64字符串
    
    :return: BGR格式的numpy数组，形状为(H, W, 3)
    返回值示例：
    np.ndarray(shape=(720, 1280, 3), dtype=np.uint8)  # BGR图像数组
    
    调用关系：
    被调用：
    - parse_rgbd_json() 调用此函数
    - load_rgbd_from_file() 调用此函数
    
    调用：
    - 无直接调用其他函数
    
    可能用到的库函数：base64, cv2, numpy
    """
    try:
        img_bytes = base64.b64decode(b64_string)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("无法解码base64图像字符串")
        
        return img
    except Exception as e:
        raise ValueError(f"base64图像解码失败: {str(e)}")

def parse_rgbd_json(json_data: Union[str, Dict]) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    """
    输入：RGB-D JSON数据（字符串或字典格式）
    处理：解析JSON数据，提取RGB图像、深度图和元数据
    输出：RGB图像、深度图和元数据的元组
    
    :param json_data: JSON数据（字符串或字典）
    示例
    json_data = {
        "frame_id": 1,
        "timestamp": "2024-01-15T10:30:00",
        "camera": {"resolution": [1280, 720]},
        "rgb_data": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDA...",
        "depth_data": None,
        "eye_centers": {"left": None, "right": None},
        "pupil_centers": {"left": None, "right": None},
        "iris_boundaries": {"left": [], "right": []}
    }
    
    :return: (rgb_image, depth_map, metadata) 元组
    返回值示例：
    rgb_image: np.ndarray(shape=(720, 1280, 3), dtype=np.uint8)  # BGR格式图像
    depth_map: np.ndarray(shape=(720, 1280), dtype=np.float32) 或 None  # 深度图（米单位）
    metadata: dict  # 包含frame_id, timestamp, camera等信息
    
    调用关系：
    被调用：
    - load_rgbd_from_file() 调用此函数
    
    调用：
    - decode_base64_image() (内部函数)
    
    可能用到的库函数：json, numpy, cv2
    """
    # 如果输入是字符串，先解析为字典
    if isinstance(json_data, str):
        try:
            data_dict = json.loads(json_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON解析失败: {str(e)}")
    else:
        data_dict = json_data
    
    # 验证必要字段
    required_fields = ["rgb_data", "frame_id", "timestamp"]
    for field in required_fields:
        if field not in data_dict:
            raise ValueError(f"缺少必要字段: {field}")
    
    # 解析RGB图像
    rgb_b64 = data_dict["rgb_data"]
    if not rgb_b64:
        raise ValueError("rgb_data字段为空")
    
    rgb_image = decode_base64_image(rgb_b64)
    
    # 解析深度图（可选）
    depth_map = None
    depth_data = data_dict.get("depth_data")
    if depth_data is not None:
        if isinstance(depth_data, str):
            # 如果深度数据是base64字符串
            depth_map = decode_base64_image(depth_data)
            if len(depth_map.shape) == 3:
                # 如果是3通道，转换为单通道
                depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)
        elif isinstance(depth_data, list):
            # 如果深度数据是列表
            depth_map = np.array(depth_data, dtype=np.float32)
    
    # 提取元数据
    metadata = {
        "frame_id": data_dict["frame_id"],
        "timestamp": data_dict["timestamp"],
        "camera": data_dict.get("camera", {}),
        "eye_centers": data_dict.get("eye_centers", {"left": None, "right": None}),
        "pupil_centers": data_dict.get("pupil_centers", {"left": None, "right": None}),
        "iris_boundaries": data_dict.get("iris_boundaries", {"left": [], "right": []})
    }
    
    return rgb_image, depth_map, metadata

def load_rgbd_from_file(file_path: str) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    """
    输入：JSON文件路径
    处理：从JSON文件加载RGB-D数据
    输出：RGB图像、深度图和元数据的元组
    
    :param file_path: JSON文件路径
    示例
    file_path = "data/rgbd_input.json"  # JSON文件路径
    
    :return: (rgb_image, depth_map, metadata) 元组
    返回值示例：
    rgb_image: np.ndarray(shape=(720, 1280, 3), dtype=np.uint8)  # BGR格式图像
    depth_map: np.ndarray(shape=(720, 1280), dtype=np.float32) 或 None  # 深度图（米单位）
    metadata: dict  # 包含frame_id, timestamp, camera等信息
    
    调用关系：
    被调用：
    - 无直接被调用
    
    调用：
    - parse_rgbd_json() (内部函数)
    
    可能用到的库函数：json, pathlib
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except Exception as e:
        raise ValueError(f"文件读取失败: {str(e)}")
    
    return parse_rgbd_json(json_data)

def encode_image_to_base64(image: np.ndarray, format: str = '.jpg') -> str:
    """
    输入：BGR格式的numpy数组图像
    处理：将numpy数组图像编码为base64字符串
    输出：base64编码的字符串
    
    :param image: BGR格式的numpy数组，形状为(H, W, 3)
    示例
    image = np.ndarray(shape=(720, 1280, 3), dtype=np.uint8)  # BGR图像
    
    :param format: 图像格式，默认为'.jpg'
    示例
    format = '.jpg'  # 图像格式
    
    :return: base64编码的字符串
    返回值示例：
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDA..."  # base64字符串
    
    调用关系：
    被调用：
    - data_writer.py 中的 convert_to_json() 调用此函数
    
    调用：
    - 无直接调用其他函数
    
    可能用到的库函数：cv2, base64, numpy
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("输入必须是numpy数组")
    
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("输入图像必须是3通道BGR格式")
    
    try:
        _, buffer = cv2.imencode(format, image)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        raise ValueError(f"图像编码失败: {str(e)}")

def validate_image_format(image: np.ndarray) -> bool:
    """
    输入：待验证的图像
    处理：验证图像格式是否符合要求
    输出：是否有效的布尔值
    
    :param image: 待验证的图像
    示例
    image = np.ndarray(shape=(720, 1280, 3), dtype=np.uint8)  # BGR图像
    
    :return: 是否有效
    返回值示例：
    True  # 图像格式有效
    False  # 图像格式无效
    
    验证条件：
    1. 必须是numpy数组
    2. 必须是3维数组
    3. 第三维必须是3（BGR格式）
    4. 数据类型必须是uint8
    
    调用关系：
    被调用：
    - 无直接被调用
    
    调用：
    - 无直接调用其他函数
    
    可能用到的库函数：numpy
    """
    if not isinstance(image, np.ndarray):
        return False
    
    if len(image.shape) != 3:
        return False
    
    if image.shape[2] != 3:
        return False
    
    if image.dtype != np.uint8:
        return False
    
    return True 