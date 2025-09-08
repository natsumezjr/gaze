"""
眼动追踪数据接口类
支持模拟数据和真实摄像头数据

使用方法：
1. 修改 _get_real_camera_data() 方法中的硬件接口代码
2. 实现 _capture_eye_centers(), _capture_pupil_centers() 等方法
3. 运行程序时选择"使用真实摄像头"模式
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple
import os
import json
from datetime import datetime

# ==================== 可选的硬件库导入 ====================
# 取消注释你需要的库：

# USB摄像头支持
# import cv2

# 眼动仪支持（如Tobii）
# import tobii_research as tr

# 串口通信支持
# import serial

# 图像处理支持
# from PIL import Image
# import matplotlib.pyplot as plt

# 深度学习支持（如果需要）
# import torch
# import torchvision


class EyeTrackingDataInterface:
    """眼动追踪数据接口类 - 支持模拟数据和真实摄像头数据"""
    
    def __init__(self, use_real_camera=False, camera_config=None):
        """
        初始化数据接口
        
        Args:
            use_real_camera: 是否使用真实摄像头
            camera_config: 摄像头配置字典
        """
        self.use_real_camera = use_real_camera
        self.camera_config = camera_config or self._get_default_camera_config()
        
        # 硬件相关属性
        self.cap = None  # USB摄像头对象
        self.eyetracker = None  # 眼动仪对象
        self.serial_conn = None  # 串口连接对象
        
        if use_real_camera:
            self._init_real_camera()
    
    def _get_default_camera_config(self):
        """获取默认摄像头配置"""
        return {
            'resolution': (1920, 1080),  # 屏幕分辨率
            'focal_length': 1000,        # 焦距（像素）
            'principal_point': (960, 540),  # 主点
            'camera_matrix': np.array([[1000, 0, 960],
                                     [0, 1000, 540],
                                     [0, 0, 1]]),
            'sample_rate': 30,           # 采样率（Hz）
            'calibration_points': 9      # 校准点数量
        }
    
    def _init_real_camera(self):
        """初始化真实摄像头（预留接口）"""
        try:
            # ==================== 在这里添加你的摄像头初始化代码 ====================
            
            # 示例1：初始化USB摄像头
            # self._init_usb_camera()
            
            # 示例2：初始化眼动仪
            # self._init_eyetracker()
            
            # 示例3：初始化串口连接
            # self._init_serial_connection()
            
            # ==================== 临时：显示初始化信息 ====================
            logging.info("真实摄像头模式已启用")
            logging.info(f"摄像头配置: {self.camera_config}")
            logging.info("注意：真实摄像头接口尚未完全实现，将回退到模拟数据")
            
        except Exception as e:
            logging.error(f"摄像头初始化失败: {e}")
            self.use_real_camera = False
            logging.info("回退到模拟数据模式")
    
    def _init_usb_camera(self):
        """初始化USB摄像头"""
        try:
            # import cv2  # 取消注释这行
            # self.cap = cv2.VideoCapture(0)  # 打开默认摄像头
            # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # if not self.cap.isOpened():
            #     raise Exception("无法打开USB摄像头")
            
            logging.info("USB摄像头初始化成功")
            
        except Exception as e:
            logging.error(f"USB摄像头初始化失败: {e}")
            raise
    
    def _init_eyetracker(self):
        """初始化眼动仪（如Tobii）"""
        try:
            # import tobii_research as tr  # 取消注释这行
            # eyetrackers = tr.find_all_eyetrackers()
            # if not eyetrackers:
            #     raise Exception("未找到眼动仪设备")
            
            # self.eyetracker = eyetrackers[0]
            # logging.info(f"眼动仪初始化成功: {self.eyetracker.address}")
            
            logging.info("眼动仪初始化成功")
            
        except Exception as e:
            logging.error(f"眼动仪初始化失败: {e}")
            raise
    
    def _init_serial_connection(self):
        """初始化串口连接"""
        try:
            # import serial  # 取消注释这行
            # self.serial_conn = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
            # logging.info("串口连接初始化成功")
            
            logging.info("串口连接初始化成功")
            
        except Exception as e:
            logging.error(f"串口连接初始化失败: {e}")
            raise
    
    def get_calibration_data(self, num_samples=10):
        """
        获取校准数据
        
        Args:
            num_samples: 样本数量
            
        Returns:
            dict: 包含eyes, pupils, target_pixels, K的数据字典
        """
        if self.use_real_camera:
            return self._get_real_camera_data(num_samples)
        else:
            return self._get_simulated_data(num_samples)
    
    def _get_simulated_data(self, num_samples):
        """获取模拟数据（用于测试）"""
        np.random.seed(42)
        
        # 模拟眼球中心和瞳孔中心
        eyes = np.random.randn(num_samples, 3) * 0.1
        pupils = np.random.randn(num_samples, 3) * 0.05
        
        # 使用配置中的相机参数
        K = self.camera_config['camera_matrix']
        
        # 模拟目标像素坐标（在校准点附近）
        resolution = self.camera_config['resolution']
        target_pixels = np.random.randint(0, min(resolution), (num_samples, 2))
        
        return {
            'eyes': eyes,
            'pupils': pupils,
            'target_pixels': target_pixels,
            'K': K,
            'data_type': 'simulated'
        }
    
    def _get_real_camera_data(self, num_samples):
        """
        获取真实摄像头数据
        
        ==================== 重要：在这里实现你的硬件接口 ====================
        
        你需要实现以下功能：
        1. 获取眼球中心坐标
        2. 获取瞳孔中心坐标
        3. 获取屏幕注视点
        4. 获取相机内参
        
        示例代码已经提供，请根据你的硬件进行修改
        """
        try:
            # ==================== 方法1：USB摄像头 + 图像处理 ====================
            if self.cap is not None:
                return self._get_usb_camera_data(num_samples)
            
            # ==================== 方法2：眼动仪数据 ====================
            elif self.eyetracker is not None:
                return self._get_eyetracker_data(num_samples)
            
            # ==================== 方法3：串口数据 ====================
            elif self.serial_conn is not None:
                return self._get_serial_data(num_samples)
            
            # ==================== 方法4：从文件读取数据 ====================
            else:
                return self._get_file_data(num_samples)
            
        except Exception as e:
            logging.error(f"获取真实摄像头数据失败: {e}")
            logging.info("回退到模拟数据模式")
            return self._get_simulated_data(num_samples)
    
    def _get_usb_camera_data(self, num_samples):
        """从USB摄像头获取数据"""
        try:
            # import cv2  # 取消注释这行
            
            # 获取图像
            # ret, frame = self.cap.read()
            # if not ret:
            #     raise Exception("无法从摄像头读取图像")
            
            # 检测眼球中心
            # eyes = self._detect_eye_centers(frame, num_samples)
            
            # 检测瞳孔中心
            # pupils = self._detect_pupil_centers(frame, num_samples)
            
            # 获取屏幕注视点
            # target_pixels = self._get_screen_gaze_targets(num_samples)
            
            # 获取相机内参
            # K = self._get_camera_intrinsics()
            
            # ==================== 临时：返回模拟数据 ====================
            logging.warning("USB摄像头接口尚未完全实现，返回模拟数据")
            return self._get_simulated_data(num_samples)
            
        except Exception as e:
            logging.error(f"USB摄像头数据获取失败: {e}")
            raise
    
    def _get_eyetracker_data(self, num_samples):
        """从眼动仪获取数据"""
        try:
            # import tobii_research as tr  # 取消注释这行
            
            # 获取眼动数据
            # gaze_data = self.eyetracker.get_gaze_data()
            
            # 处理数据
            # eyes = self._process_eyetracker_data(gaze_data, 'eye_center', num_samples)
            # pupils = self._process_eyetracker_data(gaze_data, 'pupil_center', num_samples)
            # target_pixels = self._process_eyetracker_data(gaze_data, 'gaze_point', num_samples)
            
            # ==================== 临时：返回模拟数据 ====================
            logging.warning("眼动仪接口尚未完全实现，返回模拟数据")
            return self._get_simulated_data(num_samples)
            
        except Exception as e:
            logging.error(f"眼动仪数据获取失败: {e}")
            raise
    
    def _get_serial_data(self, num_samples):
        """从串口获取数据"""
        try:
            # import serial  # 取消注释这行
            
            # 读取串口数据
            # data = self.serial_conn.readline().decode().strip()
            
            # 解析数据
            # eyes, pupils, target_pixels = self._parse_serial_data(data, num_samples)
            
            # ==================== 临时：返回模拟数据 ====================
            logging.warning("串口接口尚未完全实现，返回模拟数据")
            return self._get_simulated_data(num_samples)
            
        except Exception as e:
            logging.error(f"串口数据获取失败: {e}")
            raise
    
    def _get_file_data(self, num_samples):
        """从文件读取数据（支持JSON格式）"""
        try:
            # 优先检查JSON文件
            json_file = "eye_tracking_data.json"
            npy_file = "eye_tracking_data.npy"
            
            if os.path.exists(json_file):
                # 从JSON文件加载数据
                data = self._load_json_data(json_file)
                logging.info(f"从JSON文件加载数据: {json_file}")
                return data
            elif os.path.exists(npy_file):
                # 从NPY文件加载数据
                data = np.load(npy_file, allow_pickle=True).item()
                logging.info(f"从NPY文件加载数据: {npy_file}")
                return data
            else:
                # 创建示例JSON数据文件
                self._create_sample_json_file(json_file)
                logging.info(f"创建示例JSON数据文件: {json_file}")
                return self._get_simulated_data(num_samples)
                
        except Exception as e:
            logging.error(f"文件数据获取失败: {e}")
            return self._get_simulated_data(num_samples)
    
    def _load_json_data(self, json_file):
        """从JSON文件加载眼动追踪数据"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 验证JSON数据格式
            if not self._validate_json_data(data):
                raise ValueError("JSON数据格式不正确")
            
            # 转换JSON数据为numpy数组
            converted_data = self._convert_json_to_numpy(data)
            
            return converted_data
            
        except Exception as e:
            logging.error(f"JSON文件读取失败: {e}")
            raise
    
    def _validate_json_data(self, data):
        """验证JSON数据格式"""
        required_keys = ['eyes', 'pupils', 'target_pixels', 'camera_matrix']
        
        # 检查必需字段
        for key in required_keys:
            if key not in data:
                logging.error(f"缺少必需字段: {key}")
                return False
        
        # 检查数据类型
        if not isinstance(data['eyes'], list) or not isinstance(data['pupils'], list):
            logging.error("eyes和pupils必须是数组")
            return False
        
        if not isinstance(data['target_pixels'], list):
            logging.error("target_pixels必须是数组")
            return False
        
        if not isinstance(data['camera_matrix'], list):
            logging.error("camera_matrix必须是数组")
            return False
        
        return True
    
    def _convert_json_to_numpy(self, json_data):
        """将JSON数据转换为numpy数组格式"""
        try:
            # 转换眼球中心坐标
            eyes = np.array(json_data['eyes'], dtype=np.float32)
            
            # 转换瞳孔中心坐标
            pupils = np.array(json_data['pupils'], dtype=np.float32)
            
            # 转换目标像素坐标
            target_pixels = np.array(json_data['target_pixels'], dtype=np.float32)
            
            # 转换相机内参矩阵
            camera_matrix = np.array(json_data['camera_matrix'], dtype=np.float32)
            
            # 检查数据维度
            if eyes.shape[1] != 3 or pupils.shape[1] != 3:
                raise ValueError("眼球和瞳孔坐标必须是3D坐标")
            
            if target_pixels.shape[1] != 2:
                raise ValueError("目标像素坐标必须是2D坐标")
            
            if camera_matrix.shape != (3, 3):
                raise ValueError("相机内参矩阵必须是3x3")
            
            # 检查数据一致性
            if not (eyes.shape[0] == pupils.shape[0] == target_pixels.shape[0]):
                raise ValueError("所有数据数组的样本数量必须一致")
            
            return {
                'eyes': eyes,
                'pupils': pupils,
                'target_pixels': target_pixels,
                'K': camera_matrix,
                'data_type': 'json_file'
            }
            
        except Exception as e:
            logging.error(f"JSON数据转换失败: {e}")
            raise
    
    def _create_sample_json_file(self, filename):
        """创建示例JSON数据文件"""
        sample_data = {
            "metadata": {
                "description": "眼动追踪校准数据示例",
                "created_at": datetime.now().isoformat(),
                "data_format": "JSON",
                "coordinate_system": "camera_coordinates_mm",
                "sample_count": 10
            },
            "eyes": [
                [0.0, 0.0, 0.0],      # 眼球中心1 (mm)
                [0.1, 0.0, 0.0],      # 眼球中心2
                [0.0, 0.1, 0.0],      # 眼球中心3
                [-0.1, 0.0, 0.0],     # 眼球中心4
                [0.0, -0.1, 0.0],     # 眼球中心5
                [0.05, 0.05, 0.0],    # 眼球中心6
                [-0.05, 0.05, 0.0],   # 眼球中心7
                [0.05, -0.05, 0.0],   # 眼球中心8
                [-0.05, -0.05, 0.0],  # 眼球中心9
                [0.0, 0.0, 0.1]       # 眼球中心10
            ],
            "pupils": [
                [0.0, 0.0, 0.1],      # 瞳孔中心1 (mm)
                [0.1, 0.0, 0.1],      # 瞳孔中心2
                [0.0, 0.1, 0.1],      # 瞳孔中心3
                [-0.1, 0.0, 0.1],     # 瞳孔中心4
                [0.0, -0.1, 0.1],     # 瞳孔中心5
                [0.05, 0.05, 0.1],    # 瞳孔中心6
                [-0.05, 0.05, 0.1],   # 瞳孔中心7
                [0.05, -0.05, 0.1],   # 瞳孔中心8
                [-0.05, -0.05, 0.1],  # 瞳孔中心9
                [0.0, 0.0, 0.2]       # 瞳孔中心10
            ],
            "target_pixels": [
                [100, 100],            # 屏幕目标点1 (像素)
                [960, 100],            # 屏幕目标点2
                [1820, 100],           # 屏幕目标点3
                [100, 540],            # 屏幕目标点4
                [960, 540],            # 屏幕目标点5
                [1820, 540],           # 屏幕目标点6
                [100, 980],            # 屏幕目标点7
                [960, 980],            # 屏幕目标点8
                [1820, 980],           # 屏幕目标点9
                [960, 540]             # 屏幕目标点10
            ],
            "camera_matrix": [
                [1000.0, 0.0, 960.0],   # 相机内参矩阵
                [0.0, 1000.0, 540.0],   # fx, fy, cx, cy
                [0.0, 0.0, 1.0]
            ],
            "screen_resolution": [1920, 1080],
            "units": {
                "coordinates": "millimeters",
                "pixels": "pixels",
                "angles": "degrees"
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)
        
        logging.info(f"示例JSON数据文件已创建: {filename}")
    
    def save_data_to_json(self, data, filename="eye_tracking_data.json"):
        """将数据保存为JSON格式"""
        try:
            # 转换numpy数组为Python列表
            json_data = {
                "metadata": {
                    "description": "眼动追踪校准数据",
                    "created_at": datetime.now().isoformat(),
                    "data_format": "JSON",
                    "coordinate_system": "camera_coordinates_mm",
                    "sample_count": len(data['eyes'])
                },
                "eyes": data['eyes'].tolist(),
                "pupils": data['pupils'].tolist(),
                "target_pixels": data['target_pixels'].tolist(),
                "camera_matrix": data['K'].tolist(),
                "screen_resolution": self.camera_config['resolution'],
                "units": {
                    "coordinates": "millimeters",
                    "pixels": "pixels",
                    "angles": "degrees"
                }
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            logging.info(f"数据已保存到JSON文件: {filename}")
            return True
            
        except Exception as e:
            logging.error(f"保存JSON文件失败: {e}")
            return False
    
    def _detect_eye_centers(self, frame, num_samples):
        """
        从图像中检测眼球中心
        
        Args:
            frame: 摄像头图像
            num_samples: 需要的样本数量
            
        Returns:
            np.ndarray: 形状为 (num_samples, 3) 的眼球中心坐标
        """
        # ==================== 在这里实现你的眼球检测算法 ====================
        
        # 示例1：简单的颜色阈值检测
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # # 你的检测代码...
        
        # 示例2：使用深度学习模型
        # model = self._load_eye_detection_model()
        # results = model(frame)
        # # 处理结果...
        
        # 示例3：使用OpenCV的Haar级联分类器
        # eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        # eyes = eye_cascade.detectMultiScale(gray)
        # # 处理检测结果...
        
        # ==================== 临时：返回模拟数据 ====================
        logging.warning("眼球检测算法尚未实现，返回模拟数据")
        return np.random.randn(num_samples, 3) * 0.1
    
    def _detect_pupil_centers(self, frame, num_samples):
        """
        从图像中检测瞳孔中心
        
        Args:
            frame: 摄像头图像
            num_samples: 需要的样本数量
            
        Returns:
            np.ndarray: 形状为 (num_samples, 3) 的瞳孔中心坐标
        """
        # ==================== 在这里实现你的瞳孔检测算法 ====================
        
        # 示例1：Hough圆检测
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
        #                           param1=50, param2=30, minRadius=0, maxRadius=0)
        # # 处理检测结果...
        
        # 示例2：轮廓检测
        # _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # # 处理轮廓...
        
        # ==================== 临时：返回模拟数据 ====================
        logging.warning("瞳孔检测算法尚未实现，返回模拟数据")
        return np.random.randn(num_samples, 3) * 0.05
    
    def _get_screen_gaze_targets(self, num_samples):
        """
        获取屏幕上的注视点坐标
        
        Args:
            num_samples: 需要的样本数量
            
        Returns:
            np.ndarray: 形状为 (num_samples, 2) 的屏幕坐标
        """
        # ==================== 在这里实现你的注视点获取 ====================
        
        # 示例1：9点校准模式
        calibration_points = [
            [100, 100], [960, 100], [1820, 100],
            [100, 540], [960, 540], [1820, 540],
            [100, 980], [960, 980], [1820, 980]
        ]
        
        # 随机选择校准点
        if num_samples <= len(calibration_points):
            selected_indices = np.random.choice(len(calibration_points), num_samples, replace=False)
            return np.array([calibration_points[i] for i in selected_indices])
        else:
            # 如果需要的样本数超过校准点数，重复选择
            return np.array(calibration_points * (num_samples // len(calibration_points) + 1))[:num_samples]
    
    def _get_camera_intrinsics(self):
        """
        获取相机内参
        
        Returns:
            np.ndarray: 3x3相机内参矩阵
        """
        # ==================== 在这里实现你的相机标定 ====================
        
        # 示例1：使用预定义的参数
        # return self.camera_config['camera_matrix']
        
        # 示例2：从文件加载标定结果
        # calib_file = "camera_calibration.npy"
        # if os.path.exists(calib_file):
        #     return np.load(calib_file)
        
        # 示例3：实时标定
        # return self._calibrate_camera_realtime()
        
        # ==================== 临时：返回默认参数 ====================
        return self.camera_config['camera_matrix']
    
    def _calibrate_camera_realtime(self):
        """实时相机标定"""
        # ==================== 在这里实现实时标定 ====================
        
        # 示例：使用OpenCV的相机标定
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # objp = np.zeros((6*7,3), np.float32)
        # objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
        # objpoints = []
        # imgpoints = []
        # # 标定过程...
        
        # ==================== 临时：返回默认参数 ====================
        return self.camera_config['camera_matrix']
    
    def update_camera_config(self, new_config):
        """更新摄像头配置"""
        self.camera_config.update(new_config)
        if 'camera_matrix' in new_config:
            # 重新计算相机内参矩阵
            focal_length = new_config.get('focal_length', self.camera_config['focal_length'])
            principal_point = new_config.get('principal_point', self.camera_config['principal_point'])
            
            self.camera_config['camera_matrix'] = np.array([
                [focal_length, 0, principal_point[0]],
                [0, focal_length, principal_point[1]],
                [0, 0, 1]
            ])
        
        logging.info(f"摄像头配置已更新: {self.camera_config}")
    
    def switch_to_real_camera(self):
        """切换到真实摄像头模式"""
        self.use_real_camera = True
        self._init_real_camera()
    
    def switch_to_simulation(self):
        """切换到模拟数据模式"""
        self.use_real_camera = False
        logging.info("已切换到模拟数据模式")
    
    def get_camera_info(self):
        """获取摄像头信息"""
        return {
            'mode': 'real_camera' if self.use_real_camera else 'simulation',
            'config': self.camera_config.copy(),
            'status': 'ready',
            'hardware_connected': any([self.cap, self.eyetracker, self.serial_conn])
        }
    
    def cleanup(self):
        """清理资源"""
        if self.cap is not None:
            # self.cap.release()
            self.cap = None
        
        if self.eyetracker is not None:
            # self.eyetracker.disconnect()
            self.eyetracker = None
        
        if self.serial_conn is not None:
            # self.serial_conn.close()
            self.serial_conn = None
        
        logging.info("硬件资源已清理")
    
    def __del__(self):
        """析构函数，确保资源被清理"""
        self.cleanup()
