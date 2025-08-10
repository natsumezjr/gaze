import numpy as np
import json
from typing import Dict, Optional, Tuple
import cv2
import os

class CameraCalibrator:
    """
    相机标定单例类
    负责相机参数的加载、验证和调整
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(CameraCalibrator, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, file_path: str = None, rgb_d=False):
        if self._initialized:
            return
        
        self._rgb_d = rgb_d
        
        # 设置配置文件路径
        if file_path is None:
            from project.recognition.config.settings import CAMERA_PARAMS_PATH
            self._config_file_path = CAMERA_PARAMS_PATH
        else:
            self._config_file_path = file_path
        
        # 初始化摄像头
        self._cap = self._initialize_camera()
        if self._cap is None:
            print("警告：无法初始化摄像头")
            
        
        
        self._initialized = True
    
    
    # ==================== 公共接口 ====================
    def get_cap(self)->cv2.VideoCapture:
        return self._cap
    
    
    def load_camera_params(self) -> Dict:
        """
        加载相机参数并返回最终字典
        
        :return: 验证和调整后的相机参数字典
        """
        try:
            if not self._rgb_d:
                config = self._get_default_config()
            else:
                # 加载配置文件
                if os.path.exists(self._config_file_path):
                    with open(self._config_file_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    print(f"从配置文件加载参数: {self._config_file_path}")
                else:
                    print("配置文件不存在，使用默认参数")
                    config = self._get_default_config()
            
            # 验证和调整参数
            validated_config = self._validate_and_adjust_params(config)
            
            # 保存调整后的参数
            self._save_config(validated_config)
            
            return validated_config
            
        except Exception as e:
            print(f"加载相机参数失败: {e}")
            return self._get_default_config()
    
    def get_image_resolution(self) -> Tuple[int, int]:
        """
        获取图像分辨率
        
        :return: (width, height) 元组
        """
        if self._cap is None:
            return (640, 480)  # 默认分辨率
        
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)
    
    def get_intrinsics(self) -> Dict:
        """
        获取相机内参
        
        :return: 内参字典 {"fx": float, "fy": float, "cx": float, "cy": float}
        """
        config = self.load_camera_params()
        return config.get("intrinsic_params", {})
    
    def get_depth_scale(self) -> float:
        """
        获取深度缩放因子
        
        :return: 深度缩放因子
        """
        config = self.load_camera_params()
        return config.get("depth_scale", 0.001)
    
    # ==================== 私有方法 ====================
    
    def _initialize_camera(self) -> cv2.VideoCapture:
        """
        初始化摄像头
        
        :return: 摄像头对象
        """
        
        if self._rgb_d:
            cap = self._open_rgb_d()
        else:
            cap = cv2.VideoCapture(0)
            
        return cap
    
    def _open_rgb_d(self)->cv2.VideoCapture:
        cap = cv2.VideoCapture(0)
        return cap
    
    def _validate_and_adjust_params(self, config: Dict) -> Dict:
        """
        检查并修改字典，返回修改后的字典
        
        :param config: 原始配置字典
        :return: 验证和调整后的配置字典
        """
        print("开始验证和调整相机参数...")
        
        # 1. 字段约定检查
        config = self._check_required_fields(config)
        
        # 2. 分辨率检查
        config = self._check_and_adjust_resolution(config)
        
        # 3. 内参检查
        config = self._check_and_adjust_intrinsics(config)
        
        # 4. 其他参数检查
        config = self._check_other_params(config)
        
        print("参数验证和调整完成")
        return config
    
    def _check_required_fields(self, config: Dict) -> Dict:
        """
        检查必需字段
        
        :param config: 配置字典
        :return: 调整后的配置字典
        """
        required_fields = [
            "camera_name", "camera_type", "intrinsic_params", 
            "image_resolution", "depth_scale"
        ]
        
        for field in required_fields:
            if field not in config:
                print(f"缺少必需字段: {field}")
                if field == "camera_name":
                    config[field] = "Auto Detected Camera"
                elif field == "camera_type":
                    config[field] = "RGB"
                elif field == "intrinsic_params":
                    config[field] = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}
                elif field == "image_resolution":
                    config[field] = {"width": 640, "height": 480}
                elif field == "depth_scale":
                    config[field] = 0.001
        
        return config
    
    def _check_and_adjust_resolution(self, config: Dict) -> Dict:
        """
        检查并调整分辨率
        
        :param config: 配置字典
        :return: 调整后的配置字典
        """
        if self._cap is None:
            print("摄像头未初始化，跳过分辨率检查")
            return config
        
        # 获取实际分辨率
        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        config_resolution = config.get("image_resolution", {})
        config_width = config_resolution.get("width", 0)
        config_height = config_resolution.get("height", 0)
        
        print(f"配置分辨率: {config_width}x{config_height}")
        print(f"实际分辨率: {actual_width}x{actual_height}")
        
        # 检查分辨率是否匹配
        if config_width != actual_width or config_height != actual_height:
            print("分辨率不匹配，进行调整")
            config["image_resolution"] = {
                "width": actual_width,
                "height": actual_height
            }
            
            # 同时调整内参
            config = self._adjust_intrinsics_for_resolution(config, actual_width, actual_height)
        
        return config
    
    def _check_and_adjust_intrinsics(self, config: Dict) -> Dict:
        """
        检查并调整内参
        
        :param config: 配置字典
        :return: 调整后的配置字典
        """
        intrinsics = config.get("intrinsic_params", {})
        
        # 检查内参字段
        required_intrinsics = ["fx", "fy", "cx", "cy"]
        for param in required_intrinsics:
            if param not in intrinsics or intrinsics[param] <= 0:
                print(f"内参 {param} 无效，使用默认值")
                if param in ["fx", "fy"]:
                    intrinsics[param] = 500.0
                elif param == "cx":
                    intrinsics[param] = config["image_resolution"]["width"] / 2.0
                elif param == "cy":
                    intrinsics[param] = config["image_resolution"]["height"] / 2.0
        
        # 检查内参合理性
        width = config["image_resolution"]["width"]
        height = config["image_resolution"]["height"]
        
        # 焦距应该在合理范围内
        if intrinsics["fx"] < width * 0.1 or intrinsics["fx"] > width * 2.0:
            print(f"焦距fx={intrinsics['fx']}超出合理范围，进行调整")
            intrinsics["fx"] = width * 0.7
        
        if intrinsics["fy"] < height * 0.1 or intrinsics["fy"] > height * 2.0:
            print(f"焦距fy={intrinsics['fy']}超出合理范围，进行调整")
            intrinsics["fy"] = height * 0.7
        
        # 主点应该在图像范围内
        if intrinsics["cx"] < 0 or intrinsics["cx"] > width:
            print(f"主点cx={intrinsics['cx']}超出图像范围，进行调整")
            intrinsics["cx"] = width / 2.0
        
        if intrinsics["cy"] < 0 or intrinsics["cy"] > height:
            print(f"主点cy={intrinsics['cy']}超出图像范围，进行调整")
            intrinsics["cy"] = height / 2.0
        
        config["intrinsic_params"] = intrinsics
        return config
    
    def _adjust_intrinsics_for_resolution(self, config: Dict, width: int, height: int) -> Dict:
        """
        根据分辨率调整内参
        
        :param config: 配置字典
        :param width: 实际宽度
        :param height: 实际高度
        :return: 调整后的配置字典
        """
        # 预设参数
        preset_params = {
            "640x480": {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0},
            "1280x720": {"fx": 1000.0, "fy": 1000.0, "cx": 640.0, "cy": 360.0},
            "1920x1080": {"fx": 1500.0, "fy": 1500.0, "cx": 960.0, "cy": 540.0}
        }
        
        resolution = f"{width}x{height}"
        
        if resolution in preset_params:
            config["intrinsic_params"] = preset_params[resolution]
            print(f"使用预设内参: {preset_params[resolution]}")
        else:
            # 估算内参
            config["intrinsic_params"] = {
                "fx": width * 0.7,
                "fy": height * 0.7,
                "cx": width / 2.0,
                "cy": height / 2.0
            }
            print(f"使用估算内参: {config['intrinsic_params']}")
        
        return config
    
    def _check_other_params(self, config: Dict) -> Dict:
        """
        检查其他参数
        
        :param config: 配置字典
        :return: 调整后的配置字典
        """
        # 检查深度缩放因子
        if "depth_scale" not in config or config["depth_scale"] <= 0:
            config["depth_scale"] = 0.001
            print("设置默认深度缩放因子: 0.001")
        
        # 检查深度范围
        if "min_depth" not in config or config["min_depth"] < 0:
            config["min_depth"] = 0.1
            print("设置默认最小深度: 0.1")
        
        if "max_depth" not in config or config["max_depth"] <= config["min_depth"]:
            config["max_depth"] = 10.0
            print("设置默认最大深度: 10.0")
        
        # 检查畸变系数
        if "distortion_coeffs" not in config:
            config["distortion_coeffs"] = {
                "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0
            }
            print("设置默认畸变系数")
        
        return config
    
    def _get_default_config(self) -> Dict:
        """
        获取默认配置 - 通过PowerShell获取相机信息
        
        :return: 默认配置字典
        """
        try:
            # 通过PowerShell获取相机信息
            camera_info = self._get_camera_info_from_powershell()
            
            # 获取实际分辨率
            width, height = self.get_image_resolution()
            
            # 根据相机类型和分辨率设置参数
            if camera_info:
                print(f"检测到相机: {camera_info}")
                config = self._create_config_from_camera_info(camera_info, width, height)
            else:
                print("无法获取相机信息，使用通用预设")
                config = self._create_generic_config(width, height)
            
            return config
            
        except Exception as e:
            print(f"获取默认配置失败: {e}")
            return self._create_generic_config(640, 480)
        
    def _get_camera_info_from_powershell(self) -> Optional[Dict]:
        """
        通过PowerShell获取相机信息
        
        :return: 相机信息字典
        """
        try:
            import subprocess
            
            # 获取相机设备信息
            cmd = [
                'powershell', 
                'Get-WmiObject -Class Win32_PnPEntity | Where-Object {$_.Name -like "*camera*" -or $_.Name -like "*webcam*" -or $_.Name -like "*USB Video*"} | Select-Object Name, DeviceID, Manufacturer, Description | ConvertTo-Json'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and result.stdout.strip():
                import json
                cameras = json.loads(result.stdout)
                
                # 如果是单个相机，转换为列表
                if isinstance(cameras, dict):
                    cameras = [cameras]
                
                if cameras:
                    # 选择第一个相机
                    camera = cameras[0]
                    return {
                        "name": camera.get("Name", "Unknown Camera"),
                        "device_id": camera.get("DeviceID", ""),
                        "manufacturer": camera.get("Manufacturer", "Unknown"),
                        "description": camera.get("Description", "")
                    }
            
            return None
            
        except Exception as e:
            print(f"PowerShell获取相机信息失败: {e}")
            return None
        
    def _create_config_from_camera_info(self, camera_info: Dict, width: int, height: int) -> Dict:
        """
        根据相机信息创建配置
        
        :param camera_info: 相机信息
        :param width: 图像宽度
        :param height: 图像高度
        :return: 配置字典
        """
        camera_name = camera_info.get("name", "Unknown Camera")
        manufacturer = camera_info.get("manufacturer", "Unknown")
        
        # 根据制造商和分辨率设置预设参数
        preset_params = self._get_preset_params_by_manufacturer(manufacturer, width, height)
        
        return {
            "camera_name": camera_name,
            "camera_type": "RGB",
            "intrinsic_params": preset_params,
            "image_resolution": {
                "width": width,
                "height": height
            },
            "depth_scale": 0.001,
            "min_depth": 0.1,
            "max_depth": 10.0,
            "distortion_coeffs": {
                "k1": 0.0,
                "k2": 0.0,
                "p1": 0.0,
                "p2": 0.0,
                "k3": 0.0
            },
            "calibration_date": "2024-01-15",
            "calibration_method": "PowerShell Auto Detection",
            "notes": f"通过PowerShell自动检测获得，制造商: {manufacturer}"
        }
        
    def _get_preset_params_by_manufacturer(self, manufacturer: str, width: int, height: int) -> Dict:
        """
        根据制造商和分辨率获取预设参数
        
        :param manufacturer: 制造商
        :param width: 图像宽度
        :param height: 图像高度
        :return: 内参字典
        """
        # 制造商特定的预设参数
        manufacturer_presets = {
            "Logitech": {
                "640x480": {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0},
                "1280x720": {"fx": 1000.0, "fy": 1000.0, "cx": 640.0, "cy": 360.0},
                "1920x1080": {"fx": 1500.0, "fy": 1500.0, "cx": 960.0, "cy": 540.0}
            },
            "Microsoft": {
                "640x480": {"fx": 450.0, "fy": 450.0, "cx": 320.0, "cy": 240.0},
                "1280x720": {"fx": 900.0, "fy": 900.0, "cx": 640.0, "cy": 360.0},
                "1920x1080": {"fx": 1350.0, "fy": 1350.0, "cx": 960.0, "cy": 540.0}
            },
            "Intel": {
                "640x480": {"fx": 462.5, "fy": 616.7, "cx": 320.0, "cy": 240.0},
                "1280x720": {"fx": 925.0, "fy": 1233.4, "cx": 640.0, "cy": 360.0},
                "1920x1080": {"fx": 1387.5, "fy": 1850.1, "cx": 960.0, "cy": 540.0}
            },
            "Generic": {
                "640x480": {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0},
                "1280x720": {"fx": 1000.0, "fy": 1000.0, "cx": 640.0, "cy": 360.0},
                "1920x1080": {"fx": 1500.0, "fy": 1500.0, "cx": 960.0, "cy": 540.0}
            }
        }
        
        # 确定制造商
        if "Logitech" in manufacturer:
            manufacturer_key = "Logitech"
        elif "Microsoft" in manufacturer:
            manufacturer_key = "Microsoft"
        elif "Intel" in manufacturer:
            manufacturer_key = "Intel"
        else:
            manufacturer_key = "Generic"
        
        # 确定分辨率
        resolution = f"{width}x{height}"
        
        # 获取预设参数
        if manufacturer_key in manufacturer_presets and resolution in manufacturer_presets[manufacturer_key]:
            params = manufacturer_presets[manufacturer_key][resolution]
            print(f"使用{manufacturer_key}的{resolution}预设参数: {params}")
            return params
        else:
            # 使用通用预设
            params = manufacturer_presets["Generic"].get(resolution, {
                "fx": width * 0.7,
                "fy": height * 0.7,
                "cx": width / 2.0,
                "cy": height / 2.0
            })
            print(f"使用通用{resolution}预设参数: {params}")
            return params
        
    def _create_generic_config(self, width: int, height: int) -> Dict:
        """
        创建通用配置
        
        :param width: 图像宽度
        :param height: 图像高度
        :return: 配置字典
        """
        return {
            "camera_name": "Generic USB Camera",
            "camera_type": "RGB",
            "intrinsic_params": {
                "fx": width * 0.7,
                "fy": height * 0.7,
                "cx": width / 2.0,
                "cy": height / 2.0
            },
            "image_resolution": {
                "width": width,
                "height": height
            },
            "depth_scale": 0.001,
            "min_depth": 0.1,
            "max_depth": 10.0,
            "distortion_coeffs": {
                "k1": 0.0,
                "k2": 0.0,
                "p1": 0.0,
                "p2": 0.0,
                "k3": 0.0
            },
            "calibration_date": "2024-01-15",
            "calibration_method": "Generic Preset",
            "notes": "使用通用预设参数"
        }
    
    def _save_config(self, config: Dict) -> None:
        """
        保存配置到文件
        
        :param config: 配置字典
        """
        try:
            os.makedirs(os.path.dirname(self._config_file_path), exist_ok=True)
            with open(self._config_file_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            print(f"配置已保存到: {self._config_file_path}")
        except Exception as e:
            print(f"保存配置失败: {e}")
    
    def __del__(self):
        """
        析构函数，释放摄像头资源
        """
        if hasattr(self, '_cap') and self._cap is not None:
            self._cap.release()

# ==================== 测试接口 ====================

def main():
    """
    测试接口的main函数
    """
    print("=== 相机参数加载测试 ===")
    
    # 创建相机标定器实例
    print("\n开始相机参数加载测试...")
    calibrator = CameraCalibrator(rgb_d=True)
    
    # 加载相机参数
    camera_params = calibrator.load_camera_params()
    
    if camera_params:
        print("✓ 相机参数加载成功")
        print(f"相机名称: {camera_params.get('camera_name')}")
        print(f"图像分辨率: {calibrator.get_image_resolution()}")
        print(f"内参: {calibrator.get_intrinsics()}")
        print(f"深度缩放因子: {calibrator.get_depth_scale()}")
    else:
        print("✗ 相机参数加载失败")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    main()