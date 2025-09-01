# 眼动追踪摄像头数据集成指南

## 🎯 **概述**

这个项目已经为你准备好了接入真实摄像头的接口！现在你可以：

1. **使用模拟数据进行测试**（默认模式）
2. **切换到真实摄像头模式**（预留接口）
3. **自定义摄像头参数**（焦距、主点等）

## 🚀 **快速开始**

### 运行程序
```bash
cd /Users/andrew/PycharmProjects/大创/gaze
python -m project.fitting.main
```

### 选择数据模式
程序运行后会提示你选择：
```
摄像头配置选项:
1. 使用模拟数据（默认）
2. 使用真实摄像头
3. 自定义摄像头配置
```

## 🔧 **集成真实摄像头**

### 1. **修改数据接口类**

编辑文件：`project/fitting/core/eye_tracking_data_interface.py`

在 `_get_real_camera_data()` 方法中实现你的硬件接口：

```python
def _get_real_camera_data(self, num_samples):
    """获取真实摄像头数据"""
    try:
        # ==================== 在这里实现你的硬件接口 ====================
        
        # 示例1：OpenCV摄像头
        # import cv2
        # cap = cv2.VideoCapture(0)  # 打开摄像头
        # ret, frame = cap.read()
        # if ret:
        #     # 处理图像，检测眼球和瞳孔
        #     eyes = self._detect_eye_centers(frame, num_samples)
        #     pupils = self._detect_pupil_centers(frame, num_samples)
        
        # 示例2：Tobii眼动仪
        # import tobii_research as tr
        # eyetracker = tr.EyeTracker("your_device_address")
        # gaze_data = eyetracker.get_gaze_data()
        # eyes = self._process_tobii_data(gaze_data, 'eye_center')
        # pupils = self._process_tobii_data(gaze_data, 'pupil_center')
        
        # 示例3：自定义硬件
        # eyes = self._read_from_custom_hardware('eye_centers')
        # pupils = self._read_from_custom_hardware('pupil_centers')
        
        # 获取屏幕注视点
        target_pixels = self._get_screen_gaze_targets(num_samples)
        
        # 获取相机内参
        K = self._get_camera_intrinsics()
        
        return {
            'eyes': eyes,
            'pupils': pupils,
            'target_pixels': target_pixels,
            'K': K,
            'data_type': 'real_camera'
        }
        
    except Exception as e:
        logging.error(f"获取真实摄像头数据失败: {e}")
        return self._get_simulated_data(num_samples)
```

### 2. **实现具体的方法**

#### **眼球中心检测**
```python
def _detect_eye_centers(self, frame, num_samples):
    """从图像中检测眼球中心"""
    # 这里实现你的眼球检测算法
    # 可以使用OpenCV、深度学习模型等
    
    # 示例：简单的颜色阈值检测
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # # 你的检测代码...
    
    # 返回3D坐标数组 (num_samples, 3)
    return np.array([[x1, y1, z1], [x2, y2, z2], ...])
```

#### **瞳孔中心检测**
```python
def _detect_pupil_centers(self, frame, num_samples):
    """从图像中检测瞳孔中心"""
    # 实现瞳孔检测算法
    # 可以使用Hough圆检测、轮廓检测等
    
    # 返回3D坐标数组 (num_samples, 3)
    return np.array([[x1, y1, z1], [x2, y2, z2], ...])
```

#### **屏幕注视点获取**
```python
def _get_screen_gaze_targets(self, num_samples):
    """获取屏幕上的注视点坐标"""
    # 这里需要与你的校准程序同步
    # 可以是预定义的点，或者实时获取
    
    # 示例：9点校准
    calibration_points = [
        [100, 100], [960, 100], [1820, 100],
        [100, 540], [960, 540], [1820, 540],
        [100, 980], [960, 980], [1820, 980]
    ]
    
    # 随机选择或按顺序选择
    selected_points = np.random.choice(len(calibration_points), num_samples, replace=False)
    return np.array([calibration_points[i] for i in selected_points])
```

### 3. **相机标定**

如果你的相机内参未知，需要先进行标定：

```python
def _calibrate_camera(self):
    """相机标定"""
    # 使用OpenCV的相机标定功能
    # import cv2
    
    # 1. 准备标定板图像
    # 2. 检测角点
    # 3. 计算相机矩阵
    # 4. 保存标定结果
    
    # 示例代码：
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(...)
    
    pass
```

## 📊 **数据格式要求**

### **眼球中心坐标**
- 格式：`np.ndarray`，形状 `(num_samples, 3)`
- 单位：毫米（mm）
- 坐标系：相机坐标系，原点在相机光心

### **瞳孔中心坐标**
- 格式：`np.ndarray`，形状 `(num_samples, 3)`
- 单位：毫米（mm）
- 坐标系：相机坐标系

### **屏幕注视点**
- 格式：`np.ndarray`，形状 `(num_samples, 2)`
- 单位：像素
- 坐标系：屏幕坐标系，左上角为原点

### **相机内参矩阵**
- 格式：`np.ndarray`，形状 `(3, 3)`
- 标准形式：
```python
K = np.array([
    [fx, 0,  cx],  # fx, fy: 焦距，cx, cy: 主点
    [0,  fy, cy],
    [0,  0,  1]
])
```

## 🔄 **测试和调试**

### 1. **先用模拟数据测试**
确保kappa校准算法工作正常

### 2. **逐步替换真实数据**
- 先替换眼球中心
- 再替换瞳孔中心
- 最后替换注视点

### 3. **验证数据质量**
- 检查坐标范围是否合理
- 验证数据一致性
- 监控kappa角估计结果

## 🛠 **常见硬件接口**

### **USB摄像头**
```python
import cv2

def setup_usb_camera(self):
    self.cap = cv2.VideoCapture(0)
    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

### **Tobii眼动仪**
```python
import tobii_research as tr

def setup_tobii(self):
    eyetrackers = tr.find_all_eyetrackers()
    self.eyetracker = eyetrackers[0]
    self.eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, self.on_gaze_data)
```

### **自定义硬件**
```python
import serial

def setup_serial_connection(self):
    self.ser = serial.Serial('/dev/ttyUSB0', 115200)
```

## 📝 **配置文件**

你可以创建配置文件来管理不同的摄像头设置：

```python
# camera_config.json
{
    "usb_camera": {
        "device_id": 0,
        "resolution": [640, 480],
        "focal_length": 800,
        "principal_point": [320, 240]
    },
    "tobii": {
        "device_address": "your_device_address",
        "sample_rate": 60,
        "focal_length": 1000,
        "principal_point": [960, 540]
    }
}
```

## 🎉 **完成集成后**

一旦你实现了真实摄像头接口：

1. **选择模式2**：使用真实摄像头
2. **程序会自动调用**你的硬件接口
3. **获取真实数据**进行kappa校准
4. **得到准确的**个人kappa角参数

## 🆘 **遇到问题？**

1. **检查数据格式**：确保返回的数据符合要求
2. **验证坐标系统**：确保所有坐标使用相同的参考系
3. **调试硬件接口**：单独测试每个数据获取方法
4. **查看日志输出**：程序会显示详细的调试信息

## 🚀 **下一步**

完成摄像头集成后，你可以：
- 进行真实的个人kappa校准
- 收集更多用户数据
- 优化算法参数
- 开发实时眼动追踪应用

祝你集成顺利！🎯
