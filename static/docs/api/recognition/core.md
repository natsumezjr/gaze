# 识别模块数据结构接口

## 主要数据结构

### 1. 图像数据
- **BGR图像**：`np.ndarray`，形状`(H, W, 3)`，类型`np.uint8`
- **格式**：✅ **BGR格式统一**（OpenCV默认格式）
- **单位**：像素值范围 0-255

**示例：**
```python
# BGR图像示例
bgr_image = np.ndarray(
    data=[[[255, 0, 0], [0, 255, 0], [0, 0, 255]],  # 第一行：蓝、绿、红
         [[128, 64, 32], [64, 128, 64], [32, 64, 128]]],  # 第二行：混合色
    shape=(2, 3, 3),  # 高度=2, 宽度=3, 通道=3
    dtype=np.uint8  # 无符号8位整数
)
```

### 2. 深度图
- **类型**：`np.ndarray`，形状`(H, W)`
- **单位**：✅ **深度图单位统一（米）**
- **类型**：`np.float32`

**示例：**
```python
# 深度图示例
depth_map = np.ndarray(
    data=[[0.5, 0.6, 0.7],  # 第一行深度值（米）
          [0.4, 0.5, 0.6],  # 第二行深度值（米）
          [0.3, 0.4, 0.5]], # 第三行深度值（米）
    shape=(3, 3),  # 高度=3, 宽度=3
    dtype=np.float32  # 32位浮点数
)
```

### 3. 关键点
- **单个关键点**：`List[float]` [x, y, z]
- **关键点列表**：`List[List[float]]`
- **关键点字典**：`Dict[str, List[float]]`

**示例：**
```python
# 单个关键点示例
single_landmark = [640.5, 360.2, 0.8]  # [x像素, y像素, z相对深度]

# 关键点列表示例
landmarks_list = [
    [100.0, 200.0, 0.1],  # 关键点1
    [150.0, 250.0, 0.2],  # 关键点2
    [200.0, 300.0, 0.3]   # 关键点3
]

# 关键点字典示例
landmarks_dict = {
    "left_eye": [120.5, 180.3, 0.15],
    "right_eye": [280.7, 180.1, 0.16],
    "nose": [200.0, 220.0, 0.20]
}
```

### 4. 三维坐标
- **单个三维坐标**：`np.ndarray`，形状`(3,)`
- **三维坐标字典**：`Dict[str, np.ndarray]`
- **三维坐标列表**：`List[np.ndarray]`

**示例：**
```python
# 单个三维坐标示例
coord_3d = np.ndarray(
    data=[0.12, 0.06, 0.82],  # [x米, y米, z米]
    shape=(3,),  # 3个元素
    dtype=np.float32  # 32位浮点数
)

# 三维坐标字典示例
coords_dict = {
    "left_eye": np.array([0.10, 0.05, 0.80], dtype=np.float32),
    "right_eye": np.array([0.15, 0.05, 0.80], dtype=np.float32)
}

# 三维坐标列表示例
coords_list = [
    np.array([0.10, 0.05, 0.80], dtype=np.float32),  # 坐标1
    np.array([0.15, 0.05, 0.80], dtype=np.float32),  # 坐标2
    np.array([0.12, 0.08, 0.82], dtype=np.float32)   # 坐标3
]
```

### 5. 相机参数
- **类型**：`dict`
- **主要字段**：`intrinsic_params`、`image_resolution`、`depth_scale`

**示例：**
```python
# 相机参数示例
camera_params = {
    "camera_name": "Intel RealSense D435i",
    "camera_type": "RGB-D",
    "intrinsic_params": {
        "fx": 925.0,  # 焦距x（像素）
        "fy": 925.0,  # 焦距y（像素）
        "cx": 640.0,  # 主点x坐标（像素）
        "cy": 360.0   # 主点y坐标（像素）
    },
    "image_resolution": {
        "width": 1280,   # 图像宽度（像素）
        "height": 720    # 图像高度（像素）
    },
    "depth_scale": 0.001,  # 深度值缩放因子（米/单位）
    "min_depth": 0.1,      # 最小有效深度（米）
    "max_depth": 10.0      # 最大有效深度（米）
}
```

## 核心模块接口

### Core模块

#### 1. landmark_extractor.py
```python
def extract_landmarks(bgr_image: np.ndarray) -> List[List[float]]
# 示例：
bgr_image = np.ndarray(data=[[[255, 0, 0], [0, 255, 0]], [[128, 64, 32], [64, 128, 64]]], shape=(2, 2, 3), dtype=np.uint8)
landmarks = extract_landmarks(bgr_image)
# 返回：[[x1, y1, z1], [x2, y2, z2], ..., [x478, y478, z478]]

def get_landmark_indices() -> Dict[str, List[int]]
# 示例：
indices = get_landmark_indices()
# 返回：{"right_eye_contour": [33, 7, 163, ...], "left_eye_contour": [362, 382, 381, ...], ...}

def validate_landmarks(landmarks: List[List[float]]) -> bool
# 示例：
landmarks = [[100.0, 200.0, 0.1], [150.0, 250.0, 0.2], [200.0, 300.0, 0.3]]
is_valid = validate_landmarks(landmarks)
# 返回：True 或 False

def get_pupil_center_landmarks(landmarks: List[List[float]]) -> Dict[str, Optional[List[float]]]
# 示例：
landmarks = [[100.0, 200.0, 0.1], [150.0, 250.0, 0.2], [200.0, 300.0, 0.3]]
pupils = get_pupil_center_landmarks(landmarks)
# 返回：{"left": [120.5, 180.3, 0.15], "right": [280.7, 180.1, 0.16]}

def get_eye_contours_landmarks(landmarks: List[List[float]]) -> Dict[str, List[List[float]]]
# 示例：
landmarks = [[100.0, 200.0, 0.1], [150.0, 250.0, 0.2], [200.0, 300.0, 0.3]]
eye_contours = get_eye_contours_landmarks(landmarks)
# 返回：{"left": [[x1, y1, z1], ...], "right": [[x1, y1, z1], ...]}

def get_iris_boundaries_landmarks(landmarks: List[List[float]]) -> Dict[str, List[List[float]]]
# 示例：
landmarks = [[100.0, 200.0, 0.1], [150.0, 250.0, 0.2], [200.0, 300.0, 0.3]]
iris = get_iris_boundaries_landmarks(landmarks)
# 返回：{"left": [[x1, y1, z1], ...], "right": [[x1, y1, z1], ...]}
```

#### 2. detector.py (FaceDetector类)
```python
def __init__(self, camera_params: dict)
# 示例：
camera_params = {"intrinsic_params": {"fx": 925.0, "fy": 925.0, "cx": 640.0, "cy": 360.0}}
detector = FaceDetector(camera_params)

def detect_face(self, bgr_image: np.ndarray, depth_map: np.ndarray) -> bool
# 示例：
bgr_image = np.ndarray(data=[[[255, 0, 0], [0, 255, 0]], [[128, 64, 32], [64, 128, 64]]], shape=(2, 2, 3), dtype=np.uint8)
depth_map = np.ndarray(data=[[0.5, 0.6], [0.4, 0.5]], shape=(2, 2), dtype=np.float32)
success = detector.detect_face(bgr_image, depth_map)
# 返回：True 或 False

def get_eyes_contours(self) -> Dict[str, List[np.ndarray]]
# 示例：
eye_contours = detector.get_eyes_contours()
# 返回：{"left": [np.array([x1, y1, z1]), ...], "right": [np.array([x1, y1, z1]), ...]}

def get_pupil_center(self) -> Dict[str, np.ndarray]
# 示例：
pupil_center = detector.get_pupil_center()
# 返回：{"left": np.array([0.12, 0.06, 0.82], dtype=np.float32), "right": np.array([0.18, 0.06, 0.82], dtype=np.float32)}

def get_iris_boundaries(self) -> Dict[str, List[np.ndarray]]
# 示例：
iris_boundaries = detector.get_iris_boundaries()
# 返回：{"left": [np.array([0.11, 0.05, 0.81]), np.array([0.13, 0.05, 0.81])], "right": [np.array([0.17, 0.05, 0.81]), np.array([0.19, 0.05, 0.81])]}

def get_detection_confidence(self) -> float
# 示例：
confidence = detector.get_detection_confidence()
# 返回：0.85

def get_detection_status(self) -> str
# 示例：
status = detector.get_detection_status()
# 返回："检测成功" 或 "检测失败"
```

#### 3. coordinate_converter.py
```python
def pixel_to_3d(pixel_coords: Tuple[int, int], depth_map_meters: np.ndarray, camera_params: dict) -> np.ndarray
# 示例：
pixel_coords = (640, 360)
depth_map_meters = np.ndarray(data=[[0.5, 0.6], [0.4, 0.5]], shape=(2, 2), dtype=np.float32)
camera_params = {"intrinsic_params": {"fx": 925.0, "fy": 925.0, "cx": 640.0, "cy": 360.0}}
coord_3d = pixel_to_3d(pixel_coords, depth_map_meters, camera_params)
# 返回：np.array([0.12, 0.06, 0.82], dtype=np.float32)

def batch_convert_landmarks(landmarks: List[Tuple[float, float, float]], depth_map_meters: np.ndarray, camera_params: dict) -> List[np.ndarray]
# 示例：
landmarks = [(100.0, 200.0, 0.1), (150.0, 250.0, 0.2)]
depth_map_meters = np.ndarray(data=[[0.5, 0.6], [0.4, 0.5]], shape=(2, 2), dtype=np.float32)
camera_params = {"intrinsic_params": {"fx": 925.0, "fy": 925.0, "cx": 640.0, "cy": 360.0}}
coords_3d = batch_convert_landmarks(landmarks, depth_map_meters, camera_params)
# 返回：[np.array([0.10, 0.05, 0.80], dtype=np.float32), np.array([0.15, 0.05, 0.80], dtype=np.float32)]

def validate_3d_coordinates(coords_3d: List[np.ndarray]) -> bool
# 示例：
coords_3d = [np.array([0.10, 0.05, 0.80], dtype=np.float32), np.array([0.15, 0.05, 0.80], dtype=np.float32)]
is_valid = validate_3d_coordinates(coords_3d)
# 返回：True 或 False
```

### Utils模块

#### 1. data_manager.py
```python
# DataManager单例类
class DataManager:
    def __init__(self) -> None
    # 单例模式初始化
    
    def add_frame(self, frame_id: int, bgr_image: np.ndarray, depth_map: Optional[np.ndarray] = None) -> None
    # 示例：
    frame_id = 0
    bgr_image = np.ndarray(shape=(480, 640, 3), dtype=np.uint8)
    depth_map = np.ndarray(shape=(480, 640), dtype=np.float32)
    data_manager.add_frame(frame_id, bgr_image, depth_map)
    # 返回：无返回值，直接存储到内存
    
    def get_image(self, frame_id: int) -> Optional[np.ndarray]
    # 示例：
    bgr_image = data_manager.get_image(0)
    # 返回：np.ndarray(shape=(480, 640, 3), dtype=np.uint8) 或 None
    
    def get_depth(self, frame_id: int) -> Optional[np.ndarray]
    # 示例：
    depth_map = data_manager.get_depth(0)
    # 返回：np.ndarray(shape=(480, 640), dtype=np.float32) 或 None
    
    def get_frame_data(self, frame_id: int) -> Optional[Dict]
    # 示例：
    frame_data = data_manager.get_frame_data(0)
    # 返回：{"bgr_image": np.ndarray, "depth_map": np.ndarray, "timestamp": str} 或 None
    
    def get_resolution(self) -> Optional[Tuple[int, int]]
    # 示例：
    resolution = data_manager.get_resolution()
    # 返回：(width, height) 元组，如 (640, 480)
    
    def get_frame_count(self) -> int
    # 示例：
    count = data_manager.get_frame_count()
    # 返回：当前存储的帧数量
    
    def clear_all(self) -> None
    # 示例：
    data_manager.clear_all()
    # 返回：无返回值，清空所有数据
    
    def remove_frame(self, frame_id: int) -> bool
    # 示例：
    success = data_manager.remove_frame(0)
    # 返回：True 或 False

# 便捷函数接口
def add_frame(frame_id: int, bgr_image: np.ndarray, depth_map: Optional[np.ndarray] = None) -> None
# 示例：
add_frame(0, bgr_image, depth_map)
# 返回：无返回值

def get_image(frame_id: int) -> Optional[np.ndarray]
# 示例：
bgr_image = get_image(0)
# 返回：np.ndarray(shape=(H,W,3), dtype=np.uint8) 或 None

def get_depth(frame_id: int) -> Optional[np.ndarray]
# 示例：
depth_map = get_depth(0)
# 返回：np.ndarray(shape=(H,W), dtype=np.float32) 或 None

def get_resolution() -> Optional[Tuple[int, int]]
# 示例：
resolution = get_resolution()
# 返回：(width, height) 元组

def get_frame_count() -> int
# 示例：
count = get_frame_count()
# 返回：当前帧数量
```

#### 2. camera_calibration.py
```python
# CameraCalibrator单例类
class CameraCalibrator:
    def __init__(self, file_path: str = None, rgb_d=False) -> None
    # 示例：
    calibrator = CameraCalibrator("config/camera_params.json", rgb_d=True)
    
    def get_cap(self) -> cv2.VideoCapture
    # 示例：
    cap = calibrator.get_cap()
    # 返回：cv2.VideoCapture对象
    
    def load_camera_params(self) -> Dict
    # 示例：
    camera_params = calibrator.load_camera_params()
    # 返回：{"intrinsic_params": {"fx": 925.0, "fy": 925.0, "cx": 640.0, "cy": 360.0}, ...}
    
    def get_image_resolution(self) -> Tuple[int, int]
    # 示例：
    resolution = calibrator.get_image_resolution()
    # 返回：(width, height) 元组，如 (1280, 720)
    
    def get_intrinsics(self) -> Dict
    # 示例：
    intrinsics = calibrator.get_intrinsics()
    # 返回：{"fx": 925.0, "fy": 925.0, "cx": 640.0, "cy": 360.0}
    
    def get_depth_scale(self) -> float
    # 示例：
    depth_scale = calibrator.get_depth_scale()
    # 返回：0.001
```

## 统一约定

### ✅ 图像格式统一
- **BGR格式**：所有图像输入输出统一使用BGR格式（OpenCV默认）
- **数据类型**：`np.uint8`，像素值范围 0-255
- **形状**：`(H, W, 3)` 或 `(H, W)`（深度图）

### ✅ 深度图单位统一
- **单位**：米（meters）
- **数据类型**：`np.float32`
- **形状**：`(H, W)`

### ✅ 坐标系定义统一
- **OpenCV标准坐标系**：
  - X轴：水平方向，向右为正
  - Y轴：垂直方向，向下为正
  - Z轴：深度方向，向前为正（相机光轴方向）

### ✅ 类型定义统一
- **关键点**：`List[float]` [x, y, z]
- **三维坐标**：`np.ndarray` 形状 `(3,)`
- **相机参数**：`dict` 类型

### ✅ 职责分离统一
- **数据解析**：`utils/data_manager.py` 专门处理数据存储和管理
- **业务逻辑**：`core/` 模块专注于算法处理
- **工具函数**：`utils/` 模块提供辅助功能

## 数据流规范

### 标准数据流