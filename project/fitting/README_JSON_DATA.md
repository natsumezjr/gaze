# JSON数据格式使用指南

## 🎯 **概述**

现在你可以使用JSON文件来提供眼动追踪数据，而不需要修改代码！这种方式非常适合：

- **数据交换**：与其他系统共享数据
- **离线分析**：保存和加载校准数据
- **调试测试**：使用预定义的数据集
- **批量处理**：处理多个用户的数据

## 📁 **JSON数据格式**

### **基本结构**
```json
{
  "metadata": {
    "description": "眼动追踪校准数据",
    "created_at": "2024-12-01T10:30:00.000000",
    "data_format": "JSON",
    "coordinate_system": "camera_coordinates_mm",
    "sample_count": 9
  },
  "eyes": [
    [x1, y1, z1],  # 眼球中心坐标 (mm)
    [x2, y2, z2],
    ...
  ],
  "pupils": [
    [x1, y1, z1],  # 瞳孔中心坐标 (mm)
    [x2, y2, z2],
    ...
  ],
  "target_pixels": [
    [u1, v1],       # 屏幕目标点 (像素)
    [u2, v2],
    ...
  ],
  "camera_matrix": [
    [fx, 0, cx],    # 相机内参矩阵
    [0, fy, cy],
    [0, 0, 1]
  ]
}
```

### **字段说明**

| 字段 | 类型 | 说明 | 单位 |
|------|------|------|------|
| `eyes` | 数组 | 眼球中心3D坐标 | 毫米 (mm) |
| `pupils` | 数组 | 瞳孔中心3D坐标 | 毫米 (mm) |
| `target_pixels` | 数组 | 屏幕目标点2D坐标 | 像素 |
| `camera_matrix` | 数组 | 3x3相机内参矩阵 | 像素 |
| `sample_count` | 整数 | 样本数量 | - |

## 🚀 **使用方法**

### **步骤1：准备JSON数据文件**

创建文件 `eye_tracking_data.json`，包含你的校准数据：

```json
{
  "metadata": {
    "description": "我的眼动校准数据",
    "sample_count": 9
  },
  "eyes": [
    [0.0, 0.0, 0.0],
    [0.05, 0.0, 0.0],
    [0.1, 0.0, 0.0],
    [0.0, 0.05, 0.0],
    [0.05, 0.05, 0.0],
    [0.1, 0.05, 0.0],
    [0.0, 0.1, 0.0],
    [0.05, 0.1, 0.0],
    [0.1, 0.1, 0.0]
  ],
  "pupils": [
    [0.0, 0.0, 0.1],
    [0.05, 0.0, 0.1],
    [0.1, 0.0, 0.1],
    [0.0, 0.05, 0.1],
    [0.05, 0.05, 0.1],
    [0.1, 0.05, 0.1],
    [0.0, 0.1, 0.1],
    [0.05, 0.1, 0.1],
    [0.1, 0.1, 0.1]
  ],
  "target_pixels": [
    [192, 108],
    [960, 108],
    [1728, 108],
    [192, 540],
    [960, 540],
    [1728, 540],
    [192, 972],
    [960, 972],
    [1728, 972]
  ],
  "camera_matrix": [
    [1000.0, 0.0, 960.0],
    [0.0, 1000.0, 540.0],
    [0.0, 0.0, 1.0]
  ]
}
```

### **步骤2：运行程序**

```bash
cd /Users/andrew/PycharmProjects/大创/gaze
python -m project.fitting.main
```

### **步骤3：选择JSON数据模式**

```
摄像头配置选项:
1. 使用模拟数据（默认）
2. 使用真实摄像头
3. 自定义摄像头配置
4. 从JSON文件读取数据

请选择 (1/2/3/4，直接回车使用默认): 4
```

程序会自动：
1. 检查 `eye_tracking_data.json` 文件
2. 验证数据格式
3. 加载JSON数据
4. 进行kappa校准

## 🔧 **数据格式要求**

### **坐标系统**
- **眼球中心**：相机坐标系，原点在相机光心
- **瞳孔中心**：相机坐标系，原点在相机光心
- **屏幕目标**：屏幕坐标系，左上角为原点

### **数据一致性**
- 所有数组的样本数量必须一致
- 眼球和瞳孔坐标必须是3D坐标 (x, y, z)
- 屏幕目标坐标必须是2D坐标 (u, v)
- 相机内参矩阵必须是3x3矩阵

### **单位说明**
- **坐标**：毫米 (mm)
- **像素**：像素 (pixels)
- **角度**：度 (degrees)

## 📊 **示例数据**

### **9点校准网格**
```json
"target_pixels": [
  [192, 108],   # 左上角
  [960, 108],   # 上中
  [1728, 108],  # 右上角
  [192, 540],   # 左中
  [960, 540],   # 中心
  [1728, 540],  # 右中
  [192, 972],   # 左下角
  [960, 972],   # 下中
  [1728, 972]   # 右下角
]
```

### **6点校准（六边形）**
```json
"target_pixels": [
  [640, 216],   # 左上
  [1280, 216],  # 右上
  [320, 540],   # 左中
  [1600, 540],  # 右中
  [640, 864],   # 左下
  [1280, 864]   # 右下
]
```

## 🛠 **高级功能**

### **自动创建示例文件**
如果JSON文件不存在，程序会自动创建示例文件：

```python
# 程序会自动调用
interface._create_sample_json_file("eye_tracking_data.json")
```

### **数据验证**
程序会自动验证JSON数据：
- 检查必需字段
- 验证数据类型
- 检查数据维度
- 确保数据一致性

### **错误处理**
如果JSON数据有问题，程序会：
1. 显示详细错误信息
2. 自动回退到模拟数据模式
3. 继续运行kappa校准演示

## 📝 **数据来源建议**

### **从眼动追踪设备获取**
```python
# 示例：从Tobii眼动仪获取数据
import tobii_research as tr

eyetracker = tr.EyeTracker("your_device_address")
gaze_data = eyetracker.get_gaze_data()

# 转换为JSON格式
json_data = {
    "eyes": [...],
    "pupils": [...],
    "target_pixels": [...],
    "camera_matrix": [...]
}
```

### **从图像处理获取**
```python
# 示例：从OpenCV检测结果获取
import cv2

# 检测眼球和瞳孔
eye_centers = detect_eye_centers(frame)
pupil_centers = detect_pupil_centers(frame)

# 转换为JSON格式
json_data = {
    "eyes": eye_centers.tolist(),
    "pupils": pupil_centers.tolist(),
    "target_pixels": target_points,
    "camera_matrix": camera_matrix.tolist()
}
```

### **从其他系统导入**
```python
# 示例：从CSV文件导入
import pandas as pd

df = pd.read_csv("calibration_data.csv")
json_data = {
    "eyes": df[['eye_x', 'eye_y', 'eye_z']].values.tolist(),
    "pupils": df[['pupil_x', 'pupil_y', 'pupil_z']].values.tolist(),
    "target_pixels": df[['target_u', 'target_v']].values.tolist(),
    "camera_matrix": camera_matrix.tolist()
}
```

## 🎉 **优势总结**

### **使用JSON的好处**
1. **标准化**：JSON是通用的数据交换格式
2. **可读性**：人类可读，便于调试
3. **兼容性**：几乎所有编程语言都支持
4. **灵活性**：可以轻松添加元数据
5. **可扩展**：支持嵌套结构和复杂数据

### **与现有系统的集成**
- **前端界面**：可以直接生成JSON数据
- **数据采集**：实时保存为JSON格式
- **数据分析**：使用Python、MATLAB等工具分析
- **数据共享**：在不同系统间传输数据

## 🚀 **下一步**

1. **创建你的JSON数据文件**
2. **运行程序选择JSON模式**
3. **验证数据加载是否正常**
4. **查看kappa校准结果**
5. **根据需要调整数据格式**

现在你就可以使用JSON文件来提供眼动追踪数据了！这种方式既简单又灵活，非常适合各种应用场景。

