ji# 眼动追踪校准界面使用说明

## 功能概述

这是一个类似苹果系统眼动追踪校准的前端界面，用于收集用户的眼动校准数据并保存为JSON格式。

## 主要特性

- **9点校准模式**：标准的3x3网格校准点布局
- **苹果风格UI**：现代化的渐变背景和动画效果
- **实时进度显示**：校准进度条和点状态指示
- **数据收集**：模拟眼动数据收集（可替换为真实设备数据）
- **JSON导出**：完整的校准数据保存功能

## 使用方法

### 1. 启动校准
1. 打开 `eye_tracking_calibration.html` 文件
2. 点击"开始校准"按钮
3. 确保头部稳定，不要移动

### 2. 校准过程
- 系统会依次显示9个校准点
- 每个点显示3秒，请注视该点
- 保持眼睛睁开，尽量减少眨眼
- 校准点会从白色变为蓝色（激活状态），然后变为绿色（完成状态）

### 3. 数据导出
- 校准完成后，点击"下载数据"按钮
- 系统会生成包含完整校准数据的JSON文件
- 文件名格式：`eye_tracking_calibration_YYYY-MM-DDTHH-MM-SS.json`

## 数据结构

导出的JSON文件包含以下信息：

```json
{
  "metadata": {
    "version": "1.0",
    "calibrationType": "9-point",
    "timestamp": "2024-01-01T12:00:00.000Z",
    "duration": 27000,
    "pointCount": 9
  },
  "calibrationPoints": [
    {"x": 10, "y": 10},
    {"x": 50, "y": 10},
    // ... 其他点
  ],
  "calibrationData": [
    {
      "pointIndex": 0,
      "targetPosition": {"x": 10, "y": 10},
      "timestamp": 1704110400000,
      "duration": 3000,
      "gazeData": [
        {
          "timestamp": 1704110400000,
          "x": 10.5,
          "y": 10.2,
          "confidence": 0.85,
          "pupilDiameter": 3.7,
          "fixation": true
        }
        // ... 更多采样点
      ],
      "systemInfo": {
        "screenWidth": 1920,
        "screenHeight": 1080,
        "userAgent": "...",
        "timestamp": "2024-01-01T12:00:00.000Z"
      }
    }
    // ... 其他校准点数据
  ]
}
```

## 自定义配置

### 修改校准参数
在 `EyeTrackingCalibration` 类的构造函数中可以调整：

```javascript
this.pointDuration = 3000;        // 每个点显示时间（毫秒）
this.transitionDuration = 1000;   // 点之间过渡时间（毫秒）
```

### 修改校准点布局
可以自定义校准点的位置：

```javascript
this.calibrationPoints = [
    { x: 10, y: 10 }, { x: 50, y: 10 }, { x: 90, y: 10 },
    { x: 10, y: 50 }, { x: 50, y: 50 }, { x: 90, y: 50 },
    { x: 10, y: 90 }, { x: 50, y: 90 }, { x: 90, y: 90 }
];
```

### 集成真实眼动设备
替换 `simulateGazeData` 方法中的数据收集逻辑：

```javascript
simulateGazeData(targetPoint) {
    // 替换为真实的眼动设备API调用
    // 例如：Tobii、EyeLink、Pupil Labs等
    return realEyeTrackingData;
}
```

## 技术实现

### 核心类
- `EyeTrackingCalibration`：主要的校准逻辑类
- 管理校准点显示、数据收集、进度更新

### 关键方法
- `startCalibration()`：开始校准过程
- `showNextPoint()`：显示下一个校准点
- `collectData()`：收集当前点的眼动数据
- `downloadData()`：导出JSON数据文件

### 样式特性
- 响应式设计，适配不同屏幕尺寸
- 平滑的CSS动画和过渡效果
- 苹果风格的UI设计语言

## 浏览器兼容性

- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+

## 注意事项

1. **头部稳定**：校准过程中保持头部稳定，避免大幅移动
2. **环境光线**：确保充足的光线，避免强光直射眼睛
3. **设备距离**：保持适当的观看距离（通常50-70cm）
4. **数据质量**：校准质量直接影响后续眼动追踪的准确性

## 扩展功能

可以考虑添加的功能：
- 多用户支持
- 校准质量评估
- 实时眼动追踪预览
- 网络数据上传
- 校准历史记录
