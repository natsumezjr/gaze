# 眼动校准程序

本目录包含基于 Vue.js 框架实现的眼动校准程序。

## 1. 概述

本文档详细说明了眼动校准程序的功能、改进和使用方法。该程序基于Vue.js框架开发，用于进行眼动追踪设备的校准，支持真实视线输入和模拟数据生成，并预留了与后端系统集成的接口。

## 2. 主要功能

### 2.1 校准功能

- **多点校准**：支持六边形布局（6点）和矩形布局（5点）两种校准模式
- **视线监测**：实时显示用户视线位置，并判断是否在目标点范围内
- **注视判断**：要求用户注视目标点一定时间才能继续校准
- **数据采集**：采集校准过程中的视线数据，用于后续分析和校准
- **主题切换**：支持多种颜色主题（白色、黑色、浅色）
- **流畅的动画效果**：平滑的校准点移动和视线反馈
- **响应式设计**：适配不同屏幕尺寸

### 2.2 改进功能

- **视线估计优化**：提高了真实注视点估计的灵敏度，减小抖动范围，增加平滑处理
- **注视判断工作流**：实现了注视点在误差范围内才能继续校准的工作流程
- **后端数据交换**：预留了与后端交换数据并进行kappa角补偿的接口

## 3. 系统架构

### 3.1 前端架构

```
src/
├── assets/           # 静态资源
│   └── styles/       # 样式文件
├── components/       # 组件
│   └── CalibrationContainer.vue  # 校准容器组件
├── composables/      # 组合式API
│   └── useCalibration.js         # 校准功能组合式API
├── config/           # 配置文件
│   ├── calibration.config.js     # 校准配置
│   └── integration.config.js     # 集成配置
├── services/         # 服务
│   ├── api.js                    # API服务
│   ├── calibration.js            # 校准反馈服务
│   └── websocket.js              # WebSocket服务
├── App.vue           # 主应用组件
└── main.js           # 入口文件
```

### 3.2 数据流

1. 用户启动校准程序
2. 程序显示校准点，用户注视校准点
3. 程序采集用户视线数据
4. 数据发送到后端进行处理
5. 后端返回校准结果和kappa角参数
6. 前端应用kappa角补偿，修正视线位置

## 4. 配置说明

### 4.1 校准配置 (calibration.config.js)

```javascript
export default {
  // 校准点布局配置
  points: {
    // 六边形布局（默认）
    hexagon: [
      { x: 33, y: 20 },   // 左上（左三分之一线上）
      { x: 67, y: 20 },   // 右上（右三分之一线上）
      { x: 10, y: 50 },   // 左中（靠近左侧边缘）
      { x: 90, y: 50 },   // 右中（靠近右侧边缘）
      { x: 33, y: 80 },   // 左下（左三分之一线上）
      { x: 67, y: 80 }    // 右下（右三分之一线上）
    ],
    // 矩形布局（可选）
    rectangle: [
      { x: 20, y: 20 },   // 左上
      { x: 80, y: 20 },   // 右上
      { x: 80, y: 80 },   // 右下
      { x: 20, y: 80 },   // 左下
      { x: 50, y: 50 }    // 中心
    ]
  },
  
  // 主题配置
  themes: { ... },
  
  // 数据采集配置
  collection: {
    sampleInterval: 100,     // 采样间隔（毫秒）
    sampleDuration: 2000,    // 每点采样时长（毫秒）
    minMoveTime: 0.6,        // 最小移动时间（秒）
    maxMoveTime: 1.4,        // 最大移动时间（秒）
    baseSpeed: 100           // 基准移动速度（距离单位/秒）
  },
  
  // 视线判断配置
  gaze: {
    showRealGaze: true,           // 是否显示真实视线点
    accuracyThreshold: 5,          // 视线精度阈值（百分比）
    requiredFixationTime: 500,     // 需要注视的时间（毫秒）
    gazePointSize: 15,             // 视线点大小（像素）
    validColor: '#4CAF50',         // 有效视线颜色（绿色）
    invalidColor: '#FFC107'        // 无效视线颜色（黄色）
  }
}
```

### 4.2 集成配置 (integration.config.js)

```javascript
export default {
  // 是否使用真实后端
  USE_REAL_BACKEND: true,
  
  // 后端API基础URL
  BACKEND_URL: 'http://localhost:8000',
  
  // WebSocket端口
  WS_PORT: 8765,
  
  // 校准点采样时长（毫秒）
  SAMPLE_DURATION: 2000,
  
  // 视线注视判定时间（毫秒）
  FIXATION_DURATION: 500,
  
  // 视线与目标点距离阈值（屏幕对角线长度的百分比）
  TARGET_THRESHOLD_PERCENT: 5,
  
  // 校准精度阈值（像素）
  ACCURACY_THRESHOLD: 30
}
```

## 5. 接口说明

### 5.1 API接口 (api.js)

```javascript
// 开始校准会话
export async function startCalibrationSession()

// 开始采集校准点数据
export async function startPointCollection(pointIndex, screenXY, duration)

// 停止采集校准点数据
export async function stopPointCollection(pointIndex)

// 完成校准
export async function completeCalibration(calibrationData)

// 获取校准状态
export async function getCalibrationStatus()

// 获取实时视线数据
export async function getRealTimeGazeData()

// 验证视线位置是否在目标点范围内
export async function validateGazePosition(pointIndex, gazePosition, threshold)
```

### 5.2 WebSocket接口 (websocket.js)

```javascript
// 初始化WebSocket连接
export function initGazeWebSocket(url, options = {})

// 关闭WebSocket连接
export function closeGazeWebSocket()

// 设置视线数据回调
export function onGazeData(callback)

// 发送消息到服务器
export function sendMessage(data)

// 请求实时视线数据
export function requestGazeData()

// 停止实时视线数据
export function stopGazeData()
```

### 5.3 校准反馈接口 (calibration.js)

```javascript
// 应用kappa角补偿
export function applyKappaCompensation(gazePoint, kappaParams)

// 计算校准精度
export function calculateCalibrationAccuracy(calibrationData)

// 保存校准模型
export async function saveCalibrationModel(calibrationResult)

// 加载校准模型
export async function loadCalibrationModel(modelId)

// 应用校准模型进行视线修正
export function applyCalibrationModel(gazePoint, calibrationModel)

// 获取当前校准模型
export function getCurrentCalibrationModel()

// 应用当前校准模型进行视线修正
export function applyCurrentCalibrationModel(gazePoint)
```

## 6. 数据格式

### 6.1 校准数据格式

```javascript
// 发送到后端的校准数据格式
const calibrationData = {
  metadata: {
    description: "眼动追踪校准数据",
    created_at: "2023-01-01T12:00:00.000Z",
    data_format: "JSON",
    coordinate_system: "camera_coordinates_mm",
    sample_count: 120,
    calibration_type: "6_point_hexagon",
    hardware: "web_browser"
  },
  eyes: [],         // 眼球中心坐标数组 [[x, y, z], ...]
  pupils: [],       // 瞳孔中心坐标数组 [[x, y, z], ...]
  target_pixels: [], // 目标像素坐标数组 [[x, y], ...]
  camera_matrix: [  // 相机内参矩阵
    [1000.0, 0.0, 960.0],
    [0.0, 1000.0, 540.0],
    [0.0, 0.0, 1.0]
  ],
  screen_resolution: [1920, 1080],
  units: {
    coordinates: "millimeters",
    pixels: "pixels",
    angles: "degrees"
  },
  calibration_info: {
    method: "6_point_hexagon",
    duration_seconds: 2,
    samples_per_point: 20,
    point_order: ["point_1", "point_2", "point_3", "point_4", "point_5", "point_6"]
  }
}
```

### 6.2 校准结果格式

```javascript
// 后端返回的校准结果格式
const calibrationResult = {
  status: "success",
  model_id: "calibration_model_1234567890",
  accuracy: 95.5,
  kappa_params: {
    left_eye: {
      axis: [0.1, 0.2, 0.0],
      angle: 2.3
    },
    right_eye: {
      axis: [0.1, 0.2, 0.0],
      angle: 2.1
    }
  },
  samples_processed: 120
}
```

## 7. 使用说明

### 7.1 启动程序

```bash
# 安装依赖
npm install

# 开发模式运行
npm run serve

# 构建生产版本
npm run build
```

如果您不需要修改源代码，也可以直接打开`vue_index.html`文件在浏览器中查看已构建好的版本。

### 7.2 校准流程

1. 点击"开始校准"按钮
2. 按照屏幕提示，注视移动的校准点
3. 保持注视直到校准点变色并移动到下一个位置
4. 完成所有校准点后，系统会自动处理数据并显示结果

### 7.3 注意事项

- 校准过程中保持头部稳定
- 确保眼动追踪设备正确连接并工作
- 校准环境光线应适中，避免强光直射
- 如果使用真实后端，确保后端服务已启动

## 8. 自定义开发

### 8.1 实现kappa角补偿算法

在`src/services/calibration.js`文件中，实现`applyKappaCompensation`函数：

```javascript
export function applyKappaCompensation(gazePoint, kappaParams) {
  // 实现自定义的kappa角补偿算法
  // ...
  return compensatedGazePoint;
}
```

### 8.2 实现校准精度计算

在`src/services/calibration.js`文件中，实现`calculateCalibrationAccuracy`函数：

```javascript
export function calculateCalibrationAccuracy(calibrationData) {
  // 实现自定义的校准精度计算算法
  // ...
  return accuracy;
}
```

### 8.3 自定义校准点布局

在`src/config/calibration.config.js`文件中，修改`points`配置：

```javascript
points: {
  // 自定义布局
  custom: [
    { x: 25, y: 25 },
    { x: 75, y: 25 },
    { x: 50, y: 50 },
    { x: 25, y: 75 },
    { x: 75, y: 75 }
  ]
}
```

## 9. 故障排除

### 9.1 常见问题

- **界面空白**：检查Vue.js脚本路径是否正确，确保使用相对路径`./src/main.js`
- **WebSocket连接失败**：检查后端服务是否启动，端口是否正确
- **视线点不显示**：检查`showRealGaze`配置是否为`true`
- **校准点不移动**：检查浏览器控制台是否有错误信息

### 9.2 调试方法

- 打开浏览器开发者工具，查看控制台输出
- 检查网络请求，确认API调用是否成功
- 使用Vue.js开发者工具检查组件状态
- 临时启用模拟数据模式（设置`USE_REAL_BACKEND: false`）

## 10. 后续开发计划

- 支持多种眼动追踪设备
- 增加校准结果可视化
- 实现更复杂的kappa角补偿算法
- 添加用户校准历史记录
- 优化移动设备支持

---

## 附录：与后端集成

### A.1 后端API要求

后端需要实现以下API接口：

- `POST /start_session`：开始校准会话
- `POST /start_point`：开始采集校准点数据
- `POST /stop_point`：停止采集校准点数据
- `POST /complete_calibration`：完成校准
- `GET /calibration_status`：获取校准状态
- `GET /gaze_data`：获取实时视线数据
- `POST /validate_gaze`：验证视线位置

### A.2 WebSocket接口

后端需要提供WebSocket服务，用于实时传输视线数据：

- 连接地址：`ws://{BACKEND_URL}:{WS_PORT}/ws/gaze`
- 消息格式：JSON
- 数据字段：`gaze_position`（视线位置）、`confidence`（置信度）等

### A.3 集成步骤

1. 配置后端URL和端口：修改`integration.config.js`
2. 启用真实后端：设置`USE_REAL_BACKEND: true`
3. 实现kappa角补偿算法：修改`calibration.js`
4. 启动后端服务
5. 启动前端应用并进行校准测试
