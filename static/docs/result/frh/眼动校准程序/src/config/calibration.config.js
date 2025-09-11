/**
 * 眼动校准系统配置
 */

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
  themes: {
    // 白色主题（默认）
    light: [
      { bg: '#DC3545', fg: '#C82333' }, // 深红色系
      { bg: '#20C997', fg: '#1E7E34' }, // 深绿色系
      { bg: '#0D6EFD', fg: '#0A58CA' }, // 深蓝色系
      { bg: '#198754', fg: '#155724' }, // 深薄荷绿系
      { bg: '#FD7E14', fg: '#E8590C' }, // 深橙色系
      { bg: '#6610F2', fg: '#520DC2' }  // 深紫色系
    ],
    // 黑色主题
    dark: [
      { bg: '#000000', fg: '#FFFFFF' },  // 黑底白字
      { bg: '#000000', fg: '#FFFFFF' },
      { bg: '#000000', fg: '#FFFFFF' },
      { bg: '#000000', fg: '#FFFFFF' },
      { bg: '#000000', fg: '#FFFFFF' },
      { bg: '#000000', fg: '#FFFFFF' }
    ],
    // 浅色主题
    pastel: [
      { bg: '#E8F5E9', fg: '#1B5E20' },  // 浅绿
      { bg: '#E3F2FD', fg: '#0D47A1' },  // 浅蓝
      { bg: '#FFF3E0', fg: '#E65100' },  // 浅橙
      { bg: '#F3E5F5', fg: '#4A148C' },  // 浅紫
      { bg: '#E0F7FA', fg: '#006064' },  // 浅青
      { bg: '#FFF8E1', fg: '#F57F17' }   // 浅黄
    ]
  },
  
  // 数据采集配置
  collection: {
    sampleInterval: 100,     // 采样间隔（毫秒）
    sampleDuration: 2000,    // 每点采样时长（毫秒）
    minMoveTime: 0.6,        // 最小移动时间（秒）
    maxMoveTime: 1.4,        // 最大移动时间（秒）
    baseSpeed: 100           // 基准移动速度（距离单位/秒）
  },
  
  // 后端集成配置
  backend: {
    enabled: true,          // 是否启用真实后端
    baseUrl: 'http://localhost:8000',  // 后端API基础URL
    endpoints: {
      startSession: '/start_session',
      startPoint: '/start_point',
      stopPoint: '/stop_point',
      completeCalibration: '/complete_calibration',
      calibrationStatus: '/calibration_status'
    }
  },
  
  // 界面配置
  ui: {
    countdownDuration: 3,    // 倒计时时长（秒）
    circleSize: 50,          // 校准圆圈大小（像素）
    completionDisplayTime: 3000,  // 完成信息显示时长（毫秒）
    successMessageTime: 4000,     // 成功消息显示时长（毫秒）
    errorMessageTime: 6000        // 错误消息显示时长（毫秒）
  },
  
  // 缩放配置
  scaling: {
    preventZoom: true,       // 是否阻止页面缩放
    useViewportUnits: true,  // 是否使用视口单位（vw/vh）而非像素
    scaleCircleWithViewport: true  // 校准圆圈是否随视口缩放
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
};