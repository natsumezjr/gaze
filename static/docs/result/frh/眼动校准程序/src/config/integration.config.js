/**
 * 集成配置
 * 用于配置与后端的集成参数
 */

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
};