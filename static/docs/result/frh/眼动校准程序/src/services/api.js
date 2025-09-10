/**
 * 眼动校准系统API服务
 * 负责与后端通信的所有功能
 */

// 后端API基础URL
const API_BASE_URL = 'http://localhost:8000';

/**
 * 开始校准会话
 * @returns {Promise<Object>} 会话信息
 */
export async function startCalibrationSession() {
  try {
    const response = await fetch(`${API_BASE_URL}/start_session`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        timestamp: Date.now(),
        browser: navigator.userAgent,
        screen_resolution: {
          width: window.screen.width,
          height: window.screen.height
        }
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('开始校准会话失败:', error);
    throw error;
  }
}

/**
 * 开始采集校准点数据
 * @param {Number} pointIndex 校准点索引
 * @param {Array} screenXY 屏幕坐标 [x, y]
 * @param {Number} duration 采集时长(毫秒)
 * @returns {Promise<Object>} 响应结果
 */
export async function startPointCollection(pointIndex, screenXY, duration) {
  try {
    const response = await fetch(`${API_BASE_URL}/start_point`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        point_index: pointIndex,
        screen_xy: screenXY,
        duration: duration
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('开始采集校准点数据失败:', error);
    throw error;
  }
}

/**
 * 停止采集校准点数据
 * @param {Number} pointIndex 校准点索引
 * @returns {Promise<Object>} 采集到的样本数据
 */
export async function stopPointCollection(pointIndex) {
  try {
    const response = await fetch(`${API_BASE_URL}/stop_point`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        point_index: pointIndex
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('停止采集校准点数据失败:', error);
    throw error;
  }
}

/**
 * 完成校准
 * @param {Object} calibrationData 校准数据
 * @returns {Promise<Object>} 校准结果
 */
export async function completeCalibration(calibrationData) {
  try {
    const response = await fetch(`${API_BASE_URL}/complete_calibration`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(calibrationData)
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('完成校准失败:', error);
    throw error;
  }
}

/**
 * 获取校准状态
 * @returns {Promise<Object>} 校准状态
 */
export async function getCalibrationStatus() {
  try {
    const response = await fetch(`${API_BASE_URL}/calibration_status`);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('获取校准状态失败:', error);
    throw error;
  }
}

/**
 * 获取实时视线数据
 * @returns {Promise<Object>} 视线数据
 */
export async function getRealTimeGazeData() {
  try {
    const response = await fetch(`${API_BASE_URL}/gaze_data`);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('获取实时视线数据失败:', error);
    throw error;
  }
}

/**
 * 验证视线位置是否在目标点范围内
 * @param {Number} pointIndex 校准点索引
 * @param {Array} gazePosition 视线位置 [x, y]
 * @param {Number} threshold 阈值（百分比）
 * @returns {Promise<Object>} 验证结果
 */
export async function validateGazePosition(pointIndex, gazePosition, threshold) {
  try {
    const response = await fetch(`${API_BASE_URL}/validate_gaze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        point_index: pointIndex,
        gaze_position: gazePosition,
        threshold: threshold
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('验证视线位置失败:', error);
    throw error;
  }
}