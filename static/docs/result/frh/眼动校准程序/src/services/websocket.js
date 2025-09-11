/**
 * WebSocket服务模块
 * 用于处理实时视线数据的接收
 */

// WebSocket连接状态
const WS_STATE = {
  CONNECTING: 0,
  OPEN: 1,
  CLOSING: 2,
  CLOSED: 3
};

// 存储WebSocket实例
let gazeWebSocket = null;

// 回调函数集合
const callbacks = {
  onGazeData: null,
  onOpen: null,
  onClose: null,
  onError: null
};

/**
 * 初始化WebSocket连接
 * @param {String} url WebSocket服务URL
 * @param {Object} options 配置选项
 */
export function initGazeWebSocket(url, options = {}) {
  // 关闭可能存在的连接
  closeGazeWebSocket();
  
  try {
    // 创建新连接
    gazeWebSocket = new WebSocket(url);
    
    // 设置回调函数
    gazeWebSocket.onopen = (event) => {
      console.log('视线追踪WebSocket连接已建立');
      if (callbacks.onOpen) {
        callbacks.onOpen(event);
      }
    };
    
    gazeWebSocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.gaze_position && callbacks.onGazeData) {
          callbacks.onGazeData(data);
        }
      } catch (error) {
        console.error('解析WebSocket消息失败:', error);
      }
    };
    
    gazeWebSocket.onclose = (event) => {
      console.log('视线追踪WebSocket连接已关闭');
      if (callbacks.onClose) {
        callbacks.onClose(event);
      }
    };
    
    gazeWebSocket.onerror = (event) => {
      console.error('视线追踪WebSocket连接错误');
      if (callbacks.onError) {
        callbacks.onError(event);
      }
    };
    
    return true;
  } catch (error) {
    console.error('初始化WebSocket连接失败:', error);
    if (callbacks.onError) {
      callbacks.onError(error);
    }
    return false;
  }
}

/**
 * 关闭WebSocket连接
 */
export function closeGazeWebSocket() {
  if (gazeWebSocket && gazeWebSocket.readyState !== WS_STATE.CLOSED) {
    gazeWebSocket.close();
    gazeWebSocket = null;
  }
}

/**
 * 设置视线数据回调
 * @param {Function} callback 回调函数
 */
export function onGazeData(callback) {
  callbacks.onGazeData = callback;
}

/**
 * 设置连接打开回调
 * @param {Function} callback 回调函数
 */
export function onOpen(callback) {
  callbacks.onOpen = callback;
}

/**
 * 设置连接关闭回调
 * @param {Function} callback 回调函数
 */
export function onClose(callback) {
  callbacks.onClose = callback;
}

/**
 * 设置错误回调
 * @param {Function} callback 回调函数
 */
export function onError(callback) {
  callbacks.onError = callback;
}

/**
 * 获取连接状态
 * @returns {Number} 连接状态
 */
export function getConnectionState() {
  return gazeWebSocket ? gazeWebSocket.readyState : WS_STATE.CLOSED;
}

/**
 * 检查连接是否打开
 * @returns {Boolean} 连接是否打开
 */
export function isConnected() {
  return gazeWebSocket && gazeWebSocket.readyState === WS_STATE.OPEN;
}

/**
 * 发送消息到服务器
 * @param {Object} data 要发送的数据
 * @returns {Boolean} 是否发送成功
 */
export function sendMessage(data) {
  if (!isConnected()) {
    console.error('WebSocket未连接，无法发送消息');
    return false;
  }
  
  try {
    gazeWebSocket.send(JSON.stringify(data));
    return true;
  } catch (error) {
    console.error('发送WebSocket消息失败:', error);
    return false;
  }
}

/**
 * 请求实时视线数据
 * @returns {Boolean} 是否请求成功
 */
export function requestGazeData() {
  return sendMessage({ type: 'request_gaze_data' });
}

/**
 * 停止实时视线数据
 * @returns {Boolean} 是否请求成功
 */
export function stopGazeData() {
  return sendMessage({ type: 'stop_gaze_data' });
}