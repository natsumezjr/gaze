/**
 * 摄像头服务模块
 * 用于访问用户摄像头和处理视频流
 */

// 存储摄像头流和视频元素
let cameraStream = null;
let videoElement = null;
let canvasElement = null;
let canvasContext = null;

// 摄像头状态
const CAMERA_STATE = {
  INACTIVE: 'inactive',
  REQUESTING: 'requesting',
  ACTIVE: 'active',
  ERROR: 'error'
};

// 当前状态
let currentState = CAMERA_STATE.INACTIVE;

// 回调函数集合
const callbacks = {
  onFrame: null,
  onError: null,
  onStateChange: null
};

// 帧处理定时器
let frameProcessingInterval = null;

/**
 * 初始化摄像头
 * @param {Object} options 配置选项
 * @returns {Promise<boolean>} 是否成功初始化
 */
export async function initCamera(options = {}) {
  // 默认配置
  const config = {
    videoElementId: options.videoElementId || 'camera-video',
    canvasElementId: options.canvasElementId || 'camera-canvas',
    width: options.width || 640,
    height: options.height || 480,
    frameRate: options.frameRate || 30,
    facingMode: options.facingMode || 'user', // 'user'前置摄像头，'environment'后置摄像头
    createElements: options.createElements || false, // 是否自动创建视频和画布元素
    parentElement: options.parentElement || document.body, // 父元素
    hidden: options.hidden || true // 是否隐藏视频元素
  };
  
  try {
    // 更新状态
    updateState(CAMERA_STATE.REQUESTING);
    
    // 如果需要创建元素
    if (config.createElements) {
      createVideoAndCanvasElements(config);
    } else {
      // 获取已有元素
      videoElement = document.getElementById(config.videoElementId);
      canvasElement = document.getElementById(config.canvasElementId);
      
      // 如果元素不存在，创建它们
      if (!videoElement || !canvasElement) {
        createVideoAndCanvasElements(config);
      }
    }
    
    // 设置画布上下文
    canvasContext = canvasElement.getContext('2d');
    
    // 请求摄像头权限
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: config.width },
        height: { ideal: config.height },
        frameRate: { ideal: config.frameRate },
        facingMode: config.facingMode
      },
      audio: false
    });
    
    // 保存流
    cameraStream = stream;
    
    // 设置视频源
    videoElement.srcObject = stream;
    videoElement.play();
    
    // 等待视频元素加载完成
    await new Promise((resolve) => {
      videoElement.onloadedmetadata = () => {
        resolve();
      };
    });
    
    // 更新状态
    updateState(CAMERA_STATE.ACTIVE);
    
    // 开始处理帧
    startFrameProcessing(config.frameRate);
    
    return true;
  } catch (error) {
    console.error('初始化摄像头失败:', error);
    updateState(CAMERA_STATE.ERROR);
    
    if (callbacks.onError) {
      callbacks.onError(error);
    }
    
    return false;
  }
}

/**
 * 创建视频和画布元素
 * @param {Object} config 配置选项
 */
function createVideoAndCanvasElements(config) {
  // 创建视频元素
  videoElement = document.createElement('video');
  videoElement.id = config.videoElementId;
  videoElement.width = config.width;
  videoElement.height = config.height;
  videoElement.autoplay = true;
  videoElement.playsInline = true; // 对iOS很重要
  
  // 如果需要隐藏
  if (config.hidden) {
    videoElement.style.position = 'absolute';
    videoElement.style.opacity = '0';
    videoElement.style.pointerEvents = 'none';
  }
  
  // 创建画布元素
  canvasElement = document.createElement('canvas');
  canvasElement.id = config.canvasElementId;
  canvasElement.width = config.width;
  canvasElement.height = config.height;
  
  // 如果需要隐藏
  if (config.hidden) {
    canvasElement.style.position = 'absolute';
    canvasElement.style.opacity = '0';
    canvasElement.style.pointerEvents = 'none';
  }
  
  // 添加到父元素
  config.parentElement.appendChild(videoElement);
  config.parentElement.appendChild(canvasElement);
}

/**
 * 开始处理视频帧
 * @param {number} frameRate 帧率
 */
function startFrameProcessing(frameRate) {
  // 计算帧间隔（毫秒）
  const frameInterval = 1000 / frameRate;
  
  // 清除可能存在的旧定时器
  if (frameProcessingInterval) {
    clearInterval(frameProcessingInterval);
  }
  
  // 创建新定时器
  frameProcessingInterval = setInterval(() => {
    // 如果摄像头不活跃，停止处理
    if (currentState !== CAMERA_STATE.ACTIVE) {
      clearInterval(frameProcessingInterval);
      return;
    }
    
    // 将视频帧绘制到画布上
    canvasContext.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
    
    // 获取图像数据
    const imageData = canvasContext.getImageData(0, 0, canvasElement.width, canvasElement.height);
    
    // 如果有帧处理回调，调用它
    if (callbacks.onFrame) {
      callbacks.onFrame(imageData, canvasElement, canvasContext);
    }
  }, frameInterval);
}

/**
 * 停止摄像头
 */
export function stopCamera() {
  // 停止帧处理
  if (frameProcessingInterval) {
    clearInterval(frameProcessingInterval);
    frameProcessingInterval = null;
  }
  
  // 停止视频流
  if (cameraStream) {
    cameraStream.getTracks().forEach(track => track.stop());
    cameraStream = null;
  }
  
  // 清除视频源
  if (videoElement) {
    videoElement.srcObject = null;
  }
  
  // 更新状态
  updateState(CAMERA_STATE.INACTIVE);
}

/**
 * 更新摄像头状态
 * @param {string} newState 新状态
 */
function updateState(newState) {
  currentState = newState;
  
  if (callbacks.onStateChange) {
    callbacks.onStateChange(newState);
  }
}

/**
 * 获取当前摄像头状态
 * @returns {string} 当前状态
 */
export function getCameraState() {
  return currentState;
}

/**
 * 检查摄像头是否活跃
 * @returns {boolean} 是否活跃
 */
export function isCameraActive() {
  return currentState === CAMERA_STATE.ACTIVE;
}

/**
 * 设置帧处理回调
 * @param {Function} callback 回调函数
 */
export function onFrame(callback) {
  callbacks.onFrame = callback;
}

/**
 * 设置错误回调
 * @param {Function} callback 回调函数
 */
export function onError(callback) {
  callbacks.onError = callback;
}

/**
 * 设置状态变化回调
 * @param {Function} callback 回调函数
 */
export function onStateChange(callback) {
  callbacks.onStateChange = callback;
}

/**
 * 获取摄像头视频元素
 * @returns {HTMLVideoElement} 视频元素
 */
export function getVideoElement() {
  return videoElement;
}

/**
 * 获取画布元素
 * @returns {HTMLCanvasElement} 画布元素
 */
export function getCanvasElement() {
  return canvasElement;
}

/**
 * 获取画布上下文
 * @returns {CanvasRenderingContext2D} 画布上下文
 */
export function getCanvasContext() {
  return canvasContext;
}

/**
 * 拍摄当前帧
 * @returns {string} 图像数据URL
 */
export function captureFrame() {
  if (!canvasElement || currentState !== CAMERA_STATE.ACTIVE) {
    return null;
  }
  
  // 将当前视频帧绘制到画布上
  canvasContext.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
  
  // 返回图像数据URL
  return canvasElement.toDataURL('image/png');
}

/**
 * 检查浏览器是否支持摄像头API
 * @returns {boolean} 是否支持
 */
export function isCameraSupported() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}