/**
 * 眼睛检测和视线估计服务
 * 使用TensorFlow.js和face-api.js进行眼睛检测和视线估计
 */

// 导入依赖
import * as tf from '@tensorflow/tfjs';
import * as faceapi from 'face-api.js';
import * as cameraService from './camera';

// 模型加载状态
let modelsLoaded = false;

// 视线估计状态
let gazeEstimationActive = false;

// 回调函数集合
const callbacks = {
  onGazeData: null,
  onFaceDetected: null,
  onEyesDetected: null,
  onModelLoaded: null,
  onError: null
};

// 视线数据处理定时器
let gazeProcessingInterval = null;

// 最后一次检测到的面部特征点
let lastDetectedFaceLandmarks = null;

// 最后一次估计的视线数据
let lastGazeData = {
  timestamp: 0,
  gazeX: 50,
  gazeY: 50,
  confidence: 0,
  leftEye: null,
  rightEye: null,
  facePosition: null
};

/**
 * 初始化眼睛检测和视线估计
 * @param {Object} options 配置选项
 * @returns {Promise<boolean>} 是否成功初始化
 */
export async function initEyeTracking(options = {}) {
  // 默认配置
  const config = {
    modelPath: options.modelPath || '/models',
    useCamera: options.useCamera !== undefined ? options.useCamera : true,
    cameraOptions: options.cameraOptions || {},
    detectionInterval: options.detectionInterval || 100, // 检测间隔（毫秒）
    minConfidence: options.minConfidence || 0.5, // 最小置信度
    useTinyModel: options.useTinyModel !== undefined ? options.useTinyModel : true // 是否使用小型模型
  };
  
  try {
    // 加载TensorFlow.js
    await tf.ready();
    console.log('TensorFlow.js已加载');
    
    // 加载face-api.js模型
    await loadModels(config.modelPath, config.useTinyModel);
    console.log('Face-API.js模型已加载');
    
    // 如果需要使用摄像头
    if (config.useCamera) {
      // 初始化摄像头
      const cameraInitialized = await cameraService.initCamera(config.cameraOptions);
      
      if (!cameraInitialized) {
        throw new Error('摄像头初始化失败');
      }
      
      // 设置帧处理回调
      cameraService.onFrame((imageData, canvas, context) => {
        // 处理视频帧，进行面部和眼睛检测
        processVideoFrame(imageData, canvas, context, config);
      });
    }
    
    // 开始视线估计
    startGazeEstimation(config.detectionInterval, config.minConfidence);
    
    return true;
  } catch (error) {
    console.error('初始化眼睛检测和视线估计失败:', error);
    
    if (callbacks.onError) {
      callbacks.onError(error);
    }
    
    return false;
  }
}

/**
 * 加载face-api.js模型
 * @param {string} modelPath 模型路径
 * @param {boolean} useTinyModel 是否使用小型模型
 */
async function loadModels(modelPath, useTinyModel) {
  // 设置模型路径
  const path = `${modelPath}/`;
  
  try {
    // 加载人脸检测模型
    await faceapi.nets.tinyFaceDetector.loadFromUri(path);
    
    // 加载面部特征点检测模型
    await faceapi.nets.faceLandmark68Net.loadFromUri(path);
    
    // 设置模型已加载标志
    modelsLoaded = true;
    
    // 调用模型加载回调
    if (callbacks.onModelLoaded) {
      callbacks.onModelLoaded();
    }
  } catch (error) {
    console.error('加载模型失败:', error);
    throw error;
  }
}

/**
 * 处理视频帧
 * @param {ImageData} imageData 图像数据
 * @param {HTMLCanvasElement} canvas 画布元素
 * @param {CanvasRenderingContext2D} context 画布上下文
 * @param {Object} config 配置选项
 */
async function processVideoFrame(imageData, canvas, context, config) {
  // 如果模型未加载或视线估计未激活，不处理
  if (!modelsLoaded || !gazeEstimationActive) {
    return;
  }
  
  try {
    // 创建HTML图像元素，用于face-api.js处理
    const tempImage = await createImageFromImageData(imageData);
    
    // 检测人脸
    const detections = await faceapi.detectAllFaces(
      tempImage,
      new faceapi.TinyFaceDetectorOptions({ minConfidence: config.minConfidence })
    ).withFaceLandmarks();
    
    // 如果检测到人脸
    if (detections && detections.length > 0) {
      // 获取第一个检测到的人脸
      const detection = detections[0];
      
      // 保存面部特征点
      lastDetectedFaceLandmarks = detection.landmarks;
      
      // 获取眼睛位置
      const leftEye = getEyePosition(detection.landmarks.getLeftEye());
      const rightEye = getEyePosition(detection.landmarks.getRightEye());
      
      // 获取面部位置
      const facePosition = {
        x: detection.detection.box.x + detection.detection.box.width / 2,
        y: detection.detection.box.y + detection.detection.box.height / 2,
        width: detection.detection.box.width,
        height: detection.detection.box.height
      };
      
      // 调用面部检测回调
      if (callbacks.onFaceDetected) {
        callbacks.onFaceDetected(detection, facePosition);
      }
      
      // 调用眼睛检测回调
      if (callbacks.onEyesDetected) {
        callbacks.onEyesDetected(leftEye, rightEye);
      }
      
      // 估计视线方向
      const gazeData = estimateGaze(leftEye, rightEye, facePosition, canvas.width, canvas.height);
      
      // 更新最后一次视线数据
      lastGazeData = {
        timestamp: Date.now(),
        gazeX: gazeData.x,
        gazeY: gazeData.y,
        confidence: gazeData.confidence,
        leftEye,
        rightEye,
        facePosition
      };
      
      // 调用视线数据回调
      if (callbacks.onGazeData) {
        callbacks.onGazeData(lastGazeData);
      }
      
      // 可选：在画布上绘制调试信息
      if (config.debug) {
        drawDebugInfo(context, detection, leftEye, rightEye, gazeData);
      }
    }
  } catch (error) {
    console.error('处理视频帧失败:', error);
  }
}

/**
 * 从ImageData创建Image对象
 * @param {ImageData} imageData 图像数据
 * @returns {Promise<HTMLImageElement>} 图像元素
 */
function createImageFromImageData(imageData) {
  return new Promise((resolve, reject) => {
    // 创建临时画布
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = imageData.width;
    tempCanvas.height = imageData.height;
    
    // 获取上下文并绘制图像数据
    const tempContext = tempCanvas.getContext('2d');
    tempContext.putImageData(imageData, 0, 0);
    
    // 创建图像元素
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = tempCanvas.toDataURL();
  });
}

/**
 * 获取眼睛位置
 * @param {Array} eyePoints 眼睛特征点数组
 * @returns {Object} 眼睛位置和大小
 */
function getEyePosition(eyePoints) {
  // 计算眼睛的边界框
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  
  // 遍历所有眼睛特征点
  eyePoints.forEach(point => {
    minX = Math.min(minX, point.x);
    minY = Math.min(minY, point.y);
    maxX = Math.max(maxX, point.x);
    maxY = Math.max(maxY, point.y);
  });
  
  // 计算眼睛中心和大小
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  const width = maxX - minX;
  const height = maxY - minY;
  
  return {
    x: centerX,
    y: centerY,
    width,
    height,
    points: eyePoints
  };
}

/**
 * 估计视线方向
 * @param {Object} leftEye 左眼位置
 * @param {Object} rightEye 右眼位置
 * @param {Object} facePosition 面部位置
 * @param {number} canvasWidth 画布宽度
 * @param {number} canvasHeight 画布高度
 * @returns {Object} 视线数据
 */
function estimateGaze(leftEye, rightEye, facePosition, canvasWidth, canvasHeight) {
  // 计算两眼中心点
  const eyesCenterX = (leftEye.x + rightEye.x) / 2;
  const eyesCenterY = (leftEye.y + rightEye.y) / 2;
  
  // 计算眼睛方向向量（从面部中心到眼睛中心）
  const directionX = eyesCenterX - facePosition.x;
  const directionY = eyesCenterY - facePosition.y;
  
  // 归一化方向向量
  const length = Math.sqrt(directionX * directionX + directionY * directionY);
  const normalizedDirectionX = directionX / length;
  const normalizedDirectionY = directionY / length;
  
  // 计算视线位置（简化模型）
  // 这里使用眼睛位置和方向的线性组合来估计视线
  // 实际应用中可能需要更复杂的模型
  const gazeOffsetX = normalizedDirectionX * facePosition.width * 2;
  const gazeOffsetY = normalizedDirectionY * facePosition.height * 2;
  
  // 计算屏幕上的视线位置
  let gazeX = eyesCenterX + gazeOffsetX;
  let gazeY = eyesCenterY + gazeOffsetY;
  
  // 将像素坐标转换为百分比（0-100）
  gazeX = (gazeX / canvasWidth) * 100;
  gazeY = (gazeY / canvasHeight) * 100;
  
  // 确保值在0-100范围内
  gazeX = Math.max(0, Math.min(100, gazeX));
  gazeY = Math.max(0, Math.min(100, gazeY));
  
  // 计算置信度（简化）
  // 这里使用眼睛大小作为置信度的一个因素
  const eyeSize = (leftEye.width + leftEye.height + rightEye.width + rightEye.height) / 4;
  const maxEyeSize = Math.min(canvasWidth, canvasHeight) * 0.1; // 假设最大眼睛大小为画布尺寸的10%
  const confidence = Math.min(1, eyeSize / maxEyeSize);
  
  return {
    x: gazeX,
    y: gazeY,
    confidence
  };
}

/**
 * 在画布上绘制调试信息
 * @param {CanvasRenderingContext2D} context 画布上下文
 * @param {Object} detection 人脸检测结果
 * @param {Object} leftEye 左眼位置
 * @param {Object} rightEye 右眼位置
 * @param {Object} gazeData 视线数据
 */
function drawDebugInfo(context, detection, leftEye, rightEye, gazeData) {
  // 绘制人脸边界框
  context.strokeStyle = '#00ff00';
  context.lineWidth = 2;
  context.strokeRect(
    detection.detection.box.x,
    detection.detection.box.y,
    detection.detection.box.width,
    detection.detection.box.height
  );
  
  // 绘制面部特征点
  context.fillStyle = '#00ff00';
  detection.landmarks.positions.forEach(point => {
    context.beginPath();
    context.arc(point.x, point.y, 2, 0, 2 * Math.PI);
    context.fill();
  });
  
  // 绘制眼睛
  context.strokeStyle = '#0000ff';
  context.lineWidth = 2;
  
  // 左眼
  context.strokeRect(
    leftEye.x - leftEye.width / 2,
    leftEye.y - leftEye.height / 2,
    leftEye.width,
    leftEye.height
  );
  
  // 右眼
  context.strokeRect(
    rightEye.x - rightEye.width / 2,
    rightEye.y - rightEye.height / 2,
    rightEye.width,
    rightEye.height
  );
  
  // 绘制视线点
  const gazeX = (gazeData.x / 100) * context.canvas.width;
  const gazeY = (gazeData.y / 100) * context.canvas.height;
  
  context.fillStyle = '#ff0000';
  context.beginPath();
  context.arc(gazeX, gazeY, 10, 0, 2 * Math.PI);
  context.fill();
  
  // 绘制视线数据文本
  context.fillStyle = '#ffffff';
  context.font = '14px Arial';
  context.fillText(`Gaze: (${gazeData.x.toFixed(1)}%, ${gazeData.y.toFixed(1)}%)`, 10, 20);
  context.fillText(`Confidence: ${gazeData.confidence.toFixed(2)}`, 10, 40);
}

/**
 * 开始视线估计
 * @param {number} interval 检测间隔（毫秒）
 * @param {number} minConfidence 最小置信度
 */
export function startGazeEstimation(interval = 100, minConfidence = 0.5) {
  // 设置视线估计为活跃状态
  gazeEstimationActive = true;
  
  // 清除可能存在的旧定时器
  if (gazeProcessingInterval) {
    clearInterval(gazeProcessingInterval);
  }
  
  // 创建新定时器，定期发送视线数据
  gazeProcessingInterval = setInterval(() => {
    // 如果有视线数据回调且最后一次视线数据的置信度大于最小置信度
    if (callbacks.onGazeData && lastGazeData.confidence >= minConfidence) {
      callbacks.onGazeData({
        timestamp: Date.now(),
        gaze_position: [lastGazeData.gazeX, lastGazeData.gazeY],
        confidence: lastGazeData.confidence
      });
    }
  }, interval);
}

/**
 * 停止视线估计
 */
export function stopGazeEstimation() {
  // 设置视线估计为非活跃状态
  gazeEstimationActive = false;
  
  // 清除定时器
  if (gazeProcessingInterval) {
    clearInterval(gazeProcessingInterval);
    gazeProcessingInterval = null;
  }
}

/**
 * 停止眼睛检测和视线估计
 */
export function stopEyeTracking() {
  // 停止视线估计
  stopGazeEstimation();
  
  // 停止摄像头
  cameraService.stopCamera();
}

/**
 * 设置视线数据回调
 * @param {Function} callback 回调函数
 */
export function onGazeData(callback) {
  callbacks.onGazeData = callback;
}

/**
 * 设置面部检测回调
 * @param {Function} callback 回调函数
 */
export function onFaceDetected(callback) {
  callbacks.onFaceDetected = callback;
}

/**
 * 设置眼睛检测回调
 * @param {Function} callback 回调函数
 */
export function onEyesDetected(callback) {
  callbacks.onEyesDetected = callback;
}

/**
 * 设置模型加载回调
 * @param {Function} callback 回调函数
 */
export function onModelLoaded(callback) {
  callbacks.onModelLoaded = callback;
}

/**
 * 设置错误回调
 * @param {Function} callback 回调函数
 */
export function onError(callback) {
  callbacks.onError = callback;
}

/**
 * 获取最后一次视线数据
 * @returns {Object} 视线数据
 */
export function getLastGazeData() {
  return lastGazeData;
}

/**
 * 检查模型是否已加载
 * @returns {boolean} 是否已加载
 */
export function areModelsLoaded() {
  return modelsLoaded;
}

/**
 * 检查视线估计是否活跃
 * @returns {boolean} 是否活跃
 */
export function isGazeEstimationActive() {
  return gazeEstimationActive;
}