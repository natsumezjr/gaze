import { ref, reactive, onMounted, onUnmounted, computed } from 'vue';
import calibrationConfig from '../config/calibration.config';
import * as apiService from '../services/api';
import INTEGRATION_CONFIG from '../config/integration.config';
import { initGazeWebSocket, closeGazeWebSocket, onGazeData as onWebSocketGazeData, onError as onWebSocketError } from '../services/websocket';
import * as calibrationService from '../services/calibration';
import * as cameraService from '../services/camera';
import * as eyeTrackingService from '../services/eyeTracking';

/**
 * 校准功能组合式API
 */
export function useCalibration(options = {}) {
  // 校准点坐标数组 - 从配置中获取六边形布局
  const calibrationPoints = reactive(calibrationConfig.points.hexagon || [
    { x: 33, y: 20 },   // 左上（左三分之一线上）
    { x: 67, y: 20 },   // 右上（右三分之一线上）
    { x: 10, y: 50 },   // 左中（靠近左侧边缘）
    { x: 90, y: 50 },   // 右中（靠近右侧边缘）
    { x: 33, y: 80 },   // 左下（左三分之一线上）
    { x: 67, y: 80 }    // 右下（右三分之一线上）
  ]);

  // 校准点颜色数组 - 从配置中获取白色主题
  const calibrationColors = reactive(calibrationConfig.themes.light || [
    { bg: '#DC3545', fg: '#C82333' }, // 深红色系
    { bg: '#20C997', fg: '#1E7E34' }, // 深绿色系
    { bg: '#0D6EFD', fg: '#0A58CA' }, // 深蓝色系
    { bg: '#198754', fg: '#155724' }, // 深薄荷绿系
    { bg: '#FD7E14', fg: '#E8590C' }, // 深橙色系
    { bg: '#6610F2', fg: '#520DC2' }  // 深紫色系
  ]);

  // 状态变量
  const isCalibrating = ref(false);
  const currentPointIndex = ref(0);
  const collectedData = ref([]);
  
  // 真实视线输入相关状态
  const realGazePosition = ref({ x: 50, y: 50 }); // 用户当前真实注视点位置
  const showRealGaze = ref(false); // 是否显示真实注视点
  const isWithinTargetThreshold = ref(false); // 是否在目标点阈值范围内
  const gazeAccuracyThreshold = ref(calibrationConfig.gaze.accuracyThreshold || 5); // 视线精度阈值（百分比）
  const requiredFixationTime = ref(calibrationConfig.gaze.requiredFixationTime || 500); // 需要注视的时间（毫秒）
  const fixationStartTime = ref(null); // 开始注视的时间戳
  const canProceedToNextPoint = ref(false); // 是否可以继续到下一个校准点
  
  // 视线平滑处理
  const gazeHistorySize = 5; // 历史数据大小
  const gazeHistory = ref([]); // 视线历史数据
  
  // 眼动追踪状态
  const eyeTrackingInitialized = ref(false);
  const usingCameraTracking = ref(false);
  
  // 模拟视线数据定时器
  let mockGazeInterval = null;
  
  // 更新布局函数 - 支持动态切换布局
  function updateLayout(layout) {
    if (!isCalibrating.value && calibrationConfig.points[layout]) {
      Object.assign(calibrationPoints, calibrationConfig.points[layout]);
      console.log(`布局已更新为: ${layout}`);
    } else if (isCalibrating.value) {
      console.warn('校准过程中无法更改布局');
    }
  }
  
  // 更新主题函数 - 支持动态切换主题
  function updateTheme(theme) {
    if (!isCalibrating.value && calibrationConfig.themes[theme]) {
      Object.assign(calibrationColors, calibrationConfig.themes[theme]);
      console.log(`主题已更新为: ${theme}`);
    } else if (isCalibrating.value) {
      console.warn('校准过程中无法更改主题');
    }
  }
  
  // 更新后端集成配置
  function updateBackendConfig(useRealBackend) {
    if (!isCalibrating.value) {
      INTEGRATION_CONFIG.USE_REAL_BACKEND = useRealBackend;
      console.log(`后端集成已${useRealBackend ? '启用' : '禁用'}`);
    } else if (isCalibrating.value) {
      console.warn('校准过程中无法更改后端配置');
    }
  }
  
  // DOM元素引用
  let movingCircle = null;
  let innerCircle = null;
  let calibrationContainer = null;

  /**
   * 初始化眼动追踪
   */
  async function initEyeTracking() {
    try {
      // 检查浏览器是否支持摄像头API
      if (!cameraService.isCameraSupported()) {
        throw new Error('浏览器不支持摄像头API');
      }
      
      // 初始化眼睛检测和视线估计
      const initialized = await eyeTrackingService.initEyeTracking({
        modelPath: './models',
        useCamera: true,
        cameraOptions: {
          width: 640,
          height: 480,
          frameRate: 30,
          createElements: true,
          hidden: false,
          parentElement: document.getElementById('calibration-container') || document.body
        },
        detectionInterval: 100,
        minConfidence: 0.5,
        useTinyModel: true,
        debug: true
      });
      
      if (!initialized) {
        throw new Error('初始化眼睛检测和视线估计失败');
      }
      
      // 设置视线数据回调
      eyeTrackingService.onGazeData((gazeData) => {
        // 更新真实视线位置
        updateGazePositionWithSmoothing({
          x: gazeData.gaze_position[0],
          y: gazeData.gaze_position[1]
        });
        
        // 确保显示真实视线点
        if (!showRealGaze.value && isCalibrating.value) {
          showRealGaze.value = true;
          console.log('显示真实视线点');
        }
      });
      
      // 设置错误回调
      eyeTrackingService.onError((error) => {
        console.error('眼睛检测和视线估计错误:', error);
        // 如果使用摄像头追踪失败，关闭视线显示
        usingCameraTracking.value = false;
        showRealGaze.value = false;
      });
      
      // 设置眼动追踪初始化状态
      eyeTrackingInitialized.value = true;
      usingCameraTracking.value = true;
      
      // 立即显示真实视线点
      showRealGaze.value = true;
      
      console.log('眼动追踪初始化成功，显示真实视线点');
      return true;
    } catch (error) {
      console.error('初始化眼动追踪失败:', error);
      // 如果初始化失败，关闭视线显示
      eyeTrackingInitialized.value = false;
      usingCameraTracking.value = false;
      showRealGaze.value = false;
      return false;
    }
  }

  /**
   * 开始校准流程
   */
  async function startCalibration() {
    if (isCalibrating.value) return;
    
    console.log('开始校准流程');
    isCalibrating.value = true;
    collectedData.value = [];
    currentPointIndex.value = 0;
    canProceedToNextPoint.value = false;
    gazeHistory.value = [];
    
    // 隐藏标题和按钮
    if (options.onCalibrationStart) {
      options.onCalibrationStart();
    }
    
    // 显示进度指示器
    showProgressIndicator();
    
    // 强制显示真实视线点
    showRealGaze.value = true;
    console.log('强制显示真实视线点');
    
    // 初始化眼动追踪（如果尚未初始化）
    if (!eyeTrackingInitialized.value) {
      const initialized = await initEyeTracking();
      if (!initialized) {
        console.warn('眼动追踪初始化失败，将不显示视线点');
        showRealGaze.value = false;
      } else {
        // 只有在成功初始化眼动追踪后才显示真实视线点
        showRealGaze.value = true;
        console.log('眼动追踪初始化成功，显示真实视线点');
      }
    } else {
      // 已经初始化过，检查是否使用摄像头追踪
      showRealGaze.value = usingCameraTracking.value;
      console.log(`眼动追踪已初始化，${showRealGaze.value ? '显示' : '不显示'}真实视线点`);
    }
    
    // 开始倒计时
    startCountdown();
  }

  /**
   * 开始倒计时
   */
  function startCountdown() {
    console.log('开始倒计时');
    
    // 获取校准容器
    calibrationContainer = document.getElementById('calibration-container');
    
    // 创建移动圆圈
    movingCircle = document.getElementById('moving-circle');
    
    // 确保移动圆圈可见并设置初始样式
    if (movingCircle) {
      movingCircle.classList.add('countdown');
      movingCircle.style.opacity = '1';
      movingCircle.style.transition = 'all 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94)';
      // 确保圆圈在屏幕中央
      movingCircle.style.left = '50%';
      movingCircle.style.top = '50%';
      console.log('倒计时圆圈已显示');
    }
    
    // 3秒后移动到第一个点
    setTimeout(() => {
      // 倒计时结束，移除倒计时样式
      if (movingCircle) {
        movingCircle.classList.remove('countdown');
        console.log('倒计时结束，准备移动到第一个校准点');
      }
      moveToFirstPoint();
    }, 3000);
  }

  /**
   * 移动到第一个校准点
   */
  function moveToFirstPoint() {
    console.log('移动到第一个校准点');
    
    const firstPoint = calibrationPoints[0];
    const firstColor = calibrationColors[0];
    
    // 开始移动到第一个点
    if (movingCircle) {
      // 设置移动动画
      movingCircle.style.transition = 'all 1.0s cubic-bezier(0.25, 0.46, 0.45, 0.94)';
      movingCircle.style.left = `${firstPoint.x}%`;
      movingCircle.style.top = `${firstPoint.y}%`;
      movingCircle.style.backgroundColor = firstColor.fg; // 变为内圈亮色
      movingCircle.style.opacity = '1';
      console.log(`移动到第一个点，颜色渐变: ${firstColor.fg}`);
    }
    
    // 移动完成后创建内圈并开始校准
    setTimeout(() => {
      if (movingCircle) {
        // 圆圈变为外圈暗色
        movingCircle.style.backgroundColor = firstColor.bg;
        
        // 创建内圈亮色
        createInnerCircle(firstColor.fg);
      }
      startCalibrationAtPoint(0);
    }, 1000);
  }
  
  /**
   * 创建内圈
   */
  function createInnerCircle(color) {
    // 通知组件显示内圈
    if (options.onCreateInnerCircle) {
      options.onCreateInnerCircle(color);
    }
    console.log(`创建内圈，颜色: ${color}`);
  }

  /**
   * 在当前点开始校准
   */
  function startCalibrationAtPoint(pointIndex) {
    console.log(`校准点 ${pointIndex + 1}/${calibrationPoints.length}`);
    updateProgress(pointIndex + 1, calibrationPoints.length);
    
    // 更新当前点索引
    currentPointIndex.value = pointIndex;
    canProceedToNextPoint.value = false;
    
    // 通知点变化
    if (options.onPointChange) {
      options.onPointChange(pointIndex);
    }
    
    // 采集数据
    collectGazeData(calibrationPoints[pointIndex]);
    
    // 等待用户注视点达到要求后再继续
    waitForFixationAndProceed(pointIndex);
  }

  /**
   * 等待用户注视并继续到下一个点
   */
  function waitForFixationAndProceed(pointIndex) {
    // 创建监测定时器
    const checkInterval = setInterval(() => {
      if (!isCalibrating.value) {
        clearInterval(checkInterval);
        return;
      }
      
      // 如果可以继续到下一个点
      if (canProceedToNextPoint.value) {
        clearInterval(checkInterval);
        
        // 如果是最后一个点，完成校准
        if (pointIndex + 1 >= calibrationPoints.length) {
          finishCalibration();
        } else {
          // 否则移动到下一个点
          moveToNextPoint(pointIndex + 1);
        }
      }
    }, 100); // 每100ms检查一次
  }

  /**
   * 计算两点间距离
   */
  function calculateDistance(point1, point2) {
    const dx = point1.x - point2.x;
    const dy = point1.y - point2.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  /**
   * 根据距离计算移动时间（统一速度）
   */
  function calculateMoveTime(currentIndex, nextIndex) {
    const currentPoint = calibrationPoints[currentIndex];
    const nextPoint = calibrationPoints[nextIndex];
    const distance = calculateDistance(currentPoint, nextPoint);
    
    // 设定基准速度（距离100对应1秒）
    const baseSpeed = 100; // 每100单位距离用时1秒
    const minTime = 0.6; // 最小时间0.6秒
    const maxTime = 1.4; // 最大时间1.4秒
    
    const calculatedTime = distance / baseSpeed;
    return Math.max(minTime, Math.min(maxTime, calculatedTime));
  }

  /**
   * 移动到下一个校准点
   */
  function moveToNextPoint(nextIndex) {
    const nextPoint = calibrationPoints[nextIndex];
    const nextColor = calibrationColors[nextIndex];
    
    // 计算移动时间（统一速度）
    const moveTime = calculateMoveTime(currentPointIndex.value, nextIndex);
    
    console.log(`移动到校准点 ${nextIndex + 1}，距离: ${calculateDistance(calibrationPoints[currentPointIndex.value], nextPoint).toFixed(1)}，时间: ${moveTime.toFixed(2)}s`);
    
    // 重置状态
    canProceedToNextPoint.value = false;
    
    // 移动完成后开始下一轮校准
    setTimeout(() => {
      // 开始下一个点的校准
      setTimeout(() => {
        startCalibrationAtPoint(nextIndex);
      }, 100);
    }, moveTime * 1000); // 根据实际移动时间调整
  }

  /**
   * 完成校准
   */
  function finishCalibration() {
    console.log('所有校准点完成');
    
    // 完成校准流程
    completeCalibration();
  }

  /**
   * 采集眼动数据
   * @param {Object} point - 当前校准点坐标
   */
  function collectGazeData(point) {
    console.log(`正在采集点 (${point.x}, ${point.y}) 的数据...`);
    
    // 确保显示真实视线点
    if (isCalibrating.value && usingCameraTracking.value) {
      showRealGaze.value = true;
      console.log('采集数据时显示真实视线点');
    }
    
    if (INTEGRATION_CONFIG.USE_REAL_BACKEND) {
      // 真实后端集成
      collectRealGazeData(point);
    } else {
      // 模拟数据采集
      collectMockGazeData(point);
    }
    
    // 启动视线位置监测
    startGazeMonitoring(point);
  }

  /**
   * 真实后端数据采集（集成接口）
   * @param {Object} point - 当前校准点坐标
   */
  async function collectRealGazeData(point) {
    try {
      // 告诉后端开始采集这个校准点
      const startResult = await apiService.startPointCollection(
        currentPointIndex.value,
        [point.x, point.y],
        INTEGRATION_CONFIG.SAMPLE_DURATION
      );
      
      console.log(`后端开始采集点 ${currentPointIndex.value + 1} 的数据:`, startResult);
      
      // 等待采集完成
      setTimeout(async () => {
        try {
          const result = await apiService.stopPointCollection(currentPointIndex.value);
          console.log(`完成采集点 ${currentPointIndex.value + 1}:`, result);
          
          // 将后端采集的数据添加到collectedData中
          if (result.samples) {
            result.samples.forEach((sample, index) => {
              const gazeData = {
                pointId: currentPointIndex.value,
                targetX: point.x,
                targetY: point.y,
                timestamp: new Date(sample.timestamp).getTime(),
                sampleIndex: index,
                gazeX: sample.pupil_center.left[0],
                gazeY: sample.pupil_center.left[1],
                confidence: sample.confidence,
                pupilDiameter: calculatePupilDiameter(sample.pupil_center)
              };
              collectedData.value.push(gazeData);
            });
          }
          
        } catch (error) {
          console.error('停止采集失败:', error);
        }
      }, INTEGRATION_CONFIG.SAMPLE_DURATION);
      
    } catch (error) {
      console.error('开始采集失败:', error);
      // 回退到模拟数据
      collectMockGazeData(point);
    }
  }

  /**
   * 模拟数据采集（开发测试用）
   * @param {Object} point - 当前校准点坐标
   */
  function collectMockGazeData(point) {
    const sampleInterval = 100; // 每100ms采样一次
    const sampleCount = 2000 / sampleInterval; // 2秒内采样次数
    
    for (let i = 0; i < sampleCount; i++) {
      setTimeout(() => {
        // 模拟采集到的眼动数据 - 减小抖动范围，提高灵敏度
        const jitter = 2.0; // 减小抖动范围（原来是5）
        const gazeData = {
          pointId: currentPointIndex.value,
          targetX: point.x,
          targetY: point.y,
          timestamp: Date.now(),
          sampleIndex: i,
          gazeX: point.x + (Math.random() - 0.5) * jitter,
          gazeY: point.y + (Math.random() - 0.5) * jitter,
          confidence: 0.9 + Math.random() * 0.1, // 提高置信度
          pupilDiameter: 3.0 + Math.random() * 1.0
        };
        
        collectedData.value.push(gazeData);
        
        if (i === 0) {
          console.log(`开始采集点 ${currentPointIndex.value + 1} 的数据`);
        } else if (i === sampleCount - 1) {
          console.log(`完成采集点 ${currentPointIndex.value + 1} 的数据`);
        }
        
      }, i * sampleInterval);
    }
  }

  /**
   * 更新视线位置并应用平滑处理
   * @param {Object} newPosition - 新的视线位置
   */
  function updateGazePositionWithSmoothing(newPosition) {
    // 添加到历史数据
    gazeHistory.value.push(newPosition);
    
    // 保持历史数据大小
    if (gazeHistory.value.length > gazeHistorySize) {
      gazeHistory.value.shift();
    }
    
    // 计算平均位置
    if (gazeHistory.value.length > 0) {
      const avgX = gazeHistory.value.reduce((sum, pos) => sum + pos.x, 0) / gazeHistory.value.length;
      const avgY = gazeHistory.value.reduce((sum, pos) => sum + pos.y, 0) / gazeHistory.value.length;
      
      // 更新平滑后的视线位置
      realGazePosition.value = {
        x: avgX,
        y: avgY
      };
      
      // 始终显示视线点
      showRealGaze.value = true;
      console.log(`视线位置更新: (${avgX.toFixed(1)}, ${avgY.toFixed(1)})`);
      
      // 如果在校准中，检查是否在目标点附近
      if (isCalibrating.value && currentPointIndex.value >= 0) {
        const targetPoint = calibrationPoints[currentPointIndex.value];
        const distance = Math.sqrt(
          Math.pow(avgX - targetPoint.x, 2) + 
          Math.pow(avgY - targetPoint.y, 2)
        );
        
        isWithinTargetThreshold.value = distance < gazeAccuracyThreshold.value;
      }
    }
  }
  
  /**
   * 启动模拟视线数据生成
   */
  function startSimulatedGazeData() {
    // 如果已经在使用摄像头追踪，则不需要模拟数据
    if (usingCameraTracking.value && eyeTrackingInitialized.value) {
      console.log('使用真实眼动追踪，不启动模拟数据');
      return;
    }
    
    console.log('启动模拟视线数据生成');
    
    // 清除可能存在的旧定时器
    stopSimulatedGazeData();
    
    // 强制显示视线点
    showRealGaze.value = true;
    
    // 创建模拟视线数据定时器
    mockGazeInterval = setInterval(() => {
      if (!isCalibrating.value) {
        stopSimulatedGazeData();
        return;
      }
      
      // 获取当前目标点
      const targetPoint = calibrationPoints[currentPointIndex.value];
      
      // 计算模拟视线位置（逐渐靠近目标点）
      const currentGaze = realGazePosition.value;
      const moveSpeed = 0.1; // 移动速度因子
      
      // 添加一些随机抖动
      const jitterAmount = 2.0;
      const jitterX = (Math.random() - 0.5) * jitterAmount;
      const jitterY = (Math.random() - 0.5) * jitterAmount;
      
      // 计算新位置（向目标点移动）
      const newX = currentGaze.x + (targetPoint.x - currentGaze.x) * moveSpeed + jitterX;
      const newY = currentGaze.y + (targetPoint.y - currentGaze.y) * moveSpeed + jitterY;
      
      // 直接更新视线位置，不经过平滑处理
      realGazePosition.value = {
        x: newX,
        y: newY
      };
      
      // 强制显示模拟视线点
      showRealGaze.value = true;
      
      // 输出调试信息
      console.log(`模拟视线位置: (${newX.toFixed(1)}, ${newY.toFixed(1)}), 目标点: (${targetPoint.x}, ${targetPoint.y})`);
      
      // 检查是否在目标点附近
      const distance = Math.sqrt(
        Math.pow(newX - targetPoint.x, 2) + 
        Math.pow(newY - targetPoint.y, 2)
      );
      
      isWithinTargetThreshold.value = distance < gazeAccuracyThreshold.value;
    }, 50); // 每50ms更新一次
  }
  
  /**
   * 停止模拟视线数据生成
   */
  function stopSimulatedGazeData() {
    if (mockGazeInterval) {
      clearInterval(mockGazeInterval);
      mockGazeInterval = null;
      console.log('停止模拟视线数据生成');
    }
  }

  /**
   * 计算瞳孔直径（辅助函数）
   * @param {Object} pupilCenter - 瞳孔中心数据
   * @returns {number} 瞳孔直径
   */
  function calculatePupilDiameter(pupilCenter) {
    // 简单估算，实际应用中可能需要更复杂的计算
    return 3.0 + Math.random() * 1.0;
  }

  /**
   * 显示进度指示器
   */
  function showProgressIndicator() {
    let progressIndicator = document.querySelector('.progress-indicator');
    
    if (progressIndicator) {
      progressIndicator.classList.add('visible');
      updateProgress(0, calibrationPoints.length);
    }
  }

  /**
   * 更新进度显示
   */
  function updateProgress(current, total) {
    const progressIndicator = document.querySelector('.progress-indicator');
    if (progressIndicator) {
      progressIndicator.textContent = `校准进度: ${current}/${total}`;
    }
  }

  /**
   * 隐藏进度指示器
   */
  function hideProgressIndicator() {
    const progressIndicator = document.querySelector('.progress-indicator');
    if (progressIndicator) {
      progressIndicator.classList.remove('visible');
    }
  }

  /**
   * 完成校准流程
   */
  function completeCalibration() {
    console.log('校准完成');
    isCalibrating.value = false;
    
    // 发送数据到服务器
    sendDataToServer(collectedData.value);
    
    // 隐藏进度指示器
    hideProgressIndicator();
    
    // 隐藏真实视线点
    showRealGaze.value = false;
    
    // 停止眼动追踪
    if (usingCameraTracking.value) {
      eyeTrackingService.stopGazeEstimation();
    }
    
    // 通知校准结束
    if (options.onCalibrationEnd) {
      options.onCalibrationEnd();
    }
  }

  /**
   * 重置校准状态
   */
  function resetCalibration() {
    currentPointIndex.value = 0;
    isCalibrating.value = false;
    collectedData.value = [];
    showRealGaze.value = false;
    canProceedToNextPoint.value = false;
    gazeHistory.value = [];
    
    // 停止眼动追踪
    if (usingCameraTracking.value) {
      eyeTrackingService.stopGazeEstimation();
    }
    
    // 停止模拟视线数据
    stopSimulatedGazeData();
    
    console.log('校准状态已重置');
  }

  /**
   * 将采集到的数据发送到服务器
   */
  async function sendDataToServer(data) {
    console.log('将所有采集数据发送到服务器...');
    console.log('采集到的数据总数:', data.length);
    
    if (INTEGRATION_CONFIG.USE_REAL_BACKEND) {
      // 真实后端集成
      await sendDataToRealBackend(data);
    } else {
      // 模拟发送
      sendDataToMockBackend(data);
    }
  }

  /**
   * 发送数据到真实后端（集成接口）
   */
  async function sendDataToRealBackend(data) {
    try {
      // 转换数据格式 - 按照前后端集成接口文档要求
      const calibrationData = {
        metadata: {
          description: "眼动追踪校准数据",
          created_at: new Date().toISOString(),
          data_format: "JSON",
          coordinate_system: "camera_coordinates_mm",
          sample_count: data.length,
          calibration_type: "6_point_hexagon",
          hardware: "web_browser"
        },
        eyes: [],
        pupils: [],
        target_pixels: [],
        camera_matrix: [
          [1000.0, 0.0, 960.0],
          [0.0, 1000.0, 540.0],
          [0.0, 0.0, 1.0]
        ],
        screen_resolution: [window.screen.width, window.screen.height],
        units: {
          coordinates: "millimeters",
          pixels: "pixels",
          angles: "degrees"
        },
        calibration_info: {
          method: "6_point_hexagon",
          duration_seconds: INTEGRATION_CONFIG.SAMPLE_DURATION / 1000,
          samples_per_point: data.filter(d => d.pointId === 0).length,
          point_order: calibrationPoints.map((_, index) => `point_${index + 1}`)
        }
      };
      
      // 处理数据
      data.forEach(sample => {
        // 如果使用摄像头追踪，使用真实的眼球和瞳孔数据
        if (usingCameraTracking.value && eyeTrackingService.getLastGazeData()) {
          const lastGazeData = eyeTrackingService.getLastGazeData();
          if (lastGazeData.leftEye && lastGazeData.rightEye) {
            // 使用真实的眼球和瞳孔数据
            const leftEye = lastGazeData.leftEye;
            const rightEye = lastGazeData.rightEye;
            
            // 转换为后端需要的格式
            calibrationData.eyes.push([leftEye.x / window.innerWidth, leftEye.y / window.innerHeight, 0]);
            calibrationData.pupils.push([rightEye.x / window.innerWidth, rightEye.y / window.innerHeight, 0.1]);
          } else {
            // 如果没有真实数据，使用模拟数据
            const eyeCenter = simulateEyeCenter(sample);
            const pupilCenter = simulatePupilCenter(sample);
            
            calibrationData.eyes.push(eyeCenter);
            calibrationData.pupils.push(pupilCenter);
          }
        } else {
          // 否则使用模拟数据
          const eyeCenter = simulateEyeCenter(sample);
          const pupilCenter = simulatePupilCenter(sample);
          
          calibrationData.eyes.push(eyeCenter);
          calibrationData.pupils.push(pupilCenter);
        }
        
        calibrationData.target_pixels.push([sample.targetX * window.screen.width / 100, sample.targetY * window.screen.height / 100]);
      });
      
      // 发送完成校准请求
      const result = await apiService.completeCalibration(calibrationData);
      console.log('✓ 校准数据发送成功:', result);
      
      // 处理kappa角补偿参数
      if (result && result.kappa_params) {
        // 保存kappa角参数，用于后续视线修正
        saveKappaParams(result.kappa_params);
      }
      
    } catch (error) {
      console.error('发送数据到真实后端失败:', error);
    }
  }

  /**
   * 模拟眼球中心坐标（实际应用中需要从眼动追踪设备获取）
   */
  function simulateEyeCenter(sample) {
    // 模拟眼球中心在相机坐标系中的位置
    const baseX = 0.0;
    const baseY = 0.0;
    const baseZ = 0.0;
    
    // 根据目标位置添加一些变化
    const offsetX = (sample.targetX - 50) * 0.001; // 转换为毫米
    const offsetY = (sample.targetY - 50) * 0.001;
    
    return [baseX + offsetX, baseY + offsetY, baseZ];
  }

  /**
   * 模拟瞳孔中心坐标（实际应用中需要从眼动追踪设备获取）
   */
  function simulatePupilCenter(sample) {
    // 模拟瞳孔中心相对于眼球中心的偏移
    const eyeCenter = simulateEyeCenter(sample);
    const pupilOffset = 0.1; // 10mm偏移
    
    return [eyeCenter[0], eyeCenter[1], eyeCenter[2] + pupilOffset];
  }

  /**
   * 保存kappa角参数
   */
  function saveKappaParams(kappaParams) {
    // 使用校准服务保存kappa角参数
    calibrationService.saveCalibrationModel({
      kappa_params: kappaParams
    });
  }

  /**
   * 模拟后端响应（开发测试用）
   */
  function sendDataToMockBackend(data) {
    setTimeout(() => {
      const mockResult = {
        status: 'success',
        model_id: `calibration_model_${Date.now()}`,
        accuracy: 92.5 + Math.random() * 5,
        message: '校准成功，模型已生成',
        kappa_params: {
          left_eye: { axis: [0.1, 0.2, 0.0], angle: 2.3 },
          right_eye: { axis: [0.1, 0.2, 0.0], angle: 2.1 }
        },
        samples_processed: data.length
      };
      
      console.log('✓ 模拟校准数据发送成功');
      
      // 保存kappa角参数
      saveKappaParams(mockResult.kappa_params);
    }, 1500);
  }
  
  /**
   * 启动视线位置监测
   * @param {Object} targetPoint - 目标校准点
   */
  function startGazeMonitoring(targetPoint) {
    // 重置注视状态
    isWithinTargetThreshold.value = false;
    fixationStartTime.value = null;
    canProceedToNextPoint.value = false;
    
    // 清除可能存在的旧定时器
    if (gazeMonitoringInterval) {
      clearInterval(gazeMonitoringInterval);
    }
    
    // 强制显示真实视线点
    if (isCalibrating.value && usingCameraTracking.value) {
      showRealGaze.value = true;
      console.log('开始监测时强制显示真实视线点');
    }
    
    // 创建监测定时器
    gazeMonitoringInterval = setInterval(() => {
      if (!isCalibrating.value) {
        clearInterval(gazeMonitoringInterval);
        return;
      }
      
      // 确保在校准过程中显示真实视线点
      if (isCalibrating.value && !showRealGaze.value && usingCameraTracking.value) {
        showRealGaze.value = true;
        console.log('监测过程中显示真实视线点');
      }
      
      // 计算视线位置与目标点的距离
      const distance = calculateDistance(
        realGazePosition.value,
        targetPoint
      );
      
      // 判断是否在阈值范围内
      const withinThreshold = distance <= gazeAccuracyThreshold.value;
      isWithinTargetThreshold.value = withinThreshold;
      
      // 处理注视时间逻辑
      const now = Date.now();
      
      if (withinThreshold) {
        if (fixationStartTime.value === null) {
          // 开始注视
          fixationStartTime.value = now;
          console.log(`开始注视目标点 ${currentPointIndex.value + 1}`);
        } else if (now - fixationStartTime.value >= requiredFixationTime.value) {
          // 注视时间达到要求，可以继续到下一个点
          console.log(`视线已稳定在目标点 ${currentPointIndex.value + 1} 上，开始采集数据`);
          
          // 设置可以继续到下一个点
          if (!canProceedToNextPoint.value) {
            canProceedToNextPoint.value = true;
            console.log(`可以继续到下一个校准点`);
          }
        }
      } else {
        // 视线移出目标区域，重置注视时间
        if (fixationStartTime.value !== null) {
          console.log(`视线移出目标点 ${currentPointIndex.value + 1}，重置注视计时`);
          fixationStartTime.value = null;
          
          // 如果之前已经可以继续，现在视线移出，取消继续状态
          if (canProceedToNextPoint.value) {
            canProceedToNextPoint.value = false;
            console.log(`需要重新注视目标点`);
          }
        }
      }
    }, 30); // 提高检查频率（原来是50ms）
  }
  
  // 存储WebSocket和定时器引用
  let gazeMonitoringInterval = null;
  
  // 生成用户ID
  function generateUserId() {
    return `user_${Date.now()}_${Math.floor(Math.random() * 1000)}`;
  }
  
  // 清理函数
  onUnmounted(() => {
    // 关闭WebSocket连接
    closeGazeWebSocket();
    
    // 停止眼动追踪
    if (eyeTrackingInitialized.value) {
      eyeTrackingService.stopEyeTracking();
    }
    
    // 清除定时器
    if (mockGazeInterval) {
      clearInterval(mockGazeInterval);
    }
    
    if (gazeMonitoringInterval) {
      clearInterval(gazeMonitoringInterval);
    }
    
    // 停止模拟视线数据
    stopSimulatedGazeData();
  });

  return {
    calibrationPoints, 
    calibrationColors,
    isCalibrating,
    currentPointIndex,
    startCalibration,
    resetCalibration,
    updateLayout,
    updateTheme,
    updateBackendConfig,
    // 导出真实视线相关状态和函数
    realGazePosition,
    showRealGaze,
    isWithinTargetThreshold,
    canProceedToNextPoint,
    // 内圈创建回调
    createInnerCircle,
    // 眼动追踪状态
    eyeTrackingInitialized,
    usingCameraTracking,
    initEyeTracking
  };
}