<template>
  <div class="main-container">
    <h1 class="title" v-if="!isCalibrating">眼动校准程序</h1>
    
    <div class="camera-status" v-if="!isCalibrating">
      <div class="status-indicator" :class="{ active: cameraActive, error: cameraError }">
        摄像头状态: {{ cameraStatusText }}
      </div>
      <button class="camera-button" @click="requestCameraAccess" v-if="!cameraActive && !cameraRequesting">
        启用摄像头
      </button>
    </div>
    
    <button id="start-btn" class="start-button" v-if="!isCalibrating" @click="startCalibration">
      开始校准
    </button>
    
    <div id="calibration-container" class="calibration-container"></div>
    <CalibrationContainer 
      :startCalibration="startCalibrationFlag" 
      :selectedTheme="selectedTheme"
      :selectedLayout="selectedLayout"
      :useRealBackend="useRealBackend"
      :useCameraTracking="cameraActive"
      @calibration-complete="onCalibrationComplete"
    />
  </div>
</template>

<script>
import { ref, onMounted, onUnmounted } from 'vue';
import CalibrationContainer from './components/CalibrationContainer.vue';
import calibrationConfig from './config/calibration.config';
import * as cameraService from './services/camera';
import * as eyeTrackingService from './services/eyeTracking';

export default {
  name: 'App',
  components: {
    CalibrationContainer
  },
  setup() {
    const isCalibrating = ref(false);
    const startCalibrationFlag = ref(false);
    const selectedTheme = ref('light');
    const selectedLayout = ref('hexagon');
    const useRealBackend = ref(calibrationConfig.backend.enabled);
    
    // 摄像头状态
    const cameraActive = ref(false);
    const cameraRequesting = ref(false);
    const cameraError = ref(false);
    const cameraStatusText = ref('未启用');
    
    // 请求摄像头访问权限
    async function requestCameraAccess() {
      try {
        // 检查浏览器是否支持摄像头API
        if (!cameraService.isCameraSupported()) {
          throw new Error('浏览器不支持摄像头API');
        }
        
        // 更新状态
        cameraRequesting.value = true;
        cameraStatusText.value = '请求访问中...';
        
        // 初始化摄像头
        const initialized = await cameraService.initCamera({
          videoElementId: 'camera-video',
          canvasElementId: 'camera-canvas',
          width: 640,
          height: 480,
          createElements: true,
          hidden: true,
          parentElement: document.getElementById('calibration-container')
        });
        
        if (initialized) {
          cameraActive.value = true;
          cameraError.value = false;
          cameraStatusText.value = '已启用';
          console.log('摄像头初始化成功');
        } else {
          throw new Error('摄像头初始化失败');
        }
      } catch (error) {
        console.error('请求摄像头访问失败:', error);
        cameraError.value = true;
        cameraStatusText.value = `错误: ${error.message}`;
      } finally {
        cameraRequesting.value = false;
      }
    }
    
    // 开始校准
    function startCalibration() {
      isCalibrating.value = true;
      startCalibrationFlag.value = true;
    }
    
    // 校准完成回调
    function onCalibrationComplete() {
      isCalibrating.value = false;
      startCalibrationFlag.value = false;
      console.log('校准完成');
    }
    
    // 键盘事件处理
    function handleKeyDown(event) {
      if (event.key === 'Escape' && isCalibrating.value) {
        console.log('用户按下Esc键，中断校准');
        onCalibrationComplete();
      }
    }
    
    // 挂载时添加键盘事件监听
    onMounted(() => {
      document.addEventListener('keydown', handleKeyDown);
      
      // 检查浏览器是否支持摄像头API
      if (cameraService.isCameraSupported()) {
        cameraStatusText.value = '可用但未启用';
      } else {
        cameraStatusText.value = '浏览器不支持';
        cameraError.value = true;
      }
    });
    
    // 卸载时移除键盘事件监听
    onUnmounted(() => {
      document.removeEventListener('keydown', handleKeyDown);
      
      // 停止摄像头
      if (cameraActive.value) {
        cameraService.stopCamera();
      }
    });
    
    return {
      isCalibrating,
      startCalibrationFlag,
      selectedTheme,
      selectedLayout,
      useRealBackend,
      startCalibration,
      onCalibrationComplete,
      // 摄像头状态
      cameraActive,
      cameraRequesting,
      cameraError,
      cameraStatusText,
      requestCameraAccess
    };
  }
};
</script>

<style>
@import './assets/styles/main.css';

.camera-status {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-bottom: 20px;
}

.status-indicator {
  display: flex;
  align-items: center;
  margin-bottom: 10px;
  padding: 8px 16px;
  border-radius: 20px;
  background-color: #f0f0f0;
  font-size: 14px;
}

.status-indicator::before {
  content: '';
  display: inline-block;
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background-color: #ccc;
  margin-right: 8px;
}

.status-indicator.active::before {
  background-color: #4CAF50;
}

.status-indicator.error::before {
  background-color: #f44336;
}

.camera-button {
  padding: 8px 16px;
  border-radius: 20px;
  background-color: #2196F3;
  color: white;
  border: none;
  cursor: pointer;
  font-size: 14px;
  transition: background-color 0.3s;
}

.camera-button:hover {
  background-color: #0b7dda;
}

.camera-button:active {
  background-color: #0a69b7;
}
</style>