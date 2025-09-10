<template>
  <div id="calibration-container" class="calibration-container">
    <!-- 移动校准圆圈 -->
    <div 
      id="moving-circle" 
      class="moving-circle" 
      :class="{ visible: isCalibrating || isCountingDown }"
      :style="{
        left: !isCountingDown && currentPoint ? `${currentPoint.x}%` : '50%',
        top: !isCountingDown && currentPoint ? `${currentPoint.y}%` : '50%',
        backgroundColor: isCountingDown ? '#FFFFFF' : (currentColor ? currentColor.bg : '#FFFFFF')
      }"
    >
      <div 
        v-if="innerCircleVisible && !isCountingDown"
        class="inner-shrink" 
        :class="{ shrinking: isCollecting }"
        :style="{
          backgroundColor: currentColor ? currentColor.fg : '#000000'
        }"
      ></div>
    </div>
    
    <!-- 真实视线注视点 -->
    <GazePoint
      v-if="showRealGaze"
      :position="realGazePosition"
      :visible="showRealGaze"
      :isWithinTarget="isWithinTargetThreshold"
    />
    
    <div class="progress-indicator" :class="{ visible: isCalibrating }">
      校准进度: {{ currentPointIndex + 1 }}/{{ calibrationPoints.length }}
    </div>
    
    <div class="countdown-prompt" :class="{ visible: isCountingDown }">
      眼睛跟随圆点在屏幕上移动。
    </div>
    
    <div 
      class="countdown-circle" 
      :class="{ visible: isCountingDown }"
      v-if="isCountingDown"
    >
      {{ countdownValue }}
    </div>
  </div>
</template>

<script>
import { ref, computed, watch, onMounted } from 'vue';
import { useCalibration } from '../composables/useCalibration';
import GazePoint from './GazePoint.vue';

export default {
  name: 'CalibrationContainer',
  components: {
    GazePoint
  },
  props: {
    startCalibration: {
      type: Boolean,
      default: false
    },
    selectedTheme: {
      type: String,
      default: 'light'
    },
    selectedLayout: {
      type: String,
      default: 'hexagon'
    },
    useRealBackend: {
      type: Boolean,
      default: false
    },
    useCameraTracking: {
      type: Boolean,
      default: false
    }
  },
  emits: ['calibration-complete'],
  setup(props, { emit }) {
    // 使用校准组合式API
    const { 
      calibrationPoints, 
      calibrationColors,
      isCalibrating,
      currentPointIndex,
      startCalibration: startCalibrationProcess,
      resetCalibration,
      updateLayout,
      updateTheme,
      updateBackendConfig,
      // 真实视线相关状态
      realGazePosition,
      showRealGaze,
      isWithinTargetThreshold,
      canProceedToNextPoint,
      // 创建内部圆圈回调
      createInnerCircle,
      // 眼动追踪状态
      eyeTrackingInitialized,
      usingCameraTracking,
      initEyeTracking
    } = useCalibration({
      onCalibrationEnd: () => {
        emit('calibration-complete');
      }
    });
    
    // 当前点和颜色
    const currentPoint = computed(() => {
      return calibrationPoints[currentPointIndex.value];
    });
    
    const currentColor = computed(() => {
      return calibrationColors[currentPointIndex.value % calibrationColors.length];
    });
    
    // 倒计时相关
    const isCountingDown = ref(false);
    const countdownValue = ref(3);
    
    // 数据采集状态
    const isCollecting = ref(false);
    const innerCircleVisible = ref(true);
    
    // 监听开始校准属性
    watch(() => props.startCalibration, (newValue) => {
      if (newValue) {
        startCountdown();
      }
    });
    
    // 监听主题选择
    watch(() => props.selectedTheme, (newValue) => {
      updateTheme(newValue);
    });
    
    // 监听布局选择
    watch(() => props.selectedLayout, (newValue) => {
      updateLayout(newValue);
    });
    
    // 监听后端配置
    watch(() => props.useRealBackend, (newValue) => {
      updateBackendConfig(newValue);
    });
    
    // 监听摄像头追踪状态
    watch(() => props.useCameraTracking, (newValue) => {
      console.log(`摄像头追踪状态变更: ${newValue}`);
      if (newValue && !eyeTrackingInitialized.value) {
        // 如果启用摄像头追踪但尚未初始化，尝试初始化
        initEyeTracking().then(success => {
          console.log(`眼动追踪初始化${success ? '成功' : '失败'}`);
        });
      }
    }, { immediate: true });
    
    // 开始倒计时
    function startCountdown() {
      isCountingDown.value = true;
      countdownValue.value = 3;
      innerCircleVisible.value = false; // 确保倒计时期间不显示内圈
      
      const countdownInterval = setInterval(() => {
        countdownValue.value -= 1;
        
        if (countdownValue.value <= 0) {
          clearInterval(countdownInterval);
          // 倒计时结束后，延迟一小段时间再开始校准，确保数字消失
          setTimeout(() => {
            isCountingDown.value = false;
            startCalibrationProcess();
          }, 100);
        }
      }, 1000);
    }
    
    // 监听校准状态变化
    watch(isCalibrating, (newValue) => {
      if (!newValue) {
        // 校准结束，重置状态
        resetCalibration();
      }
    });
    
    // 监听数据采集状态
    watch(canProceedToNextPoint, (newValue) => {
      isCollecting.value = newValue;
    });
    
    // 监听当前点变化
    watch(currentPointIndex, () => {
      innerCircleVisible.value = true;
    });
    
    // 添加内圈创建回调
    const onCreateInnerCircle = (color) => {
      innerCircleVisible.value = true;
      console.log(`组件接收到内圈创建请求，颜色: ${color}`);
    };
    
    // 设置回调
    useCalibration({
      onCalibrationEnd: () => {
        emit('calibration-complete');
      },
      onCreateInnerCircle: onCreateInnerCircle
    });
    
    return {
      // 校准状态
      isCalibrating,
      currentPointIndex,
      calibrationPoints,
      // 计算属性
      currentPoint,
      currentColor,
      // 倒计时
      isCountingDown,
      countdownValue,
      // 数据采集
      isCollecting,
      innerCircleVisible,
      // 真实视线
      realGazePosition,
      showRealGaze,
      isWithinTargetThreshold
    };
  }
};
</script>

<style>
@import '../assets/styles/main.css';
</style>