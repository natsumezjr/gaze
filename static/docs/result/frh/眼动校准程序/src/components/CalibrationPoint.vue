<template>
  <div 
    class="calibration-point" 
    :style="{
      left: `${point.x}%`,
      top: `${point.y}%`,
      backgroundColor: color.bg,
      opacity: isVisible ? 1 : 0,
      transform: isVisible ? 'translate(-50%, -50%) scale(1)' : 'translate(-50%, -50%) scale(0.5)'
    }"
  >
    <div 
      class="inner-circle" 
      :style="{
        backgroundColor: color.fg,
        width: isShrinking ? '25%' : '100%',
        height: isShrinking ? '25%' : '100%'
      }"
      :class="{ 'shrinking': isShrinking }"
    ></div>
  </div>
</template>

<script>
import { ref, computed, watch } from 'vue';

export default {
  name: 'CalibrationPoint',
  props: {
    point: {
      type: Object,
      required: true
    },
    color: {
      type: Object,
      default: () => ({ bg: '#FFFFFF', fg: '#000000' })
    },
    isActive: {
      type: Boolean,
      default: false
    },
    isCollecting: {
      type: Boolean,
      default: false
    }
  },
  setup(props) {
    const isVisible = ref(false);
    const isShrinking = ref(false);
    
    // 监听活动状态变化
    watch(() => props.isActive, (newValue) => {
      isVisible.value = newValue;
      
      // 如果变为活动状态，延迟100ms后重置内圈
      if (newValue) {
        setTimeout(() => {
          isShrinking.value = false;
        }, 100);
      }
    });
    
    // 监听采集状态变化
    watch(() => props.isCollecting, (newValue) => {
      // 如果开始采集，延迟100ms后开始收缩动画
      if (newValue) {
        setTimeout(() => {
          isShrinking.value = true;
        }, 100);
      } else {
        isShrinking.value = false;
      }
    });
    
    return {
      isVisible,
      isShrinking
    };
  }
};
</script>

<style scoped>
.calibration-point {
  position: absolute;
  width: 50px;
  height: 50px;
  border-radius: 50%;
  transform: translate(-50%, -50%);
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94);
  box-shadow: 0 0 20px rgba(0, 0, 0, 0.2);
}

.inner-circle {
  border-radius: 50%;
  transition: all 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94);
}

.inner-circle.shrinking {
  animation: shrink 2s cubic-bezier(0.25, 0.46, 0.45, 0.94);
}

@keyframes shrink {
  0% {
    width: 100%;
    height: 100%;
  }
  100% {
    width: 25%;
    height: 25%;
  }
}

@media (max-width: 768px) {
  .calibration-point {
    width: 40px;
    height: 40px;
  }
}
</style>