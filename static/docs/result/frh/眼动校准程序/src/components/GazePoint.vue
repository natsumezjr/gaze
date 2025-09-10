<template>
  <div 
    class="gaze-point" 
    :style="{
      left: `${position.x}%`,
      top: `${position.y}%`,
      opacity: visible ? 0.7 : 0,
      backgroundColor: isWithinTarget ? '#4CAF50' : '#FFC107'
    }"
  ></div>
</template>

<script>
import { computed } from 'vue';

export default {
  name: 'GazePoint',
  props: {
    position: {
      type: Object,
      required: true,
      default: () => ({ x: 50, y: 50 })
    },
    visible: {
      type: Boolean,
      default: true
    },
    isWithinTarget: {
      type: Boolean,
      default: false
    }
  },
  setup(props) {
    return {};
  }
}
</script>

<style scoped>
.gaze-point {
  position: absolute;
  width: 15px;
  height: 15px;
  border-radius: 50%;
  transform: translate(-50%, -50%);
  pointer-events: none;
  transition: all 0.1s ease;
  z-index: 100;
  box-shadow: 0 0 5px rgba(0, 0, 0, 0.3);
}
</style>