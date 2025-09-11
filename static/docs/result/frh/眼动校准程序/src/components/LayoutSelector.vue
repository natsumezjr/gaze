<template>
  <div class="layout-selector">
    <div class="layout-label">布局：</div>
    <div class="layout-options">
      <button 
        v-for="(layout, index) in layouts" 
        :key="index"
        class="layout-option"
        :class="{ active: currentLayout === layout.value }"
        @click="selectLayout(layout.value)"
      >
        <div class="layout-preview" :class="layout.value">
          <div v-for="(_, i) in getPointsForPreview(layout.value)" :key="i" class="preview-point"></div>
        </div>
        <span>{{ layout.label }}</span>
      </button>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue';
import calibrationConfig from '../config/calibration.config';

export default {
  name: 'LayoutSelector',
  emits: ['layout-change'],
  setup(props, { emit }) {
    const layouts = [
      { label: '六边形', value: 'hexagon' },
      { label: '矩形', value: 'rectangle' }
    ];
    
    const currentLayout = ref('hexagon');
    
    function selectLayout(layout) {
      currentLayout.value = layout;
      emit('layout-change', layout);
    }
    
    function getPointsForPreview(layout) {
      return calibrationConfig.points[layout] || [];
    }
    
    return {
      layouts,
      currentLayout,
      selectLayout,
      getPointsForPreview
    };
  }
};
</script>

<style scoped>
.layout-selector {
  display: flex;
  align-items: center;
  margin-bottom: 20px;
}

.layout-label {
  margin-right: 10px;
  font-weight: 500;
}

.layout-options {
  display: flex;
  gap: 10px;
}

.layout-option {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 8px;
  border-radius: 8px;
  border: 1px solid #e0e0e0;
  background: #f5f5f5;
  cursor: pointer;
  transition: all 0.2s ease;
}

.layout-option:hover {
  background: #eeeeee;
}

.layout-option.active {
  background: #e3f2fd;
  border-color: #2196f3;
}

.layout-preview {
  width: 60px;
  height: 60px;
  border: 1px solid #ccc;
  border-radius: 4px;
  margin-bottom: 6px;
  position: relative;
  background: #fff;
}

.preview-point {
  position: absolute;
  width: 6px;
  height: 6px;
  background-color: #333;
  border-radius: 50%;
  transform: translate(-50%, -50%);
}

/* 六边形布局预览点 */
.layout-preview.hexagon .preview-point:nth-child(1) { left: 50%; top: 20%; }
.layout-preview.hexagon .preview-point:nth-child(2) { left: 80%; top: 40%; }
.layout-preview.hexagon .preview-point:nth-child(3) { left: 80%; top: 70%; }
.layout-preview.hexagon .preview-point:nth-child(4) { left: 50%; top: 90%; }
.layout-preview.hexagon .preview-point:nth-child(5) { left: 20%; top: 70%; }
.layout-preview.hexagon .preview-point:nth-child(6) { left: 20%; top: 40%; }

/* 矩形布局预览点 */
.layout-preview.rectangle .preview-point:nth-child(1) { left: 20%; top: 20%; }
.layout-preview.rectangle .preview-point:nth-child(2) { left: 80%; top: 20%; }
.layout-preview.rectangle .preview-point:nth-child(3) { left: 80%; top: 80%; }
.layout-preview.rectangle .preview-point:nth-child(4) { left: 20%; top: 80%; }
.layout-preview.rectangle .preview-point:nth-child(5) { left: 50%; top: 50%; }

@media (max-width: 768px) {
  .layout-selector {
    flex-direction: column;
    align-items: flex-start;
  }
  
  .layout-label {
    margin-bottom: 8px;
  }
  
  .layout-preview {
    width: 50px;
    height: 50px;
  }
}
</style>