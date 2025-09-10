<template>
  <div class="theme-switcher">
    <div class="theme-label">主题：</div>
    <div class="theme-options">
      <button 
        v-for="(theme, index) in themes" 
        :key="index"
        class="theme-option"
        :class="{ active: currentTheme === theme.value }"
        @click="selectTheme(theme.value)"
      >
        <div class="theme-preview" :style="getPreviewStyle(theme.value)"></div>
        <span>{{ theme.label }}</span>
      </button>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue';
import calibrationConfig from '../config/calibration.config';

export default {
  name: 'ThemeSwitcher',
  emits: ['theme-change'],
  setup(props, { emit }) {
    const themes = [
      { label: '白色', value: 'light' },
      { label: '黑色', value: 'dark' },
      { label: '浅色', value: 'pastel' }
    ];
    
    const currentTheme = ref('light');
    
    function selectTheme(theme) {
      currentTheme.value = theme;
      emit('theme-change', theme);
    }
    
    function getPreviewStyle(theme) {
      const themeColors = calibrationConfig.themes[theme];
      if (!themeColors || themeColors.length === 0) return {};
      
      // 使用主题的第一个颜色作为预览
      return {
        backgroundColor: themeColors[0].bg,
        borderColor: themeColors[0].fg
      };
    }
    
    return {
      themes,
      currentTheme,
      selectTheme,
      getPreviewStyle
    };
  }
};
</script>

<style scoped>
.theme-switcher {
  display: flex;
  align-items: center;
  margin-bottom: 20px;
}

.theme-label {
  margin-right: 10px;
  font-weight: 500;
}

.theme-options {
  display: flex;
  gap: 10px;
}

.theme-option {
  display: flex;
  align-items: center;
  padding: 6px 12px;
  border-radius: 20px;
  border: 1px solid #e0e0e0;
  background: #f5f5f5;
  cursor: pointer;
  transition: all 0.2s ease;
}

.theme-option:hover {
  background: #eeeeee;
}

.theme-option.active {
  background: #e3f2fd;
  border-color: #2196f3;
}

.theme-preview {
  width: 16px;
  height: 16px;
  border-radius: 50%;
  margin-right: 6px;
  border: 2px solid;
}

@media (max-width: 768px) {
  .theme-switcher {
    flex-direction: column;
    align-items: flex-start;
  }
  
  .theme-label {
    margin-bottom: 8px;
  }
  
  .theme-option {
    padding: 4px 8px;
    font-size: 14px;
  }
}
</style>