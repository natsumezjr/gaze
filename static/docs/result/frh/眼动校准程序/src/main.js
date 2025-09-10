import { createApp } from 'vue'
import App from './App.vue'
import './assets/styles/main.css'

document.addEventListener('DOMContentLoaded', function() {
  // 检查是否存在#app元素
  const appElement = document.getElementById('app');
  if (appElement) {
    createApp(App).mount('#app');
  } else {
    // 如果不存在#app元素，则直接挂载到body下的.main-container
    const mainContainer = document.querySelector('.main-container');
    if (mainContainer) {
      createApp(App).mount('.main-container');
    } else {
      console.error('找不到挂载点：既没有#app元素，也没有.main-container元素');
    }
  }
});