/**
 * 下载face-api.js模型的脚本
 * 运行此脚本将下载所需的模型文件到public/models目录
 */

const fs = require('fs');
const path = require('path');
const https = require('https');

// 模型文件URL
const MODEL_URLS = {
  tinyFaceDetector: 'https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/tiny_face_detector_model-weights_manifest.json',
  tinyFaceDetectorWeights: 'https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/tiny_face_detector_model.weights',
  faceLandmark68: 'https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/face_landmark_68_model-weights_manifest.json',
  faceLandmark68Weights: 'https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/face_landmark_68_model.weights'
};

// 目标目录
const MODEL_DIR = path.join(__dirname, 'public', 'models');

// 确保目录存在
function ensureDirectoryExists(directory) {
  if (!fs.existsSync(directory)) {
    fs.mkdirSync(directory, { recursive: true });
    console.log(`创建目录: ${directory}`);
  }
}

// 下载文件
function downloadFile(url, destination) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(destination);
    
    https.get(url, (response) => {
      if (response.statusCode !== 200) {
        reject(new Error(`下载失败，状态码: ${response.statusCode}`));
        return;
      }
      
      response.pipe(file);
      
      file.on('finish', () => {
        file.close();
        console.log(`下载完成: ${destination}`);
        resolve();
      });
    }).on('error', (error) => {
      fs.unlink(destination, () => {}); // 删除部分下载的文件
      reject(error);
    });
  });
}

// 主函数
async function downloadModels() {
  try {
    // 确保目录存在
    ensureDirectoryExists(MODEL_DIR);
    
    // 下载所有模型文件
    const downloads = [];
    
    for (const [name, url] of Object.entries(MODEL_URLS)) {
      const filename = path.basename(url);
      const destination = path.join(MODEL_DIR, filename);
      
      console.log(`开始下载 ${name}: ${url}`);
      downloads.push(downloadFile(url, destination));
    }
    
    // 等待所有下载完成
    await Promise.all(downloads);
    
    console.log('所有模型文件下载完成！');
  } catch (error) {
    console.error('下载模型文件时出错:', error);
  }
}

// 执行下载
downloadModels();