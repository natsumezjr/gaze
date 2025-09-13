// 注视点测试页面 JavaScript

// API配置
const API_CONFIG = {
    BASE_URL: 'http://localhost:5000',
    ENDPOINTS: {
        CAMERA_START: '/api/camera/start',
        CAMERA_STOP: '/api/camera/stop',
        CALIBRATION_STATUS: '/api/calibration/status',
        GAZE_CURRENT: '/api/gaze/current'
    }
};

// 全局状态变量
let backendConnected = false;
let cameraActive = false;
let trackingActive = false;
let gazeUpdateInterval = null;
let gazeFeedbackElement = null;
let updateCount = 0;
let lastUpdateTime = Date.now();

// DOM元素
const backendStatusDot = document.getElementById('backend-status');
const cameraStatusDot = document.getElementById('camera-status');
const trackingStatusDot = document.getElementById('tracking-status');
const startCameraBtn = document.getElementById('start-camera-btn');
const startTrackingBtn = document.getElementById('start-tracking-btn');

// 信息显示元素
const gazeCoordsSpan = document.getElementById('gaze-coords');
const gazeConfidenceSpan = document.getElementById('gaze-confidence');
const gazeAccuracySpan = document.getElementById('gaze-accuracy');
const updateRateSpan = document.getElementById('update-rate');

// 调试信息显示元素
const debugPupilCountSpan = document.getElementById('debug-pupil-count');
const debugPupilRawSpan = document.getElementById('debug-pupil-raw');
const debugPupilNormSpan = document.getElementById('debug-pupil-norm');
const debugCoarseFittingSpan = document.getElementById('debug-coarse-fitting');
const debugEyeCenterSpan = document.getElementById('debug-eye-center');
const debug3DOffsetSpan = document.getElementById('debug-3d-offset');
const debugResolutionSpan = document.getElementById('debug-resolution');

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    console.log('注视点测试页面已加载');
    gazeFeedbackElement = document.getElementById('gaze-feedback');
    checkBackendConnection();
});

/**
 * 检查后端连接
 */
async function checkBackendConnection() {
    try {
        console.log('检查后端连接...');
        const response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.CALIBRATION_STATUS}`);
        if (response.ok) {
            backendConnected = true;
            updateStatus('backend', true);
            console.log('后端连接成功');

            // 检查摄像头状态
            const status = await response.json();
            if (status.is_camera_active) {
                cameraActive = true;
                updateStatus('camera', true);
                startCameraBtn.textContent = '摄像头已启动';
                startCameraBtn.disabled = true;
                startTrackingBtn.disabled = false;
                console.log('摄像头已激活');
            }
        } else {
            throw new Error(`HTTP ${response.status}`);
        }
    } catch (error) {
        console.warn('后端连接失败:', error);
        backendConnected = false;
        updateStatus('backend', false);
    }
}

/**
 * 更新状态指示器
 */
function updateStatus(type, connected) {
    const statusDot = document.getElementById(`${type}-status`);
    if (connected) {
        statusDot.classList.add('connected');
    } else {
        statusDot.classList.remove('connected');
    }
}

/**
 * 启动摄像头
 */
async function startCamera() {
    if (!backendConnected) {
        alert('后端未连接，请检查服务器是否启动');
        return;
    }

    try {
        console.log('启动摄像头...');
        startCameraBtn.textContent = '启动中...';
        startCameraBtn.disabled = true;

        const response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.CAMERA_START}`, {
            method: 'POST'
        });

        if (response.ok) {
            const result = await response.json();
            console.log('摄像头启动结果:', result);

            if (result.success) {
                cameraActive = true;
                updateStatus('camera', true);
                startCameraBtn.textContent = '摄像头已启动';
                startTrackingBtn.disabled = false;

                // 等待几秒让系统稳定
                setTimeout(() => {
                    console.log('摄像头已稳定，可以开始追踪');
                }, 2000);
            } else {
                throw new Error(result.message || '摄像头启动失败');
            }
        } else {
            throw new Error(`HTTP ${response.status}`);
        }
    } catch (error) {
        console.error('启动摄像头失败:', error);
        alert(`摄像头启动失败: ${error.message}`);
        startCameraBtn.textContent = '启动摄像头';
        startCameraBtn.disabled = false;
    }
}

/**
 * 开始注视追踪
 */
function startTracking() {
    if (!cameraActive) {
        alert('请先启动摄像头');
        return;
    }

    console.log('开始注视点追踪');
    trackingActive = true;
    updateStatus('tracking', true);

    startTrackingBtn.textContent = '追踪中...';
    startTrackingBtn.disabled = true;

    // 显示注视点反馈
    if (gazeFeedbackElement) {
        gazeFeedbackElement.classList.add('visible');
    }

    // 重置计数器
    updateCount = 0;
    lastUpdateTime = Date.now();

    // 立即执行一次更新
    updateGazeFeedback();

    // 启动定期更新 (10fps)
    gazeUpdateInterval = setInterval(updateGazeFeedback, 100);
}

/**
 * 停止注视追踪
 */
function stopTracking() {
    console.log('停止注视点追踪');
    trackingActive = false;
    updateStatus('tracking', false);

    startTrackingBtn.textContent = '开始追踪';
    startTrackingBtn.disabled = false;

    // 隐藏注视点反馈
    if (gazeFeedbackElement) {
        gazeFeedbackElement.classList.remove('visible');
    }

    // 停止定期更新
    if (gazeUpdateInterval) {
        clearInterval(gazeUpdateInterval);
        gazeUpdateInterval = null;
    }

    // 清空信息显示
    gazeCoordsSpan.textContent = '-';
    gazeConfidenceSpan.textContent = '-';
    gazeAccuracySpan.textContent = '-';
    updateRateSpan.textContent = '-';

    // 清空调试信息显示
    debugPupilCountSpan.textContent = '-';
    debugPupilRawSpan.textContent = '-';
    debugPupilNormSpan.textContent = '-';
    debugCoarseFittingSpan.textContent = '-';
    debugEyeCenterSpan.textContent = '-';
    debug3DOffsetSpan.textContent = '-';
    debugResolutionSpan.textContent = '-';
}

/**
 * 更新注视点反馈
 */
async function updateGazeFeedback() {
    if (!trackingActive || !backendConnected || !gazeFeedbackElement) {
        return;
    }

    try {
        const response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.GAZE_CURRENT}`);
        if (response.ok) {
            const result = await response.json();

            if (result.success && result.gaze) {
                const gaze = result.gaze;

                // 更新注视点位置
                gazeFeedbackElement.style.left = `${gaze.gaze_x}px`;
                gazeFeedbackElement.style.top = `${gaze.gaze_y}px`;

                // 更新外观基于置信度
                updateGazeFeedbackAppearance(gaze);

                // 更新信息显示
                updateGazeInfo(gaze, result.debug);

                // 计算更新频率
                updateCount++;
                const now = Date.now();
                if (now - lastUpdateTime >= 1000) { // 每秒更新一次频率显示
                    const fps = updateCount / ((now - lastUpdateTime) / 1000);
                    updateRateSpan.textContent = `${fps.toFixed(1)} FPS`;
                    updateCount = 0;
                    lastUpdateTime = now;
                }

                // 详细的控制台输出
                if (updateCount % 10 === 0) { // 每10次更新输出一次，避免日志过多
                    console.log(`注视点: (${gaze.gaze_x.toFixed(1)}, ${gaze.gaze_y.toFixed(1)}), 置信度: ${gaze.confidence.toFixed(3)}`);
                }

            } else {
                console.warn('注视点数据获取失败:', result.message || '未知错误');
                gazeCoordsSpan.textContent = '数据获取失败';
                gazeConfidenceSpan.textContent = '-';
                gazeAccuracySpan.textContent = result.accuracy || 'poor';
            }
        } else {
            console.warn('注视点API请求失败:', response.status, response.statusText);
            gazeCoordsSpan.textContent = `API错误 ${response.status}`;
        }
    } catch (error) {
        console.error('更新注视点反馈失败:', error);
        gazeCoordsSpan.textContent = '连接错误';

        // 如果是网络错误，可能需要重新检查后端连接
        if (error instanceof TypeError && error.message.includes('fetch')) {
            backendConnected = false;
            updateStatus('backend', false);
            stopTracking();
            alert('后端连接丢失，请检查服务器状态');
        }
    }
}

/**
 * 更新注视点反馈外观
 */
function updateGazeFeedbackAppearance(gaze) {
    if (!gazeFeedbackElement) return;

    // 清除之前的状态类
    gazeFeedbackElement.classList.remove('accurate', 'low-confidence');

    // 根据置信度设置外观
    if (gaze.confidence < 0.4) {
        gazeFeedbackElement.classList.add('low-confidence');
    } else if (gaze.confidence > 0.7) {
        gazeFeedbackElement.classList.add('accurate');
    }

    // 根据置信度调整大小
    const scale = Math.max(0.7, Math.min(1.5, 0.8 + (gaze.confidence * 0.7)));
    gazeFeedbackElement.style.transform = `translate(-50%, -50%) scale(${scale})`;
}

/**
 * 更新注视信息显示
 */
function updateGazeInfo(gaze, debugInfo) {
    gazeCoordsSpan.textContent = `(${gaze.gaze_x.toFixed(0)}, ${gaze.gaze_y.toFixed(0)})`;
    gazeConfidenceSpan.textContent = `${(gaze.confidence * 100).toFixed(1)}%`;

    // 根据置信度显示准确度
    let accuracyText = 'poor';
    let accuracyColor = '#ef4444';

    if (gaze.confidence > 0.8) {
        accuracyText = 'excellent';
        accuracyColor = '#22c55e';
    } else if (gaze.confidence > 0.6) {
        accuracyText = 'good';
        accuracyColor = '#f59e0b';
    } else if (gaze.confidence > 0.4) {
        accuracyText = 'fair';
        accuracyColor = '#f97316';
    }

    gazeAccuracySpan.textContent = accuracyText;
    gazeAccuracySpan.style.color = accuracyColor;
    gazeAccuracySpan.style.fontWeight = 'bold';

    // 更新调试信息
    if (debugInfo) {
        debugPupilCountSpan.textContent = debugInfo.raw_pupil_data_count || '-';

        if (debugInfo.latest_pupil_raw) {
            debugPupilRawSpan.textContent = `(${debugInfo.latest_pupil_raw[0].toFixed(1)}, ${debugInfo.latest_pupil_raw[1].toFixed(1)})`;
        }

        if (debugInfo.pupil_position_normalized) {
            const norm = debugInfo.pupil_position_normalized;
            debugPupilNormSpan.textContent = `(${norm.x.toFixed(2)}, ${norm.y.toFixed(2)})`;
        }

        debugCoarseFittingSpan.textContent = debugInfo.has_coarse_fitting ? '已完成' : '未完成';

        if (debugInfo.has_coarse_fitting && debugInfo.eyeball_center_3d) {
            const center = debugInfo.eyeball_center_3d;
            debugEyeCenterSpan.textContent = `(${center[0]}, ${center[1]}, ${center[2]})mm`;
        } else {
            debugEyeCenterSpan.textContent = '未检测到';
        }

        if (debugInfo.has_coarse_fitting && debugInfo.pupil_3d_offset) {
            const offset = debugInfo.pupil_3d_offset;
            debug3DOffsetSpan.textContent = `(${offset.x}, ${offset.y}, ${offset.z})mm`;
        } else {
            debug3DOffsetSpan.textContent = '-';
        }

        if (debugInfo.camera_resolution && debugInfo.screen_resolution) {
            debugResolutionSpan.textContent = `${debugInfo.camera_resolution} → ${debugInfo.screen_resolution}`;
        }

        // 详细的调试日志（每5次更新输出一次）
        if (updateCount % 5 === 0) {
            console.log('详细调试信息:', {
                gaze: gaze,
                debug: debugInfo
            });
        }
    }
}

// 错误处理和重连机制
window.addEventListener('error', function(e) {
    console.error('页面错误:', e.error);
});

// 页面卸载时清理资源
window.addEventListener('beforeunload', function() {
    if (trackingActive) {
        stopTracking();
    }
});

// 定期检查连接状态
setInterval(async function() {
    if (backendConnected && !trackingActive) {
        // 只在不追踪时检查连接，避免干扰
        try {
            const response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.CALIBRATION_STATUS}`, {
                method: 'GET',
                cache: 'no-cache'
            });
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
        } catch (error) {
            console.warn('连接检查失败，后端可能已断开');
            backendConnected = false;
            updateStatus('backend', false);
        }
    }
}, 5000); // 每5秒检查一次