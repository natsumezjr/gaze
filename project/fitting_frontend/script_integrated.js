// 眼动校准系统 - 前后端集成版本
// 保持原有6点六边形布局和动画效果，添加真实摄像头数据集成

// 校准点坐标数组 - 6点六边形布局
const calibrationPoints = [
    { x: 33, y: 20 },   // 左上（左三分之一线上）
    { x: 67, y: 20 },   // 右上（右三分之一线上）
    { x: 10, y: 50 },   // 左中（靠近左侧边缘）
    { x: 90, y: 50 },   // 右中（靠近右侧边缘）
    { x: 33, y: 80 },   // 左下（左三分之一线上）
    { x: 67, y: 80 }    // 右下（右三分之一线上）
];

// 校准点颜色数组 - 6点白色主题配色
const calibrationColors = [
    { bg: '#DC3545', fg: '#C82333' }, // 深红色系
    { bg: '#20C997', fg: '#1E7E34' }, // 深绿色系
    { bg: '#0D6EFD', fg: '#0A58CA' }, // 深蓝色系
    { bg: '#198754', fg: '#155724' }, // 深薄荷绿系
    { bg: '#FD7E14', fg: '#E8590C' }, // 深橙色系
    { bg: '#6610F2', fg: '#520DC2' }  // 深紫色系
];

// API配置
const API_CONFIG = {
    BASE_URL: 'http://localhost:5000',
    ENDPOINTS: {
        CAMERA_START: '/api/camera/start',
        CAMERA_STOP: '/api/camera/stop',
        CALIBRATION_STATUS: '/api/calibration/status',
        CALIBRATION_START: '/api/calibration/start',
        CALIBRATION_COLLECT: '/api/calibration/collect',
        CALIBRATION_COMPLETE: '/api/calibration/complete'
    }
};

// 获取DOM元素
const startBtn = document.getElementById('start-btn');
const calibrationContainer = document.getElementById('calibration-container');
const title = document.querySelector('.title');

// 校准状态变量
let isCalibrating = false;
let currentPointIndex = 0;
let collectedData = [];
let movingCircle = null;
let innerCircle = null;
let backendConnected = false;
let calibrationSession = null;

// 注视点反馈变量
let gazeTrackingActive = false;
let gazeUpdateInterval = null;
let gazeFeedbackElement = null;
let gazeAccuracyIndicator = null;
let gazeTrailElements = [];

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    console.log('眼动校准系统已加载 - 集成版本');
    initializeSystem();
    startBtn.addEventListener('click', startCalibrationFlow);
});

/**
 * 初始化系统
 */
async function initializeSystem() {
    try {
        // 更新状态
        if (window.updateSystemStatus) {
            window.updateSystemStatus('processing', '连接后端中...');
        }
        
        // 检查后端连接
        const response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.CALIBRATION_STATUS}`);
        if (response.ok) {
            backendConnected = true;
            console.log('后端连接成功');
            updateStartButtonText('检查摄像头状态中...');
            
            if (window.updateSystemStatus) {
                window.updateSystemStatus('processing', '启动摄像头中...');
            }
            
            // 启动摄像头
            await startCamera();
        } else {
            throw new Error('后端连接失败');
        }
    } catch (error) {
        console.warn('后端连接失败，使用纯前端模式:', error);
        backendConnected = false;
        updateStartButtonText('开始校准 (演示模式)');
        
        if (window.updateSystemStatus) {
            window.updateSystemStatus('', '演示模式');
        }
    }
}

/**
 * 启动摄像头
 */
async function startCamera() {
    try {
        updateStartButtonText('启动摄像头中...');
        
        const response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.CAMERA_START}`, {
            method: 'POST'
        });
        
        if (response.ok) {
            const result = await response.json();
            console.log('摄像头启动结果:', result);
            
            if (result.success) {
                // 等待粗拟合完成
                await waitForCoarseFitting();
            } else {
                throw new Error(result.message);
            }
        } else {
            throw new Error('摄像头启动失败');
        }
    } catch (error) {
        console.error('摄像头启动失败:', error);
        updateStartButtonText('摄像头启动失败 - 使用演示模式');
        backendConnected = false;
    }
}

/**
 * 等待粗拟合完成
 */
async function waitForCoarseFitting() {
    updateStartButtonText('检测人脸中...');
    console.log('等待粗拟合完成...');
    
    if (window.updateSystemStatus) {
        window.updateSystemStatus('processing', '等待人脸检测...');
    }
    
    const maxWaitTime = 30000; // 最大等待30秒
    const startTime = Date.now();
    
    const checkStatus = async () => {
        try {
            const response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.CALIBRATION_STATUS}`);
            if (response.ok) {
                const status = await response.json();
                console.log('校准状态:', status);
                
                if (status.is_coarse_fitting_done) {
                    console.log('粗拟合完成！');
                    updateStartButtonText('开始校准');
                    startBtn.disabled = false;
                    
                    if (window.updateSystemStatus) {
                        window.updateSystemStatus('connected', '准备就绪');
                    }
                    
                    return true;
                } else {
                    // 更新状态提示
                    if (!status.is_camera_active) {
                        updateStartButtonText('摄像头未激活');
                        if (window.updateSystemStatus) {
                            window.updateSystemStatus('', '摄像头未激活');
                        }
                    } else {
                        updateStartButtonText('请确保人脸在摄像头视野内...');
                        if (window.updateSystemStatus) {
                            window.updateSystemStatus('processing', '检测人脸中...');
                        }
                    }
                    
                    // 检查超时
                    if (Date.now() - startTime < maxWaitTime) {
                        setTimeout(checkStatus, 1000); // 每秒检查一次
                    } else {
                        console.warn('粗拟合等待超时');
                        updateStartButtonText('初始化超时 - 使用演示模式');
                        backendConnected = false;
                        
                        if (window.updateSystemStatus) {
                            window.updateSystemStatus('', '初始化超时');
                        }
                    }
                }
            }
        } catch (error) {
            console.error('检查校准状态失败:', error);
            updateStartButtonText('连接失败 - 使用演示模式');
            backendConnected = false;
            
            if (window.updateSystemStatus) {
                window.updateSystemStatus('', '连接失败');
            }
        }
    };
    
    await checkStatus();
}

/**
 * 更新开始按钮文字
 */
function updateStartButtonText(text) {
    startBtn.textContent = text;
}

/**
 * 开始校准流程
 */
async function startCalibrationFlow() {
    if (isCalibrating) {
        console.log('校准已在进行中...');
        return;
    }
    
    console.log('开始眼动校准...');
    isCalibrating = true;
    currentPointIndex = 0;
    collectedData = [];
    
    // 隐藏按钮和标题
    startBtn.style.display = 'none';
    title.style.display = 'none';
    
    // 显示进度指示器
    showProgressIndicator();
    
    // 初始化注视点反馈
    initializeGazeFeedback();
    
    if (backendConnected) {
        // 启动后端校准
        await startBackendCalibration();
        // 启动实时注视点追踪
        startGazeTracking();
    }
    
    // 开始前端动画
    startCountdown();
}

/**
 * 启动后端校准
 */
async function startBackendCalibration() {
    try {
        const response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.CALIBRATION_START}`, {
            method: 'POST'
        });
        
        if (response.ok) {
            const result = await response.json();
            console.log('后端校准启动结果:', result);
            calibrationSession = result;
        } else {
            console.error('启动后端校准失败');
            backendConnected = false;
        }
    } catch (error) {
        console.error('启动后端校准失败:', error);
        backendConnected = false;
    }
}

/**
 * 开始倒计时 (保持原有动画)
 */
function startCountdown() {
    // 创建提示文字
    const prompt = document.createElement('div');
    prompt.className = 'countdown-prompt';
    prompt.textContent = '眼睛跟随圆点在屏幕上移动。';
    document.body.appendChild(prompt);
    
    // 创建唯一的移动圆圈
    movingCircle = document.createElement('div');
    movingCircle.id = 'moving-circle';
    movingCircle.className = 'countdown';
    
    movingCircle.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, calc(-50% - 3px));
        opacity: 0;
        transition: all 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    `;
    
    document.body.appendChild(movingCircle);
    
    // 显示提示和圆圈
    requestAnimationFrame(() => {
        prompt.classList.add('visible');
        movingCircle.style.opacity = '1';
    });
    
    // 3秒倒计时
    let countdown = 3;
    movingCircle.textContent = countdown;
    
    const countdownInterval = setInterval(() => {
        countdown--;
        if (countdown > 0) {
            movingCircle.textContent = countdown;
        } else {
            clearInterval(countdownInterval);
            
            // 倒计时结束动画
            movingCircle.style.transition = 'color 0.5s ease';
            movingCircle.style.color = 'rgba(0, 0, 0, 0)';
            
            setTimeout(() => {
                movingCircle.textContent = '';
                movingCircle.classList.remove('countdown');
                movingCircle.style.color = '#000000';
                movingCircle.style.transition = 'all 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94)';
                
                prompt.classList.remove('visible');
                setTimeout(() => prompt.remove(), 500);
                
                // 开始移动到第一个点
                moveToFirstPoint();
            }, 500);
        }
    }, 1000);
}

/**
 * 移动到第一个校准点 (保持原有动画)
 */
function moveToFirstPoint() {
    const firstPoint = calibrationPoints[0];
    const firstColor = calibrationColors[0];
    
    console.log('移动到第一个校准点');
    
    setTimeout(() => {
        movingCircle.style.left = `${firstPoint.x}%`;
        movingCircle.style.top = `${firstPoint.y}%`;
        movingCircle.style.transition = 'all 1.0s cubic-bezier(0.25, 0.46, 0.45, 0.94)';
        movingCircle.style.backgroundColor = firstColor.fg;
        movingCircle.style.opacity = '1';
    }, 100);
    
    setTimeout(() => {
        movingCircle.style.transition = 'none';
        movingCircle.style.backgroundColor = firstColor.bg;
        movingCircle.style.opacity = '1';
        
        createInnerCircle(firstColor.fg);
        startCalibrationAtPoint(0);
    }, 1100);
}

/**
 * 创建内圈 (保持原有动画)
 */
function createInnerCircle(color) {
    movingCircle.innerHTML = '';
    
    innerCircle = document.createElement('div');
    innerCircle.className = 'inner-shrink';
    
    innerCircle.style.cssText = `
        position: absolute;
        top: 50%;
        left: 50%;
        width: 100%;
        height: 100%;
        border-radius: 50%;
        background-color: ${color};
        transform: translate(-50%, -50%);
        z-index: 2;
    `;
    
    movingCircle.appendChild(innerCircle);
    console.log(`创建内圈，颜色: ${color}`);
}

/**
 * 在当前点开始校准 (集成真实数据收集)
 */
async function startCalibrationAtPoint(pointIndex) {
    console.log(`校准点 ${pointIndex + 1}/${calibrationPoints.length}`);
    updateProgress(pointIndex + 1, calibrationPoints.length);
    
    currentPointIndex = pointIndex;
    
    // 获取当前校准点的屏幕坐标
    const currentPoint = calibrationPoints[pointIndex];
    const screenWidth = window.screen.width || 1920;
    const screenHeight = window.screen.height || 1080;
    const targetX = Math.round((currentPoint.x / 100) * screenWidth);
    const targetY = Math.round((currentPoint.y / 100) * screenHeight);
    
    // 调试信息
    console.log('processCalibrationPoint - gazeTrackingActive:', gazeTrackingActive);
    console.log('processCalibrationPoint - backendConnected:', backendConnected);
    
    // 注视触发逻辑 - 只有在真正注视到校准点时才触发动画
    let gazeAccuracyInterval = null;
    let animationTriggered = false;
    let noGazeDataCount = 0;  // 记录无数据次数
    let consecutiveAccurateCount = 0;  // 连续准确注视次数
    const REQUIRED_ACCURATE_FRAMES = 3;  // 需要连续3次准确注视才触发
    const ACCURACY_THRESHOLD = 100;  // 注视点偏差阈值（像素）

    // 强制启用注视触发模式（如果后端连接正常）
    if (backendConnected && gazeTrackingActive) {
        console.log(`开始检查校准点 ${pointIndex + 1} 的注视准确度 - 需要连续${REQUIRED_ACCURATE_FRAMES}次准确注视，偏差小于${ACCURACY_THRESHOLD}px`);

        // 开始连续检查注视准确度
        gazeAccuracyInterval = setInterval(async () => {
            if (animationTriggered) return; // 动画已触发，停止检查

            const accuracy = await checkGazeAccuracy(targetX, targetY);
            if (accuracy) {
                noGazeDataCount = 0;  // 重置计数

                // 检查注视偏差是否在阈值内
                const isAccurate = accuracy.distance < ACCURACY_THRESHOLD &&
                                 (accuracy.level === 'excellent' || accuracy.level === 'good' || accuracy.level === 'fair');

                if (isAccurate) {
                    consecutiveAccurateCount++;
                    console.log(`校准点 ${pointIndex + 1} 注视准确 (${consecutiveAccurateCount}/${REQUIRED_ACCURATE_FRAMES}) - 偏差: ${accuracy.distance.toFixed(1)}px, 等级: ${accuracy.level}`);

                    // 注视点变为绿色表示准确
                    if (gazeFeedbackElement) {
                        gazeFeedbackElement.classList.add('accurate');
                    }

                    // 只有连续准确注视达到要求次数才触发动画
                    if (consecutiveAccurateCount >= REQUIRED_ACCURATE_FRAMES && !animationTriggered) {
                        animationTriggered = true;
                        clearInterval(gazeAccuracyInterval); // 停止检查

                        // 清除超时检测
                        if (gazeAccuracyInterval.timeoutId) {
                            clearTimeout(gazeAccuracyInterval.timeoutId);
                        }

                        console.log(`✓ 注视稳定 - 连续${REQUIRED_ACCURATE_FRAMES}次准确，触发内圈收缩动画`);
                        setTimeout(() => {
                            innerCircle.style.transition = 'none';
                            innerCircle.offsetHeight;
                            innerCircle.classList.add('shrinking');
                            console.log('内圈收缩动画已开始');
                        }, 100);
                    }
                } else {
                    // 注视不准确时重置计数器
                    if (consecutiveAccurateCount > 0) {
                        console.log(`校准点 ${pointIndex + 1} 注视不准确，重置计数器 - 偏差: ${accuracy.distance.toFixed(1)}px, 等级: ${accuracy.level}`);
                    }
                    consecutiveAccurateCount = 0;

                    // 注视不准确时移除绿色标记
                    if (gazeFeedbackElement) {
                        gazeFeedbackElement.classList.remove('accurate');
                    }
                }
            } else {
                noGazeDataCount++;
                consecutiveAccurateCount = 0;  // 无数据时也重置

                // 如果连续多次无数据，提示问题
                if (noGazeDataCount > 25) {
                    console.warn(`校准点 ${pointIndex + 1} 连续${noGazeDataCount}次未获取到注视数据，请检查摄像头和人脸检测`);

                    // 提供用户提示
                    if (gazeFeedbackElement) {
                        gazeFeedbackElement.style.borderColor = '#ff4444';
                        gazeFeedbackElement.style.borderStyle = 'dashed';
                    }
                } else if (noGazeDataCount % 10 === 0) {
                    console.log(`校准点 ${pointIndex + 1} 等待注视数据... (${noGazeDataCount})`);
                }
            }
        }, 150); // 每150ms检查一次，更快响应

        // 设置超时检测 - 如果长时间无法检测到准确注视，提示用户
        const TIMEOUT_DURATION = 30000; // 30秒超时
        const timeoutId = setTimeout(() => {
            if (!animationTriggered) {
                console.warn(`校准点 ${pointIndex + 1} 超时未检测到准确注视`);

                // 停止检查循环
                clearInterval(gazeAccuracyInterval);

                // 弹出问题诊断窗口
                showCalibrationTimeoutDialog(pointIndex + 1);
            }
        }, TIMEOUT_DURATION);

        // 存储超时ID以便在成功时取消
        gazeAccuracyInterval.timeoutId = timeoutId;

        console.log(`⚠️ 请注视校准点 ${pointIndex + 1}，30秒内需要连续${REQUIRED_ACCURATE_FRAMES}次准确注视才能继续`);

    } else {
        // 如果后端未连接，显示连接错误对话框
        console.warn('后端未连接，无法进行注视检测');
        setTimeout(() => {
            if (!animationTriggered) {
                showBackendConnectionDialog(pointIndex + 1);
            }
        }, 1000);
    }
    
    // 收集数据 (真实或模拟)
    collectGazeData(calibrationPoints[pointIndex]);
    
    // 等待动画触发后再移动到下一个点
    const checkAnimationComplete = () => {
        if (animationTriggered) {
            // 动画已触发，等待1.5秒让动画完成
            setTimeout(() => {
                if (pointIndex + 1 < calibrationPoints.length) {
                    moveToNextPoint(pointIndex + 1);
                } else {
                    finishCalibration();
                }
            }, 1500);
        } else {
            // 动画未触发，继续等待（每500ms检查一次）
            setTimeout(checkAnimationComplete, 500);
        }
    };
    
    // 延迟500ms开始检查，给用户时间调整注视点
    setTimeout(checkAnimationComplete, 500);
}

/**
 * 收集眼动数据 (集成真实后端数据)
 */
async function collectGazeData(point) {
    if (backendConnected) {
        try {
            // 转换百分比坐标到像素坐标
            const screenWidth = window.screen.width || 1920;
            const screenHeight = window.screen.height || 1080;
            const pixelX = Math.round((point.x / 100) * screenWidth);
            const pixelY = Math.round((point.y / 100) * screenHeight);
            
            console.log(`收集真实数据 - 点 ${currentPointIndex + 1}: (${pixelX}, ${pixelY})`);
            
            const response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.CALIBRATION_COLLECT}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    target_pixel: [pixelX, pixelY],
                    point_index: currentPointIndex
                })
            });
            
            if (response.ok) {
                const result = await response.json();
                console.log(`校准点 ${currentPointIndex + 1} 数据收集结果:`, result);
                
                // 记录收集的数据
                collectedData.push({
                    pointId: currentPointIndex,
                    targetX: pixelX,
                    targetY: pixelY,
                    timestamp: Date.now(),
                    backendResult: result
                });
            } else {
                console.error('数据收集失败，使用模拟数据');
                collectMockGazeData(point);
            }
        } catch (error) {
            console.error('数据收集失败:', error);
            collectMockGazeData(point);
        }
    } else {
        // 使用模拟数据
        collectMockGazeData(point);
    }
}

/**
 * 模拟数据采集
 */
function collectMockGazeData(point) {
    console.log(`收集模拟数据 - 点 ${currentPointIndex + 1}`);
    
    const gazeData = {
        pointId: currentPointIndex,
        targetX: point.x,
        targetY: point.y,
        timestamp: Date.now(),
        gazeX: point.x + (Math.random() - 0.5) * 5,
        gazeY: point.y + (Math.random() - 0.5) * 5,
        confidence: 0.8 + Math.random() * 0.2
    };
    
    collectedData.push(gazeData);
}

/**
 * 移动到下一个校准点 (保持原有动画)
 */
function moveToNextPoint(nextIndex) {
    const nextPoint = calibrationPoints[nextIndex];
    const nextColor = calibrationColors[nextIndex];
    
    const moveTime = calculateMoveTime(currentPointIndex, nextIndex);
    console.log(`移动到校准点 ${nextIndex + 1}，时间: ${moveTime.toFixed(2)}s`);
    
    movingCircle.style.transition = `all ${moveTime}s cubic-bezier(0.25, 0.46, 0.45, 0.94)`;
    if (innerCircle) {
        innerCircle.style.transition = `all ${moveTime}s cubic-bezier(0.25, 0.46, 0.45, 0.94)`;
        innerCircle.classList.remove('shrinking');
    }
    
    // 移动和变色
    movingCircle.style.left = `${nextPoint.x}%`;
    movingCircle.style.top = `${nextPoint.y}%`;
    movingCircle.style.backgroundColor = nextColor.bg;
    
    if (innerCircle) {
        innerCircle.style.width = '100%';
        innerCircle.style.height = '100%';
        innerCircle.style.backgroundColor = nextColor.fg;
    }
    
    // 到达后开始校准
    setTimeout(() => {
        startCalibrationAtPoint(nextIndex);
    }, moveTime * 1000 + 50);
}

/**
 * 计算移动时间
 */
function calculateMoveTime(currentIndex, nextIndex) {
    const currentPoint = calibrationPoints[currentIndex];
    const nextPoint = calibrationPoints[nextIndex];
    const dx = currentPoint.x - nextPoint.x;
    const dy = currentPoint.y - nextPoint.y;
    const distance = Math.sqrt(dx * dx + dy * dy);
    
    const baseSpeed = 100;
    const minTime = 0.6;
    const maxTime = 1.4;
    const calculatedTime = distance / baseSpeed;
    
    return Math.max(minTime, Math.min(maxTime, calculatedTime));
}

/**
 * 完成校准
 */
async function finishCalibration() {
    console.log('校准完成！收集的数据:', collectedData);
    
    // 停止注视点追踪
    stopGazeTracking();
    
    if (backendConnected) {
        try {
            const response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.CALIBRATION_COMPLETE}`, {
                method: 'POST'
            });
            
            if (response.ok) {
                const result = await response.json();
                console.log('校准完成结果:', result);
            }
        } catch (error) {
            console.error('完成校准失败:', error);
        }
    }
    
    // 显示完成动画
    showCompletionAnimation();
}

/**
 * 显示完成动画
 */
function showCompletionAnimation() {
    const completionMessage = document.createElement('div');
    completionMessage.className = 'completion-message';
    completionMessage.innerHTML = `
        <div class="completion-content">
            <div class="completion-icon">✓</div>
            <div class="completion-text">校准完成</div>
            <div class="completion-subtitle">共收集 ${collectedData.length} 个校准点数据</div>
            <button class="completion-button" onclick="resetCalibration()">重新校准</button>
        </div>
    `;
    
    document.body.appendChild(completionMessage);
    
    // 隐藏移动圆圈
    if (movingCircle) {
        movingCircle.style.opacity = '0';
        setTimeout(() => movingCircle.remove(), 500);
    }
    
    // 隐藏进度指示器
    const progressIndicator = document.querySelector('.progress-indicator');
    if (progressIndicator) {
        progressIndicator.classList.remove('visible');
    }
    
    // 显示完成消息
    setTimeout(() => {
        completionMessage.classList.add('visible');
    }, 600);
}

/**
 * 重置校准
 */
function resetCalibration() {
    // 清理注视点反馈
    cleanupGazeFeedback();
    
    // 重新加载页面
    location.reload();
}

/**
 * 显示进度指示器
 */
function showProgressIndicator() {
    let progressIndicator = document.querySelector('.progress-indicator');
    
    if (!progressIndicator) {
        progressIndicator = document.createElement('div');
        progressIndicator.className = 'progress-indicator';
        document.body.appendChild(progressIndicator);
    }
    
    progressIndicator.classList.add('visible');
    updateProgress(0, calibrationPoints.length);
}

/**
 * 更新进度显示
 */
function updateProgress(current, total) {
    const progressIndicator = document.querySelector('.progress-indicator');
    if (progressIndicator) {
        progressIndicator.innerHTML = `
            <div class="progress-text">校准进度</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: ${(current / total) * 100}%"></div>
            </div>
            <div class="progress-count">${current} / ${total}</div>
        `;
    }
}

// ===========================================
// 注视点反馈系统
// ===========================================

/**
 * 初始化注视点反馈元素
 */
function initializeGazeFeedback() {
    console.log('初始化注视点反馈系统');
    
    // 创建注视点反馈元素
    gazeFeedbackElement = document.createElement('div');
    gazeFeedbackElement.className = 'gaze-feedback';
    gazeFeedbackElement.id = 'gaze-feedback';
    document.body.appendChild(gazeFeedbackElement);
    
    // 创建准确度指示器
    gazeAccuracyIndicator = document.createElement('div');
    gazeAccuracyIndicator.className = 'gaze-accuracy-indicator';
    gazeAccuracyIndicator.id = 'gaze-accuracy-indicator';
    gazeAccuracyIndicator.innerHTML = `
        <div class="accuracy-dot" id="accuracy-dot"></div>
        <span id="accuracy-text">注视检测中...</span>
    `;
    document.body.appendChild(gazeAccuracyIndicator);
}

/**
 * 启动注视点追踪
 */
function startGazeTracking() {
    console.log('startGazeTracking called - gazeTrackingActive:', gazeTrackingActive, 'backendConnected:', backendConnected);

    if (gazeTrackingActive || !backendConnected) {
        console.log('startGazeTracking 提前返回 - gazeTrackingActive:', gazeTrackingActive, 'backendConnected:', backendConnected);
        return;
    }

    console.log('启动实时注视点追踪');
    gazeTrackingActive = true;

    // 确保注视点反馈元素存在并正确初始化
    if (!gazeFeedbackElement) {
        console.log('重新初始化注视点反馈元素');
        initializeGazeFeedback();
    }

    // 显示注视点反馈
    if (gazeFeedbackElement) {
        console.log('显示注视点反馈元素');
        gazeFeedbackElement.classList.add('visible');
        // 强制重绘
        gazeFeedbackElement.offsetHeight;
    }
    if (gazeAccuracyIndicator) {
        console.log('显示准确度指示器');
        gazeAccuracyIndicator.classList.add('visible');
    }

    // 显示帮助说明
    showGazeHelp();

    // 立即执行一次更新
    setTimeout(updateGazeFeedback, 100);

    // 启动定期更新
    gazeUpdateInterval = setInterval(updateGazeFeedback, 100); // 10fps更新

    console.log('注视点追踪已启动');
}

/**
 * 停止注视点追踪
 */
function stopGazeTracking() {
    console.log('停止实时注视点追踪');
    gazeTrackingActive = false;
    
    if (gazeUpdateInterval) {
        clearInterval(gazeUpdateInterval);
        gazeUpdateInterval = null;
    }
    
    // 隐藏注视点反馈
    if (gazeFeedbackElement) {
        gazeFeedbackElement.classList.remove('visible');
    }
    if (gazeAccuracyIndicator) {
        gazeAccuracyIndicator.classList.remove('visible');
    }
}

/**
 * 更新注视点反馈
 */
async function updateGazeFeedback() {
    if (!gazeTrackingActive || !backendConnected || !gazeFeedbackElement) {
        return;
    }

    try {
        const response = await fetch(`${API_CONFIG.BASE_URL}/api/gaze/current`);
        if (response.ok) {
            const result = await response.json();
            if (result.success && result.gaze) {
                const gaze = result.gaze;

                // 确保注视点反馈元素可见
                if (!gazeFeedbackElement.classList.contains('visible')) {
                    gazeFeedbackElement.classList.add('visible');
                }

                // 更新注视点位置
                gazeFeedbackElement.style.left = `${gaze.gaze_x}px`;
                gazeFeedbackElement.style.top = `${gaze.gaze_y}px`;

                // 更新外观基于置信度
                updateGazeFeedbackAppearance(gaze);

                // 更新准确度指示器
                updateAccuracyIndicator(gaze);

                // 添加轨迹点（可选）
                if (Math.random() < 0.08) { // 8%的概率添加轨迹点，避免过多
                    addGazeTrail(gaze.gaze_x, gaze.gaze_y);
                }

                console.log(`注视点更新: (${gaze.gaze_x.toFixed(1)}, ${gaze.gaze_y.toFixed(1)}) 置信度: ${gaze.confidence.toFixed(2)}`);
            } else {
                console.warn('注视点数据获取失败:', result.message || '未知错误');
            }
        } else {
            console.warn('注视点API请求失败:', response.status);
        }
    } catch (error) {
        console.error('更新注视点反馈失败:', error);
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

    // 根据置信度调整大小和透明度
    const scale = Math.max(0.7, Math.min(1.3, 0.8 + (gaze.confidence * 0.5))); // 0.7 - 1.3
    const opacity = Math.max(0.6, Math.min(1.0, 0.6 + (gaze.confidence * 0.4))); // 0.6 - 1.0

    // 应用变换（保持基础的translate(-50%, -50%)）
    gazeFeedbackElement.style.transform = `translate(-50%, -50%) scale(${scale})`;
    gazeFeedbackElement.style.opacity = opacity;
}

/**
 * 更新准确度指示器
 */
function updateAccuracyIndicator(gaze) {
    const accuracyDot = document.getElementById('accuracy-dot');
    const accuracyText = document.getElementById('accuracy-text');
    
    if (!accuracyDot || !accuracyText) return;
    
    // 清除之前的状态
    accuracyDot.classList.remove('good', 'medium', 'poor');
    
    // 设置新状态
    let statusText = '';
    if (gaze.confidence > 0.8) {
        accuracyDot.classList.add('good');
        statusText = '注视检测良好';
    } else if (gaze.confidence > 0.6) {
        accuracyDot.classList.add('medium');
        statusText = '注视检测一般';
    } else {
        accuracyDot.classList.add('poor');
        statusText = '注视检测较差';
    }
    
    accuracyText.textContent = `${statusText} (${(gaze.confidence * 100).toFixed(0)}%)`;
}

/**
 * 添加注视轨迹点
 */
function addGazeTrail(x, y) {
    const trail = document.createElement('div');
    trail.className = 'gaze-trail';
    trail.style.left = `${x}px`;
    trail.style.top = `${y}px`;

    document.body.appendChild(trail);
    gazeTrailElements.push(trail);

    // 显示轨迹点
    requestAnimationFrame(() => {
        trail.classList.add('visible');
    });

    // 1秒后开始淡出
    setTimeout(() => {
        trail.classList.remove('visible');
        trail.classList.add('fading');
    }, 800);

    // 3秒后移除元素
    setTimeout(() => {
        if (trail.parentNode) {
            trail.remove();
        }
        const index = gazeTrailElements.indexOf(trail);
        if (index > -1) {
            gazeTrailElements.splice(index, 1);
        }
    }, 3000);

    // 限制轨迹点数量
    if (gazeTrailElements.length > 8) {
        const oldTrail = gazeTrailElements.shift();
        if (oldTrail && oldTrail.parentNode) {
            oldTrail.remove();
        }
    }
}

/**
 * 检查当前注视点与校准点的准确度
 */
async function checkGazeAccuracy(targetX, targetY) {
    if (!backendConnected) {
        return null;
    }
    
    try {
        const response = await fetch(`${API_CONFIG.BASE_URL}/api/gaze/accuracy`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                target_x: targetX,
                target_y: targetY
            })
        });
        
        if (response.ok) {
            const result = await response.json();
            if (result.success) {
                return result.accuracy;
            }
        }
    } catch (error) {
        console.error('检查注视准确度失败:', error);
    }
    
    return null;
}

/**
 * 清理注视点反馈
 */
function cleanupGazeFeedback() {
    stopGazeTracking();
    
    // 移除所有相关元素
    if (gazeFeedbackElement) {
        gazeFeedbackElement.remove();
        gazeFeedbackElement = null;
    }
    
    if (gazeAccuracyIndicator) {
        gazeAccuracyIndicator.remove();
        gazeAccuracyIndicator = null;
    }
    
    // 清理轨迹点
    gazeTrailElements.forEach(trail => {
        trail.remove();
    });
    gazeTrailElements = [];
}

/**
 * 显示注视反馈帮助说明
 */
function showGazeHelp() {
    const helpElement = document.getElementById('gaze-help');
    if (helpElement) {
        helpElement.style.opacity = '1';
        
        // 5秒后自动隐藏
        setTimeout(() => {
            hideGazeHelp();
        }, 5000);
    }
}

/**
 * 隐藏注视反馈帮助说明
 */
function hideGazeHelp() {
    const helpElement = document.getElementById('gaze-help');
    if (helpElement) {
        helpElement.style.opacity = '0';
    }
}

/**
 * 显示校准超时诊断对话框
 */
function showCalibrationTimeoutDialog(pointIndex) {
    // 创建遮罩层
    const overlay = document.createElement('div');
    overlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: rgba(0, 0, 0, 0.8);
        z-index: 10000;
        display: flex;
        align-items: center;
        justify-content: center;
    `;

    // 创建对话框
    const dialog = document.createElement('div');
    dialog.style.cssText = `
        background: white;
        border-radius: 12px;
        padding: 32px;
        max-width: 500px;
        width: 90%;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        text-align: center;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    `;

    // 诊断内容
    dialog.innerHTML = `
        <div style="font-size: 24px; margin-bottom: 16px;">⚠️</div>
        <h3 style="margin: 0 0 16px 0; color: #333;">校准点 ${pointIndex} 检测超时</h3>
        <p style="color: #666; line-height: 1.5; margin-bottom: 24px;">
            系统在30秒内未检测到您对校准点的准确注视。<br>
            请检查以下问题：
        </p>
        <div style="text-align: left; background: #f8f9fa; padding: 16px; border-radius: 8px; margin-bottom: 24px;">
            <ul style="margin: 0; padding-left: 20px; color: #555;">
                <li>确保您的脸部在摄像头视野内</li>
                <li>确保摄像头没有被遮挡</li>
                <li>确保环境光线充足</li>
                <li>尝试调整坐姿，使眼部更清晰</li>
                <li>检查是否正在注视红色校准点</li>
            </ul>
        </div>
        <div style="display: flex; gap: 12px; justify-content: center;">
            <button id="retry-calibration" style="
                background: #007bff;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 16px;
            ">重试校准点 ${pointIndex}</button>
            <button id="skip-calibration" style="
                background: #6c757d;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 16px;
            ">跳过这个点</button>
            <button id="restart-calibration" style="
                background: #dc3545;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 16px;
            ">重新开始校准</button>
        </div>
    `;

    overlay.appendChild(dialog);
    document.body.appendChild(overlay);

    // 按钮事件处理
    document.getElementById('retry-calibration').addEventListener('click', () => {
        overlay.remove();
        console.log(`用户选择重试校准点 ${pointIndex}`);
        // 重新启动当前校准点
        startCalibrationAtPoint(pointIndex - 1);
    });

    document.getElementById('skip-calibration').addEventListener('click', () => {
        overlay.remove();
        console.log(`用户选择跳过校准点 ${pointIndex}`);
        // 跳到下一个校准点
        if (pointIndex < calibrationPoints.length) {
            moveToNextPoint(pointIndex);
        } else {
            // 如果是最后一个点，完成校准
            finishCalibration();
        }
    });

    document.getElementById('restart-calibration').addEventListener('click', () => {
        overlay.remove();
        console.log('用户选择重新开始校准');

        // 清理当前状态
        if (movingCircle) {
            movingCircle.remove();
            movingCircle = null;
        }
        if (innerCircle) {
            innerCircle = null;
        }

        // 重新启动校准流程
        isCalibrating = false;
        currentPointIndex = 0;
        collectedData = [];

        // 重新显示开始按钮
        startBtn.style.display = 'block';
        startBtn.textContent = '重新开始校准';
        title.style.display = 'block';

        // 隐藏进度指示器
        const progressContainer = document.querySelector('.progress-container');
        if (progressContainer) {
            progressContainer.remove();
        }
    });
}

/**
 * 显示后端连接错误对话框
 */
function showBackendConnectionDialog(pointIndex) {
    // 创建遮罩层
    const overlay = document.createElement('div');
    overlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: rgba(0, 0, 0, 0.8);
        z-index: 10000;
        display: flex;
        align-items: center;
        justify-content: center;
    `;

    // 创建对话框
    const dialog = document.createElement('div');
    dialog.style.cssText = `
        background: white;
        border-radius: 12px;
        padding: 32px;
        max-width: 500px;
        width: 90%;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        text-align: center;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    `;

    // 诊断内容
    dialog.innerHTML = `
        <div style="font-size: 24px; margin-bottom: 16px;">🔌</div>
        <h3 style="margin: 0 0 16px 0; color: #333;">后端连接失败</h3>
        <p style="color: #666; line-height: 1.5; margin-bottom: 24px;">
            无法连接到校准服务器，无法进行注视检测。<br>
            请检查以下问题：
        </p>
        <div style="text-align: left; background: #f8f9fa; padding: 16px; border-radius: 8px; margin-bottom: 24px;">
            <ul style="margin: 0; padding-left: 20px; color: #555;">
                <li>确保后端服务器正在运行 (http://localhost:5000)</li>
                <li>检查网络连接</li>
                <li>确认摄像头权限已开启</li>
                <li>尝试重新启动后端服务</li>
            </ul>
        </div>
        <div style="display: flex; gap: 12px; justify-content: center;">
            <button id="retry-connection" style="
                background: #007bff;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 16px;
            ">重试连接</button>
            <button id="exit-calibration" style="
                background: #dc3545;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 16px;
            ">退出校准</button>
        </div>
        <div style="margin-top: 16px; font-size: 14px; color: #999;">
            注：没有后端连接时，系统无法进行真正的注视追踪校准
        </div>
    `;

    overlay.appendChild(dialog);
    document.body.appendChild(overlay);

    // 按钮事件处理
    document.getElementById('retry-connection').addEventListener('click', async () => {
        overlay.remove();
        console.log('用户选择重试连接');

        // 重新检查后端连接
        try {
            const response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.CALIBRATION_STATUS}`);
            if (response.ok) {
                backendConnected = true;
                console.log('后端连接恢复成功');

                // 重新启动校准系统
                await initializeSystem();

                // 重新开始校准流程
                isCalibrating = false;
                currentPointIndex = 0;
                collectedData = [];

                // 重新显示开始按钮
                startBtn.style.display = 'block';
                startBtn.textContent = '重新开始校准';
                title.style.display = 'block';

                // 清理当前状态
                if (movingCircle) {
                    movingCircle.remove();
                    movingCircle = null;
                }
                if (innerCircle) {
                    innerCircle = null;
                }

                // 隐藏进度指示器
                const progressContainer = document.querySelector('.progress-container');
                if (progressContainer) {
                    progressContainer.remove();
                }
            } else {
                throw new Error('连接仍然失败');
            }
        } catch (error) {
            console.error('重试连接失败:', error);
            alert('后端连接仍然失败，请检查服务器状态。');
        }
    });

    document.getElementById('exit-calibration').addEventListener('click', () => {
        overlay.remove();
        console.log('用户选择退出校准');

        // 清理所有状态并返回初始界面
        isCalibrating = false;
        currentPointIndex = 0;
        collectedData = [];

        if (movingCircle) {
            movingCircle.remove();
            movingCircle = null;
        }
        if (innerCircle) {
            innerCircle = null;
        }

        // 停止注视追踪
        stopGazeTracking();

        // 重新显示开始按钮
        startBtn.style.display = 'block';
        startBtn.textContent = '开始校准';
        title.style.display = 'block';

        // 隐藏进度指示器
        const progressContainer = document.querySelector('.progress-container');
        if (progressContainer) {
            progressContainer.remove();
        }
    });
}