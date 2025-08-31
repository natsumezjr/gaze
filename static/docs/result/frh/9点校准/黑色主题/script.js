// 眼动校准系统 - 完全重写版本
// 校准点坐标数组 - 使用百分比定位
const calibrationPoints = [
    { x: 10, y: 10 },   // 左上角
    { x: 90, y: 10 },   // 右上角
    { x: 50, y: 50 },   // 中心点
    { x: 10, y: 90 },   // 左下角
    { x: 90, y: 90 },   // 右下角
    { x: 50, y: 10 },   // 上中
    { x: 50, y: 90 },   // 下中
    { x: 10, y: 50 },   // 左中
    { x: 90, y: 50 }    // 右中
];

// 校准点颜色数组 - 外圈暗色，内圈亮色（如iPad风格）
const calibrationColors = [
    { bg: '#CC4444', fg: '#FF8888' }, // 珊瑚红系：暗 -> 亮
    { bg: '#339999', fg: '#77DDDD' }, // 青绿色系：暗 -> 亮
    { bg: '#4477CC', fg: '#88BBFF' }, // 天蓝色系：暗 -> 亮
    { bg: '#449966', fg: '#88DDBB' }, // 薄荷绿系：暗 -> 亮
    { bg: '#CC9944', fg: '#FFCC88' }, // 温暖黄系：暗 -> 亮
    { bg: '#9944AA', fg: '#DD88DD' }, // 优雅紫系：暗 -> 亮
    { bg: '#449988', fg: '#88DDCC' }, // 海洋绿系：暗 -> 亮
    { bg: '#CC8844', fg: '#FFBB77' }, // 阳光金系：暗 -> 亮
    { bg: '#884488', fg: '#BB88BB' }  // 梦幻紫系：暗 -> 亮
];

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

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    console.log('眼动校准系统已加载');
    startBtn.addEventListener('click', startCalibration);
});

/**
 * 开始校准流程
 */
function startCalibration() {
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
    
    // 开始倒计时
    startCountdown();
}

/**
 * 开始倒计时
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
    movingCircle.className = 'countdown'; // 添加倒计时状态类
    
    // 设置初始倒计时样式（移除会覆盖CSS的样式）
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
            
            // 最后一个数字"1"渐变消失效果
            movingCircle.style.transition = 'color 0.5s ease';
            movingCircle.style.color = 'rgba(0, 0, 0, 0)'; // 文字渐变透明
            
            setTimeout(() => {
                // 1. 数字完全消失，变为纯白圆圈
                movingCircle.textContent = '';
                movingCircle.classList.remove('countdown'); // 移除倒计时状态类
                
                // 重置样式
                movingCircle.style.color = '#000000';
                movingCircle.style.transition = 'all 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94)';
                
                // 隐藏提示
                prompt.classList.remove('visible');
                setTimeout(() => prompt.remove(), 500);
                
                // 开始移动到第一个点
                moveToFirstPoint();
            }, 500);
        }
    }, 1000);
}

/**
 * 移动到第一个校准点
 */
function moveToFirstPoint() {
    const firstPoint = calibrationPoints[0];
    const firstColor = calibrationColors[0];
    
    console.log('移动到第一个校准点');
    
    // 开始移动
    setTimeout(() => {
        movingCircle.style.left = `${firstPoint.x}%`;
        movingCircle.style.top = `${firstPoint.y}%`;
        
        // 2. 移动途中平滑变色为内圈亮色（与位置移动同步）
        console.log(`移动到第一个点，颜色渐变: ${firstColor.fg}`);
        // 启用颜色过渡，实现平滑渐变到内圈颜色
        movingCircle.style.transition = 'all 1.0s cubic-bezier(0.25, 0.46, 0.45, 0.94)';
        movingCircle.style.backgroundColor = firstColor.fg; // 变为内圈亮色
        movingCircle.style.opacity = '1';
        
    }, 100);
    
    // 3. 到达后变为外圈暗色，创建内圈亮色并开始收缩
    setTimeout(() => {
        // 圆圈变为外圈暗色
        movingCircle.style.transition = 'none';
        movingCircle.style.backgroundColor = firstColor.bg;
        movingCircle.style.opacity = '1';
        
        // 创建内圈亮色
        createInnerCircle(firstColor.fg);
        startCalibrationAtPoint(0);
    }, 1100); // 调整为1.1秒，匹配新的移动时长
}

/**
 * 创建内圈
 */
function createInnerCircle(color) {
    // 完全清理移动圆圈内容
    movingCircle.innerHTML = '';
    
    // 创建新的内圈
    innerCircle = document.createElement('div');
    innerCircle.className = 'inner-shrink';
    
    // 使用内联样式确保颜色正确，从100%开始
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
    
    // 验证同心圆定位
    setTimeout(() => {
        const outerRect = movingCircle.getBoundingClientRect();
        const innerRect = innerCircle.getBoundingClientRect();
        const outerCenter = { x: outerRect.left + outerRect.width/2, y: outerRect.top + outerRect.height/2 };
        const innerCenter = { x: innerRect.left + innerRect.width/2, y: innerRect.top + innerRect.height/2 };
        const offset = Math.sqrt(Math.pow(outerCenter.x - innerCenter.x, 2) + Math.pow(outerCenter.y - innerCenter.y, 2));
        console.log(`同心圆检查 - 中心偏移: ${offset.toFixed(2)}px (应该接近0)`);
    }, 50);
}



/**
 * 在当前点开始校准
 */
function startCalibrationAtPoint(pointIndex) {
    console.log(`校准点 ${pointIndex + 1}/${calibrationPoints.length}`);
    updateProgress(pointIndex + 1, calibrationPoints.length);
    
    // 更新当前点索引
    currentPointIndex = pointIndex;
    
    // 采集数据
    collectGazeData(calibrationPoints[pointIndex]);
    
    // 3. 内圈开始收缩到25% - 使用优化的CSS动画
    setTimeout(() => {
        if (innerCircle) {
            // 确保动画能够执行
            innerCircle.style.transition = 'none';
            innerCircle.style.removeProperty('animation');
            
            // 强制重排后开始动画
            innerCircle.offsetHeight; // 触发重排
            
            // 开始流畅的收缩动画
            innerCircle.classList.add('shrinking');
            console.log('内圈收缩动画已开始');
        }
    }, 100);
    
    // 2秒后移动到下一个点或结束
    setTimeout(() => {
        if (pointIndex + 1 < calibrationPoints.length) {
            moveToNextPoint(pointIndex + 1);
        } else {
            // 所有点完成
            finishCalibration();
        }
    }, 2000);
}

/**
 * 计算两点间距离
 */
function calculateDistance(point1, point2) {
    const dx = point1.x - point2.x;
    const dy = point1.y - point2.y;
    return Math.sqrt(dx * dx + dy * dy);
}

/**
 * 根据距离计算移动时间（统一速度）
 */
function calculateMoveTime(currentIndex, nextIndex) {
    const currentPoint = calibrationPoints[currentIndex];
    const nextPoint = calibrationPoints[nextIndex];
    const distance = calculateDistance(currentPoint, nextPoint);
    
    // 设定基准速度（距离100对应1秒）
    const baseSpeed = 100; // 每100单位距离用时1秒
    const minTime = 0.6; // 最小时间0.6秒
    const maxTime = 1.4; // 最大时间1.4秒
    
    const calculatedTime = distance / baseSpeed;
    return Math.max(minTime, Math.min(maxTime, calculatedTime));
}

/**
 * 移动到下一个校准点
 */
function moveToNextPoint(nextIndex) {
    const nextPoint = calibrationPoints[nextIndex];
    const nextColor = calibrationColors[nextIndex];
    
    // 计算移动时间（统一速度）
    const moveTime = calculateMoveTime(currentPointIndex, nextIndex);
    
    console.log(`移动到校准点 ${nextIndex + 1}，距离: ${calculateDistance(calibrationPoints[currentPointIndex], nextPoint).toFixed(1)}，时间: ${moveTime.toFixed(2)}s`);
    
    // 设置移动动画 - 根据距离调整时间
    movingCircle.style.transition = `all ${moveTime}s cubic-bezier(0.25, 0.46, 0.45, 0.94)`;
    if (innerCircle) {
        innerCircle.style.transition = `all ${moveTime}s cubic-bezier(0.25, 0.46, 0.45, 0.94)`;
    }
    
    // 4. 移动过程中同时进行：位置移动 + 颜色渐变
    console.log(`移动到点 ${nextIndex + 1}，颜色渐变: ${nextColor.bg}`);
    
    // 移除收缩动画类，准备移动
    if (innerCircle) {
        innerCircle.classList.remove('shrinking');
    }
    
    // 同时开始位置移动和颜色渐变
    movingCircle.style.left = `${nextPoint.x}%`;
    movingCircle.style.top = `${nextPoint.y}%`;
    movingCircle.style.backgroundColor = nextColor.bg;
    
    // 内圈同时变色和放大
    if (innerCircle) {
        innerCircle.style.backgroundColor = nextColor.fg;
        innerCircle.style.width = '100%';
        innerCircle.style.height = '100%';
    }
    
    // 移动完成后开始下一轮校准（内圈已经是100%，无需重置）
    setTimeout(() => {
        // 开始下一个点的校准
        setTimeout(() => {
            startCalibrationAtPoint(nextIndex);
        }, 100);
        
    }, moveTime * 1000); // 根据实际移动时间调整
}

/**
 * 完成校准
 */
function finishCalibration() {
    console.log('所有校准点完成');
    
    // 圆圈消失
    if (movingCircle) {
        movingCircle.style.opacity = '0';
        setTimeout(() => {
            if (movingCircle.parentNode) {
                movingCircle.remove();
            }
            movingCircle = null;
            innerCircle = null;
        }, 300);
    }
    
    // 完成校准流程
    completeCalibration();
}

// ==================== 数据采集与发送函数 ====================

/**
 * 采集眼动数据
 * @param {Object} point - 当前校准点坐标
 */
function collectGazeData(point) {
    console.log(`正在采集点 (${point.x}, ${point.y}) 的数据...`);
    
    // 模拟数据采集过程
    const sampleInterval = 100; // 每100ms采样一次
    const sampleCount = 20; // 2秒内采样20次
    
    for (let i = 0; i < sampleCount; i++) {
        setTimeout(() => {
            // 模拟采集到的眼动数据
            const gazeData = {
                pointId: currentPointIndex,
                targetX: point.x,
                targetY: point.y,
                timestamp: Date.now(),
                sampleIndex: i,
                gazeX: point.x + (Math.random() - 0.5) * 5,
                gazeY: point.y + (Math.random() - 0.5) * 5,
                confidence: 0.8 + Math.random() * 0.2,
                pupilDiameter: 3.0 + Math.random() * 1.0
            };
            
            collectedData.push(gazeData);
            
            if (i === 0) {
                console.log(`开始采集点 ${currentPointIndex + 1} 的数据`);
            } else if (i === sampleCount - 1) {
                console.log(`完成采集点 ${currentPointIndex + 1} 的数据`);
            }
            
        }, i * sampleInterval);
    }
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
        progressIndicator.textContent = `校准进度: ${current}/${total}`;
    }
}

/**
 * 隐藏进度指示器
 */
function hideProgressIndicator() {
    const progressIndicator = document.querySelector('.progress-indicator');
    if (progressIndicator) {
        progressIndicator.classList.remove('visible');
    }
}

/**
 * 完成校准流程
 */
function completeCalibration() {
    console.log('校准完成');
    isCalibrating = false;
    
    // 发送数据到服务器
    sendDataToServer(collectedData);
    
    // 隐藏进度指示器
    hideProgressIndicator();
    
    // 显示完成信息
    showCompletionMessage();
    
    // 3秒后重置界面
    setTimeout(() => {
        resetInterface();
    }, 3000);
}

/**
 * 显示校准完成信息
 */
function showCompletionMessage() {
    const completionMessage = document.createElement('div');
    completionMessage.innerHTML = `
        <h2>校准完成！</h2>
        <p>眼动追踪系统已成功校准</p>
        <div style="font-size: 4rem; color: #2ecc71; margin-top: 1rem;">✓</div>
    `;
    
    completionMessage.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        text-align: center;
        color: white;
        z-index: 100;
        opacity: 0;
        transition: opacity 0.5s ease;
    `;
    
    document.body.appendChild(completionMessage);
    
    // 显示动画
    requestAnimationFrame(() => {
        completionMessage.style.opacity = '1';
    });
    
    // 2.5秒后开始淡出
    setTimeout(() => {
        completionMessage.style.opacity = '0';
        setTimeout(() => {
            if (completionMessage.parentNode) {
                completionMessage.remove();
            }
        }, 500);
    }, 2500);
}

/**
 * 重置界面到初始状态
 */
function resetInterface() {
    startBtn.style.display = 'inline-block';
    title.style.display = 'block';
    
    currentPointIndex = 0;
    isCalibrating = false;
    collectedData = [];
    movingCircle = null;
    innerCircle = null;
    
    console.log('界面已重置，可以重新开始校准');
}

/**
 * 将采集到的数据发送到服务器
 */
async function sendDataToServer(data) {
    console.log('将所有采集数据发送到服务器...');
    console.log('采集到的数据总数:', data.length);
    
    // 模拟发送成功
    setTimeout(() => {
        const mockResult = {
            status: 'success',
            model_id: `calibration_model_${Date.now()}`,
            accuracy: 92.5 + Math.random() * 5,
            message: '校准成功，模型已生成'
        };
        
        console.log('✓ 校准数据发送成功');
        showResponseMessage('success', '校准成功，模型已生成', mockResult);
    }, 1500);
}

/**
 * 显示响应消息 - iOS风格
 */
function showResponseMessage(type, message, details = {}) {
    const messageElement = document.createElement('div');
    
    const isSuccess = type === 'success';
    const icon = isSuccess ? '✓' : '✗';
    const iconColor = isSuccess ? '#34C759' : '#FF3B30';
    
    messageElement.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: rgba(28, 28, 30, 0.95);
        backdrop-filter: blur(40px);
        -webkit-backdrop-filter: blur(40px);
        color: white;
        padding: 35px 45px;
        border-radius: 22px;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", sans-serif;
        font-size: 1.1rem;
        font-weight: 500;
        text-align: center;
        z-index: 1000;
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.15);
        max-width: 400px;
        opacity: 0;
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        letter-spacing: -0.2px;
    `;
    
    messageElement.innerHTML = `
        <div style="font-size: 2.5rem; margin-bottom: 15px; font-weight: 300; color: ${iconColor};">${icon}</div>
        <div style="font-weight: 600; margin-bottom: 12px; font-size: 1.3rem; letter-spacing: -0.3px;">${message}</div>
        ${details.model_id ? `<div style="font-size: 0.85rem; opacity: 0.8; margin-bottom: 4px;">模型ID: ${details.model_id}</div>` : ''}
        ${details.accuracy ? `<div style="font-size: 0.85rem; opacity: 0.8;">精度: ${Math.round(details.accuracy)}%</div>` : ''}
    `;
    
    document.body.appendChild(messageElement);
    
    requestAnimationFrame(() => {
        messageElement.style.opacity = '1';
        messageElement.style.transform = 'translate(-50%, -50%) scale(1)';
    });
    
    setTimeout(() => {
        messageElement.style.opacity = '0';
        messageElement.style.transform = 'translate(-50%, -50%) scale(0.95)';
        setTimeout(() => {
            if (messageElement.parentNode) {
                messageElement.remove();
            }
        }, 400);
    }, isSuccess ? 4000 : 6000);
}

/**
 * 处理键盘事件
 */
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape' && isCalibrating) {
        console.log('用户中断校准');
        
        // 清理圆圈
        if (movingCircle && movingCircle.parentNode) {
            movingCircle.remove();
        }
        
        resetInterface();
        hideProgressIndicator();
    }
});
