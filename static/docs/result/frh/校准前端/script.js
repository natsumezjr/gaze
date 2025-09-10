// 定义校准点（9点布局）和颜色
const calibrationPoints = [
    { x: '50%', y: '50%' }, // 中心
    { x: '10%', y: '10%' }, // 左上
    { x: '90%', y: '10%' }, // 右上
    { x: '10%', y: '90%' }, // 左下
    { x: '90%', y: '90%' }, // 右下
    { x: '50%', y: '10%' }, // 中上
    { x: '10%', y: '50%' }, // 中左
    { x: '90%', y: '50%' }, // 中右
    { x: '50%', y: '90%' }  // 中下
];
const colors = ['#E74C3C', '#8E44AD', '#3498DB', '#1ABC9C', '#F1C40F', '#E67E22', '#2ECC71', '#D35400', '#7F8C8D'];

// 获取DOM元素
const calibrationContainer = document.getElementById('calibration-container');
const startButton = document.getElementById('start-button');
const title = document.querySelector('.title'); // 修正为使用类选择器
const gazePoint = document.getElementById('gaze-point');

// 状态变量
let isCalibrating = false;
let currentPointIndex = 0;
let calibrationData = [];
let gazePosition = { x: 0, y: 0 };
let focusStartTime = null;
const FOCUS_DURATION_MS = 1000; // 注视持续时间（毫秒）

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    startButton.addEventListener('click', startCalibration);
});

// 监听鼠标移动作为模拟的注视点
document.addEventListener('mousemove', (e) => {
    gazePosition.x = e.clientX;
    gazePosition.y = e.clientY;
    requestAnimationFrame(() => {
        gazePoint.style.transform = `translate(${gazePosition.x}px, ${gazePosition.y}px)`;
    });
});

/**
 * 开始校准流程
 */
function startCalibration() {
    isCalibrating = true;
    calibrationData = [];
    currentPointIndex = 0;
    startButton.style.display = 'none';
    title.style.display = 'none';
    gazePoint.style.opacity = '1';
    calibrationContainer.innerHTML = ''; // 清空容器
    
    // 创建进度指示器
    const progressIndicator = document.getElementById('progress-indicator');
    progressIndicator.textContent = `校准点: 0/${calibrationPoints.length}`;
    progressIndicator.classList.add('visible');
    
    // 开始倒计时
    startCountdown();
}

/**
 * 开始倒计时
 */
function startCountdown() {
    // 显示倒计时提示
    const countdownPrompt = document.getElementById('countdown-prompt');
    countdownPrompt.textContent = '请注视屏幕中心';
    countdownPrompt.classList.add('visible');
    
    // 创建移动圆圈
    const movingCircle = document.createElement('div');
    movingCircle.id = 'moving-circle';
    movingCircle.classList.add('countdown');
    movingCircle.style.left = '50%';
    movingCircle.style.top = '50%';
    calibrationContainer.appendChild(movingCircle);
    
    // 创建倒计时数字
    let count = 3;
    const countdownNumber = document.createElement('div');
    countdownNumber.textContent = count;
    countdownNumber.style.fontSize = '1.5rem';
    countdownNumber.style.fontWeight = 'bold';
    countdownNumber.style.color = '#FFFFFF';
    movingCircle.appendChild(countdownNumber);
    
    // 显示圆圈
    setTimeout(() => {
        movingCircle.classList.add('visible');
    }, 100);
    
    // 倒计时
    const countdownInterval = setInterval(() => {
        count--;
        if (count > 0) {
            countdownNumber.textContent = count;
        } else {
            clearInterval(countdownInterval);
            countdownPrompt.classList.remove('visible');
            movingCircle.classList.remove('countdown');
            movingCircle.innerHTML = '';
            
            // 创建内圈
            createInnerCircle(movingCircle);
            
            // 移动到第一个点
            setTimeout(() => {
                moveToFirstPoint(movingCircle);
            }, 500);
        }
    }, 1000);
}

/**
 * 创建内圈
 */
function createInnerCircle(movingCircle) {
    const innerShrink = document.createElement('div');
    innerShrink.className = 'inner-shrink';
    movingCircle.appendChild(innerShrink);
}

/**
 * 移动到第一个校准点
 */
function moveToFirstPoint(movingCircle) {
    const firstPoint = calibrationPoints[0];
    const firstColor = colors[0];
    
    // 设置移动动画
    movingCircle.style.transition = 'all 1s cubic-bezier(0.25, 0.46, 0.45, 0.94)';
    movingCircle.style.left = firstPoint.x;
    movingCircle.style.top = firstPoint.y;
    movingCircle.style.backgroundColor = firstColor;
    
    // 更新进度指示器
    updateProgress(1, calibrationPoints.length);
    
    // 移动完成后开始校准
    setTimeout(() => {
        startCalibrationAtPoint(0);
    }, 1000);
}

/**
 * 更新进度指示器
 */
function updateProgress(current, total) {
    const progressIndicator = document.getElementById('progress-indicator');
    progressIndicator.textContent = `校准点: ${current}/${total}`;
}

/**
 * 显示下一个校准点
 */
function displayNextPoint() {
    if (currentPointIndex >= calibrationPoints.length) {
        finishCalibration();
        return;
    }

    calibrationContainer.innerHTML = ''; // 清理上一个点
    const point = calibrationPoints[currentPointIndex];
    const targetPoint = createTargetPoint(point);
    calibrationContainer.appendChild(targetPoint);

    focusStartTime = null; // 重置聚焦开始时间
    requestAnimationFrame(checkFocus); // 开始检测循环
}

/**
 * 创建校准目标点元素
 * @param {object} point - 坐标对象 {x, y}
 */
function createTargetPoint(point) {
    const target = document.createElement('div');
    target.className = 'calibration-point';
    target.style.left = point.x;
    target.style.top = point.y;
    target.style.backgroundColor = colors[currentPointIndex % colors.length];
    return target;
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
        const innerCircle = document.querySelector('.inner-shrink');
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
    const dx = parseFloat(point1.x) - parseFloat(point2.x);
    const dy = parseFloat(point1.y) - parseFloat(point2.y);
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
    const nextColor = colors[nextIndex % colors.length];
    
    // 计算移动时间（统一速度）
    const moveTime = calculateMoveTime(currentPointIndex, nextIndex);
    
    console.log(`移动到校准点 ${nextIndex + 1}，距离: ${calculateDistance(calibrationPoints[currentPointIndex], nextPoint).toFixed(1)}，时间: ${moveTime.toFixed(2)}s`);
    
    // 获取移动圆圈
    const movingCircle = document.getElementById('moving-circle');
    const innerCircle = document.querySelector('.inner-shrink');
    
    // 设置移动动画 - 根据距离调整时间
    movingCircle.style.transition = `all ${moveTime}s cubic-bezier(0.25, 0.46, 0.45, 0.94)`;
    if (innerCircle) {
        innerCircle.style.transition = `all ${moveTime}s cubic-bezier(0.25, 0.46, 0.45, 0.94)`;
    }
    
    // 4. 移动过程中同时进行：位置移动 + 颜色渐变
    console.log(`移动到点 ${nextIndex + 1}，颜色渐变: ${nextColor}`);
    
    // 移除收缩动画类，准备移动
    if (innerCircle) {
        innerCircle.classList.remove('shrinking');
    }
    
    // 同时开始位置移动和颜色渐变
    movingCircle.style.left = `${nextPoint.x}`;
    movingCircle.style.top = `${nextPoint.y}`;
    movingCircle.style.backgroundColor = nextColor;
    
    // 内圈同时变色和放大
    if (innerCircle) {
        innerCircle.style.backgroundColor = '#FFFFFF';
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
 * 检测注视点是否在目标上
 */
function checkFocus() {
    if (!isCalibrating || currentPointIndex >= calibrationPoints.length) {
        return;
    }

    const target = calibrationContainer.querySelector('.calibration-point');
    if (!target) {
        requestAnimationFrame(checkFocus); // 如果目标不存在，继续等待
        return;
    }

    const targetRect = target.getBoundingClientRect();
    const isGazeInside =
        gazePosition.x >= targetRect.left &&
        gazePosition.x <= targetRect.right &&
        gazePosition.y >= targetRect.top &&
        gazePosition.y <= targetRect.bottom;

    if (isGazeInside) {
        if (focusStartTime === null) {
            focusStartTime = Date.now();
            target.classList.add('focus'); // 开始聚焦动画
            const innerCircle = document.querySelector('.inner-shrink');
            if (innerCircle) {
                innerCircle.classList.add('shrinking');
            }
        } else {
            const elapsedTime = Date.now() - focusStartTime;
            if (elapsedTime >= FOCUS_DURATION_MS) {
                collectData(targetRect);
                currentPointIndex++;
                displayNextPoint(); // 切换到下一个点
                return; // 成功采集，跳出当前帧的检测
            }
        }
    } else {
        focusStartTime = null;
        target.classList.remove('focus'); // 移出，移除聚焦状态
        const innerCircle = document.querySelector('.inner-shrink');
        if (innerCircle) {
            innerCircle.classList.remove('shrinking');
        }
    }

    requestAnimationFrame(checkFocus); // 继续下一帧检测
}

/**
 * 收集数据
 * @param {DOMRect} targetRect - 目标点的边界矩形
 */
function collectData(targetRect) {
    const targetPixel = {
        x: targetRect.left + targetRect.width / 2,
        y: targetRect.top + targetRect.height / 2,
    };

    // 生成写死的、符合格式的模拟3D坐标
    const eyeballCenter3dMm = { x: 1.0, y: 2.0, z: 3.0 };
    const pupilCenter3dMm = { x: 4.0, y: 5.0, z: 6.0 };

    calibrationData.push({
        target_pixel: targetPixel,
        eyeball_center_3d_mm: eyeballCenter3dMm,
        pupil_center_3d_mm: pupilCenter3dMm,
    });
    console.log(`Collected data for point ${currentPointIndex}:`, calibrationData[calibrationData.length - 1]);
}

/**
 * 采集眼动数据
 * @param {Object} point - 当前校准点坐标
 */
function collectGazeData(point) {
    console.log(`正在采集点 (${point.x}, ${point.y}) 的数据...`);
    
    // 模拟数据采集
    collectMockGazeData(point);
}

/**
 * 模拟数据采集
 * @param {Object} point - 当前校准点坐标
 */
function collectMockGazeData(point) {
    // 模拟数据采集过程
    console.log(`模拟采集点 (${point.x}, ${point.y}) 的数据`);
    
    // 这里可以添加模拟数据生成逻辑
    const mockData = {
        point_index: currentPointIndex,
        screen_xy: [parseFloat(point.x), parseFloat(point.y)],
        gaze_data: Array.from({length: 10}, () => ({
            timestamp: Date.now(),
            left_eye: {x: Math.random(), y: Math.random(), z: Math.random()},
            right_eye: {x: Math.random(), y: Math.random(), z: Math.random()}
        }))
    };
    
    console.log('模拟数据:', mockData);
    
    // 在实际应用中，这里会发送数据到后端
}

/**
 * 完成校准
 */
function completeCalibration() {
    console.log('校准完成，准备提交数据');
    // 这里可以添加校准完成后的逻辑
}

/**
 * 结束校准并提交数据
 */
async function finishCalibration() {
    isCalibrating = false;
    gazePoint.style.opacity = '0';
    
    // 隐藏移动圆圈
    const movingCircle = document.getElementById('moving-circle');
    if (movingCircle) {
        movingCircle.style.opacity = '0';
        setTimeout(() => {
            if (movingCircle.parentNode) {
                movingCircle.remove();
            }
        }, 300);
    }
    
    // 隐藏进度指示器
    const progressIndicator = document.getElementById('progress-indicator');
    progressIndicator.classList.remove('visible');
    
    // 移除内圈收缩动画
    const innerCircle = document.querySelector('.inner-shrink');
    if (innerCircle) {
        innerCircle.classList.remove('shrinking');
    }
    
    calibrationContainer.innerHTML = '<div class="message">校准完成！正在提交数据...</div>';

    try {
        // 替换为您的后端API地址
        const response = await fetch('http://localhost:8000/api/calibration', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                calibration_points: calibrationData,
                screen_width: window.screen.width,
                screen_height: window.screen.height,
            }),
        });

        if (response.ok) {
            calibrationContainer.innerHTML = '<div class="message">数据提交成功！</div>';
        } else {
            const errorData = await response.json();
            calibrationContainer.innerHTML = `<div class="message">数据提交失败: ${errorData.detail || response.statusText}</div>`;
        }
    } catch (error) {
        console.error('Error submitting calibration data:', error);
        calibrationContainer.innerHTML = `<div class="message">数据提交出错: ${error.message}</div>`;
    }

    // 3秒后重置UI
    setTimeout(() => {
        startButton.style.display = 'block';
        title.style.display = 'block';
        calibrationContainer.innerHTML = '';
    }, 3000);
}
