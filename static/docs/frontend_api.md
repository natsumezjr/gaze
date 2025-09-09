# 眼动追踪系统前端接口文档

## 概述

本文档描述了眼动追踪系统与前端的数据交互接口，包括粗拟合结果的数据格式和API函数。

## 数据格式

### 粗拟合结果数据结构

```python
coarse_fitting_results = {
    'kappa_angle': float,      # Kappa角度（度）
    'fit_quality': dict,       # 拟合质量信息
    'status': str,             # 状态：'pending'/'success'/'failed'
    'message': str             # 状态消息
}
```

### 详细字段说明

#### kappa_angle
- **类型**: `float`
- **单位**: 度（degrees）
- **描述**: 计算得到的Kappa角度值
- **示例**: `12.5`

#### fit_quality
- **类型**: `dict`
- **描述**: 拟合质量评估结果
- **结构**:
  ```python
  {
      'pixel_error': float,    # 像素误差
      'angular_error': float,  # 角度误差
      'rmse': float,          # 均方根误差
      'r_squared': float      # 决定系数
  }
  ```

#### status
- **类型**: `str`
- **可能值**:
  - `'pending'`: 等待中
  - `'success'`: 成功
  - `'failed'`: 失败
- **描述**: 粗拟合的执行状态

#### message
- **类型**: `str`
- **描述**: 状态描述信息
- **示例**: `"粗拟合完成"`, `"粗拟合失败"`

## API函数

### 1. get_coarse_fitting_results()

**功能**: 获取当前粗拟合结果

**参数**: 无

**返回值**: `dict` - 粗拟合结果字典的副本

**示例**:
```python
results = get_coarse_fitting_results()
print(f"状态: {results['status']}")
print(f"Kappa角: {results['kappa_angle']}°")
```

### 2. send_to_frontend(data)

**功能**: 发送数据给前端

**参数**:
- `data` (dict): 要发送的数据

**返回值**: 无

**示例**:
```python
# 发送粗拟合结果
send_to_frontend(coarse_fitting_results)

# 发送自定义数据
custom_data = {
    'type': 'calibration_update',
    'timestamp': time.time(),
    'data': coarse_fitting_results
}
send_to_frontend(custom_data)
```

## 数据流示例

### 成功流程
```python
# 1. 初始状态
{
    'kappa_angle': None,
    'fit_quality': None,
    'status': 'pending',
    'message': ''
}

# 2. 粗拟合成功
{
    'kappa_angle': 12.5,
    'fit_quality': {
        'pixel_error': 2.3,
        'angular_error': 1.2,
        'rmse': 1.8,
        'r_squared': 0.95
    },
    'status': 'success',
    'message': '粗拟合完成'
}
```

### 失败流程
```python
# 粗拟合失败
{
    'kappa_angle': None,
    'fit_quality': None,
    'status': 'failed',
    'message': '粗拟合失败'
}
```

## 前端集成建议

### 1. 轮询方式
```javascript
// 前端JavaScript示例
function pollCoarseFittingResults() {
    fetch('/api/coarse-fitting-results')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                console.log('Kappa角:', data.kappa_angle);
                console.log('拟合质量:', data.fit_quality);
                // 更新UI显示
            } else if (data.status === 'failed') {
                console.error('粗拟合失败:', data.message);
                // 显示错误信息
            }
        })
        .catch(error => {
            console.error('获取结果失败:', error);
        });
}

// 每2秒轮询一次
setInterval(pollCoarseFittingResults, 2000);
```

### 2. WebSocket方式
```javascript
// WebSocket连接示例
const ws = new WebSocket('ws://localhost:8080/calibration');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    if (data.type === 'coarse_fitting_complete') {
        console.log('粗拟合完成:', data.results);
        // 更新UI
    }
};
```

### 3. 事件驱动方式
```python
# 后端Python示例
import threading
import time

def notify_frontend():
    """通知前端粗拟合完成"""
    results = get_coarse_fitting_results()
    if results['status'] == 'success':
        # 发送WebSocket消息
        send_websocket_message({
            'type': 'coarse_fitting_complete',
            'results': results,
            'timestamp': time.time()
        })

# 在粗拟合完成后调用
notify_frontend()
```

## 错误处理

### 常见错误情况

1. **数据不足**
   - 状态: `'failed'`
   - 消息: `"会话中有 0 个样本，需要至少1个样本进行kappa校准"`

2. **拟合失败**
   - 状态: `'failed'`
   - 消息: `"眼球形状拟合失败"`

3. **Kappa估计失败**
   - 状态: `'failed'`
   - 消息: `"Kappa估计失败"`

### 前端错误处理建议

```javascript
function handleCoarseFittingResults(data) {
    switch(data.status) {
        case 'success':
            showSuccessMessage('粗拟合完成');
            updateCalibrationUI(data);
            break;
        case 'failed':
            showErrorMessage(`粗拟合失败: ${data.message}`);
            showRetryButton();
            break;
        case 'pending':
            showLoadingMessage('正在进行粗拟合...');
            break;
        default:
            console.warn('未知状态:', data.status);
    }
}
```

## 扩展接口

### 精细拟合结果
```python
fine_fitting_results = {
    'kappa_angle': float,
    'fit_quality': dict,
    'calibration_points': int,  # 校准点数量
    'status': str,
    'message': str
}
```

### 实时视线数据
```python
gaze_data = {
    'left_eye': {
        'gaze_vector': [float, float, float],
        'pupil_center': [float, float, float],
        'confidence': float
    },
    'right_eye': {
        'gaze_vector': [float, float, float],
        'pupil_center': [float, float, float],
        'confidence': float
    },
    'timestamp': float
}
```

## 注意事项

1. **数据更新频率**: 粗拟合结果只在完成时更新一次
2. **线程安全**: 所有API函数都是线程安全的
3. **数据持久化**: 结果存储在内存中，程序重启后会丢失
4. **错误恢复**: 建议前端实现重试机制
5. **性能考虑**: 避免过于频繁的轮询请求

## 更新日志

- **v1.0.0** (2025-09-09): 初始版本，支持粗拟合结果接口
