# UI 和 Main 之间的回调接口规范

## 概述

本文档定义了 UI 线程（EyeCalibrationApp）和主线程/拟合线程之间的回调接口规范。

## 数据流

```
拟合线程 (FittingManager)
  ↓ emit('calibration_start_request')
UI 线程 (EyeCalibrationApp._on_calibration_start_request)
  ↓ 启动校准流程
UI 线程 (EyeCalibrationApp.start_calibration)
  ↓ emit('calibration_data_request', target_pixel)
拟合线程 (FittingManager._on_calibration_data_request)
  ↓ 计算视线交点
拟合线程 (FittingManager)
  ↓ emit('calibration_data_submit', left_eye_data, right_eye_data)
UI 线程 (EyeCalibrationApp.submit_data_callback)
  ↓ 收集数据
UI 线程 (EyeCalibrationApp)
  ↓ emit('calibration_data_ready', left_eye_data, right_eye_data)
拟合线程 (FittingManager._on_calibration_data_ready)
  ↓ 处理数据
UI 线程 (EyeCalibrationApp.finish_calibration)
  ↓ emit('calibration_complete', calibration_result)
拟合线程 (FittingManager._on_calibration_complete)
```

## 回调接口定义

### 1. calibration_start_request

**事件名称**: `calibration_start_request`

**触发者**: FittingManager（拟合线程）

**接收者**: EyeCalibrationApp（UI 线程）

**参数**: 无

**说明**: 
- 当拟合线程检测到需要校准时触发
- UI 收到此回调后，如果当前状态为 IDLE，则启动校准流程

**回调函数签名**:
```python
def _on_calibration_start_request(self) -> None:
    """收到校准启动请求"""
    pass
```

### 2. calibration_data_request

**事件名称**: `calibration_data_request`

**触发者**: FittingManager（拟合线程）

**接收者**: EyeCalibrationApp（UI 线程）

**参数**:
- `target_pixel` (Point2D): 目标像素坐标

**说明**:
- 当拟合线程需要为某个校准点收集数据时触发
- UI 收到此回调后，显示该目标点，并进入等待数据状态

**回调函数签名**:
```python
def _on_calibration_data_request(self, target_pixel: Point2D) -> None:
    """收到校准数据请求"""
    pass
```

### 3. calibration_data_submit

**事件名称**: `calibration_data_submit`

**触发者**: FittingManager（拟合线程）

**接收者**: EyeCalibrationApp（UI 线程）

**参数**:
- `left_eye_data` (Point2D): 左眼视线交点
- `right_eye_data` (Point2D): 右眼视线交点

**说明**:
- 拟合线程计算完视线交点后，通过此回调提交给 UI
- UI 收到数据后，如果当前状态为 WAITING_FOR_DATA，则收集数据

**回调函数签名**:
```python
def submit_data_callback(self, left_eye_data: Point2D, right_eye_data: Point2D) -> bool:
    """外部通过回调提交数据"""
    pass
```

**返回值**: 
- `bool`: 是否成功接收数据

### 4. calibration_data_ready

**事件名称**: `calibration_data_ready`

**触发者**: EyeCalibrationApp（UI 线程）

**接收者**: FittingManager（拟合线程）

**参数**:
- `left_eye_data` (Point2D): 左眼数据
- `right_eye_data` (Point2D): 右眼数据

**说明**:
- UI 收集完一个校准点的数据后，通过此回调通知拟合线程
- 拟合线程收到后，可以继续处理或请求下一个校准点

**回调函数签名**:
```python
def _on_calibration_data_ready(self, left_eye_data: Point2D, right_eye_data: Point2D) -> None:
    """校准数据就绪回调"""
    pass
```

### 5. calibration_complete

**事件名称**: `calibration_complete`

**触发者**: EyeCalibrationApp（UI 线程）

**接收者**: FittingManager（拟合线程）

**参数**:
- `calibration_result` (dict): 校准结果
  ```python
  {
      "left_eye": CalibrationResult,
      "right_eye": CalibrationResult
  }
  ```

**说明**:
- UI 完成所有校准点收集后，通过此回调通知拟合线程
- 拟合线程收到后，可以提取校准数据并计算 Kappa 角

**回调函数签名**:
```python
def _on_calibration_complete(self, calibration_result: dict) -> None:
    """校准完成回调"""
    pass
```

## 实现示例

### FittingManager 中的使用

```python
class FittingManager:
    def __init__(self, recg_fit_data_manager, callback_manager=None):
        self.callback_manager = callback_manager or CALLBACK_MANAGER
        
        # 注册回调
        self.callback_manager.register(
            'calibration_data_ready',
            self._on_calibration_data_ready
        )
        self.callback_manager.register(
            'calibration_complete',
            self._on_calibration_complete
        )
        
        # 等待数据的同步机制
        self._calibration_data = None
        self._calibration_ready = threading.Event()
        self._calibration_result = None
        self._calibration_complete = threading.Event()
    
    def _on_calibration_data_ready(self, left_eye_data, right_eye_data):
        """校准数据就绪回调"""
        self._calibration_data = (left_eye_data, right_eye_data)
        self._calibration_ready.set()
    
    def _on_calibration_complete(self, calibration_result):
        """校准完成回调"""
        self._calibration_result = calibration_result
        self._calibration_complete.set()
    
    def run(self):
        if not self.kapa_storage.is_pixel_valid():
            # 请求启动校准
            self.callback_manager.emit('calibration_start_request')
            
            # 等待 UI 准备好（可以通过状态检查或回调确认）
            time.sleep(0.5)  # 给 UI 一些时间启动
            
            # 请求校准数据
            target_pixel = self.intersections["left"]
            self.callback_manager.emit(
                'calibration_data_request',
                target_pixel=target_pixel
            )
            
            # 提交当前计算的视线交点
            self.callback_manager.emit(
                'calibration_data_submit',
                left_eye_data=self.intersections["left"],
                right_eye_data=self.intersections["right"]
            )
            
            # 等待数据就绪
            if self._calibration_ready.wait(timeout=30.0):
                left_data, right_data = self._calibration_data
                # 处理数据
                self.kapa_storage.set_pixel(left_data, right_data)
```

### EyeCalibrationApp 中的使用

```python
class EyeCalibrationApp:
    def __init__(self, callback_manager=None):
        self.callback_manager = callback_manager or CALLBACK_MANAGER
        
        # 注册回调
        self.callback_manager.register(
            'calibration_start_request',
            self._on_calibration_start_request
        )
        self.callback_manager.register(
            'calibration_data_request',
            self._on_calibration_data_request
        )
        self.callback_manager.register(
            'calibration_data_submit',
            self.submit_data_callback
        )
    
    def _on_calibration_start_request(self):
        """收到校准启动请求"""
        if self.state == CalibrationState.IDLE:
            # 在主线程中启动校准（Tkinter 要求）
            self.root.after(0, self.start_calibration)
    
    def _on_calibration_data_request(self, target_pixel):
        """收到校准数据请求"""
        if self.state == CalibrationState.CALIBRATING:
            # 在主线程中更新 UI
            self.root.after(0, lambda: self._handle_data_request(target_pixel))
    
    def _handle_data_request(self, target_pixel):
        """处理数据请求（在主线程中执行）"""
        self.current_target_pixel = target_pixel
        self.state = CalibrationState.WAITING_FOR_DATA
        self.show_waiting_for_data(target_pixel)
    
    def submit_data_callback(self, left_eye_data, right_eye_data):
        """外部通过回调提交数据"""
        if self.state == CalibrationState.WAITING_FOR_DATA:
            # 在主线程中处理数据
            self.root.after(0, lambda: self._process_submitted_data(left_eye_data, right_eye_data))
            return True
        return False
    
    def _process_submitted_data(self, left_eye_data, right_eye_data):
        """处理提交的数据（在主线程中执行）"""
        self.collected_left_eye_points.append(left_eye_data)
        self.collected_right_eye_points.append(right_eye_data)
        
        # 通知拟合线程数据已接收
        self.callback_manager.emit(
            'calibration_data_ready',
            left_eye_data=left_eye_data,
            right_eye_data=right_eye_data
        )
        
        # 继续下一个点
        self.continue_to_next_point()
    
    def finish_calibration(self):
        """完成校准"""
        self.state = CalibrationState.COMPLETED
        
        # 通过回调通知校准完成
        result = self.get_calibration_result()
        self.callback_manager.emit(
            'calibration_complete',
            calibration_result=result
        )
```

## 注意事项

1. **线程安全**: 
   - 所有回调函数都应该是线程安全的
   - UI 相关的操作必须在主线程（Tkinter 主循环）中执行，使用 `root.after(0, ...)` 调度

2. **数据传递**:
   - 使用不可变对象（如 Point2D）传递数据，避免共享可变状态
   - 复杂对象应该深拷贝后再传递

3. **错误处理**:
   - 回调函数应该捕获异常，避免影响其他回调
   - 使用超时机制避免无限等待

4. **生命周期管理**:
   - 在模块销毁时注销回调，避免内存泄漏
   - 使用单例模式确保回调管理器唯一

5. **状态同步**:
   - UI 状态检查应该在主线程中进行
   - 使用事件（Event）或条件变量（Condition）进行线程间同步
