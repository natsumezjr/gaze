# 数据模型与数据来源说明

本文档说明本项目在“识别→拟合→标定/补偿”流程中的核心数据结构、字段含义、单位与来源，确保前后端与各模块协同一致、可扩展、易维护。

## 1. 总览

- 摄像头帧数据：由 `project/recognition/utils/camera_data_manager.py` 统一管理。
- 识别输出三维点：由 `project/recg_fit_data/data_manager.py` 的全局实例 `RECG_FIT_DATA_MANAGER` 统一管理。
- 拟合结果、标定样本、Kappa 模型与补偿视线：由 `project/fitting/app/state.py` 的全局实例 `SESSION_MANAGER` 统一管理。

数据主链路：
1) 摄像头 → `camera_data_manager.add_frame()` → `get_image/get_depth`
2) 人脸/眼睛识别 → 三维关键点 → `RECG_FIT_DATA_MANAGER`
3) 眼球拟合 → `fit_all_eyes()` → 结果写入 `SESSION_MANAGER.last_fitting`
4) 前端标定 → `push_sample()` → 样本进入 `SESSION_MANAGER.samples`
5) `fit_kappa()` 生成 Kappa 模型 → `compute_gaze()` 得到补偿后视线

## 2. 摄像头帧数据（CameraDataManager）

- 模块：`project/recognition/utils/camera_data_manager.py`
- 接口：
  - `add_frame(frame_id, bgr_image, depth_map)`
  - `get_image(frame_id) -> np.ndarray(H,W,3), uint8`
  - `get_depth(frame_id) -> np.ndarray(H,W), float32`
  - `get_resolution() -> (width, height)`
- 单位：
  - BGR 图像：uint8, 0~255
  - 深度图：float32，原始单位（在 detector 中转换为米）

## 3. 识别输出三维点（RECG_FIT_DATA_MANAGER）

- 模块：`project/recg_fit_data/data_manager.py`
- 眼与类型：
  - `EYE_TYPE = ["left", "right"]`
  - `FITTING_TYPE = ["pupil", "iris", "inner_canthus", "upper_eyelid", "lower_eyelid", "outer_canthus"]`
- 存储格式：
  - 返回列表[List[np.ndarray]]，单点 `shape=(4,), dtype=float32`，含义 `[x, y, z, visibility]`
  - 坐标单位：毫米（内部自动从米转换到毫米）

## 4. 拟合结果（FittingResultLite）

- 模块：`project/fitting/app/state.py`
- 字段：
  - `center: np.ndarray(3,), mm`
  - `radius: float, mm`
  - `confidence: float (0~1)`
  - `converged: bool`
  - `final_residual_norm: float`
  - `strategy_used: str`
- 来源：`fit_all_eyes()` 完成后在 `project/fitting/main.py` 写入 `SESSION_MANAGER.set_last_fitting(eye, lite)`

## 5. 标定样本与会话（CalibrationSample / CalibrationSession）

- 模块：`project/fitting/app/state.py`
- 样本字段：
  - `timestamp: float`
  - `target_pixel: (int, int)` 屏幕像素坐标
  - `eye_center: np.ndarray(3,), mm`
  - `pupil_center: np.ndarray(3,), mm`
  - `theoretical_gaze: np.ndarray(3,), unit`
  - `quality: float (0~1)`
  - `frame_id: Optional[int]`
- 会话字段：
  - `session_id: str`
  - `stage: pre_fitting | collecting | calibrated`
  - `samples: List[CalibrationSample]`
  - `kappa_model: Optional[KappaModel]`
  - `last_fitting: Dict[Eye, FittingResultLite]`

样本字段来源：
- `target_pixel`：前端标定页面点击/指定的像素点
- `eye_center`：来自 `SESSION_MANAGER.last_fitting[eye].center`
- `pupil_center`：来自 `RECG_FIT_DATA_MANAGER.get_coordinate_point(eye, "pupil")` 的摄像头识别结果（毫米）
- `theoretical_gaze`：`compute_theoretical_gaze(eye_center, pupil_center)`
- `quality`：先取 `last_fitting.confidence`，可拓展融合更多质量因子

## 6. Kappa 模型与补偿视线（KappaModel / GazeVector）

- 模块：`project/fitting/app/state.py`, `project/fitting/app/api.py`, `project/fitting/core/kappa_calibrator_pro.py`
- `KappaModel`：
  - `axis: np.ndarray(3,), unit`
  - `angle_deg: float`
  - `samples_count: int`
  - `quality: float (0~1)`
- `GazeVector`：
  - `eye: "left"|"right"`
  - `origin: np.ndarray(3,), mm`
  - `direction: np.ndarray(3,), unit`
  - `target_pixel: Optional[(int,int)]`

API（`project/fitting/app/api.py`）：
- `start_session(session_id)`：创建/重置当前会话
- `push_sample(payload)`：融合前端像素点与当前帧三维数据，写入样本
- `fit_kappa(session_id, K)`：估计 Kappa，写入 `SESSION_MANAGER.kappa_model`
- `compute_gaze(eye, target_pixel)`：输出补偿后视线

## 7. 单位与坐标系

- 三维坐标：毫米（mm），右手坐标系
- 视线向量：单位向量
- 像素坐标：以屏幕分辨率为基准，左上角为原点

## 8. 可拓展性

- `CalibrationSample` 可扩：稳定性评分、眨眼标记、屏幕物理尺寸映射
- `KappaModel` 可扩：置信区间、时间衰减、版本
- `api.py` 可抽象为服务层（HTTP/WebSocket/队列）
- `push_sample` 已接入识别模块的瞳孔中心，后续可接入稳定滤波结果

## 9. 依赖关系

- 识别 → `RECG_FIT_DATA_MANAGER`：写入三维点（毫米）
- 拟合 → `SESSION_MANAGER.last_fitting`：写入轻量拟合结果
- 标定 → `SESSION_MANAGER.samples`：存放样本 → `fit_kappa()` 生成 `KappaModel`
- 推理 → `compute_gaze()`：输出补偿后视线


