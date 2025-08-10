# 拟合模块（fitting）结构与功能说明

> 目标：对识别模块输出的三维点（瞳孔中心、虹膜边界点）进行鲁棒球面拟合得到眼球球心；据此计算理论视线向量，并通过个体标定估计 Kappa 角，输出补偿后的最终视线向量；提供桌面应用（App）进行 Kappa 标定与视线可视化。

## 一、模块概述

拟合模块的目标是：基于 `recognition` 模块输出的三维关键点，鲁棒估计眼球几何（球心、半径），计算理论视线向量，并通过个体标定得到 Kappa 角以补偿输出最终视线。同时提供桌面应用，指导用户注视定点完成 Kappa 标定，并在界面上实时显示视线与电脑屏幕示意。

- 不包含质量校验（按要求暂不实现）
- `config/settings.py` 提供“电脑屏幕在摄像机坐标系下的三维坐标”

---

## 二、文件结构与调用关系

```
fitting/
├── __init__.py
├── main.py                       # 可选，跑通模块级流水线
├── core/
│   ├── __init__.py
│   ├── ransac_sphere_fitter.py   # RANSAC 球面拟合
│   ├── gaze_estimator.py         # 理论视线与Kappa补偿
│   └── kappa_calibrator.py       # 个体Kappa标定
├── utils/
│   ├── __init__.py
│   └── geometry.py               # 向量/角度/旋转等几何基础
├── config/
│   ├── __init__.py
│   ├── settings.py               # 屏幕在摄像机坐标系下的三维坐标（关键）
│   └── constants.py              # 常量（最小集）
└── app/
    ├── __init__.py
    ├── calibration_ui.py         # 桌面应用UI（Tkinter）
    └── run.py                    # 启动入口
```

- 上游：`recognition` 输出三维点
- `core/ransac_sphere_fitter.py`：拟合球心、半径
- `core/gaze_estimator.py`：理论视线 → Kappa 补偿 → 最终视线
- `core/kappa_calibrator.py`：根据标定样本估计 Kappa（角度与轴）
- `app/`：桌面校准与显示（非网页）
- `config/settings.py`：屏幕三维坐标（相机坐标系）

---

## 三、各文件功能

### 3.1 核心（core/）
- `ransac_sphere_fitter.py`：
  - 功能：对虹膜边界三维点进行鲁棒球面拟合，眼睛周围三维点作为限制，得到眼球球心 `C_eye=[c_x,c_y,c_z]` 与半径 `r`
  - 接口（占位）：`fit_eye_sphere(points_3d, threshold, max_trials) -> (center_xyz, radius, inlier_mask)`；`refit_with_inliers(points_3d, inlier_mask) -> (center_xyz, radius)`
- `gaze_estimator.py`：
  - 功能：计算理论视线与补偿后的最终视线，算出补偿后的视线与屏幕的焦点。
  - 接口（占位）：`compute_theoretical_gaze(c_eye, c_pupil)`；`apply_kappa_compensation(gaze_vec, kappa_axis, kappa_angle)`；`estimate_final_gaze(c_eye, c_pupil, kappa_params)`
- `kappa_calibrator.py`：
  - 功能：基于多帧“已知注视点”样本，估计个体 Kappa（角度与旋转轴）
  - 接口（占位）：`estimate_kappa(samples) -> (kappa_axis, kappa_angle, summary)`；`pack_kappa_params(kappa_axis, kappa_angle) -> dict`

### 3.2 工具（utils/）
- `geometry.py`：
  - 向量归一化、夹角、轴角旋转（Rodrigues/Rotation）等基础几何（占位）

### 3.3 配置（config/）
- `settings.py`（必须）
  - `SCREEN_3D_CORNERS_IN_CAMERA`：电脑屏幕四角在摄像机坐标系下的三维坐标（米），键：
    - `TOP_LEFT`, `TOP_RIGHT`, `BOTTOM_LEFT`, `BOTTOM_RIGHT` → `[x, y, z]`
  - 可选：`SCREEN_CENTER`, `SCREEN_NORMAL`
- `constants.py`（最小集）
  - 状态码、默认键名

### 3.4 应用（app/）
- `calibration_ui.py`：桌面应用（Tkinter）
  - 引导用户按顺序注视定点（典型 5~9 点）
  - 采集多帧样本（`C_eye`、`C_pupil`、目标点三维坐标）
  - 调用 `core.kappa_calibrator` 估计 Kappa
  - 实时显示：当前视线向量、与屏幕的交点示意
- `run.py`：启动入口，提供 `main()` 启动 UI

---

## 四、接口规范（最小）

- 输入：
  - `iris_boundary_points_3d`：形如 `(N,3)` 的三维点
  - `pupil_center_3d`：形如 `(3,)` 的三维点
  - 标定：由 App 给出屏幕定点顺序，按 `SCREEN_3D_CORNERS_IN_CAMERA` 映射到三维
- 输出：
  - `eye_center_3d`：眼球球心
  - `gaze_vector_final`：最终视线单位向量
  - `kappa_params`：`{axis, angle}` 或 `{R}`

---

## 五、App 标定与显示

- 标定流程：
  - 界面按序显示屏幕定点，提示用户注视
  - 每点采集若干帧：`C_eye`、`C_pupil` 与“目标点三维坐标”
  - 估计 `kappa_axis` 与 `kappa_angle`，保存为 `kappa_params`
- 实时显示：
  - 每帧计算 gaze（含 Kappa 补偿），在界面画出视线与屏幕交点

---

## 六、性能与精度（最小）
- 实时性：单帧拟合与估计 < 10 ms（标定除外）
- 精度：补偿后角度误差中位数目标 < 1.5°

---

## 七、测试（最小）
- 几何函数正确性
- RANSAC 在含离群点下的鲁棒性（合成数据）
- Kappa 标定在已知旋转下的可还原性


