# 单帧深度诊断与 SGBM 问题分析

本文档基于实际运行日志整理「单帧深度诊断」示例，并结合当前 SGBM 方法与参数做问题分析，便于排查深度 nan、中心无匹配、关键点采样失败等现象。

---

## 项目当前状态与背景条件

- **标定**：rectified 内参来自 `camera_params.json` 的 `stereo.P1` / `P2`，`stereo.image_size` 为 1280×1080。
- **几何关系**：rectified fx = P1[0,0] ≈ 475.5 px，基线 B ≈ -P2[0,3]/fx ≈ 52.1 mm，故 fB ≈ 24770 (px·mm)。视差与距离关系 **d ≈ fB/Z**：
  - Z=700 mm ⇒ d ≈ 35 px；Z=400 mm ⇒ d ≈ 62 px；Z=2000 mm ⇒ d ≈ 12 px。
  - **结论**：目标距离 0.4~0.7 m 对应视差约 35~62 px，**完全在 [0, 96) 内**，numDisparities=96 **不是** 视差范围不够的根因。
- **当前管线**：
  - **Resize 与 remap 顺序**：已按「方案 1」：split 得到 1920×1080 后**先 resize 到 1280×1080**，再 remap（maps/Q 均为 1280 系）。无需改顺序。
  - **Disparity 缩放**：**当前未对 SGBM 输出做 /16**。`StereoSGBM.compute()` 输出为 CV_16S，OpenCV 惯例为 Q16 定点（真实视差×16）。当前代码将 `disparity` 直接传入 `reprojectImageTo3D(disparity, Q)`，若 Q 按「像素视差」设计，则会导致深度尺度错误（约 16 倍偏近）、无效标记混乱（无效像素常为 -16 被误读为 0 等）。**这是 65% nan + 首次成功 Z≈0.12 m + min≈16 mm 的最可疑根因。**
- **SGBM 参数**：uniquenessRatio=10、speckleWindowSize=100、disp12MaxDiff=1 对人脸/弱纹理偏严，会放大 nan，但多为「放大器」而非唯一根因。

**为何「输入分辨率」与「标定 image_size」不同？**  
- **输入分辨率**（如 3840×1080）= 摄像头原始输出：左目+右目并排的一整帧。  
- **标定 image_size**（如 1280×1080）= 校正与匹配的**工作分辨率**：R1/R2/P1/P2/Q 与 remap 映射均按该尺寸计算。  
- 管线为：先按标定尺寸对左/右半幅 **resize**（3840 整帧 → 左 1920×1080 → resize 到 1280×1080），再 **remap**，再 SGBM。因此「3840×1080 输入」与「1280×1080 标定」不同是**预期行为**，标定正确即指在该工作分辨率下内参/极线正确。

---

## 一、单帧深度诊断示例（来自 gaze_20260302.log）

**时间戳**：2026-03-02 12:21:55（人脸检测成功后的第一帧）

### 1. 显示与图像

| 项 | 值 |
|----|-----|
| 图像尺寸 | h=1080, w=1280（左目校正图，与 stereo image_size 一致） |
| 显示尺寸 | 1280×1080，宽高比 1.185 |

### 2. 立体输出（DataManager 内 raw Z）

| 指标 | 值 | 说明 |
|------|-----|------|
| raw Z 范围 | min=16.316, max=2755.529, mean=276.963 (mm) | 单位由 Q 矩阵决定，本工程为 mm |
| nan 占比 | 65.5% | 约 2/3 像素无有效立体匹配 |
| 中心 Z | nan | 图像中心 (640, 540) 处无有效深度 |
| 中心 disparity | 0.0 | 中心点视差为 0 或无效（SGBM 无效时常为 minDisparity-1 或 0） |

### 3. 单位转换（Detector to_meters）

| 阶段 | 单位 | min | max | mean |
|------|------|-----|-----|------|
| to_meters 前 | camera_unit (mm) | 16.316 | 2755.529 | 276.963 |
| to_meters 后 | m | 0.0163 | 2.7555 | 0.2770 |

- depth_scale = 0.001（mm→m），转换正确。
- 若人脸实际约 0.4 m，则 camera_unit 期望约 400；当前 mean≈277 说明有效区域中近距离/噪声占比不小。

### 4. 关键点采样深度（coordinate_converter）

| 关键点 | 像素坐标 (x,y) | 采样深度 (m) | 有效 |
|--------|----------------|-------------|------|
| 0  (鼻根等) | (690, 675) | **nan** | False |
| 468 (左眼) | (680, 621) | **0.1964** | True |
| 473 (右眼) | (720, 619) | **nan** | False |

- 深度图与图像同尺寸 1080×1280，逐像素对应。
- 同一帧内：468 有深度，0 与 473 为 nan → 说明是**逐像素**立体匹配失败，不是整体单位或对齐错误。

### 5. 深度图统计（转米后）

- min=0.0163 m, max=2.7555 m, mean=0.2770 m, **nan 占比=65.5%**。

### 6. 首次成功 3D 与 nan 追溯

- **首次成功 pixel_to_3d** 得到 Z(米)=**0.1200**（期望约 0.4 m）→ 第一个通过深度阈值的关键点落在较近或噪声区域。
- 日志中 **深度追溯** 示例：  
  `[深度追溯] 关键点 473 像素(730,618) 深度=nan: 立体匹配在该像素无有效视差(或深度图与图像未对齐)，可检查标定/光照/遮挡及 image 与 depth 是否同源同尺寸`  
  → 确认 nan 来源于该像素在立体深度图中即为无效（无有效视差或 reproject 后 Z≤0/非有限）。

---

## 二、当前 SGBM 方法与参数（stereo_config.py + data_manager）

### 2.1 流程简述

1. **输入**：双目并排 3840×1080 → 左/右各 1920×1080，再按标定 `image_size` resize 到 1280×1080（若需）。
2. **校正**：`initUndistortRectifyMap` 得到左右 remap，`remap` 得到 `rect_left`、`rect_right`。
3. **灰度**：`rect_left` / `rect_right` 转灰度。
4. **视差**：`cv2.StereoSGBM_create(...).compute(gray_l, gray_r)` → `disparity`（16 位，无效为负或 0）。
5. **3D**：`cv2.reprojectImageTo3D(disparity, Q)` → 取 `points_3d[:,:,2]` 为 Z(mm)；非有限或 Z≤0 置为 `np.nan`。

### 2.2 当前 SGBM 参数

| 参数 | 当前值 | 含义与影响 |
|------|--------|------------|
| **minDisparity** | 0 | 最小视差；若真实视差为负（标定/极线问题）会全无效。 |
| **numDisparities** | 16×6 = **96** | 视差搜索范围 [0, 96)；**必须为 16 的整数倍**。越大可测越远，但计算量增大、易噪。 |
| **blockSize** | 5 | 匹配块边长（奇数）。小→细节好、噪声大；大→平滑、边缘差。 |
| **P1** | 8×3×5² = 600 | 视差平滑项一阶惩罚。 |
| **P2** | 32×3×5² = 2400 | 视差平滑项二阶惩罚。P2 大→视差更平滑、弱纹理更易错。 |
| **disp12MaxDiff** | 1 | 左右一致性检查允许差异；过严会多无效。 |
| **uniquenessRatio** | 10 | 唯一性约束；大→更少误匹配、更多无匹配。 |
| **speckleWindowSize** | 100 | 斑点滤波窗口；大→大块无效区。 |
| **speckleRange** | 32 | 斑点内视差差阈值。 |
| **preFilterCap** | 63 | x-derivative 预滤波截断。 |
| **mode** | STEREO_SGBM_MODE_SGBM_3WAY | 3-way 优化。 |

### 2.3 无效深度来源（与日志对应）

- **disparity 无效**：SGBM 未找到可靠匹配时，OpenCV 通常输出 `minDisparity - 1`（当前即 -1）或 0。
- **reprojectImageTo3D**：视差无效或超出有效范围时，Z 会为 inf、极大或 ≤0。
- **代码**：`depth_mm[~np.isfinite(depth_mm) | (depth_mm <= 0)] = np.nan` → 这些点变为 nan。
- 日志中「中心 disparity=0 或 -16、中心Z=nan」与上述行为一致：**该像素 SGBM 未给出有效视差**。

---

## 关键代码（可直接作为 ChatGPT 询问依据）

以下为与单帧深度诊断、SGBM 立体深度、单位转换、关键点采样相关的核心代码，便于复制到对话中作为上下文。

### 1. SGBM 参数配置 `project/config/stereo_config.py`

```python
# StereoSGBM 立体匹配参数（numDisparities 须为 16 的整数倍）
SGBM_MIN_DISPARITY = 0
SGBM_NUM_DISPARITIES = 16 * 6  # 96
SGBM_BLOCK_SIZE = 5
SGBM_P1 = 8 * 3 * 5 ** 2
SGBM_P2 = 32 * 3 * 5 ** 2
SGBM_DISP12_MAX_DIFF = 1
SGBM_UNIQUENESS_RATIO = 10
SGBM_SPECKLE_WINDOW_SIZE = 100
SGBM_SPECKLE_RANGE = 32
SGBM_PREFILTER_CAP = 63
```

### 2. 立体深度流程与无效深度处理 `project/data/data_manager.py`

**创建 SGBM 与校正映射（_ensure_stereo_processor）：**

```python
self._stereo_sgbm = cv2.StereoSGBM_create(
    minDisparity=SGBM_MIN_DISPARITY,
    numDisparities=SGBM_NUM_DISPARITIES,
    blockSize=SGBM_BLOCK_SIZE,
    P1=SGBM_P1,
    P2=SGBM_P2,
    disp12MaxDiff=SGBM_DISP12_MAX_DIFF,
    uniquenessRatio=SGBM_UNIQUENESS_RATIO,
    speckleWindowSize=SGBM_SPECKLE_WINDOW_SIZE,
    speckleRange=SGBM_SPECKLE_RANGE,
    preFilterCap=SGBM_PREFILTER_CAP,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
)
self._stereo_Q = Q
self._stereo_image_size = image_size  # 如 (1280, 1080)
```

**计算立体深度与 nan 处理（_compute_stereo_depth 核心）：**

```python
# 顺序：split → 若尺寸≠标定则 resize 到 (w_cal, h_cal) → remap
half = w // 2
left_bgr = frame_bgr[:, :half]
right_bgr = frame_bgr[:, half:]
if left_bgr.shape[1] != w_cal or left_bgr.shape[0] != h_cal:
    left_bgr = cv2.resize(left_bgr, (w_cal, h_cal), interpolation=cv2.INTER_LINEAR)
# right 同理
rect_left = cv2.remap(left_bgr, self._stereo_map1x, self._stereo_map1y, cv2.INTER_LINEAR)
rect_right = cv2.remap(right_bgr, self._stereo_map2x, self._stereo_map2y, cv2.INTER_LINEAR)
gray_l = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
gray_r = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)
disparity = self._stereo_sgbm.compute(gray_l, gray_r)  # CV_16S, Q16 定点
# 【当前未做 /16】若 Q 按像素视差设计，应改为 disp_float = disparity.astype(np.float32)/16.0 再 reproject
points_3d = cv2.reprojectImageTo3D(disparity, self._stereo_Q)
depth_mm = points_3d[:, :, 2].astype(np.float32)
invalid = np.logical_or(~np.isfinite(depth_mm), depth_mm <= 0)
depth_mm[invalid] = np.nan
# 日志：[深度调试] 立体输出 raw Z min/max/mean, nan占比, 中心Z, 中心disparity
```

### 3. 深度单位转换 `project/data/data_models.py` + Detector

**DepthMap.to_meters：**

```python
def to_meters(self, depth_scale: float = 0.001) -> "DepthMap":
    if self.unit == "meter":
        return DepthMap(data=self.data, unit="meter")
    elif self.unit == "camera_unit":
        return DepthMap(data=self.data * depth_scale, unit="meter")  # mm→m 时 depth_scale=0.001
    else:
        raise ValueError("unit必须是camera_unit或meter")
```

**Detector 内对深度图做 to_meters（存为米制供下游）：**

```python
depth_scale = self.camera_params.get("depth_scale", 0.001)
self._depth_map = depth_map.to_meters(depth_scale)  # 下游 coordinate_converter 使用 self._depth_map，单位米
```

### 4. 关键点深度采样与深度追溯 `project/core/recognition/coordinate_converter.py`

**单点插值取深度：**

```python
def get_depth_interpolated(depth_map: DepthMap, pixel: Point2D) -> float:
    x, y = pixel.x, pixel.y
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    map_x = np.array([[x]], dtype=np.float32)
    map_y = np.array([[y]], dtype=np.float32)
    interpolated = cv2.remap(depth_map.data, map_x, map_y,
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    return float(interpolated[0, 0])
```

**pixel_to_3d 中深度校验（无效则抛异常）：**

```python
DEPTH_VALIDATION_THRESHOLD = 0.05  # 米，低于此视为无效
depth_meters = get_depth_interpolated(depth_map, pixel)
if not np.isfinite(depth_meters) or depth_meters <= DEPTH_VALIDATION_THRESHOLD:
    raise ValueError(f"无效的深度值：{depth_meters}，必须大于{DEPTH_VALIDATION_THRESHOLD}米")
# 转换：X = (x - cx)*Z/fx, Y = (y - cy)*Z/fy, Z = depth_meters
```

**batch_convert_landmarks 中关键点 0/468/473 采样与深度追溯（节选）：**

```python
# 每前几帧或每 60 帧打一次 [深度调试] / [深度追溯]
for idx in [0, 468, 473]:
    if idx < len(landmarks):
        lm = landmarks[idx]
        pixel = lm.to_point2d()
        raw_z = get_depth_interpolated(depth_map, pixel)
        valid = np.isfinite(raw_z) and raw_z > DEPTH_VALIDATION_THRESHOLD
        logger.info("[深度调试] 关键点 %d 像素(%.0f,%.0f) 采样深度=%.4f m 有效=%s ...", idx, pixel.x, pixel.y, raw_z, valid, ...)
        if not np.isfinite(raw_z) or np.isnan(raw_z):
            logger.info(
                "[深度追溯] 关键点 %d 像素(%.0f,%.0f) 深度=nan: 立体匹配在该像素无有效视差(或深度图与图像未对齐)，可检查标定/光照/遮挡及 image 与 depth 是否同源同尺寸",
                idx, pixel.x, pixel.y,
            )
# 深度图统计
logger.info("[深度调试] 深度图统计: min=%.4f max=%.4f mean=%.4f m nan占比=%.1f%%", ...)
# 首次成功 pixel_to_3d 时
logger.info("[深度调试] coordinate_converter 首次成功 pixel_to_3d 得到 Z(米)=%.4f  (期望约0.4)", point_3d.z)
```

---

## 三、问题分析（结合单帧诊断与 SGBM）

### 3.1 高 nan 占比（约 65%）— 根因排序与修正结论

- **现象**：单帧深度图 nan 占比 63–67%，中心经常无有效 Z，首次成功 Z≈0.12 m、min≈16 mm。
- **根因排序**（按概率）：  
  1. **Disparity 缩放未处理（最高优先）**：SGBM 输出 CV_16S、Q16 定点，当前直接用 raw 送入 `reprojectImageTo3D`，导致深度约 16 倍偏近、无效像素 -16 被误读、大量不合理近距离值。  
  2. **Resize/remap 顺序**：本项目已为「先 resize 到 1280×1080 再 remap」，无需改。  
  3. **参数过严**：uniquenessRatio=10、speckleWindowSize=100、disp12MaxDiff=1 会放大 nan，属放大器，通常非唯一根因。  
- **纠正**：**numDisparities=96 对 0.4~0.7 m 并不偏小**（见上文 fB/Z：d 约 35~62 px，在 [0,96) 内），不必优先增大。

**一锤定音（disparity 是否需 /16）**：在立体深度计算处打印：
- `disparity.dtype`、`disparity.min()`、`disparity.max()`、`disparity[cy,cx]`；
- `disp_f = disparity.astype(np.float32)/16.0` 的 min/max/center（仅统计有效，如 disp_f>0）；
- 用 `disp_f` 做 reproject 后的 Z 均值/中值是否接近 0.4~0.7 m。  
若经常看到 `disparity.min()==-16`、有效区 `disp_f` 约 30~70，且除以 16 后深度接近工作距离，则**必须**在 reproject 前做 `/16.0`。

**修复顺序建议**：  
1. 确认并修正 disparity 缩放（reproject 前 `disp_float = disparity.astype(np.float32)/16.0`，用 `disp_float` 调 `reprojectImageTo3D`）。  
2. 保持当前 resize→remap 顺序。  
3. 用「诊断参数」把 nan 压下来：uniquenessRatio→5（或 0~3 仅诊断）、speckleWindowSize→0 或 50、disp12MaxDiff→5~10；可选 blockSize 7 或 9 并同步 P1/P2。  
4. 关键点深度从单像素插值升级为**局部 ROI 中值**（如 5×5/7×7 内 finite 深度取 median），减少 468/473 偶发 nan 导致整帧失败。

### 3.2 关键点处深度为 nan（如 0、473）

- **现象**：关键点 0 (690,675)、473 (720,619) 采样深度 nan；468 (680,621) 有深度 0.1964 m。  
- **结论**：人脸区域部分像素有有效视差、部分无，与「高 nan 占比」一致；**深度图与图像对齐正常**（同源、同尺寸），nan 来自立体匹配本身。  
- **深度追溯** 已指明：该像素立体匹配无有效视差或 reproject 后无效。

### 3.3 首次成功 Z=0.12 m，期望约 0.4 m

- **现象**：coordinate_converter 首次成功得到的 Z=0.12 m，小于期望 0.4 m。  
- **可能原因**：  
  1. **Disparity 未 /16**：若 raw 视差被当成像素视差 reproject，Z 会约 16 倍偏近（0.4 m→0.025 m 量级），与 0.12 m、min 16 mm 等一致，优先排查。  
  2. 第一个通过深度阈值的关键点恰好在近处或噪声；人脸主体处多为 nan，「首次成功」落在边缘或噪声区。  
- 与 mean=0.277 m、nan 65% 一致：有效点中近处/噪声拉低均值。

### 3.4 中心 disparity=0 或 -16、中心 Z=nan

- **现象**：图像中心经常 disparity=0 或 **-16**（SGBM 无效像素常为 minDisparity-1 的 Q16 值，即 -1×16=-16）、Z=nan。  
- **含义**：中心点 SGBM 未给出有效视差；若日志里把 -16 显示成 0，是显示/类型解读问题，根因仍是该点无效。  
- 修正 disparity 缩放后，无效点仍应用「≤0 或非有限」判为 nan，不改变无效判定逻辑。

---

## 四、参数与代码位置速查

| 项目 | 位置 |
|------|------|
| SGBM 参数 | `project/config/stereo_config.py` |
| 立体流程与 nan 处理 | `project/data/data_manager.py` → `_compute_stereo_depth`、`_ensure_stereo_processor` |
| 深度单位转换 | `project/data/data_manager.py`（存 camera_unit）、`detector` 中 to_meters、`depth_data_flow.md` |
| 关键点采样与深度追溯 | `project/core/recognition/coordinate_converter.py` |
| 深度图统计/单帧调试日志 | `data_manager`（立体输出）、`detector`（to_meters）、`coordinate_converter`（关键点 0/468/473、深度图统计、深度追溯） |

---

## 五、单帧诊断检查清单（复现问题时可用）

1. **Disparity 一锤定音**：在立体计算处打印 `disparity.dtype`、`min/max/center`，以及 `disp_f=disparity/16` 的有效区 min/max/center；用 disp_f 做 reproject 看 Z 是否接近 0.4~0.7 m。若 center 常为 -16、有效 disp_f 约 30~70 → 必须 reproject 前 `/16`。  
2. **[深度调试] 立体输出 raw Z**：看 nan 占比、中心 Z、中心 disparity（-16 表示无效）。  
3. **[深度调试] to_meters 前/后**：确认 camera_unit 与米制范围、depth_scale。  
4. **[深度调试] 关键点 0/468/473**：看采样深度是否 nan、像素坐标是否在脸内。  
5. **[深度追溯]**：对 nan 关键点确认「无有效视差」或「图像与深度未对齐」类提示。  
6. 使用 **有限帧深度测试脚本**（见下）可快速打出 disparity 与 /16 对比，无需跑完整 gaze 流程。

**有限帧深度测试脚本**：`python -m project.data.depth_test_limited_frames [--frames N] [--video PATH]`，仅处理 N 帧立体深度并输出 disparity 诊断与 raw/div16 深度对比，便于一锤定音。
