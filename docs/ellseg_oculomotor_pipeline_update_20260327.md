# EllSeg 视动分析增强链路改造说明

日期：2026-03-27  
项目：`gaze`  
主题：基于 EllSeg 的眼部 ROI、质量控制、局部跟踪与视动分析链路增强

## 1. 文档目的

本文档用于记录本次围绕眼部视动分析前端所完成的系统改造，包括：

- 改动内容
- 改动原因
- 当前效果
- 当前版本适用范围
- 仍然存在的问题
- 后续建议工作

本文档对应的目标场景为：

- 不改变现有硬件条件
- 眼睛距离摄像头约 `0.5m - 0.7m`
- 基于单目眼部 ROI 做相对视动分析

## 2. 初始问题

在本次改造开始前，系统主要存在以下问题：

### 2.1 ROI 过小，眼部有效像素不足

早期单眼 ROI 较小且偏扁，导致：

- 瞳孔仅为十来个像素量级
- 虹膜也只有二十来个像素量级
- pupil / iris 容易贴边
- 头动、反光、遮挡会显著放大抖动

### 2.2 EllSeg 单帧输出不够稳定

主要表现为：

- `pupil_center` 抖动较大
- `iris ellipse` 有明显跳动
- 部分帧存在向眼角吸附
- mask 已经异常时，仍可能输出看似有效的几何结果

### 2.3 显示链与分析链混在一起

原始输出、显示平滑、分析信号之间没有清晰分离，容易出现：

- 为了让显示更稳而引入 `fallback` / `EMA`
- 但这些处理会污染真正用于视动分析的信号

### 2.4 几何拟合前缺少足够质量控制

系统缺少对以下问题的系统拦截：

- mask 主体不稳定
- pupil / iris 尺寸比例不合理
- pupil 与 iris 几何关系不合理

## 3. 本次改造总体思路

本次改造的核心思想不是继续“裸用 EllSeg”，而是构建一条增强链路：

```text
原始图像
    ->
更大的稳定 ROI
    ->
EllSeg 原始输出
    ->
mask 质量分析
    ->
几何合理性检查
    ->
pupil center 融合（network + mask + tracking）
    ->
display / analysis 分流
```

目标是让 EllSeg 从“单帧直接输出器”变成“受控的前端观测模块”。

## 4. 已完成的改动

## 4.1 ROI 放大与稳定

### 做了什么

- 重新评估并调整了单眼 ROI 的大小与位置
- 增大了眼部在 ROI 中的有效像素占比
- 避免 pupil / iris 贴边

### 为什么这么做

在当前硬件不变的条件下，最有效的提升方式就是增加眼部在图像中的像素密度。  
如果 pupil / iris 过小，后续一切分割、椭圆拟合、中心估计都会非常敏感。

### 结果

最新阶段中，眼部量级已提升为：

- 瞳孔：
  - `pupil_major ≈ 18.16 px`
  - `pupil_minor ≈ 14.09 px`
- 虹膜：
  - `iris_major ≈ 32.56 px`
  - `iris_minor ≈ 24.91 px`

相比早期数据：

- 瞳孔从约 `10 - 15 px` 提升到 `14 - 18 px`
- 虹膜从约 `19 - 27 px` 提升到 `25 - 33 px`

这意味着当前版本在 ROI 有效像素密度上明显优于初始版本。

## 4.2 增强 debug / stability 工具

### 主要文件

- [`project/tools/ellseg_observation_stability.py`](/Users/andrew/Downloads/gaze/project/tools/ellseg_observation_stability.py)

### 做了什么

为每帧输出更完整的记录与统计：

- raw 几何
- mask 质量相关字段
- geometry 合法性字段
- display / analysis 两条链路的字段
- fallback / baseline / jump 相关字段

并生成：

- `records.jsonl`
- `summary.json`
- `roi_vis`
- `mask_color`
- `mask_component_debug`

### 为什么这么做

原先问题更多依赖人工观察，缺少可量化对比。  
增加这些字段后，可以把“感觉抖”转换成：

- 中心抖动标准差
- 面积统计
- 边界余量
- 几何拒绝原因
- 分析链通过率

## 4.3 增加 mask 质量分析

### 主要文件

- [`project/core/recognition/roi/observation.py`](/Users/andrew/Downloads/gaze/project/core/recognition/roi/observation.py)
- [`project/core/recognition/roi/mask_metrics.py`](/Users/andrew/Downloads/gaze/project/core/recognition/roi/mask_metrics.py)

### 做了什么

新增或补全了以下信息：

- `pupil_mask_area`
- `iris_mask_area`
- 主连通域面积
- 连通域数量
- `mask centroid`
- 边界余量
- clean/raw 面积比例

### 为什么这么做

EllSeg 能否给出一个椭圆，并不代表该帧足够可信。  
mask 质量分析用于判断：

- pupil 是否真的被分出来
- iris 是否被边界裁切
- 当前 frame 是否值得进入后续几何拟合或分析链

## 4.4 增加几何合理性门控

### 主要文件

- [`project/core/recognition/roi/geometry.py`](/Users/andrew/Downloads/gaze/project/core/recognition/roi/geometry.py)
- [`project/core/recognition/roi/constants.py`](/Users/andrew/Downloads/gaze/project/core/recognition/roi/constants.py)
- [`project/core/recognition/landmark_extractor.py`](/Users/andrew/Downloads/gaze/project/core/recognition/landmark_extractor.py)

### 做了什么

新增几何合理性约束，例如：

- pupil 是否在 iris 内
- pupil / iris 的中心关系
- pupil / iris 尺寸比例范围

### 为什么这么做

目的是拦截“看起来有结果，但其实不合理”的几何观测，避免它们继续进入三角化、眼球拟合或分析链。

## 4.5 修正 3D 眼球拟合初始化

### 主要文件

- [`project/core/fitting/fitting_strategy.py`](/Users/andrew/Downloads/gaze/project/core/fitting/fitting_strategy.py)

### 做了什么

- 修正了圆心初始化算法
- 去掉了随机抽样带来的不稳定性

### 为什么这么做

如果 2D 观测本身已有少量噪声，3D 初始化再带随机性，会进一步放大不稳定。

## 4.6 拆分 display signal 与 analysis signal

### 主要文件

- [`project/tools/ellseg_observation_stability.py`](/Users/andrew/Downloads/gaze/project/tools/ellseg_observation_stability.py)

### 做了什么

将输出明确拆为两条链路：

#### Display signal

特征：

- 可使用 `fallback`
- 可使用 `EMA`
- 重点是可视化更稳定

#### Analysis signal

特征：

- 不做强补偿
- 坏帧直接记为 `invalid`
- 输出归一化的 `analysis_nx / analysis_ny`
- 重点是保留真实视动信息

### 为什么这么做

显示稳定和分析可信是两个不同目标。  
如果把两者混在一起，分析结果容易被平滑或补帧污染。

## 4.7 新增 local pupil tracking 与中心融合

### 主要文件

- [`project/core/recognition/pupil_tracking.py`](/Users/andrew/Downloads/gaze/project/core/recognition/pupil_tracking.py)
- [`project/core/recognition/landmark_extractor.py`](/Users/andrew/Downloads/gaze/project/core/recognition/landmark_extractor.py)
- [`project/tools/ellseg_observation_stability.py`](/Users/andrew/Downloads/gaze/project/tools/ellseg_observation_stability.py)

### 做了什么

新增轻量局部跟踪器：

- 以前一帧 pupil 中心为参考
- 在当前 ROI 内做小范围模板匹配
- 结合以下三类候选中心做融合：
  - EllSeg network center
  - mask centroid
  - local tracking center

新增字段包括：

- `pupil_center_raw_roi`
- `pupil_center_tracked_roi`
- `pupil_center_fused_roi`
- `pupil_center_source`

### 为什么这么做

单纯依赖单帧 EllSeg 的 `pupil_center` 在动态场景下仍有抖动。  
局部 tracking 的作用是提供更连续的时序约束。

## 4.8 收紧 tracker 模板刷新条件

### 主要文件

- [`project/core/recognition/pupil_tracking.py`](/Users/andrew/Downloads/gaze/project/core/recognition/pupil_tracking.py)

### 做了什么

只有高质量的 tracking 结果才允许更新内部模板，例如：

- `track_mask`
- `track_network`
- 或高分 `track_only`

而：

- `mask_only`
- `network_only`

不用于模板刷新。

### 为什么这么做

避免偶发坏帧污染 tracker 模板，从而导致后续连续漂移。

## 4.9 优化 mask / network 融合策略

### 主要文件

- [`project/core/recognition/pupil_tracking.py`](/Users/andrew/Downloads/gaze/project/core/recognition/pupil_tracking.py)

### 做了什么

当当前帧没有可靠 template tracking 命中，而只能在 `mask` 和 `network` 之间选择时：

- 不再固定加权平均
- 优先选取“更接近上一帧中心”的那一路

### 为什么这么做

这样可以减少 `mask_network` 分支带来的小幅来回抖动。

## 4.10 对 display 用的椭圆几何做平滑

### 主要文件

- [`project/tools/ellseg_observation_stability.py`](/Users/andrew/Downloads/gaze/project/tools/ellseg_observation_stability.py)

### 做了什么

除了中心外，还对 display 层的：

- pupil / iris 长轴
- pupil / iris 短轴
- pupil / iris 角度

增加了 EMA 平滑。

### 为什么这么做

这一步不改变 analysis signal，  
但能显著减少可视化中“中间圈”和“外层圈”的呼吸感和抖动感。

## 5. 当前增强链路结构

```text
原始图像
    ->
大 ROI / 稳定 ROI
    ->
EllSeg 原始输出
    - segmentation_mask
    - pupil_center
    - pupil_ellipse
    - iris_ellipse
    ->
mask 质量分析
    ->
几何合理性检查
    ->
pupil center 融合
    - network center
    - mask centroid
    - local tracking center
    ->
信号分流
    - display signal
    - analysis signal
```

## 6. 当前效果

基于最新测试结果：

- [`project/debug/ellseg_stability/ellseg_stability_left_20260327_175019/summary.json`](/Users/andrew/Downloads/gaze/project/debug/ellseg_stability/ellseg_stability_left_20260327_175019/summary.json)

当前主要指标为：

- `accepted_frames = 198`
- `fallback_frames = 2`
- `frames_analysis_accept = 197 / 200`
- `analysis_accept_rate = 0.985`
- `raw_frames_dist_gt_4 = 2`
- `frames_mask_bad = 2`
- `frames_geometry_bad = 1`

主要分析稳定性指标：

- `pupil_minus_iris_dist std ≈ 0.398 px`
- `analysis_nx std ≈ 0.042`
- `analysis_ny std ≈ 0.033`

这表明：

- 大部分帧已经可以稳定进入分析链
- 系统对相对视动信号的估计已经较稳定
- 极少需要 fallback

## 7. 是否存在“越到后面越抖”的问题

对最新一组数据前后半段做对比后，未发现明显时间劣化。

前 100 帧：

- `output_dist std ≈ 0.402`
- `analysis_nx std ≈ 0.044`
- `analysis_ny std ≈ 0.033`

后 100 帧：

- `output_dist std ≈ 0.392`
- `analysis_nx std ≈ 0.038`
- `analysis_ny std ≈ 0.034`

结论：

- 目前没有证据表明 tracking 会随着时间推移逐渐发散
- 也没有出现明显“后面比前面更抖”的问题

## 8. 当前仍面临的问题

尽管整体已经明显改善，但仍需明确以下事实：

### 8.1 display 观感仍可能有少量“圈抖动”

虽然当前分析链已经较稳，但从可视化感受上看：

- 虹膜圈和瞳孔圈仍可能存在轻微呼吸感
- 这主要来自椭圆轴长与角度的逐帧变化

这更偏向显示层观感问题，而不是分析链路失效。

### 8.2 当前更适合相对视动分析，而非专业眼动仪级别任务

当前版本适合：

- 相对视动分析
- 方向变化分析
- 趋势性注视 / 扫视研究

当前版本不应直接等同于：

- 专业红外眼动仪
- 医疗级绝对 gaze 系统
- 极高精度单帧几何拟合器

### 8.3 最稳定的是中心关系，不是所有单帧椭圆参数

当前最可信的量包括：

- `pupil center`
- `iris center`
- `analysis_nx / analysis_ny`

相对不适合作为高可信单帧真值的量包括：

- 单帧 `pupil major/minor`
- 单帧 `iris angle`

## 9. 当前版本是否满足项目要求

如果当前要求是：

- 不改硬件
- 在 `0.5m - 0.7m` 范围内
- 做相对视动追踪 / 视动分析

则当前版本已经基本满足要求。

理由：

- 眼部像素量级明显优于初始版本
- 分析链通过率很高
- 原始异常帧占比很低
- 没有明显时间漂移
- 中心关系信号已经足够稳定

如果未来目标升级为：

- 更高精度绝对 gaze
- 高精度生理测量
- 更接近专业眼动仪的稳健性

则当前版本仍然不是最终答案。

## 10. 后续建议

## 10.1 当前阶段建议冻结主参数

不建议继续大幅调整：

- ROI 大小
- analysis gate
- jump / fallback 逻辑
- major/minor ratio 阈值

原因是当前版本已经进入“够用且稳定”的状态。

## 10.2 建议进入验证阶段，而不是继续大修

推荐后续做以下验证测试：

1. 静止注视测试
2. 小幅水平眼动测试
3. 小幅垂直眼动测试
4. 大幅扫视测试
5. 轻微头动测试

目标是确认当前增强链路在不同使用场景下的稳定边界。

## 10.3 若还想优化，只建议做显示层小修

如果后续仍希望提升“圈看起来更稳”的观感，建议仅继续优化：

- display 椭圆参数的平滑策略

不建议继续修改 analysis 主链。

## 11. 总结

本次改造完成后，系统已经从“直接依赖 EllSeg 单帧输出、容易抖动和误判”的状态，提升为：

- ROI 更合理
- mask 与 geometry 有质量控制
- pupil center 有融合与 tracking 支持
- display 与 analysis 明确分流
- analysis 信号可用于相对视动分析

本次最重要的结果不是单一参数优化，而是建立了一条完整的增强链路：

```text
ROI + 质控 + 融合 + 分流
```

这条链路使得当前系统在不改硬件的前提下，已经可以作为相对视动分析的稳定前端使用。
