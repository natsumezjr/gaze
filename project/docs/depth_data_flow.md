# 深度值数据流与单位约定

## 数据流图（单位与转换）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. 摄像头 / 立体匹配                                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  • RGB-D 双目 (3840×1080) → 左半帧 → 校正 → rect_left (1280×1080)                    │
│  • cv2.reprojectImageTo3D(disparity, Q) → points_3d[:,:,2] = Z                  │
│  • 单位：与标定 Q 一致，本工程为 毫米 (mm) → 变量名 depth_mm                           │
│  • 无效/非正 → 置为 np.nan                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. DataManager.add_frame / get_depth                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  • 存储：frame_data["depth_map"] = depth_mm (np.float32, H×W)                    │
│  • 返回：DepthMap(data=depth_mm, unit="camera_unit")                           │
│  • 约定：camera_unit = 毫米 (mm)，不在此处做数值转换                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. FaceDetector.detect_face                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  • 输入：DepthMap(unit="camera_unit")                                          │
│  • 转换：depth_map.to_meters(depth_scale) 其中 depth_scale = camera_params     │
│          ["depth_scale"]，默认 0.001 (mm → m)                                    │
│  • 内部保存：self._depth_map = DepthMap(data=data*scale, unit="meter")          │
│  • 下游只使用 self._depth_map，单位统一为 米 (m)                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. coordinate_converter (pixel_to_3d / get_depth_interpolated)                │
├─────────────────────────────────────────────────────────────────────────────┤
│  • 输入：DepthMap(unit="meter")，camera_params (fx,fy,cx,cy)                    │
│  • get_depth_interpolated：从 depth_map.data 插值得到 depth_value (米)           │
│  • 校验：depth_meters > DEPTH_VALIDATION_THRESHOLD (米)，否则 ValueError         │
│  • 公式：X = (x - cx)*Z/fx, Y = (y - cy)*Z/fy, Z = depth_meters                 │
│  • 输出：Point3D 坐标 (米)                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 单位汇总

| 阶段           | 数据/变量           | 单位        | 说明                    |
|----------------|--------------------|-------------|-------------------------|
| 立体输出       | depth_mm / Z       | 毫米 (mm)   | OpenCV Q 标定单位       |
| 存储/DepthMap  | data, unit="camera_unit" | 毫米 (mm) | 不做转换，仅标记        |
| to_meters 后   | DepthMap.data      | 米 (m)      | data × depth_scale      |
| coordinate_converter | depth_meters, Point3D | 米 (m) | 阈值与输出均为米        |

## 图像与深度对齐

- **RGB**：add_frame 存 `rect_left`（左目校正图），get_image 返回同一帧。
- **深度**：与 rect_left 同分辨率（stereo image_size，如 1280×1080），逐像素对应。
- **显示**：recognition 主循环用 get_image 的同一 image 画关键点，与检测/深度坐标系一致。

## 配置

- `camera_params.json` → `depth_scale`（如 0.001）由 `_adapt_stereo_config` 写入，供 detector 的 to_meters 使用。
- `DEPTH_VALIDATION_THRESHOLD`（coordinate_converter）：有效深度下限（米），小于等于该值的深度视为无效（当前 0.05 m，约 5 cm）。

## 延伸阅读

- **[单帧深度诊断与 SGBM 问题分析](depth_diagnosis_single_frame.md)**：基于日志的单帧深度诊断示例、当前 SGBM 参数说明，以及高 nan 占比/关键点 nan/中心无匹配等现象的排查与调参建议。
