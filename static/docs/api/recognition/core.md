# 识别模块数据结构接口

## 主要数据结构

### 1. RGB图像
- **类型**：`np.ndarray`，形状`(H, W, 3)`
- **格式**：✅ **RGB格式统一**（R、G、B三通道）

### 2. 深度图
- **类型**：`np.ndarray`，形状`(H, W)`
- **单位**：✅ **深度图单位统一（米）**

### 3. 关键点
- **单个关键点**：`Tuple[float, float, float]`
- **关键点列表**：`List[Tuple[float, float, float]]`
- **关键点字典**：`Dict[str, Tuple[float, float, float]]`

### 4. 三维坐标
- **单个三维坐标**：`np.ndarray`，形状`(3,)`
- **三维坐标字典**：`Dict[str, np.ndarray]`
- **三维坐标列表**：`List[np.ndarray]`

### 5. 相机参数
- **类型**：`dict`
- **主要字段**：`intrinsic_params`、`image_resolution`、`depth_scale`

## 统一约定

- ✅ **RGB图像格式统一**
- ✅ **深度图单位统一（米）**
- ✅ **坐标系定义统一（OpenCV标准）**
- ✅ **类型定义统一**

---

所有数据结构接口已统一，确保模块间数据传递的一致性。 