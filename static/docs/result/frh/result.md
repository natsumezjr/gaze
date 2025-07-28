# 拟合模块算法调研报告 —— RANSAC球面拟合方案

## 1. 模块目标

本模块目标：  
**通过对虹膜边界点的三维坐标进行球面拟合，获得眼球球心和半径，为后续视线向量计算和Kappa角补偿提供基础。**

## 2. 理论导视（白话说明）

传统方法采用最小二乘法拟合球面，假设所有点都可靠。但实际采集中，虹膜边界点可能受遮挡、反光、检测误差等影响，出现“离群点”（outlier）。这些异常点会极大影响拟合精度。

**RANSAC（随机采样一致性）**是一种鲁棒拟合算法，核心思想是：  
- 随机多次从点集中选取最小子集，拟合模型（如球面）
- 统计所有点中有多少“内点”与该模型足够接近
- 选取“内点”最多的模型作为最终结果

这样，即使有较多离群点，RANSAC也能找到最优拟合结果，极大提升鲁棒性。

## 3. 算法对比分析

| 方法         | 精度      | 鲁棒性   | 速度     | 易用性   | 依赖库         |
|--------------|-----------|----------|----------|----------|----------------|
| 最小二乘法   | 高（无异常点） | 差（对离群点敏感） | 快      | 简单      | numpy, scipy   |
| RANSAC拟合   | 高        | 强（可容忍大量离群点） | 较慢（需多次迭代） | 一般      | sklearn, numpy |
| 其它鲁棒拟合 | 视实现而定 | 视实现而定 | 视实现而定 | 复杂      | 依赖多         |

**结论：**  
- 数据干净时，最小二乘法足够且速度快  
- 数据含异常点时，RANSAC能显著提升拟合精度，推荐作为实际工程方案

## 4. 选择理由

- 实际采集虹膜边界点时，异常点不可避免
- RANSAC对离群点不敏感，拟合结果更稳定
- sklearn等库已集成RANSAC，易于实现

## 5. 定性公式实现

- **球面模型**：  
  $$
  (x-c_x)^2 + (y-c_y)^2 + (z-c_z)^2 = r^2
  $$
- **RANSAC流程**：  
  1. 随机选4个点，拟合球面参数$(c_x, c_y, c_z, r)$
  2. 计算所有点到球面的距离，统计“内点”数量（距离小于阈值）
  3. 重复N次，选内点最多的模型
  4. 用所有内点再做一次最小二乘法精细拟合

## 6. 定量Python代码实现

```python
import numpy as np
from scipy.optimize import least_squares

# 球面残差函数
def sphere_residuals(params, xyz):
    cx, cy, cz, r = params
    return np.sqrt((xyz[:,0]-cx)**2 + (xyz[:,1]-cy)**2 + (xyz[:,2]-cz)**2) - r

# RANSAC球面拟合主流程
def ransac_fit_sphere(points, threshold=0.5, max_trials=100):
    n_points = points.shape[0]
    best_inliers = []
    best_params = None

    for _ in range(max_trials):
        # 随机选4个点
        idx = np.random.choice(n_points, 4, replace=False)
        sample = points[idx]
        # 初始猜测
        center_init = np.mean(sample, axis=0)
        r_init = np.mean(np.linalg.norm(sample - center_init, axis=1))
        params_init = np.append(center_init, r_init)
        # 拟合
        try:
            result = least_squares(sphere_residuals, params_init, args=(sample,))
            params = result.x
        except:
            continue
        # 计算所有点残差
        residuals = np.abs(sphere_residuals(params, points))
        inliers = np.where(residuals < threshold)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_params = params

    # 用所有内点再精细拟合
    if best_inliers is not None and len(best_inliers) >= 4:
        inlier_points = points[best_inliers]
        center_init = np.mean(inlier_points, axis=0)
        r_init = np.mean(np.linalg.norm(inlier_points - center_init, axis=1))
        params_init = np.append(center_init, r_init)
        result = least_squares(sphere_residuals, params_init, args=(inlier_points,))
        return result.x[:3], result.x[3]
    else:
        raise RuntimeError("RANSAC未找到有效拟合")

# 示例数据
iris_points = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
    [5.0, 5.0, 5.0]  # 离群点
])
C_eye, r = ransac_fit_sphere(iris_points)
print('RANSAC拟合球心:', C_eye, '半径:', r)
```

## 7. 总结

- RANSAC球面拟合能有效提升鲁棒性，适合实际虹膜点存在异常的场景
- 推荐在工程实现中优先采用RANSAC+最小二乘法精细拟合的组合方案 