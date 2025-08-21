# 眼球中心拟合算法优化方案
## 一、拟合体选择：前期优先选择球体，保留椭球体接口，后期可使用椭球体进行优化

### 1. 屏幕投影误差公式
$$
E_{\text{screen}} = \delta_{\text{eye}} \times \frac{L_{\text{screen}}}{D_{\text{eye}}}
$$

**参数定义与科学依据**：
- $\delta_{\text{eye}}$：眼球中心定位误差（mm）
- $L_{\text{screen}}$：屏幕宽度（340mm，15.6英寸标准屏）
- $D_{\text{eye}}$：眼球到屏幕距离（700mm）
- **几何光学原理**：眼球定位误差在屏幕平面被放大，放大比例由$\frac{L_{\text{screen}}}{D_{\text{eye}}}$决定
- **解剖学依据**：人眼光学中心与角膜曲率中心的解剖偏移（0.3-0.8mm）是主要误差源

### 2. 眼球定位误差分解
$$
\delta_{\text{eye}} = \sqrt{e_{\text{geom}}^2 + \delta_{\text{sys}}^2}
$$

**参数定义**：
| 参数 | 球体模型 | 椭球模型 | 医学依据 |
|------|----------|----------|----------|
| $e_{\text{geom}}$ | 0.8 mm | 0.5 mm | 球体忽略角膜扁率(0.95)，椭球补偿曲率(半径7.8mm) |
| $\delta_{\text{sys}}$ | 0.5 mm | 0.1 mm | 球心与角膜曲率中心不重合，椭球残余巩膜非对称性 |

### 3、定量计算过程

#### (1) 眼球定位误差计算
**球体模型**：
$$
\delta_{\text{eye}}^{\text{sphere}} = \sqrt{0.8^2 + 0.5^2} = \sqrt{0.64 + 0.25} = \sqrt{0.89} = 0.94 \text{ mm}
$$

**椭球模型**：
$$
\delta_{\text{eye}}^{\text{ellipsoid}} = \sqrt{0.5^2 + 0.1^2} = \sqrt{0.25 + 0.01} = \sqrt{0.26} = 0.51 \text{ mm}
$$

#### (2) 屏幕投影误差计算
**球体模型**：
$$
E_{\text{screen}}^{\text{sphere}} = 0.94 \times \frac{340}{700} = 0.94 \times 0.4857 = 0.456 \text{ mm}
$$

**椭球模型**：
$$
E_{\text{screen}}^{\text{ellipsoid}} = 0.51 \times \frac{340}{700} = 0.51 \times 0.4857 = 0.248 \text{ mm}
$$

#### (3) 多源误差叠加计算
**总误差模型**：
$$
E_{\text{total}} = \sqrt{E_{\text{fit}}^2 + E_{\text{kappa}}^2 + E_{\text{calib}}^2}
$$

**误差分量**：
| 误差源 | 球体模型 | 椭球模型 | 来源 |
|--------|----------|----------|------|
| $E_{\text{fit}}$ | 0.46 mm | 0.25 mm | 眼球中心定位 |
| $E_{\text{kappa}}$ | ±0.3 mm | ±0.1 mm | 光学轴-视觉轴偏差(5°均值) |
| $E_{\text{calib}}$ | ±0.2 mm | ±0.1 mm | 相机标定分辨率(0.1mm/像素) |

**总误差计算**：
**球体模型**：
$$
E_{\text{total}}^{\text{sphere}} = \sqrt{0.46^2 + 0.3^2 + 0.2^2} = \sqrt{0.3116} = 0.558 \text{ mm}
$$

**椭球模型**：
$$
E_{\text{total}}^{\text{ellipsoid}} = \sqrt{0.25^2 + 0.1^2 + 0.1^2} = \sqrt{0.0825} = 0.287 \text{ mm}
$$


## 结论
1. **球体模型**：屏幕投影误差0.456mm（纯拟合）→0.558mm（总误差），适用于消费级场景
2. **椭球模型**：屏幕投影误差0.248mm（纯拟合）→0.287mm（总误差），满足医疗级精度需求
3. **优化建议**：前期优先使用球体模型，保留椭球拟合参数接口用于后期优化。
4. **具体代码优化**：



## 二、权重 $w_i$ 的设定：三层权限构成

### 1. 先验权重：拟合点与眼球表面距离权重在医学解剖学的量化

#### 瞳孔中心点权重
- **解剖偏移范围**：$0.1\text{--}0.3$ mm → 中值 $d_i = 0.2$ mm
- **距离球面标准差**：$\sigma = 1.5$ mm
- **先验权重计算**：$w_{\text{anatomy}} = \exp\left(-\frac{d_i^2}{2\sigma^2}\right) = \exp\left(-\frac{0.2^2}{2 \times 1.5^2}\right) = 0.991$

#### 虹膜边界点曲率补偿
- **自然偏移范围**：$0.5\text{--}1.2$ mm → 中值 $d_i = 0.85$ mm
- **距离球面标准差**：$\sigma = 1.5$ mm
- **先验权重计算**：$w_{\text{anatomy}} = \exp\left(-\frac{0.85^2}{2 \times 1.5^2}\right) = 0.839$

#### 眼睑点排除机制
- **到眼球表面距离**：$2.0\text{--}4.0$ mm → 中值 $d_i = 3.0$ mm
- **距离球面标准差**：$\sigma = 1.5$ mm
- **先验权重计算**：$w_{\text{anatomy}} = \exp\left(-\frac{3.0^2}{2 \times 1.5^2}\right) = 0.135$

### 2. 几何残差权重（动态变化）
- **初始化标准差**：$\sigma = d_i$（各偏移范围的中值）
  - 瞳孔中心点：$\sigma = 0.2$ mm
  - 虹膜边界点：$\sigma = 0.85$ mm  
  - 眼睑点：$\sigma = 3.0$ mm
- **实时计算**：$w_{\text{geom}} = \exp\left(-\frac{d_i^2}{2\sigma^2}\right)$
- **$d_i$为实时测量值**：点到拟合球面的实际距离
- **动态调整**：$\sigma$根据拟合质量自适应调整（待量化）

### 3. 权重融合公式
$$w_i = w_{\text{anatomy}} \times w_{\text{geom}} \times w_{\text{SNR}}$$

其中：
- $w_{\text{anatomy}}$：基于解剖偏移中值的固定先验权重
- $w_{\text{geom}}$：基于实时测量距离的动态几何权重
- $w_{\text{SNR}}$：图像质量权重（先设置为常数，后期可量化）

## 三、阈值参数的自适应调整

### 1. 核心阈值类型与初始值

#### (1) RANSAC距离阈值$\tau$
- 初始值 $\tau_{\text{init}} = 1.5$ mm  
  *覆盖虹膜点最大解剖偏移*


### 2. 残差驱动的调整机制

#### 几何拟合残差$e_{\text{geom}}$
- 要求 $e_{\text{geom}} < 0.8$ mm  
  *角膜平滑性约束*
- 若连续3帧$e_{\text{geom}} > 1.0$ mm，触发$\tau \rightarrow 1.5\tau$并启用椭球补偿
  eg：
  $$
  \tau_{\text{new}} = 
  \begin{cases} 
  0.8\tau & \text{if } e_{\text{geom}} < 0.8 \text{ and } \rho > 0.8 \\
  1.5\tau & \text{if } e_{\text{geom}} > 1.0 \text{ or } \rho < 0.5
  \end{cases}
  $$
  其中$\rho$为内点比例，$e_{\text{geom}}$为几何残差