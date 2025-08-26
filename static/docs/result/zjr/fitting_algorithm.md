# 双残差 + 质量判别激活集（无权重）LM 眼球拟合方案

## 0. 记号与单位

- 统一单位：**mm**
- 观测点 $p_i = (x_i, y_i, z_i)$，点类型 $t(i) \in \{\text{pupil}, \text{iris}, \text{sclera}\}$
- 球心 $c = (c_x, c_y, c_z)$，半径 $r$
- 到心距离 $d_i = \| p_i - c \|$
- 参数 $\theta = (c_x, c_y, c_z, r)$

### 生理"初始偏移中值"（按点类型）

| 点类型 | 到球面偏移中值 $\delta_t^{\text{surf}}$ (mm) | 到球心偏移中值 $\delta_t^{\text{cent}}$ (mm) |
|--------|---------------------------------------------|---------------------------------------------|
| pupil  | -3.8                                        | 12.0                                        |
| iris   | -0.9                                        | 11.5                                        |
| sclera | -1.5                                        | None                                        |

> **注**：巩膜不用于 f2

---

## 1. 双残差模型（无权重）

### 1.1 残差定义

$$
\begin{aligned}
f_{1,i} &= d_i - r - \delta_{t(i)}^{\text{surf}} \\
f_{2,i} &= 
\begin{cases} 
d_i - \delta_{t(i)}^{\text{cent}}, & \delta_{t(i)}^{\text{cent}} \text{ 存在} \\
\text{不使用}, & \text{否则}
\end{cases}
\end{aligned}
$$

### 1.2 目标函数（全局占比系数）

设 $\alpha \in [0,1]$ 调节两类残差占比：

$$
\Phi(\theta) = \frac{1}{2} \left[ \alpha \sum_{i \in I_1} \rho_1(f_{1,i}) + (1 - \alpha) \sum_{i \in I_2} \rho_2(f_{2,i}) \right] + R_r(r) + P_z(c_z)
$$

其中：
- $\rho_1, \rho_2$ 为鲁棒损失（Huber/Charbonnier）
- $R_r(r)$：半径软先验
- $P_z(c_z)$：深度安全软惩罚

#### Huber 鲁棒损失推荐

$$
\rho(f) = 
\begin{cases} 
\frac{1}{2} f^2, & |f| \leq \kappa \\
\kappa (|f| - \frac{1}{2} \kappa), & |f| > \kappa
\end{cases}
\quad (\kappa \approx 0.6 \text{ mm})
$$

---

## 2. 置信度质量判别 → 激活集

### 2.1 置信度定义（仅用于判别）

$$
\begin{aligned}
C_i^{(1)} &= \exp\left( -\frac{f_{1,i}^2}{2 \sigma_{1,t(i)}^2} \right) \\
C_i^{(2)} &= 
\begin{cases} 
\exp\left( -\frac{f_{2,i}^2}{2 \sigma_{2,t(i)}^2} \right), & \delta_{t(i)}^{\text{cent}} \text{ 存在} \\
\text{N/A}, & \text{否则}
\end{cases}
\end{aligned}
$$

#### σ 的标定（满足 ∣f∣=0.4 时 C=0.9）

$$
\sigma = \frac{0.4}{\sqrt{2 \ln(1/0.9)}} \approx 0.871 \text{ mm}
$$

同一 σ 下：
- $|f| \approx 0.736 \Rightarrow C \approx 0.7$
- $|f| \approx 1.028 \Rightarrow C \approx 0.5$

**推荐范围**：
- $\sigma_{1,\cdot} \in [0.8, 1.0]$ mm
- $\sigma_{2,\cdot} \in [0.8, 1.2]$ mm

### 2.2 置信度融合与激活规则

#### 融合（仅用于判别）

$$
C_i = 
\begin{cases} 
C_i^{(1)} C_i^{(2)}, & C_i^{(2)} \text{ 存在} \\
C_i^{(1)}, & \text{否则}
\end{cases}
$$

#### 激活集

$$
\begin{aligned}
I_1 &= \{ i \mid C_i^{(1)} \geq \tau_1 \} \quad \text{(参与 f1)} \\
I_2 &= \{ i \mid C_i^{(2)} \geq \tau_2 \} \quad \text{(参与 f2)}
\end{aligned}
$$

**阈值建议**：
- $\tau_1 = \tau_2 = 0.7$ 起步
- 异常多时可放宽至 0.5 做"救援"，再回收至 0.7

**诊断触发**：若 $C_i < 0.5$ 占比 > 40%，判定生理/模型不一致 → 建议切换椭球拟合或重置初值

> **注**：眼睑/外眼角不进置信度也不进激活集，只用于遮挡判定与更新许可（§5）

---

## 3. 先验与软惩罚（与置信度独立）

### 3.1 半径先验与硬盒

**软先验**：
$$
R_r(r) = \frac{1}{2} \lambda_r (r - r_0)^2, \quad r_0 \approx 12, \ \lambda_r \in [0.1, 5]
$$

**硬盒约束**：
$$
r \in [r_{\text{min}}, r_{\text{max}}] = [8, 20] \text{ mm}
$$

> 越界更新直接截断/拒绝

### 3.2 深度安全软惩罚（不推动 r）

$$
P_z(c_z) = \frac{1}{2} \lambda_z \left[ \text{softplus}_\tau( \varepsilon_z + \max_i z_i - c_z ) \right]^2
$$

**推荐参数**：
- $\varepsilon_z = 0.2$ mm
- $\lambda_z \in [1, 10]$
- $\tau = 0.05$ mm

---

## 4. 雅可比与 LM 正规方程

### 4.1 导数计算

$$
\begin{aligned}
\frac{\partial d_i}{\partial c} &= -\frac{p_i - c}{d_i} \\
\frac{\partial f_{1,i}}{\partial c} &= \frac{\partial f_{2,i}}{\partial c} = -\frac{p_i - c}{d_i} \\
\frac{\partial f_{1,i}}{\partial r} &= -1, \quad \frac{\partial f_{2,i}}{\partial r} = 0
\end{aligned}
$$

#### Huber 损失的一阶权函数

$$
\rho'(f) = 
\begin{cases} 
f, & |f| \leq \kappa \\
\kappa \cdot \text{sign}(f), & |f| > \kappa
\end{cases}
$$

> 在 LM 中用等效残差 $\tilde{f} = w_\rho f$ 与等效雅可比（$w_\rho = \rho'(f)/f$）

### 4.2 归一化与求解

列尺度 $s = [s_x, s_y, s_z, s_r]$（例如点云 RMS 或 1 mm）

$$
(\bar{J}^\top \bar{J} + \lambda \cdot \text{diag}(\bar{J}^\top \bar{J})) \Delta_{\text{norm}} = -\bar{J}^\top \bar{r}, \quad \Delta = \Delta_{\text{norm}} \odot s
$$

---

## 5. 两阶段与占比调度 + 更新许可 / 阻尼自增

### 5.1 两阶段优化（防早期 r 失控）

#### 阶段 A（稳心）
- 固定 $r = r_0$
- $\alpha = 0.2$
- 只用 $I_2$（f2）
- 迭代 3–10 步

#### 阶段 B（细化）
- 解锁 $r$
- $\alpha \uparrow 0.6 \sim 0.7$
- 启用 $I_1, I_2$
- 迭代 50–100 步

### 5.2 每步流程（激活集驱动）

1. 基于当前 $(c, r)$ 计算 $f_{1,i}, f_{2,i}$
2. 计算 $C_i^{(1)}, C_i^{(2)}$，生成 $I_1, I_2$（阈值 $\tau$）
3. 组装 $\Phi$（只对激活集求和）与雅可比，解 LM
4. **更新许可检查**（不通过则拒绝本次更新并增大阻尼 $\lambda$）：
   - 半径硬盒：$r_{\text{min}} \leq r + \Delta r \leq r_{\text{max}}$
   - 深度安全：$c_z + \Delta c_z$ 满足 $P_z$ 的阈值（或 $c_z \geq \max z_i + \varepsilon_z$）
   - 遮挡约束：更新后不应显著增加"落入眼睑遮挡区域"的点
5. **接受/拒绝与阻尼自适应**：
   - 若 $\Phi(\theta + \Delta) < \Phi(\theta)$ 且更新许可通过 → 接受 & $\lambda \downarrow$（×0.3–0.5）
   - 否则 → 拒绝 & $\lambda \uparrow$（×3–10）并重解（上限重试后进入下一步/终止）

### 5.3 收敛判据

- **参数步长**：$\| \Delta c \| < 0.01$ mm 且 $| \Delta r | < 0.005$ mm
- **目标下降**：相对下降 $< 10^{-4}$
- **迭代上限**：阶段 A 10 步，阶段 B 50–100 步

---

## 6. 防"越拟合 r 越大"机制

1. **结构上**：增大 $r$ 只会让 $f_1$ 变小，但对 $f_2$ 完全无效；而 $(1 - \alpha) \sum_{i \in I_2} \rho_2(f_{2,i})$ 会稳住"各类点到心距离分布"，阻止仅靠放大 $r$ 降目标值
2. **硬盒 + 软先验**：$r$ 有边界与 $R_r(r)$；就算 f1 有下降空间，也会被这两项抵消
3. **深度软惩罚只作用 $c_z$**：不会共同推动 $r$
4. **激活集**：异常点被剔除而不是"减权"，不会给错的趋势提供"越大越好"的统计优势