# 目标函数（总目标）

令参数 $\theta = (c_x, c_y, c_z, r)^\top$（单位 mm），观测点 $p_i = (x_i, y_i, z_i)$，点对应生理偏移 $\delta_i$（mm），几何权重 $w_i \in (0,1]$。

## 定义数据项：

$$
f_i(\theta) = d_i(\theta) - (r + \delta_i), \quad d_i(\theta) = \|p_i - c\|
$$

## 带权平方和：

$$
F(\theta) = \frac{1}{2} \sum_{i=1}^{N} w_i \, f_i(\theta)^2
$$

## 深度约束（平滑惩罚）：

$$
h(\theta) = \text{softplus}_\tau(\varepsilon + z_{\text{max}} - c_z), \quad \text{softplus}_\tau(u) = \tau \ln(1 + e^{u/\tau})
$$

（其中 $z_{\text{max}} = \max_i z_i$，$\varepsilon$ 为小正偏移）

## 半径软先验（二次型）：

$$
R(\theta) = \frac{1}{2} \lambda_r (r - r_0)^2
$$

其中 $\lambda_r$ 为可自适应的先验强度，$r_0$ 为先验均值（例如 12 mm，但具体值随后商定）。

## 总目标函数：

$$
\Phi(\theta) = F(\theta) + \frac{1}{2} \lambda_{\text{depth}} h(\theta)^2 + \frac{1}{2} \lambda_r (r - r_0)^2
$$

---

# 几何可信权重（高斯形式）

初始阶段所有 $w_i = 1$。稳定后启用（见两阶段策略）。采用高斯式可信度：

$$
w_i = \exp\left(-\frac{(f_i^{\text{norm}})^2}{2\sigma^2}\right)
$$

其中 $f_i^{\text{norm}}$ 可用标准化残差（例如 $f_i / \sigma_{\text{obs}}$），或直接用几何一致度（见实现细节）。权重取值在 $(0,1]$。

（实现上：启用权重更新的时刻由“残差均值/变化幅度”阈值决定。）

---

# Softplus 的导数、链式法则（注意负号）

设 $u(\theta) = \varepsilon + z_{\text{max}} - c_z$。

Softplus 对 $u$ 的导数：

$$
\frac{d}{du} \text{softplus}_\tau(u) = s_\tau'(u) = \frac{1}{1 + e^{-u/\tau}}
$$

由于 $u$ 对 $c_z$ 的导数是 $\partial u / \partial c_z = -1$，因此对 $c_z$ 的导数要乘上这个负号：

$$
\frac{\partial}{\partial c_z} \left( \text{softplus}_\tau(u) \right) = s_\tau'(u) \cdot \frac{\partial u}{\partial c_z} = -s_\tau'(u)
$$

因此在计算深度残差对 $c_z$ 的雅可比时务必要带上这个负号（见雅可比部分）。

---

# 残差向量与雅可比（用于 LM）

把所有残差转成向量形式以统一用 LM 处理。

## 数据残差（加权）：

$$
\mathbf{r}_f(\theta) = 
\begin{bmatrix}
\sqrt{w_1} f_1 \\
\sqrt{w_2} f_2 \\
\vdots \\
\sqrt{w_N} f_N
\end{bmatrix}
$$

## 深度残差（单项）：

$$
r_z(\theta) = \sqrt{\lambda_{\text{depth}}} \cdot \text{softplus}_\tau(u), \quad u = \varepsilon + z_{\text{max}} - c_z
$$

## 半径先验残差：

$$
r_r(\theta) = \sqrt{\lambda_r} (r - r_0)
$$

## 合并总残差：

$$
\tilde{\mathbf{r}}(\theta) = 
\begin{bmatrix}
\mathbf{r}_f \\
r_z \\
r_r
\end{bmatrix}, \quad
\Phi(\theta) = \frac{1}{2} \| \tilde{\mathbf{r}}(\theta) \|^2
$$

---

# 数据雅可比（单点 i）

令 $d_i = \|p_i - c\|$。则

$$
\frac{\partial f_i}{\partial c_x} = -\frac{x_i - c_x}{d_i}, \quad
\frac{\partial f_i}{\partial c_y} = -\frac{y_i - c_y}{d_i}, \quad
\frac{\partial f_i}{\partial c_z} = -\frac{z_i - c_z}{d_i}, \quad
\frac{\partial f_i}{\partial r} = -1
$$

故带权雅可比行：

$$
\mathbf{J}_{f,i} = \sqrt{w_i} \cdot 
\left[ -\frac{x_i - c_x}{d_i} \quad -\frac{y_i - c_y}{d_i} \quad -\frac{z_i - c_z}{d_i} \quad -1 \right]
$$

## 深度残差雅可比

$$
\frac{\partial r_z}{\partial c_z} = \sqrt{\lambda_{\text{depth}}} \cdot s_\tau'(u) \cdot (-1)
$$

其它分量为 0。（这里再次强调负号来源于链式法则）

## 半径先验雅可比

$$
\frac{\partial r_r}{\partial r} = \sqrt{\lambda_r}, \quad \text{其余为 } 0
$$

把这些行合并得到 $\tilde{\mathbf{J}}(\theta)$。

---

# LM 更新方程（带阻尼）

标准 LM（使用对角阻尼）：

$$
\left( \tilde{\mathbf{J}}^\top \tilde{\mathbf{J}} + \lambda_{\text{LM}} \, \text{diag}(\tilde{\mathbf{J}}^\top \tilde{\mathbf{J}}) \right) \Delta = -\tilde{\mathbf{J}}^\top \tilde{\mathbf{r}}
$$

更新：

$$
\theta_{\text{new}} = \theta + \Delta
$$

接受准则：若 $\Phi(\theta_{\text{new}}) < \Phi(\theta)$ 则接受并降低 $\lambda_{\text{LM}}$（例如乘以 0.1–0.5）；否则拒绝并增大 $\lambda_{\text{LM}}$（例如乘以 10），然后重解方程直至接受或达到最大阻尼。LM 阻尼策略保持不变。

---

# 自适应二次先验因子 $\lambda_r$

要求：若个体半径真有偏移（例如真实 $r$ 距离 $r_0$ 很远），则 $\lambda_r$ 应自动放低；反之如数据不支持大偏移则 $\lambda_r$ 要强以防发散。

一种自适应策略（可直接实现）：

计算当前残差归一化指标（对数据项）：

$$
E = \frac{1}{N} \sum_{i=1}^{N} \frac{|f_i|}{\sigma_{\text{obs}}}
$$

（$\sigma_{\text{obs}}$ 是点观测噪声尺度，可估或设定。）

根据估计的“半径偏移置信度”调整 $\lambda_r$：

$$
\lambda_r = 
\begin{cases}
\lambda_{r,\text{high}}, & \text{若 } E < T_{\text{tight}} \text{（数据支持先验）} \\
\lambda_{r,\text{mid}}, & \text{若 } T_{\text{tight}} \leq E < T_{\text{loose}} \\
\lambda_{r,\text{low}}, & \text{若 } E \geq T_{\text{loose}} \text{（数据可能指示个体差异）}
\end{cases}
$$

或者使用平滑形式：

$$
\lambda_r = \lambda_{r,\text{min}} + (\lambda_{r,\text{max}} - \lambda_{r,\text{min}}) \exp(-\alpha E)
$$

这里 $\alpha > 0$ 控制敏感度。如此 $\lambda_r$ 随残差增加而自动衰减，允许更大偏移。

（参数 $\lambda_{r,\text{min}}, \lambda_{r,\text{max}}, \alpha, T_{\text{tight}}, T_{\text{loose}}$ 可在实验中微调）

---

# 两阶段拟合策略

阶段 A（初始稳态）：所有几何权重 $w_i = 1$。启动 LM，深度惩罚（$\lambda_{\text{depth}}$）与半径软先验（$\lambda_r$）均生效（按初始值）。运行若干迭代（或直到 $\Phi$ 下降速率低于阈值 / 达到参数变化阈值）。

**启用权重的条件（示例判据）**：当平均数据残差 $\frac{1}{N} \sum |f_i|$ 与上一次迭代相比下降到某个阈值内（例如相对变化 $< \eta$），并且参数变化 $\|\Delta \theta\|$ 小于阈值，则认为“拟合进入稳定区”，启用几何权重更新。

阶段 B（权重更新）：启用 $w_i$ 的高斯更新（或其它鲁棒函数），例如：

$$
w_i \leftarrow \exp\left(-\frac{(f_i / \sigma)^2}{2}\right)
$$

继续 LM，但保留深度惩罚与软先验（$\lambda_r$ 仍自适应）。权重随迭代逐步稳定，异常/遮挡点的影响会被弱化。

---

# 数值保护与实现细节（直接实现）

- 保护距离分母：若 $d_i < \epsilon$（例如 $10^{-6}$ mm）则用 $\epsilon$。
- 构造 $\tilde{\mathbf{J}}^\top \tilde{\mathbf{J}}$ 时避免溢出/奇异：若矩阵条件数过差，增大 $\lambda_{\text{LM}}$。
- 线性系统求解使用稳健 solver（例如 Cholesky 带对角修正或 QR/ SVD），并包裹异常捕获。
- LM 阻尼初始与更新策略：初始 $\lambda_{\text{LM}}$ 可选小（$10^{-3}$~$1.0$ 依归一化而定），失败时乘以 10，成功时乘以 0.1（或 0.5）。
- 记录监控指标（每步）：$\Phi$、$\Delta r$、$\|\Delta c\|$、雅可比列范数 $\|\mathbf{J}_{:,r}\|, \|\mathbf{J}_{:,c_z}\|$ 等。

---

# 最终算法流程（逐步）

## 预处理
- 把输入点 $p_i$ 转成 mm（确保全部同一量纲）。
- 为每点指定 $\delta_i$（基于点类：pupil/iris/contour，使用中位生理值）。
- 初始化参数 $\theta_0 = (c_{x0}, c_{y0}, c_{z0}, r_0)$。
- 初始几何权重 $w_i \leftarrow 1$。设初始 $\lambda_{\text{depth}}, \lambda_r$。

## 阶段 A（LM，权重为 1）
用 LM 求解 $\min \Phi(\theta)$（使用 $\tilde{\mathbf{r}}, \tilde{\mathbf{J}}$），阻尼按常规更新。运行直到达到“阶段 B 启用条件”或最大迭代数。

## 启用几何权重（判断条件）
检查残差与参数变化，如果稳定则进入阶段 B。

## 阶段 B（LM，启用权重更新与自适应 $\lambda_r$）
每若干迭代更新 $w_i$（高斯式或其他鲁棒式），并根据残差量 $E$ 自适应更新 $\lambda_r$。持续 LM 迭代，直到收敛（$\|\Delta \theta\|$ 与 $\Delta \Phi$ 小于阈值），或最大迭代次数到达。

## 后验校验
输出最终参数并检查物理合理性（$r$ 在可接受范围内，$c_z > z_{\text{max}} + \varepsilon$）。若 $\lambda_r$ 被大幅降低且 $r$ 偏离先验较多，标记“可能个体差异”以供后续统计或人工审核。

---

# 监控指标（建议记录）

每次迭代：
- $\Phi$、数据项 $F$、深度项值 $h$、先验项 $R$
- $\Delta r$、$\|\Delta c\|$、$\lambda_{\text{LM}}$、$\lambda_r$、平均残差、权重分布（mean/var）
- 雅可比列范数：$\|\mathbf{J}_{:,c_x}\|, \|\mathbf{J}_{:,c_y}\|, \|\mathbf{J}_{:,c_z}\|, \|\mathbf{J}_{:,r}\|$

---

# 为什么这满足你的要求

- 不再对“点必须在球外”做乘法惩罚，而是把生理偏移 $\delta_i$ 直接并入数据残差，使数据项表示“点减去其期望偏移后应落在球面上”。
- 几何可信度用高斯公式实现，并在拟合稳定后启用，初始阶段不会干扰初值搜索。
- 半径约束为软先验且可自适应（$\lambda_r$ 随残差自动调整），不通过硬截断改变 LM 流程。
- 保留深度惩罚（只影响 $c_z$），并正确处理 softplus 的链式导数（含负号）。
- LM 阻尼、数值保护均保留，保证数值稳定性。