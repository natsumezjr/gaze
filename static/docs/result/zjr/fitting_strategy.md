# 拟合策略配置模块 - LaTeX版本

## 算法流程总结

### 1. 数据质量评估（基于Mediapipe visibility）
- **高质量**（visibility ≥ 0.9）：直接使用 **约束 LM 拟合**
- **中质量**（0.7 ≤ visibility < 0.9）：使用 **RANSAC + 约束 LM 拟合**  
  - **RANSAC 阶段**：用几何一致性判定内点  
  - **LM 阶段**：在内点集上运行带约束的 LM  
- **低质量**（visibility < 0.7）：直接返回失败

---

### 2. 约束拟合算法策略

#### 2.1 核心约束条件
- 所有关键点必须在眼球外部：$\|\mathbf{p}_i - \mathbf{c}\| \geq r$
- 瞳孔到眼球表面距离：0.1–0.3mm
- 虹膜边界到眼球表面距离：0.5–1.2mm
- 眼睛轮廓到眼球表面距离：1.0–2.0mm

#### 2.2 约束实现方式
- 在 LM 优化过程中添加约束惩罚项
- 惩罚函数：  
  $$P = \lambda \sum_i \max(0, r - \|\mathbf{p}_i - \mathbf{c}\|)^2$$
- $\lambda$ 为惩罚因子（1000.0），确保约束严格满足

#### 2.3 最小二乘法权重策略
- 参数收敛：$\|\Delta\boldsymbol{\theta}\| < \text{param\_tolerance}$
- 残差收敛：$|\Delta r| < \text{residual\_tolerance}$
- 几何权重启用条件：  
  $$\text{param\_tolerance} < \|\Delta\boldsymbol{\theta}\| < \text{param\_tolerance} \times geometric\_weight\_factor$$  
  且  
  $$\text{residual\_tolerance} < |\Delta r| < \text{residual\_tolerance} \times geometric\_weight\_factor$$

**最终权重计算策略**：
$$w_{\text{final}} = \begin{cases}
w_{\text{anat}}, & \text{if 收敛稳定} \\
w_{\text{anat}} \cdot w_{\text{geom}}, & \text{if 中间区间满足条件}
\end{cases}$$

其中：
- $w_{\text{anat}}$：解剖学权重（固定值）
- $w_{\text{geom}}$：几何权重（动态计算）
- $w_{\text{final}}$：最终使用的权重

---

### 3. 参数复用策略
- 当调用达到 `max_trials=100` 次后，半径参数已稳定
- 可直接复用历史最佳参数，避免重复计算
- 提升效率，减少冗余迭代

---

### 4. 拟合流程
1. **数据预处理**：visibility 过滤 → 点数检查 → 数据质量评估
2. **策略选择**：高质量 → 约束 LM；中质量 → RANSAC+约束 LM
3. **约束拟合**：加入解剖学约束，执行 LM
4. **结果验证**：确保所有点在球体外部
5. **参数复用**：达到 `max_trials` 后复用最佳参数

---

### 5. 量化拟合策略公式

#### 5.1 RANSAC 内点判定（几何一致性）
$$\text{Inlier}(p_i) = \mathbf{1}\left(|\|\mathbf{p}_i - \mathbf{c}\| - r| < \tau\right)$$

其中 $\tau$ 为阈值（如 0.5mm）

#### 5.2 几何权重函数
$$w_{\text{geom}} = I(\varepsilon_{\text{param}}, \varepsilon_{\text{res}}) \cdot \exp\left(-\tfrac{d^2}{2\sigma^2}\right)$$

#### 5.3 收敛判据
- 参数收敛：$\|\Delta\boldsymbol{\theta}\| < \text{param\_tolerance}$
- 残差收敛：$|\|\mathbf{f}(\theta_{\text{new}})\| - \|\mathbf{f}(\theta_{\text{old}})\|| < \text{residual\_tolerance}$
- 最大迭代：$\text{iteration\_count} < \text{max\_iterations}$


### 6. RANSAC + 约束 LM 优化

在本项目中，我们采用了 **RANSAC + 约束 Levenberg–Marquardt (LM)** 的融合优化流程。与原始的 Gauss–Newton (GN) 不同，该方法不仅利用 RANSAC 筛选内点，还在 LM 的迭代更新中引入了解剖学约束，以提高数值稳定性和物理合理性。

---

#### 1. 参数定义

- 观测点：\(\mathbf{p}_i = (x_i,y_i,z_i)^\top, \; i=1,\dots,n\)  
- 参数向量：\(\boldsymbol{\theta}=(c_x,c_y,c_z,r)^\top\)  
- 点到中心距离：
  \[
  d_i(\boldsymbol{\theta})=\sqrt{(x_i-c_x)^2+(y_i-c_y)^2+(z_i-c_z)^2}
  \]
- 数据残差（球面一致性）：
  \[
  f_i(\boldsymbol{\theta}) = d_i(\boldsymbol{\theta}) - r
  \]
- 权重矩阵：\(\mathbf{W}=\operatorname{diag}(w_i)\)

---

#### 2. 约束条件与惩罚项

#### 2.1 点必须在球外
硬约束形式：
\[
P_1(\boldsymbol{\theta}) = \lambda_1 \sum_{i=1}^n \max(0,\, r - d_i(\boldsymbol{\theta}))^2
\]

为保证可导性，使用 softplus 平滑近似：
\[
\tilde P_1(\boldsymbol{\theta}) = \lambda_1 \sum_{i=1}^n \big[\text{softplus}_\tau(r-d_i)\big]^2
\]

其中：
\[
\text{softplus}_\tau(u) = \tau \log(1+e^{u/\tau}),\quad s_\tau'(u)=\frac{1}{1+e^{-u/\tau}}
\]

#### 2.2 球心深度必须大于所有点
设 \(\bar z_{\max} = \max_i z_i\)，则：
\[
P_2(\boldsymbol{\theta}) = \lambda_2 \max(0,\, \varepsilon + \bar z_{\max} - c_z)^2
\]

同样采用 softplus 平滑：
\[
\tilde P_2(\boldsymbol{\theta}) = \lambda_2 \big[\text{softplus}_\tau(\varepsilon+\bar z_{\max}-c_z)\big]^2
\]

#### 2.3 总惩罚
\[
\tilde P(\boldsymbol{\theta}) = \tilde P_1(\boldsymbol{\theta}) + \tilde P_2(\boldsymbol{\theta})
\]

---

#### 3. 总目标函数

\[
\Phi(\boldsymbol{\theta}) = \tfrac12\, \mathbf{f}^\top \mathbf{W}\,\mathbf{f} + \tilde P(\boldsymbol{\theta})
\]

---

#### 4. 残差堆叠与雅可比

为了在 LM 中统一处理数据项与惩罚项，我们将其转化为附加残差形式：

- 数据残差：  
  \(\mathbf{r}_f = \mathbf{W}^{1/2}\mathbf{f}(\boldsymbol{\theta})\)  
- 点约束残差：  
  \(g_i(\boldsymbol{\theta}) = \sqrt{\lambda_1}\; \text{softplus}_\tau(r-d_i)\)  
- 深度约束残差：  
  \(g_z(\boldsymbol{\theta}) = \sqrt{\lambda_2}\; \text{softplus}_\tau(\varepsilon+\bar z_{\max}-c_z)\)

总残差向量：
\[
\tilde{\mathbf{r}}(\boldsymbol{\theta})=
\begin{bmatrix}
\mathbf{r}_f \\[2pt]
\mathbf{g}
\end{bmatrix}
,\quad
\tilde{\mathbf{J}}(\boldsymbol{\theta})=
\begin{bmatrix}
\mathbf{W}^{1/2}\mathbf{J}_f \\[2pt]
\mathbf{J}_g
\end{bmatrix}
\]

---

#### 5. 各部分雅可比

#### 5.1 数据项 \(f_i\)
\[
\mathbf{J}_{f,i} =
\begin{bmatrix}
-\dfrac{x_i-c_x}{d_i} &
-\dfrac{y_i-c_y}{d_i} &
-\dfrac{z_i-c_z}{d_i} &
-1
\end{bmatrix}
\]

#### 5.2 点约束 \(g_i\)
\[
\mathbf{J}_{g,i} = \sqrt{\lambda_1}\, s_\tau'(r-d_i)\,
\begin{bmatrix}
\dfrac{x_i-c_x}{d_i} & \dfrac{y_i-c_y}{d_i} & \dfrac{z_i-c_z}{d_i} & 1
\end{bmatrix}
\]

#### 5.3 深度约束 \(g_z\)
\[
\mathbf{J}_{g,z} = \sqrt{\lambda_2}\, s_\tau'(\varepsilon+\bar z_{\max}-c_z)\,
\begin{bmatrix}
0 & 0 & -1 & 0
\end{bmatrix}
\]

---

#### 6. LM 更新方程（融合约束）

LM 的正规方程为：
\[
\big(\tilde{\mathbf{J}}^\top \tilde{\mathbf{J}} + \lambda_{\text{LM}} \mathbf{I}\big)\,\Delta
= -\tilde{\mathbf{J}}^\top \tilde{\mathbf{r}}
\]

展开得到：
\[
\Big(\mathbf{J}_f^\top \mathbf{W} \mathbf{J}_f + \mathbf{J}_g^\top \mathbf{J}_g + \lambda_{\text{LM}} I\Big)\Delta
= -\Big(\mathbf{J}_f^\top \mathbf{W} \mathbf{f} + \mathbf{J}_g^\top \mathbf{g}\Big)
\]

其中：
- \(\mathbf{J}_f^\top W \mathbf{J}_f\)：数据残差近似 Hessian  
- \(\mathbf{J}_g^\top \mathbf{J}_g\)：约束惩罚贡献的曲率  
- 右侧 \(-(\mathbf{J}_f^\top W f + \mathbf{J}_g^\top g)\)：数据与约束的联合梯度

---

#### 7. 工程注意事项

1. **Softplus 平滑**：避免不连续梯度引发 LM 振荡  
2. **权重选择**：\(\lambda_1,\lambda_2\) 与观测噪声方差量级匹配  
3. **数值保护**：\(d_i\) 太小时需加 \(\epsilon\) 防止除零  
4. **LM 阻尼调节**：采用标准信赖域策略（基于实际下降比 \(\rho\)）  
5. **RANSAC 结合**：在 RANSAC 内点集上运行本约束 LM，以增强鲁棒性



