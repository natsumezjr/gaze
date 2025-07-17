# 眼动追踪后端系统

本项目基于RGB-D摄像头（如Intel RealSense D455），实现高精度的眼动追踪与人机交互。系统包含识别、拟合、追踪、交互四大核心模块，支持三维空间下的视线估计与多层级交互。

## 主要模块
- 识别模块：基于MediaPipe等工具，提取瞳孔/虹膜三维坐标。
- 拟合模块：利用PnP算法与最小二乘法，完成空间坐标转换与视线向量计算。
- 追踪模块：基于LSTM网络，建模视线动态变化，实现稳健追踪。
- 交互模块：将gaze点映射到屏幕，支持ROI优先级的人机交互。

详细结构与说明见docs/structure.md。 

## Markdown公式显示说明

本项目文档包含大量数学公式。不同Markdown阅读器对LaTeX公式支持情况如下：

### 1. Typora
- 支持`$...$`和`$$...$$`公式，直接可用。
- 无需额外配置。

### 2. VSCode
- 推荐安装插件：`Markdown+Math` 或 `Markdown Preview Enhanced`。
- 安装后支持LaTeX公式渲染。

### 3. Jupyter Notebook
- 原生支持LaTeX公式。

### 4. GitHub网页
- 原生不支持LaTeX公式，会显示为源码。
- 解决方案：
  - 使用[Chrome插件](https://chrome.google.com/webstore/detail/github-with-mathjax/ioemnmodlmafdkllaclgeombjnmnbima)（GitHub with MathJax）
  - 或将公式转为图片插入。
  - 然后使用Shift+Ctrl+v查看

### 5. 纯文本兼容
- 若需最大兼容性，建议文档中同时给出纯文本公式表达。

---

**示例：**

- LaTeX公式：
  `$x_{pixel} = x_{norm} \times (W-1)$`
- 纯文本公式：
  `x_pixel = x_norm * (W - 1)`

如需在本地渲染公式，推荐使用Typora或VSCode+插件。 