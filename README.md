## 眼动追踪项目（Gaze）

本仓库包含两个主要子模块：识别模块（`project/recognition`）与拟合/标定模块（`project/fitting`）。已统一导包为绝对导入前缀 `project.*`，并推荐使用模块方式（`python -m ...`）从仓库根目录运行。

### 目录结构（简要）

- `project/recognition/`：人脸与眼部特征识别、关键点提取与坐标转换
- `project/fitting/`：基于识别结果进行几何拟合（球/椭球中心等）、标定工具/应用
- `requirements.txt`：项目依赖清单
- `pyrightconfig.json`：类型检查配置

### 导包规范（必须）

- 统一使用绝对导入，前缀一律为 `project.*`
  - 例如：
    - `from project.recognition.core.detector import FaceDetector`
    - `from project.fitting.core.ransac_sphere_fitter import fitting_eyeballs`
- 运行时从仓库根目录启动，使解释器能找到顶层包 `project`。
- 不建议在系统层面配置环境变量；也不在包的 `__init__.py` 中“注册路径”。如需临时修改导入路径，只在“入口脚本”最顶部注入父目录到 `sys.path`。

### 环境准备

- Python 3.8+（已在 3.11 验证）
- 建议使用虚拟环境（venv/conda）

安装依赖（仓库根目录执行）：

```powershell
pip install -r requirements.txt
```

依赖清单核心条目（详见 `requirements.txt`）：
- OpenCV (cv2)
- MediaPipe
- NumPy / SciPy
- Pillow / Matplotlib / pandas（工具与可视化）

### 运行方式（推荐）

从“仓库根目录”使用模块方式运行，确保 `project` 被正确解析为包。

- 识别模块主入口：
```powershell
python -m project.recognition.main
```

- 拟合模块主入口（示例）：
```powershell
python -m project.fitting.main
```

- 直接运行单个模块（示例）：
```powershell
python -m project.fitting.core.ransac_sphere_fitter
```

说明：`-m` 会以包模块方式运行，自动以当前工作目录作为首要搜索路径，从而找到顶层包 `project`。不要从深层目录直接 `python xxx.py`，否则会出现 `ModuleNotFoundError: No module named 'project'`。

### 可编辑安装与打包（可选）

当前仓库对识别模块提供了独立的可编辑安装配置（`project/recognition/setup.py`）。

- 开发安装（推荐在开发识别模块时使用）：
```powershell
cd project/recognition
pip install -e .
cd ../..
```
完成后，可在任意位置 `import project.recognition ...`。

- 构建分发包（识别模块）：
```powershell
cd project/recognition
python setup.py sdist bdist_wheel
cd ../..
```
生成的分发文件位于 `project/recognition/dist/`。

如需将整个 `project` 作为一个总包进行安装，可后续补充仓库根的打包配置（例如 `pyproject.toml` / 顶层 `setup.py`）。目前推荐使用“仓库根运行 + `-m`”或上面的识别子包可编辑安装方案。

### 常见问题与排查

- 报错 `ModuleNotFoundError: No module named 'project'`：
  - 确认当前工作目录是仓库根目录（包含 `project/`）。
  - 使用 `python -m project.xxx.yyy` 而不是 `python project/xxx/yyy.py`。
- 导入 `np` 未定义或注解解析报错：
  - 在用到 `numpy` 的模块顶部显式 `import numpy as np`。
  - 可在文件顶部添加 `from __future__ import annotations` 以推迟类型注解求值。
- 摄像头初始化失败：
  - 识别模块默认尝试打开摄像头，请检查设备权限与驱动；必要时修改 `project/recognition/utils/camera_calibration.py` 中的相机打开逻辑。

### 约定与建议

- 入口脚本统一以 `-m` 方式从仓库根运行，避免路径歧义。
- 只在“入口脚本”必要时短暂注入 `sys.path`，不在库代码中做路径注入。
- 文档中若仍出现 `recognition.*`/`core.*` 等历史示例，请以本 README 的 `project.*` 规范为准。

---

如需将 `project/fitting` 也做成可编辑安装的独立包，或补齐仓库根的打包配置，我可以进一步完善并提供一键脚本。


