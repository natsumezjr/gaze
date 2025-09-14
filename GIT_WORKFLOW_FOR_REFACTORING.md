# Git工作流程说明 - 眼动追踪系统重构

## 一、项目结构说明

### 1.1 分支结构
```
main (当前分支)
├── 原始项目代码
├── 识别模块 (project/recognition/)
├── 拟合模块 (project/fitting/)
└── 其他配置文件

refactor/architecture-restructure (重构分支)
├── 重构后的项目结构
├── 新的模块化设计
├── 客户端应用
└── 详细文档
```

### 1.2 当前状态
- **主分支 (main)**：包含原始项目代码，作为稳定版本
- **重构分支 (refactor/architecture-restructure)**：用于进行架构重构

## 二、Git操作指南

### 2.1 分支切换
```bash
# 查看当前分支
git branch

# 切换到重构分支
git checkout refactor/architecture-restructure

# 切换到主分支
git checkout main
```

### 2.2 同步主分支更新
```bash
# 在重构分支中同步主分支的最新更改
git checkout refactor/architecture-restructure
git merge main

# 或者使用rebase保持线性历史
git rebase main
```

### 2.3 提交更改
```bash
# 查看更改状态
git status

# 添加文件到暂存区
git add <文件名>
git add .  # 添加所有更改

# 提交更改
git commit -m "描述更改内容"

# 推送到远程仓库
git push origin refactor/architecture-restructure
```

## 三、重构工作流程

### 3.1 阶段一：项目结构重构
**目标**：创建新的模块化项目结构

**操作步骤**：
1. 在重构分支中创建新目录结构
2. 移动和重组现有代码
3. 更新导入路径和依赖关系
4. 提交阶段性更改

**提交示例**：
```bash
git add .
git commit -m "重构：创建新的模块化项目结构
- 分离识别、拟合、交点计算模块
- 统一配置管理
- 更新模块间依赖关系"
```

### 3.2 阶段二：算法实现优化
**目标**：实现详细的算法逻辑

**操作步骤**：
1. 实现眼球球面拟合算法
2. 完善Kappa角标定算法
3. 优化屏幕交点计算
4. 添加算法测试和验证

**提交示例**：
```bash
git add core/fitting/
git commit -m "算法：实现眼球球面拟合算法
- 添加RANSAC + LM优化器
- 实现多种残差类型
- 添加解剖学约束
- 完善置信度计算"
```

### 3.3 阶段三：客户端应用开发
**目标**：将网页前端改为客户端应用

**操作步骤**：
1. 设计PyQt6界面
2. 实现校准流程
3. 添加实时可视化
4. VR眼镜适配

**提交示例**：
```bash
git add client/
git commit -m "客户端：实现PyQt6界面
- 主窗口和校准界面
- 实时注视点显示
- VR眼镜适配
- 系统状态监控"
```

### 3.4 阶段四：集成测试
**目标**：测试和优化整个系统

**操作步骤**：
1. 端到端功能测试
2. 性能优化
3. 错误处理完善
4. 文档更新

**提交示例**：
```bash
git add .
git commit -m "测试：完成系统集成测试
- 端到端功能验证
- 性能优化
- 错误处理完善
- 更新文档"
```

## 四、文件管理规范

### 4.1 文件命名规范
```
# 模块文件
core/recognition/detector.py
core/fitting/sphere_fitter.py
core/intersection/screen_intersection.py

# 配置文件
config/settings.py
config/camera_params.json
config/screen_config.json

# 客户端文件
client/main_window.py
client/calibration_ui.py
client/gaze_visualizer.py

# 文档文件
docs/algorithm_details.md
docs/api_reference.md
docs/user_guide.md
```

### 4.2 提交信息规范
```
格式：<类型>：<简短描述>

类型：
- 重构：项目结构或架构更改
- 算法：算法实现或优化
- 客户端：界面或交互功能
- 配置：配置文件或参数
- 测试：测试代码或验证
- 文档：文档更新或添加
- 修复：错误修复
- 优化：性能优化

示例：
重构：分离识别和拟合模块
算法：实现Kappa角标定算法
客户端：添加实时注视点显示
配置：更新屏幕参数设置
测试：添加单元测试
文档：更新API文档
```

## 五、代码审查要点

### 5.1 代码质量检查
- [ ] 模块职责清晰，单一职责原则
- [ ] 接口设计合理，易于扩展
- [ ] 错误处理完善，异常情况处理
- [ ] 代码注释充分，易于理解
- [ ] 性能优化，避免不必要的计算

### 5.2 算法正确性检查
- [ ] 数学公式实现正确
- [ ] 参数设置合理
- [ ] 边界条件处理
- [ ] 数值稳定性
- [ ] 收敛性验证

### 5.3 集成测试检查
- [ ] 模块间接口兼容
- [ ] 数据流正确
- [ ] 配置参数生效
- [ ] 错误传播合理
- [ ] 性能满足要求

## 六、常见问题解决

### 6.1 合并冲突
```bash
# 查看冲突文件
git status

# 手动解决冲突后
git add <冲突文件>
git commit -m "解决合并冲突"
```

### 6.2 回退更改
```bash
# 回退到上一个提交
git reset --hard HEAD~1

# 回退到指定提交
git reset --hard <提交哈希>

# 回退特定文件
git checkout HEAD -- <文件名>
```

### 6.3 分支同步
```bash
# 从主分支获取最新更改
git fetch origin main
git merge origin/main

# 或者使用rebase
git rebase origin/main
```

## 七、重构检查清单

### 7.1 项目结构检查
- [ ] 模块职责分离清晰
- [ ] 配置文件统一管理
- [ ] 依赖关系合理
- [ ] 接口设计一致

### 7.2 算法实现检查
- [ ] 眼球拟合算法完整
- [ ] Kappa角标定准确
- [ ] 屏幕交点计算正确
- [ ] 参数配置合理

### 7.3 客户端应用检查
- [ ] 界面设计合理
- [ ] 交互流程完整
- [ ] 实时性能良好
- [ ] VR适配功能

### 7.4 文档完整性检查
- [ ] 算法文档详细
- [ ] API文档完整
- [ ] 用户指南清晰
- [ ] 开发文档充分

## 八、最终合并准备

### 8.1 合并前检查
```bash
# 确保重构分支是最新的
git checkout refactor/architecture-restructure
git merge main

# 运行所有测试
python -m pytest tests/

# 检查代码质量
python -m flake8 .
python -m mypy .

# 生成文档
python -m sphinx docs/ docs/_build/
```

### 8.2 合并到主分支
```bash
# 切换到主分支
git checkout main

# 合并重构分支
git merge refactor/architecture-restructure

# 推送更改
git push origin main
```

## 九、注意事项

1. **保持主分支稳定**：主分支应该始终可以正常运行
2. **小步提交**：频繁提交，便于问题定位和回退
3. **详细注释**：代码注释要充分，便于后续维护
4. **测试驱动**：先写测试，再实现功能
5. **文档同步**：代码更改时同步更新文档

---

*本说明文档为重构分支的AI开发提供指导，确保重构过程有序进行。*
