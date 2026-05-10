# 语言规则
1. 所有回复、解释、注释、分析一律使用简体中文。

# 代码修改规则
1. 只做最小必要修改，不重构无关代码。
2. 保留原有风格、命名、注释、结构。
3. 不新增依赖、不改接口、不破坏原有逻辑。
4. 优先复用现有代码与工具函数。
5. 非必要不创建新文件。
6. 修改前先阅读相关上下文代码。

# 工程规则
1. 优先模块化，避免巨型函数/类。
2. 避免硬编码路径、端口、密钥。
3. 使用 pathlib 代替 os.path。
4. 日志与错误信息保持清晰可定位。
5. 优先兼容 Linux/CUDA 环境。

# 项目记忆规则
1. 优先读取 AGENTS.md、SESSION.md、TODO.md。
2. 长期信息写入 MEMORY/，避免依赖长对话上下文。
3. 完成阶段性任务后更新 SESSION.md。
4. 新增 TODO 时保持简洁、可执行。

# 调试规则
1. 先定位根因，再修改代码。
2. 优先输出最小复现与关键日志。
3. 不用“可能是”，尽量给出明确判断依据。

# 命令规则
1. 默认使用 uv/pipenv/已有环境，不主动新建环境。
2. 非必要不安装系统包。
3. 执行命令前检查当前工作目录。

# 确认规则
1. 安全操作自动执行，不打扰用户。
2. 以下操作必须人工确认：删除文件、修改配置/密钥、外网访问、批量修改、高危命令。
3. 执行的关键步骤可视化。

# CleanDiffusion 项目规则
1. 项目定位为面向教学学习的 CleanRL-style 图像生成代码库。
2. 最重要原则：每个端到端算法实现都放在一个 `.py` 文件中，读者打开一个文件即可看到配置、数据、模型、objective、sampler、训练、采样、日志、checkpoint。
3. 推荐文件形态：`clean_diffusion/ddpm.py`、`clean_diffusion/fm.py`、`clean_diffusion/rectified_flow.py`、`clean_diffusion/consistency.py`、`clean_diffusion/cfm.py`、`clean_diffusion/dmd_lite.py`。
4. 代码中必须突出算法相关部分和端到端流程，阅读时要连贯：先说明数学目标，再给关键公式函数，再给 sampler，再给 train/sample 主流程。
5. 工程辅助代码可以存在，但不能淹没算法主线；日志、路径、checkpoint 等工具函数应尽量短，并放在不打断算法阅读的位置。
6. 不使用 `objectives/`、`samplers/`、`training/` 这类深层模块拆分作为默认组织方式；只有当代码重复严重且影响教学阅读时，才允许新增很薄的共享工具文件。
7. 共享代码必须少而透明，优先放在 `clean_diffusion/common.py`，不得形成框架层。
8. 不写巨型抽象框架，不引入复杂 registry，不写过深 inheritance。
9. 算法逻辑优先可读性，允许少量重复换取清晰。
10. 同一框架下对比 DDPM / FM / RF / CT / CFM / DMD，但每次只做一个端到端文件。
11. 每个算法文件必须能独立运行，提供清晰的 `tyro`/CLI 配置和文件顶部教学说明。
12. 统一数学约定：`x0=data/image`，`x1=Gaussian noise`，`t=0 -> noise`，`t=1 -> data`。
13. 统一 path：`x_t = (1 - t) * x1 + t * x0`。
14. Flow Matching velocity：`v_target = x0 - x1`。
15. 禁止在不同 objective 中混用 `x0=noise` 与 `x1=image`。
16. 不允许破坏已有 DDPM baseline。
17. 所有新增端到端算法必须覆盖 train、sample、save、resume 和 smoke test。
