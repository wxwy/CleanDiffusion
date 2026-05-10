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
1. 项目定位为类似 CleanRL 的图像生成研究框架，优先 single-file algorithm clarity。
2. 不写巨型抽象框架，不引入复杂 registry，不写过深 inheritance。
3. 算法逻辑优先可读性，允许少量重复换取清晰。
4. 所有 objective 尽量集中在 `clean_diffusion/objectives/` 的单文件中。
5. 所有 sampler 尽量集中在 `clean_diffusion/samplers/` 的单文件中。
6. 同一框架下对比 DDPM / FM / RF / CT / CFM / DMD，但每次只做一个 objective。
7. 统一数学约定：`x0=data/image`，`x1=Gaussian noise`，`t=0 -> noise`，`t=1 -> data`。
8. 统一 path：`x_t = (1 - t) * x1 + t * x0`。
9. Flow Matching velocity：`v_target = x0 - x1`。
10. 禁止在不同 objective 中混用 `x0=noise` 与 `x1=image`。
11. 不允许破坏已有 DDPM baseline。
12. 所有新增 objective 必须覆盖 train、sample、save、resume 和 smoke test。
