# CleanDiffusion

CleanDiffusion 是一个面向教学学习的图像生成研究代码库，目标是用 CleanRL 的方式组织 diffusion / flow / consistency 算法：每个端到端算法尽量就是一个 `.py` 文件。

本项目不是企业级框架，也不追求复杂抽象。读者应该能够打开一个算法文件，从上到下看到：

- 配置
- 数据加载
- 模型
- objective
- sampler
- CFG
- 训练循环
- 采样入口
- checkpoint
- TensorBoard / `metrics.jsonl`

代码组织还必须服务阅读连贯性。单文件不是简单堆代码，而是让算法主线清楚可见：

1. 文件顶部说明算法目标、数学约定、运行命令。
2. 配置和少量工具函数先出现。
3. 数据和模型保持简洁，不抢算法主线。
4. objective / sampler / CFG 是文件中最突出的部分。
5. 训练循环按真实端到端流程书写，让读者能顺着数据流读完。
6. checkpoint、日志、图片保存等工程辅助放在不打断算法理解的位置。

## 核心原则

- 单文件端到端实现优先。
- 算法相关代码和端到端流程必须突出。
- 代码阅读顺序必须连贯，服务教学理解。
- 教学可读性优先于代码复用率。
- 允许少量重复，避免深层抽象。
- 不引入复杂 registry。
- 不写过深 inheritance。
- same model / same dataset / same logging。
- 对比实验时只改变 objective 和 sampler。

## 推荐文件结构

```text
clean_diffusion/
  ddpm.py              # DDPM + DDIM + CFG 端到端
  fm.py                # Flow Matching 端到端
  rectified_flow.py    # Rectified Flow 端到端
  consistency.py       # Teacher-free Consistency 端到端
  cfm.py               # Consistency Flow Matching 端到端
  dmd_lite.py          # DMD-lite 端到端
  common.py            # 可选：极少量共享工具
MEMORY/
AGENTS.md
CLAUDE.md
SESSION.md
TODO.md
```

不推荐默认拆成：

```text
objectives/
samplers/
training/
models/
configs/
```

这些拆分会降低教学阅读时的端到端连续性。只有当重复代码已经明显妨碍阅读时，才把少量工具放入 `common.py`。

## 计划支持

- DDPM
- DDIM
- CFG
- Flow Matching
- Rectified Flow
- Teacher-free Consistency Training
- Consistency Flow Matching
- DMD-lite
- DMD2-style experiments

## 数学约定

全项目统一：

```text
x0 = data/image
x1 = Gaussian noise
t=0 -> noise
t=1 -> data
x_t = (1 - t) * x1 + t * x0
v_target = x0 - x1
```

禁止在不同算法文件中混用 `x0=noise`、`x1=image`。

## 当前阶段

阶段 0：实现 `clean_diffusion/ddpm.py`。

`ddpm.py` 必须包含：

- DDPM noise prediction objective
- DDPM / DDIM sampler
- CFG
- 训练
- 采样
- 保存
- resume
- TensorBoard
- `metrics.jsonl`
- sample images
- checkpoint metadata

完成阶段 0 前，不进入 FM / CFM / DMD。

## 当前进展

- `clean_diffusion/ddpm.py`：已完成 DDPM + DDIM + CFG 单文件 baseline，并通过 train / resume / sample smoke test。
- `clean_diffusion/fm.py`：已完成 Flow Matching + Euler + CFG 单文件 baseline，并通过 train / resume / sample smoke test。
- `clean_diffusion/rectified_flow.py`：已完成 Rectified Flow + RF Euler + CFG 单文件 baseline，并通过 train / resume / sample smoke test。
- `clean_diffusion/consistency.py`：已完成 teacher-free consistency + one-step CFG 单文件 baseline，并通过 train / resume / sample smoke test。
