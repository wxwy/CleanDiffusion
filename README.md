# CleanDiffusion

CleanDiffusion 是一个面向图像生成研究的 diffusion / flow / consistency 框架，目标是用类似 CleanRL 的方式组织算法：少量抽象、单文件算法清晰度、易读、易改、易实验。

当前仓库处于项目初始化阶段，只包含目录结构和协作规则。第一阶段将迁移并参数化 DDPM baseline。

## 设计目标

- single-file algorithm clarity
- objective-driven research
- same model / same dataset / same training loop / same logging
- only change objective and sampler
- 最小必要 abstraction
- 快速 prototype 新 objective / sampler

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

## 目录结构

```text
clean_diffusion/
  configs/
  data/
  models/
  objectives/
  samplers/
  training/
  utils/
  scripts/
MEMORY/
AGENTS.md
CLAUDE.md
SESSION.md
TODO.md
```

## 当前阶段

阶段 0：DDPM + DDIM + CFG baseline。

下一步实现范围：

- `--objective ddpm`
- `--sample-scheduler ddpm|ddim`
- TensorBoard
- `metrics.jsonl`
- sample images
- checkpoint metadata
- 训练、采样、保存、恢复 smoke test
