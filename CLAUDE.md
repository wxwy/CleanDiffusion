# CleanDiffusion Claude 协作说明

本文件供 Claude / 类 Claude 工具读取。项目权威协作规则以 `AGENTS.md` 为准。

## 项目定位

CleanDiffusion 是一个面向教学学习的 diffusion / flow / consistency 图像生成代码库。项目目标是像 CleanRL 一样，让每个算法都以一个端到端 `.py` 文件呈现，便于阅读、修改、实验和课堂讲解。

## 当前阶段

当前为项目初始化阶段，仅保留配套协作文件和项目说明。`clean_diffusion/` 下不预设深层包结构，后续按 `clean_diffusion/ddpm.py`、`clean_diffusion/fm.py` 这类单文件算法逐步加入。不要直接实现 FM / CFM / DMD。

## 工作方式

1. 修改前先读取 `AGENTS.md`、`SESSION.md`、`TODO.md`。
2. 长期事实写入 `MEMORY/project.md`。
3. 阶段性进展写入 `SESSION.md`。
4. 优先最小修改，不做无关重构。
5. 所有回复、解释、注释使用简体中文。

## 设计边界

- 每个端到端算法优先单文件完成。
- 单文件内必须突出算法相关代码和端到端流程，阅读顺序要连贯。
- objective、sampler、CFG、训练主循环应成为文件主线，工程辅助函数不能淹没算法。
- 允许少量代码重复，换取算法清晰度和教学可读性。
- 不引入复杂 registry。
- 不做企业级训练框架。
- 不默认拆成 `objectives/`、`samplers/`、`training/` 等深层模块。
- 必要共享工具最多放入很薄的 `clean_diffusion/common.py`。
