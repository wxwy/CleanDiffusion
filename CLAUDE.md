# CleanDiffusion Claude 协作说明

本文件供 Claude / 类 Claude 工具读取。项目权威协作规则以 `AGENTS.md` 为准。

## 项目定位

CleanDiffusion 是一个 research-friendly 的 diffusion / flow / consistency 图像生成框架，目标是像 CleanRL 一样用尽量清晰的单文件算法组织方式支持 objective-driven experiments。

## 当前阶段

当前为项目初始化阶段，仅创建项目架构和配套协作文件。不要直接实现 FM / CFM / DMD。

## 工作方式

1. 修改前先读取 `AGENTS.md`、`SESSION.md`、`TODO.md`。
2. 长期事实写入 `MEMORY/project.md`。
3. 阶段性进展写入 `SESSION.md`。
4. 优先最小修改，不做无关重构。
5. 所有回复、解释、注释使用简体中文。

## 设计边界

- 允许少量代码重复，换取算法清晰度。
- 不引入复杂 registry。
- 不做企业级训练框架。
- 每个 objective / sampler 尽量单文件、易读、易改。
