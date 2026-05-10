# 项目长期记忆

## 项目定位

- 项目名：CleanDiffusion。
- 项目路径：CleanDiffusion 仓库根目录。
- 参考项目：本地 CFM 原型项目。
- 目标：将本地 CFM 原型改造成类似 CleanRL 的图像生成研究框架。
- 风格：single-file algorithm clarity、objective-driven research、最小必要 abstraction、易读易改易实验。
- 环境：优先复用现有 Python / CUDA 环境，不主动新建环境。

## 研究目标

在统一图像生成框架下对比：

- DDPM
- DDIM
- CFG
- Flow Matching
- Rectified Flow
- Teacher-free Consistency Training
- Consistency Flow Matching
- DMD-lite
- DMD2-style experiments

核心原则：

- same model
- same dataset
- same training loop
- same logging
- only change objective and sampler

## 数学约定

- `x0 = data/image`
- `x1 = Gaussian noise`
- `t=0 -> noise`
- `t=1 -> data`
- `x_t = (1 - t) * x1 + t * x0`
- `v_target = x0 - x1`

禁止在不同 objective 中混用 `x0=noise`、`x1=image`。

## 当前状态

- 当前为项目初始化阶段。
- 已创建项目架构和协作文件。
- 尚未迁移或实现训练、采样、模型、checkpoint 代码。
- 下一阶段应先做 DDPM baseline 和 `--objective ddpm` 参数化。

## 工程约束

- 不写巨型抽象框架。
- 不引入复杂 registry。
- 不写过深 inheritance。
- 每次只做一个 objective。
- 所有新增 objective 必须覆盖 train、sample、save、resume 和 smoke test。
- 不允许破坏 DDPM baseline。
