# TODO

## 当前阶段：项目初始化

- [x] 创建 Codex / Claude 协作配套文件。
- [x] 创建项目记忆文件。
- [x] 明确项目采用 CleanRL-style 单文件端到端算法组织。
- [x] 移除默认深层包结构倾向。

## 阶段 0：`clean_diffusion/ddpm.py`

- [x] 创建单文件 DDPM baseline：`clean_diffusion/ddpm.py`。
- [x] 在一个文件内完成配置、数据、模型、DDPM objective、DDPM/DDIM sampler、CFG、训练、采样、checkpoint、日志。
- [x] 调整文件顺序，让数学说明、objective、sampler、训练主流程阅读连贯。
- [x] 突出算法相关代码，避免 checkpoint / 日志等工程辅助淹没主线。
- [x] 支持 TensorBoard、`metrics.jsonl`、sample images、checkpoint metadata。
- [x] 完成 1 step 训练 smoke test。
- [x] 完成 resume smoke test。
- [x] 完成 DDIM 低步数采样 smoke test。
- [x] 在文件顶部写清楚教学说明、数学约定和运行命令。
- [x] 用可正常导入 Torch/Pillow/CLIP 的环境完成运行级验证。

## 后续单文件路线

- [x] 阶段 1：`clean_diffusion/fm.py`。
- [x] 阶段 2：`clean_diffusion/rectified_flow.py`。
- [x] 阶段 3：`clean_diffusion/consistency.py`。
- [x] 阶段 4：`clean_diffusion/cfm.py`。
- [x] 阶段 5：`clean_diffusion/dmd_lite.py`。
- [x] 阶段 6：`clean_diffusion/dmd2.py` DMD2-style experiments。

## 系统学习补充

- [x] 新增 diffusion 系统学习地图：`docs/00_diffusion_learning_map.md`。
- [x] 新增教学实验矩阵：`docs/01_experiment_matrix.md`。
- [x] 新增二维 Flow Matching toy：`clean_diffusion/toy_fm_2d.py`。
- [ ] 补充 DDPM 参数化对比：epsilon / x0 / v / score。
- [ ] 补充 sampler 对比笔记：DDPM / DDIM / Euler / Heun / DPM-Solver-lite。
- [ ] 补充 noise schedule 与 time schedule 教学笔记。
- [ ] 补充模型结构笔记：UNet、time embedding、attention、conditioning。
- [ ] 补充 latent diffusion 教学版本。
- [ ] 补充 evaluation 笔记：FID、CLIP score、precision/recall、人工检查表。
- [ ] 补充失败样例库：记录命令、现象、判断依据和下一步。

## 组织约束

- [ ] 不默认创建 `objectives/`、`samplers/`、`training/` 等深层模块。
- [ ] 如确需共享代码，先评估是否会破坏单文件教学阅读；必要时只创建很薄的 `clean_diffusion/common.py`。
