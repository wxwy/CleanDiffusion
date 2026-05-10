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
- [ ] 阶段 3：`clean_diffusion/consistency.py`。
- [ ] 阶段 4：`clean_diffusion/cfm.py`。
- [ ] 阶段 5：`clean_diffusion/dmd_lite.py`。
- [ ] 阶段 6：DMD2-style experiments。

## 组织约束

- [ ] 不默认创建 `objectives/`、`samplers/`、`training/` 等深层模块。
- [ ] 如确需共享代码，先评估是否会破坏单文件教学阅读；必要时只创建很薄的 `clean_diffusion/common.py`。
