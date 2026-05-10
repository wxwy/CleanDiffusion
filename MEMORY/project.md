# 项目长期记忆

## 项目定位

- 项目名：CleanDiffusion。
- 项目路径：CleanDiffusion 仓库根目录。
- 参考项目：本地 CFM 原型项目。
- 目标：将本地 CFM 原型改造成类似 CleanRL 的教学学习型图像生成代码库。
- 风格：端到端单文件算法、objective-driven research、最小必要 abstraction、易读易改易实验。
- 环境：优先复用现有 Python / CUDA 环境，不主动新建环境。
- 最重要组织原则：每个算法用一个 `.py` 文件完成端到端实现，例如 `clean_diffusion/ddpm.py`、`clean_diffusion/fm.py`。
- 单文件不是简单合并代码，必须突出算法相关部分和端到端流程，阅读时从数学目标、objective、sampler 到训练/采样主流程保持连贯。
- checkpoint、日志、路径等工程辅助代码应短小透明，不能淹没算法主线。
- 不默认使用 `objectives/`、`samplers/`、`training/`、`models/` 等深层包结构；只有必要共享工具才放入很薄的 `clean_diffusion/common.py`。

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
- 已创建协作文件和项目记忆。
- 用户已删除 `clean_diffusion/` 内原有包骨架，后续不要恢复深层包结构。
- 已新增 `clean_diffusion/ddpm.py` 单文件 DDPM baseline，代码内包含训练、采样、模型、DDPM objective、DDPM/DDIM sampler、CFG、checkpoint 和日志。
- `clean_diffusion/ddpm.py` 已通过静态语法检查、轻量 CLI help 检查、1 step 训练、resume 和低步数 DDIM 采样 smoke test。
- DDPM smoke test 使用现有 CFM 环境运行，首次 Torch 导入约 30 秒以上属正常环境开销。
- `ddpm_smoke` 验证产物：`checkpoints/ddpm_smoke/step=000000002`、`checkpoints/ddpm_smoke/metrics.jsonl`、TensorBoard events、`samples/ddpm_smoke/ddim_cfg3.0_steps4_000_a_small_red_car.png`。
- 已新增 `clean_diffusion/fm.py` 单文件 Flow Matching baseline，使用统一约定 `x0=data`、`x1=noise`、`t=0->noise`、`t=1->data`、`v_target=x0-x1`。
- `clean_diffusion/fm.py` 已通过静态语法检查、轻量 CLI help 检查、1 step 训练、resume 和 4 step Euler 采样 smoke test。
- `fm_smoke` 验证产物：`checkpoints/fm_smoke/step=000000002`、`checkpoints/fm_smoke/metrics.jsonl`、TensorBoard events、`samples/fm_smoke/euler_cfg3.0_steps4_000_a_small_red_car.png`。
- 已新增 `clean_diffusion/rectified_flow.py` 单文件 Rectified Flow baseline，第一版复用 FM 直线路径速度场目标，并保留独立 `rf_euler` 采样入口。
- `clean_diffusion/rectified_flow.py` 已通过静态语法检查、轻量 CLI help 检查、1 step 训练、resume 和 4 step RF Euler 采样 smoke test。
- `rf_smoke` 验证产物：`checkpoints/rf_smoke/step=000000002`、`checkpoints/rf_smoke/metrics.jsonl`、TensorBoard events、`samples/rf_smoke/rf_euler_cfg3.0_steps4_000_a_small_red_car.png`。
- 已新增 `clean_diffusion/consistency.py` 单文件 teacher-free consistency baseline，模型直接预测 endpoint `x0_hat`，EMA 仅作为当前模型滑动平均 target network。
- `clean_diffusion/consistency.py` 已通过静态语法检查、轻量 CLI help 检查、1 step 训练、resume 和 one-step consistency 采样 smoke test。
- `consistency_smoke` 验证产物：`checkpoints/consistency_smoke/step=000000002`、`checkpoints/consistency_smoke/metrics.jsonl`、TensorBoard events、`samples/consistency_smoke/consistency_cfg3.0_steps1_000_a_small_red_car.png`。
- 已新增 `clean_diffusion/cfm.py` 单文件 Consistency Flow Matching baseline，模型主输出为 velocity，并通过 velocity 推导 endpoint 做 consistency。
- `clean_diffusion/cfm.py` 已通过静态语法检查、轻量 CLI help 检查、1 step 训练、resume 和 4 step CFM Euler 采样 smoke test。
- `cfm_smoke` 验证产物：`checkpoints/cfm_smoke/step=000000002`、`checkpoints/cfm_smoke/metrics.jsonl`、TensorBoard events、`samples/cfm_smoke/cfm_cfg3.0_steps4_000_a_small_red_car.png`。

## 工程约束

- 不写巨型抽象框架。
- 不引入复杂 registry。
- 不写过深 inheritance。
- 每次只做一个端到端算法文件。
- 所有新增算法文件必须覆盖 train、sample、save、resume 和 smoke test。
- 不允许破坏 DDPM baseline。
