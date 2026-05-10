# SESSION

## 2026-05-10

- 初始化 CleanDiffusion 项目目录。
- 创建 CleanDiffusion 推荐架构目录：`clean_diffusion/configs`、`data`、`models`、`objectives`、`samplers`、`training`、`utils`、`scripts`。
- 创建协作配套文件：`AGENTS.md`、`CLAUDE.md`、`SESSION.md`、`TODO.md`、`MEMORY/project.md`、`.codex/README.md`、`.claude/README.md`。
- 当前只做项目骨架与规则沉淀，未实现 DDPM / FM / CFM / DMD 代码。
- 第一阶段建议先做 DDPM baseline 与 `--objective ddpm` 参数化，再逐个加入 FM、Rectified Flow、Consistency、CFM、DMD。
- 已初始化本地 git 仓库，默认分支为 `main`。
- 公开仓库前已执行信息暴露检查，清理本地绝对路径、账号名和认证状态记录。
- 用户明确指出项目最重要部分是每个端到端实现都在一个 `.py` 文件中完成，用于教学学习。
- 用户已删除 `clean_diffusion/` 内原有包骨架；后续不要恢复深层模块结构。
- 已更新项目规则、README、TODO、长期记忆和 Claude 说明，使项目更偏向 CleanRL-style 单文件算法。
- 用户新增要求：代码中必须突出算法相关部分和端到端流程，阅读时要连贯；不能只是把模块机械合并到一个文件。
- 新增 `clean_diffusion/ddpm.py` 单文件 DDPM baseline：包含教学说明、配置、COCO+CLIP 数据、text-conditioned UNet、DDPM objective、CFG、DDPM/DDIM sampler、训练、采样、checkpoint、TensorBoard、`metrics.jsonl` 和 sample image 保存。
- 已完成静态验证：`python -m py_compile clean_diffusion/ddpm.py` 通过；`python clean_diffusion/ddpm.py --help`、`train --help`、`sample --help` 在无 Pillow/Torch 依赖的系统 Python 下可输出轻量帮助。
- 使用现有 CFM 环境完成 DDPM 运行级验证，首次 Torch 导入约 30 秒以上属正常环境开销。
- 完成 `ddpm_smoke` 1 step 训练：`image_size=32`、`batch_size=2`、`base_channels=16`、`channel_mults=1,2,4`、`--no-amp`，step 1 loss 为 `1.2126`，checkpoint 保存到 `checkpoints/ddpm_smoke/step=000000001`。
- 完成 `ddpm_smoke` resume 验证：从 `step=000000001` 继续到 `step=000000002`，成功加载 optimizer、AMP scaler、EMA，step 2 loss 为 `1.0853`。
- 完成 DDIM 低步数采样验证：加载 `checkpoints/ddpm_smoke/step=000000002` EMA，`steps=4`、`CFG=3.0`、prompt 为 `a small red car`，图片保存到 `samples/ddpm_smoke/ddim_cfg3.0_steps4_000_a_small_red_car.png`，尺寸为 `32x32`。
- 已确认 `checkpoints/ddpm_smoke/metrics.jsonl`、TensorBoard events、checkpoint `metadata.json` 均落盘。
- 新增 `clean_diffusion/fm.py` 单文件 Flow Matching baseline：保持同一 COCO+CLIP 数据和 text-conditioned UNet，核心算法为 `x0=data`、`x1=noise`、`x_t=(1-t)*x1+t*x0`、`v_target=x0-x1`，采样使用 Euler ODE 从 noise 积分到 data。
- 完成 `fm_smoke` 1 step 训练：`image_size=32`、`batch_size=2`、`base_channels=16`、`channel_mults=1,2,4`、`--no-amp`，step 1 loss 为 `1.3920`，checkpoint 保存到 `checkpoints/fm_smoke/step=000000001`。
- 完成 `fm_smoke` resume 验证：从 `step=000000001` 继续到 `step=000000002`，成功加载 optimizer、AMP scaler、EMA，step 2 loss 为 `1.2235`。
- 完成 FM Euler 低步数采样验证：加载 `checkpoints/fm_smoke/step=000000002` EMA，`steps=4`、`CFG=3.0`、prompt 为 `a small red car`，图片保存到 `samples/fm_smoke/euler_cfg3.0_steps4_000_a_small_red_car.png`，尺寸为 `32x32`。
- 新增 `clean_diffusion/rectified_flow.py` 单文件 Rectified Flow baseline：第一版与 FM 使用相同直线路径速度场 loss，保留独立 `rectified_flow` objective 记录和 `rf_euler` 采样入口，后续可扩展 reflow / straightening。
- 完成 `rf_smoke` 1 step 训练：step 1 loss 为 `1.3920`，checkpoint 保存到 `checkpoints/rf_smoke/step=000000001`。
- 完成 `rf_smoke` resume 验证：从 `step=000000001` 继续到 `step=000000002`，成功加载 optimizer、AMP scaler、EMA，step 2 loss 为 `1.2235`。
- 完成 RF Euler 低步数采样验证：加载 `checkpoints/rf_smoke/step=000000002` EMA，`steps=4`、`CFG=3.0`、prompt 为 `a small red car`，图片保存到 `samples/rf_smoke/rf_euler_cfg3.0_steps4_000_a_small_red_car.png`，尺寸为 `32x32`。
- 新增 `clean_diffusion/consistency.py` 单文件 teacher-free consistency baseline：模型直接预测 endpoint `x0_hat`，同一条直线路径上不同时间的 endpoint 输出保持一致，并加 `f_theta(x0,t=1)≈x0` 数据端边界；EMA 是当前模型滑动平均目标，不是外部 pretrained teacher。
- 完成 `consistency_smoke` 1 step 训练：step 1 loss 为 `0.4633`，checkpoint 保存到 `checkpoints/consistency_smoke/step=000000001`。
- 完成 `consistency_smoke` resume 验证：从 `step=000000001` 继续到 `step=000000002`，成功加载 optimizer、AMP scaler、EMA，step 2 loss 为 `0.4024`。
- 完成 consistency one-step 采样验证：加载 `checkpoints/consistency_smoke/step=000000002` EMA，`CFG=3.0`、prompt 为 `a small red car`，图片保存到 `samples/consistency_smoke/consistency_cfg3.0_steps1_000_a_small_red_car.png`，尺寸为 `32x32`。
- 新增 `clean_diffusion/cfm.py` 单文件 Consistency Flow Matching baseline：模型主输出仍为 velocity，endpoint 由 `x0_hat=x_t+(1-t)*v_theta(x_t,t)` 推导，loss 为 velocity MSE + endpoint consistency + boundary。
- 完成 `cfm_smoke` 1 step 训练：step 1 loss 为 `1.4079`，checkpoint 保存到 `checkpoints/cfm_smoke/step=000000001`。
- 完成 `cfm_smoke` resume 验证：从 `step=000000001` 继续到 `step=000000002`，成功加载 optimizer、AMP scaler、EMA，step 2 loss 为 `1.2961`。
- 完成 CFM Euler 低步数采样验证：加载 `checkpoints/cfm_smoke/step=000000002` EMA，`steps=4`、`CFG=3.0`、prompt 为 `a small red car`，图片保存到 `samples/cfm_smoke/cfm_cfg3.0_steps4_000_a_small_red_car.png`，尺寸为 `32x32`。
- 新增 `clean_diffusion/dmd_lite.py` 单文件 DMD-lite baseline：student 从 Gaussian noise 一步预测 endpoint，target 来自已有 FM/RF/CFM teacher checkpoint 的多步 Euler 生成结果；当前不实现 adversarial distribution matching。
- 使用 `checkpoints/cfm_smoke` 作为 teacher 完成 `dmd_lite_smoke` 1 step 训练：step 1 loss 为 `1.0049`，checkpoint 保存到 `checkpoints/dmd_lite_smoke/step=000000001`。
- 完成 `dmd_lite_smoke` resume 验证：从 `step=000000001` 继续到 `step=000000002`，成功加载 teacher EMA、student optimizer、AMP scaler、student EMA，step 2 loss 为 `1.0234`。
- 完成 DMD-lite one-step 采样验证：加载 `checkpoints/dmd_lite_smoke/step=000000002` EMA，`CFG=3.0`、prompt 为 `a small red car`，图片保存到 `samples/dmd_lite_smoke/dmd_lite_cfg3.0_steps1_000_a_small_red_car.png`，尺寸为 `32x32`。
- 新增 `clean_diffusion/dmd2.py` 单文件 DMD2-style 教学实验：在 DMD-lite 基础上加入 fake score model、PatchDiscriminator、score-delta surrogate loss 和 GAN proxy；这是教学骨架，不声明完整论文复现。
- 使用 `checkpoints/cfm_smoke` 作为 teacher 完成 `dmd2_smoke` 1 step 训练：step 1 loss 为 `0.0699`，checkpoint 保存到 `checkpoints/dmd2_smoke/step=000000001`。
- 完成 `dmd2_smoke` resume 验证：从 `step=000000001` 继续到 `step=000000002`，成功加载 teacher EMA、student optimizer、AMP scaler、student EMA、fake_score_model、discriminator、fake_optimizer、discriminator_optimizer，step 2 loss 为 `0.0706`。
- 完成 DMD2-style one-step 采样验证：加载 `checkpoints/dmd2_smoke/step=000000002` EMA，`CFG=3.0`、prompt 为 `a small red car`，图片保存到 `samples/dmd2_smoke/dmd2_cfg3.0_steps1_000_a_small_red_car.png`，尺寸为 `32x32`。
- 阶段 0 到阶段 6 的规划单文件算法均已实现并完成 smoke test：DDPM、FM、Rectified Flow、Teacher-free Consistency、CFM、DMD-lite、DMD2-style。
- 从系统学习 diffusion 的角度新增第一批教学补充：`docs/00_diffusion_learning_map.md`、`docs/01_experiment_matrix.md`、`clean_diffusion/toy_fm_2d.py`。
- `docs/00_diffusion_learning_map.md` 用学习顺序串联 DDPM、FM、RF、Consistency、CFM、DMD-lite、DMD2-style，强调先看 2D toy，再读图像版单文件算法。
- `docs/01_experiment_matrix.md` 记录最小可运行、采样器对比、CFG 对比、学习曲线对比和失败模式记录模板。
- `clean_diffusion/toy_fm_2d.py` 是二维 Flow Matching 端到端教学脚本，包含配置、数据分布、MLP、FM objective、Euler sampler、train/sample、checkpoint 和可视化。
- 完成 `toy_fm_2d.py` 静态检查：`python -m py_compile clean_diffusion/toy_fm_2d.py` 通过。
- 完成 `toy_fm_2d_smoke` 训练验证：`max_steps=2`、`batch_size=64`、`hidden_dim=32`、`depth=2`，step 1 loss 为 `2.493605`，step 2 loss 为 `2.739242`。
- 完成 `toy_fm_2d_smoke` resume 验证：从 `outputs/toy_fm_2d_smoke/last.pt` 恢复到 step 3，step 3 loss 为 `2.468290`。
- 完成 `toy_fm_2d.py` sample 验证：加载 `outputs/toy_fm_2d_smoke/last.pt`，`steps=4`、`num_points=128`，输出保存到 `outputs/toy_fm_2d_smoke/sample_check`。

## 下次会话入口

1. 先读取 `AGENTS.md`、`SESSION.md`、`TODO.md`。
2. 长期项目事实查看 `MEMORY/project.md`。
3. 代码改动前检查当前文件结构与 git 状态。
4. 后续学习补充优先保持文档或单文件 toy 形态，不恢复深层框架结构。
5. 不要默认创建 `objectives/`、`samplers/`、`training/` 等深层包结构。
