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

## 下次会话入口

1. 先读取 `AGENTS.md`、`SESSION.md`、`TODO.md`。
2. 长期项目事实查看 `MEMORY/project.md`。
3. 代码改动前检查当前文件结构与 git 状态。
4. 第一阶段不要直接进入 FM/CFM，实现范围控制在 `clean_diffusion/ddpm.py` 单文件 DDPM baseline。
5. 不要默认创建 `objectives/`、`samplers/`、`training/` 等深层包结构。
