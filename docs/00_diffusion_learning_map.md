# Diffusion 系统学习地图

这份文档回答一个教学问题：已经有 DDPM / FM / RF / Consistency / CFM / DMD 单文件实现后，学习者应该按什么顺序理解 diffusion。

CleanDiffusion 的学习顺序不按论文时间线，而按“生成模型需要解决什么问题”来组织：

1. 如何把噪声变成数据。
2. 如何定义训练目标。
3. 如何把训练目标变成 sampler。
4. 如何用 conditioning 控制生成。
5. 如何减少采样步数。
6. 如何比较 objective 和 sampler 的差异。

## 0. 统一记号

本项目所有 flow-like 文件统一使用：

```text
x0 = data/image
x1 = Gaussian noise
t=0 -> noise
t=1 -> data
x_t = (1 - t) * x1 + t * x0
v_target = x0 - x1
```

DDPM 文件保留 diffusion 文献中常见的 noise prediction 训练目标，但学习时要注意它和 Flow Matching 的方向约定不同：DDPM 学的是“加噪过程中的噪声”，FM/RF/CFM 学的是“从噪声走向数据的速度”。

## 1. 先学 2D toy

推荐先运行：

```bash
python clean_diffusion/toy_fm_2d.py train --run-name toy_fm_2d_smoke --max-steps 200
python clean_diffusion/toy_fm_2d.py sample --ckpt outputs/toy_fm_2d_smoke/last.pt
```

2D toy 的价值是把图像生成中不容易看见的对象画出来：

- `x1`：标准高斯噪声点。
- `x0`：二维 mixture 数据点。
- `x_t`：噪声到数据之间的直线路径。
- `v_theta(x_t,t)`：模型学习的速度场。
- Euler sampler：从 `t=0` 积分到 `t=1` 的轨迹。

在 2D 上看懂这些，再读 `clean_diffusion/fm.py` 会更顺。

## 2. DDPM：从离散加噪开始

对应文件：

```text
clean_diffusion/ddpm.py
docs/02_ddpm_parameterizations.md
clean_diffusion/toy_ddpm_parameterizations.py
```

学习重点：

- forward process：把图像逐步加噪。
- reverse process：模型预测噪声并逐步去噪。
- objective：`MSE(noise_pred, noise)`。
- parameterization：epsilon / x0 / v / score 可以在同一个加噪公式下互相转换。
- sampler：DDPM 随机采样和 DDIM 确定性采样。
- CFG：条件预测和无条件预测的线性外推。

建议先只看三个函数区域：

1. beta / alpha schedule。
2. DDPM objective。
3. DDPM / DDIM sampler。

然后运行：

```bash
python clean_diffusion/toy_ddpm_parameterizations.py
```

确认模型输出为 epsilon、x0、v 或 score 时，如何统一换回 sampler 需要的 `eps_hat` / `x0_hat`。

## 3. Flow Matching：从路径和速度理解生成

对应文件：

```text
clean_diffusion/fm.py
```

学习重点：

- 不再预测噪声，而是预测速度。
- 训练路径是直线：`x_t = (1 - t) * x1 + t * x0`。
- 速度目标是常量：`v_target = x0 - x1`。
- sampler 是 ODE Euler 积分。

FM 是后续 RF、CFM、DMD 的共同基础。

## 4. Rectified Flow：把路径变直

对应文件：

```text
clean_diffusion/rectified_flow.py
```

当前第一版和 FM 使用相同 loss，学习重点不是代码差异，而是概念差异：

- FM：给定路径，学习速度场。
- RF：追求更直、更容易一步或少步采样的生成轨迹。
- 后续可加入 reflow / straightening。

## 5. Consistency：直接对齐 endpoint

对应文件：

```text
clean_diffusion/consistency.py
```

学习重点：

- 模型直接预测 `x0_hat`。
- 同一条 trajectory 上不同时间的输出应该一致。
- EMA target network 只是当前模型的滑动平均，不是外部 pretrained teacher。
- one-step sampling 为什么可行，以及它为什么更难训练稳定。

## 6. CFM：速度学习 + endpoint 一致性

对应文件：

```text
clean_diffusion/cfm.py
```

学习重点：

- 主输出仍然是 velocity。
- endpoint 由 velocity 推导：

```text
x0_hat = x_t + (1 - t) * v_theta(x_t, t)
```

- loss 同时包含 velocity MSE 和 endpoint consistency。
- 它把 FM 的可解释速度场和 Consistency 的少步倾向接起来。

## 7. DMD-lite / DMD2-style：从 teacher 到 one-step student

对应文件：

```text
clean_diffusion/dmd_lite.py
clean_diffusion/dmd2.py
```

学习重点：

- student 从噪声一步生成。
- DMD-lite 先学习已有 FM/RF/CFM teacher 的生成结果。
- DMD2-style 文件是教学实验骨架，展示 fake score model、discriminator、score-delta surrogate loss 如何放进同一个端到端流程。
- 当前代码不声明完整复现论文。

## 8. 学习时应该记录什么

每个实验至少记录：

- objective / sampler。
- 训练 loss 曲线。
- 固定 prompt 的 sample grid。
- guidance scale。
- sample steps。
- checkpoint metadata。
- 失败样例和失败原因。

如果只看最终图片，很难理解 diffusion；如果同时看路径、速度、loss、采样步数和失败模式，学习效率会高很多。

## 9. 补充主题索引

完成上面的主线后，按下面顺序补齐系统视角：

```text
docs/03_sampler_comparison.md
docs/04_noise_and_time_schedules.md
docs/05_model_conditioning_notes.md
docs/06_latent_diffusion_teaching_plan.md
docs/07_evaluation_notes.md
docs/08_failure_case_log.md
```

推荐阅读顺序：

1. 先读 sampler，对比“模型预测”和“数值更新”是两件事。
2. 再读 schedule，理解时间坐标和噪声强度如何影响训练。
3. 再读模型结构，确认 UNet / time embedding / conditioning 在数据流中的位置。
4. 再读 latent diffusion，理解为什么工业图像生成常在 latent 空间做。
5. 最后读 evaluation 和 failure case，把实验记录从“看图”升级成可复现研究。
