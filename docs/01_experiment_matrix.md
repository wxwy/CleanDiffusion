# 实验矩阵

CleanDiffusion 的核心比较方式是：

```text
same model
same dataset
same logging
only change objective and sampler
```

这份矩阵用于安排教学实验，不追求大规模 benchmark。

## 第一组：最小可运行

目标：确认每个文件能 train / resume / sample / save。

| 文件 | objective | sampler | 训练步数 | 采样步数 | 目的 |
| --- | --- | --- | --- | --- | --- |
| `ddpm.py` | noise prediction | DDIM | 1-2 | 4 | 验证 DDPM baseline |
| `fm.py` | velocity | Euler | 1-2 | 4 | 验证 flow path |
| `rectified_flow.py` | velocity | RF Euler | 1-2 | 4 | 验证 RF 入口 |
| `consistency.py` | endpoint consistency | one-step | 1-2 | 1 | 验证 endpoint 预测 |
| `cfm.py` | velocity + endpoint consistency | CFM Euler | 1-2 | 4 | 验证组合目标 |
| `dmd_lite.py` | teacher endpoint distill | one-step | 1-2 | 1 | 验证 teacher-student |
| `dmd2.py` | DMD2-style surrogate | one-step | 1-2 | 1 | 验证教学骨架 |

## 第二组：采样器对比

目标：固定 checkpoint，比较 sampler 和 step 数。

| checkpoint | sampler | steps | 观察项 |
| --- | --- | --- | --- |
| DDPM | DDPM | 10 / 50 / 100 | 随机采样质量和耗时 |
| DDPM | DDIM | 4 / 10 / 50 | 确定性采样和低步数退化 |
| FM | Euler | 4 / 10 / 50 | ODE 积分步数影响 |
| RF | RF Euler | 1 / 4 / 10 | 直线化目标的少步潜力 |
| CFM | CFM Euler | 1 / 4 / 10 | endpoint consistency 对少步的影响 |
| Consistency | one-step | 1 | 一步模型的稳定性 |

## 第三组：conditioning 对比

目标：理解 CFG 的作用和副作用。

固定 prompt，例如：

```text
a small red car
a blue bird on a branch
a photo of a dog
```

对每个 prompt 运行：

```text
guidance_scale = 0.0 / 1.0 / 3.0 / 5.0 / 7.5
```

观察：

- prompt 对齐是否增强。
- 颜色或形状是否过饱和。
- 是否出现模式坍缩。
- 不同 objective 对 CFG 是否同样敏感。

## 第四组：学习曲线对比

目标：同一训练预算下比较 objective。

建议固定：

```text
image_size = 64
base_channels = 64
batch_size = 根据显存设置
max_steps = 1000 / 5000 / 20000
```

记录：

- loss 曲线。
- 固定 seed sample。
- 固定 prompt sample。
- checkpoint metadata。
- 采样耗时。

注意：不同 objective 的 loss 数值不可直接横向比较，只能看同一 objective 内的收敛趋势和最终样本。

## 第五组：失败模式记录

每次实验失败时，优先记录为可复现条目：

```text
文件：
命令：
checkpoint：
现象：
判断依据：
下一步：
```

常见失败模式：

- loss 正常下降但图片发灰：采样 schedule 或归一化可能有问题。
- CFG 变大后图片崩：条件/无条件预测差异过大。
- FM/RF 采样发散：Euler step 太少或 velocity scale 异常。
- Consistency 输出过平滑：endpoint loss 权重或 EMA target 太强。
- DMD student 坍缩：teacher target 太弱或 one-step 目标过硬。
