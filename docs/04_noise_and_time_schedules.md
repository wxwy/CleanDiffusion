# Noise Schedule 与 Time Schedule

Schedule 决定模型在哪些噪声强度上学习，也决定 sampler 如何走过这些噪声强度。

同一个网络、同一个 objective，换 schedule 后训练难度和采样表现都会变。

## 1. DDPM noise schedule

DDPM 中常用：

```text
beta_t
alpha_t = 1 - beta_t
alpha_bar_t = prod(alpha_1 ... alpha_t)
x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
```

`beta_t` 控制每一步加多少噪声，`alpha_bar_t` 控制第 `t` 步总共保留多少数据。

常见选择：

- linear beta：简单、经典、适合 baseline。
- cosine alpha_bar：低噪声区域更平滑，很多图像任务更稳。

学习重点：

- `alpha_bar_t` 接近 1：图像只被轻微污染。
- `alpha_bar_t` 接近 0：几乎纯噪声。
- sampler 和 objective 都依赖同一套 schedule，不能训练和采样各用一套不兼容定义。

## 2. timestep sampling

训练时通常随机采样 timestep：

```text
t ~ Uniform({0, ..., T-1})
```

这意味着模型每个 batch 都看到不同噪声强度。

可研究方向：

- uniform timestep。
- 按 SNR 加权采样。
- 对高噪声或低噪声区域重采样。
- loss reweighting，而不是改变 timestep 分布。

教学阶段建议先保持 uniform，避免 schedule、weight、objective 同时变化。

## 3. SNR 视角

DDPM 的信噪比可以写成：

```text
SNR(t) = alpha_bar_t / (1 - alpha_bar_t)
```

当 `SNR` 高时，`x_t` 仍然像数据；当 `SNR` 低时，`x_t` 更像噪声。

SNR 可以帮助理解：

- 为什么不同 timestep 的 loss 难度不同。
- 为什么 epsilon / x0 / v prediction 的稳定性不同。
- 为什么一些训练方法会做 loss weighting。

## 4. Flow Matching time schedule

CleanDiffusion 的 flow-like 文件统一：

```text
x_t = (1 - t) * x1 + t * x0
t=0 -> noise
t=1 -> data
```

训练时通常：

```text
t ~ Uniform(0, 1)
```

采样时：

```text
t_i = i / steps
dt = 1 / steps
x_{i+1} = x_i + dt * v_theta(x_i, t_i)
```

可研究方向：

- 非均匀 time grid。
- 在接近 data 的区域使用更小步长。
- 在接近 noise 的区域使用更小步长。
- 对 `t` 做 logit / cosine 等重参数化。

第一阶段建议保持 uniform time grid，这样 FM / RF / CFM 的差异主要来自 objective，而不是 schedule。

## 5. DDPM timestep 与 FM time 的区别

容易混淆的一点：

```text
DDPM: t 大通常表示噪声更大
FM:   t=0 是噪声，t=1 是数据
```

因此不要把 DDPM 的 `t/T` 直接当作 FM 的 `t` 使用。

本项目的统一约定只作用于 flow-like 文件：

```text
x0 = data/image
x1 = Gaussian noise
t=0 -> noise
t=1 -> data
```

DDPM 文件保留 DDPM 文献常见 timestep 方向，并在文档中明确标注。

## 6. 实验记录建议

每次改 schedule，记录：

- `beta_schedule` 或 time grid 公式。
- 训练 loss 是否可比。
- sample steps。
- sampler 是否同步修改。
- 固定 seed sample。
- 失败样例。

如果同时改 objective、sampler 和 schedule，很难判断收益来自哪里。
