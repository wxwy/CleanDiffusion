# Sampler 对比笔记

Sampler 回答的问题是：模型已经学会某种局部预测后，如何从噪声走到数据。

在 CleanDiffusion 中，sampler 不应该被理解成工程插件，而应该被理解成 objective 的配套数值方法。

## 1. DDPM sampler

对应文件：

```text
clean_diffusion/ddpm.py
```

核心形式：

```text
x_t -> x_{t-1}
```

DDPM sampler 是 ancestral sampling：每一步用模型预测噪声，再根据后验均值和方差采样上一步。

特点：

- 随机性强。
- 采样步数通常较多。
- 和 epsilon prediction objective 连接直接。
- 适合作为最基础 baseline。

学习重点：

- `beta_t`、`alpha_t`、`alpha_bar_t` 如何进入后验均值。
- 最后一步不再加噪声。
- 同一个 checkpoint 多次采样结果可以不同。

## 2. DDIM sampler

对应文件：

```text
clean_diffusion/ddpm.py
```

DDIM 把 DDPM reverse process 改写成可跳步的更新：

```text
x_t -> x_{t_prev}
```

特点：

- 可以用较少步数采样。
- `eta=0` 时是确定性采样。
- 常用于快速观察 DDPM checkpoint。

学习重点：

- 先从 `eps_hat` 还原 `x0_hat`。
- 再由 `x0_hat` 和 `eps_hat` 合成下一个 timestep。
- `eta` 控制是否重新引入随机噪声。

## 3. Euler sampler

对应文件：

```text
clean_diffusion/fm.py
clean_diffusion/cfm.py
clean_diffusion/toy_fm_2d.py
```

Flow Matching 使用 ODE 视角：

```text
dx/dt = v_theta(x_t, t)
x_{t+dt} = x_t + dt * v_theta(x_t, t)
```

特点：

- 从 `t=0` 的 noise 积分到 `t=1` 的 data。
- 公式直观，适合教学。
- 误差主要来自步长和速度场质量。

学习重点：

- `steps` 越少，数值误差越大。
- velocity scale 异常会直接造成采样发散。
- Euler 是一阶方法，不会自动修正局部误差。

## 4. RF Euler sampler

对应文件：

```text
clean_diffusion/rectified_flow.py
```

RF Euler 第一版和 FM Euler 形式接近，但研究目标不同：

- FM：学习给定路径上的速度。
- RF：希望路径更直，少步甚至一步采样更容易。

学习时先把 RF Euler 看作 FM Euler 的对照入口，后续再加入 reflow / straightening。

## 5. Heun sampler

Heun 是二阶 ODE 方法，可以理解成 predictor-corrector：

```text
v1 = v_theta(x_t, t)
x_pred = x_t + dt * v1
v2 = v_theta(x_pred, t + dt)
x_next = x_t + dt * (v1 + v2) / 2
```

特点：

- 比 Euler 多一次模型调用。
- 通常局部误差更小。
- 在相同步数下可能更稳，但采样耗时约翻倍。

CleanDiffusion 当前还没有默认加入 Heun，是因为第一阶段优先保证每个算法主线清晰。后续如果加入，建议作为 `toy_fm_2d.py` 或独立教学脚本先实现，再决定是否进入图像版 FM。

## 6. DPM-Solver-lite

DPM-Solver 系列利用 diffusion ODE 的特殊结构做高阶快速采样。

教学上可以先理解成：

- 它不是简单 Euler。
- 它利用噪声 schedule 的解析形式。
- 它通常为 DDPM/score-style 模型设计。
- 它适合低步数高质量采样，但公式复杂度明显高于 DDIM。

CleanDiffusion 如果加入 DPM-Solver-lite，应放在独立教学文件或 DDPM sampler 对比文件中，避免把 `ddpm.py` baseline 变成难读的大文件。

## 7. 对比实验建议

固定 checkpoint 和 prompt，比较：

```text
sampler = DDPM / DDIM / Euler / Heun
steps = 1 / 4 / 10 / 50 / 100
guidance_scale = 0 / 3 / 5 / 7.5
```

记录：

- 单张采样耗时。
- 同 seed 结果。
- 多 seed 稳定性。
- 低步数退化方式。
- CFG 放大后是否崩。

不要只看“哪个最好看”。学习阶段更重要的是回答：为什么这个 sampler 在这个 objective 下失效或稳定。
