# DDPM 参数化对比

DDPM 学习时最常见的问题是：模型到底应该预测什么？

同一个 noisy sample `x_t` 可以对应四种常见输出：

- epsilon prediction：预测加进去的噪声 `eps`。
- x0 prediction：直接预测干净数据 `x0`。
- v prediction：预测 velocity-like 旋转变量 `v`。
- score prediction：预测 `nabla_x log p_t(x)`，也就是 score。

这些参数化通常不是四个完全不同的问题，而是同一个高斯加噪公式下的不同坐标系。

## 1. DDPM 加噪公式

DDPM forward process 常写成：

```text
x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
eps ~ N(0, I)
```

为了缩短记号，下文使用：

```text
a = sqrt(alpha_bar_t)
s = sqrt(1 - alpha_bar_t)
x_t = a * x0 + s * eps
```

这里的 `t` 是离散 diffusion timestep，和 Flow Matching 文件里的连续 `t=0->noise, t=1->data` 不是同一个坐标。

## 2. epsilon prediction

当前 `clean_diffusion/ddpm.py` 使用的是 epsilon prediction：

```text
model(x_t, t, condition) -> eps_hat
loss = MSE(eps_hat, eps)
```

从 `eps_hat` 可以还原 `x0_hat`：

```text
x0_hat = (x_t - s * eps_hat) / a
```

优点：

- 经典 DDPM baseline，公式简单。
- 和 DDPM / DDIM sampler 连接直接。

学习时要注意：

- 低噪声区域 `s` 很小，噪声目标的信号可能弱。
- 不同 timestep 的 loss 权重并不天然等价。

## 3. x0 prediction

x0 prediction 让模型直接输出干净数据：

```text
model(x_t, t, condition) -> x0_hat
loss = MSE(x0_hat, x0)
```

从 `x0_hat` 可以还原 `eps_hat`：

```text
eps_hat = (x_t - a * x0_hat) / s
```

优点：

- 输出空间和图像空间一致，直观。
- 方便做 reconstruction 或 endpoint consistency。

风险：

- 高噪声区域中 `x_t` 包含的数据线索很少，直接预测 `x0` 可能更难。
- 当 `s` 接近 0 时，从 `x0_hat` 换回 `eps_hat` 要做数值保护。

## 4. v prediction

v prediction 常用于 improved diffusion / latent diffusion 一类实现。它定义为：

```text
v = a * eps - s * x0
```

这可以看作对 `(x0, eps)` 的旋转坐标。由 `x_t` 和 `v_hat` 可以还原：

```text
x0_hat  = a * x_t - s * v_hat
eps_hat = s * x_t + a * v_hat
```

优点：

- 在不同噪声强度下更平衡。
- 对高噪声和低噪声区域通常更稳定。

注意：

- 这里的 `v` 是 DDPM 参数化里的 `v`，不是 Flow Matching 文件中的 `v_target = x0 - x1`。
- 两者都叫 velocity，但数学上下文不同，不能混用。

## 5. score prediction

对于高斯扰动：

```text
x_t = a * x0 + s * eps
```

条件于 `x0` 时，`x_t` 的 score 和噪声关系为：

```text
score = -eps / s
eps = -s * score
```

因此 score prediction 可以和 epsilon prediction 互相转换：

```text
score_hat = -eps_hat / s
eps_hat = -s * score_hat
```

优点：

- 和 score-based SDE / ODE 理论直接连接。
- 方便理解 probability flow ODE。

风险：

- 当 `s` 很小时，score 数值会变大，需要权重和数值保护。

## 6. 四种参数化的转换表

| 模型输出 | 还原 `x0_hat` | 还原 `eps_hat` |
| --- | --- | --- |
| `eps_hat` | `(x_t - s * eps_hat) / a` | `eps_hat` |
| `x0_hat` | `x0_hat` | `(x_t - a * x0_hat) / s` |
| `v_hat` | `a * x_t - s * v_hat` | `s * x_t + a * v_hat` |
| `score_hat` | `(x_t + s^2 * score_hat) / a` | `-s * score_hat` |

实现 sampler 时，通常先把模型输出统一转换成 `eps_hat` 或 `x0_hat`，再复用同一套 DDPM / DDIM 更新公式。

## 7. 在 CleanDiffusion 中怎么学

当前推荐顺序：

1. 先读 `clean_diffusion/ddpm.py`，理解 epsilon prediction baseline。
2. 运行 `clean_diffusion/toy_ddpm_parameterizations.py`，确认四种参数化可以互相转换。
3. 再决定是否新增独立教学文件，例如 `ddpm_vpred.py`，而不是直接把 `ddpm.py` 改成多模式大文件。

这样可以保留 DDPM baseline 的清晰度，也能系统比较不同参数化。
