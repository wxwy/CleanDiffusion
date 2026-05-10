# 模型结构与 Conditioning 笔记

CleanDiffusion 的模型不是为了追求最大性能，而是为了让不同 objective 在同一个结构下可比较。

学习模型结构时，建议按数据流读：

```text
image/noisy image -> UNet blocks -> prediction
time -> time embedding -> blocks
text -> text encoder/context -> blocks
```

## 1. UNet 主干

图像 diffusion 常用 UNet，因为它同时保留：

- 局部纹理：卷积和 residual block。
- 多尺度结构：downsample / upsample。
- 全局语义：中间层或 attention。

在教学实现中，UNet 不需要一次性做成工业级结构。读代码时优先确认：

- 输入输出 shape 是否一致。
- down path 和 up path 的 skip connection 是否对齐。
- time/text conditioning 是否进入每个关键 block。
- 最后一层预测的是 noise、velocity 还是 endpoint。

## 2. Time embedding

模型必须知道当前噪声强度或 flow time。

常见做法：

```text
t -> sinusoidal embedding -> MLP -> time_context
```

学习重点：

- DDPM 的 `t` 是离散 timestep。
- FM/RF/CFM 的 `t` 是连续 `[0, 1]`。
- time embedding 不应改变 objective 的数学方向，只提供条件信息。

## 3. Text conditioning

当前图像单文件使用 CLIP text conditioning。

典型流程：

```text
prompt -> CLIP text encoder -> text_context -> UNet
```

教学上先关心两件事：

- 有条件输入时，模型学 `p(image | text)`。
- 空文本或 dropped text 用于 classifier-free guidance。

## 4. Classifier-free guidance

CFG 的核心公式：

```text
pred = pred_uncond + scale * (pred_cond - pred_uncond)
```

这个公式可以作用在不同预测类型上：

- DDPM：通常作用在 `eps_hat`。
- FM/RF/CFM：通常作用在 `v_hat`。
- Consistency/DMD：通常作用在 `x0_hat` 或 endpoint。

学习重点：

- `scale=1` 接近正常条件预测。
- `scale=0` 是无条件预测。
- `scale` 太大可能过饱和、崩坏或模式变少。

## 5. Attention

Attention 用来让模型在空间或文本 token 之间交换信息。

教学阶段可以先区分：

- self-attention：图像特征内部交互。
- cross-attention：图像特征查询文本 token。
- pooled text conditioning：把整句文本压成一个向量。

当前项目以可读性为先，不追求完整 diffusers 级 cross-attention 结构。后续如加入更复杂 attention，建议做成独立教学文件，而不是把所有算法文件一起改复杂。

## 6. 结构对比实验

固定 objective 和 sampler，只改变模型：

```text
base_channels = 32 / 64 / 128
channel_mults = 1,2,4 / 1,2,4,8
attention = off / middle / multi-level
conditioning = text pooled / token cross-attention
```

记录：

- 参数量。
- 单 step 显存。
- 单 step 耗时。
- loss 曲线。
- 固定 prompt sample。

不要在同一个实验里同时改模型、objective、sampler、schedule，否则无法解释结果。
