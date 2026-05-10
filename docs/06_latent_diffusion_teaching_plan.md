# Latent Diffusion 教学版本计划

Latent Diffusion 的核心思想是：不要直接在像素空间做 diffusion，而是在 autoencoder 的 latent 空间做 diffusion。

这不是 CleanDiffusion 当前 baseline 的必需部分，但从系统学习 diffusion 的角度必须理解。

## 1. 像素空间 vs latent 空间

像素空间：

```text
image x in R^{3 x H x W}
diffusion model learns on image
```

latent 空间：

```text
image x -> encoder -> latent z
diffusion model learns on z
latent z -> decoder -> image x
```

优点：

- 训练和采样更省显存。
- 更容易扩大分辨率。
- 语义压缩后，生成模型可以把容量用在高层结构上。

代价：

- 需要 autoencoder。
- 生成质量受 decoder 限制。
- latent scaling、归一化和重建误差会成为新变量。

## 2. 教学版最小结构

建议新增独立文件，而不是修改现有 DDPM/FM baseline：

```text
clean_diffusion/latent_ddpm.py
```

单文件仍然包含：

- 配置。
- 数据加载。
- 简单 autoencoder 或加载预训练 VAE 的接口。
- latent 编码。
- latent DDPM objective。
- latent DDIM sampler。
- decoder 输出图片。
- checkpoint / resume / sample / metrics。

如果使用预训练 VAE，会引入外部模型依赖；如果使用小 autoencoder，教学更完整但生成质量较弱。

## 3. 最小数学流程

训练 autoencoder：

```text
z = encoder(x)
x_rec = decoder(z)
loss_ae = reconstruction + regularization
```

训练 latent diffusion：

```text
z0 = encoder(x).detach()
z_t = sqrt(alpha_bar_t) * z0 + sqrt(1 - alpha_bar_t) * eps
eps_hat = model(z_t, t, text)
loss = MSE(eps_hat, eps)
```

采样：

```text
z_T ~ N(0, I)
z0_hat = DDIM/DDPM sampler(model, z_T)
x_hat = decoder(z0_hat)
```

## 4. 必须记录的 latent 细节

latent diffusion 最容易出错的地方不是 sampler，而是尺度：

- latent shape。
- downsample factor。
- latent mean/std。
- latent scaling factor。
- decoder 输入范围。
- 输出图片 clamp / normalize。

每个 checkpoint metadata 应记录这些字段。

## 5. 教学路线建议

推荐顺序：

1. 先完成像素版 DDPM / FM / CFM 的理解。
2. 单独训练一个小 autoencoder，确认 reconstruction 正常。
3. 冻结 autoencoder，在 latent 上跑 DDPM。
4. 比较像素 DDPM 与 latent DDPM 的速度、显存、样本质量。
5. 再考虑 latent FM / latent CFM。

不要一开始就把 latent、text cross-attention、CFG、大模型和复杂 sampler 全部放在一起。

## 6. Smoke test 要求

未来实现 `latent_ddpm.py` 时，至少验证：

- autoencoder forward 可运行。
- latent diffusion 1 step train 可运行。
- resume 可恢复 model / optimizer / autoencoder 状态。
- sample 可保存图片。
- metadata 记录 latent shape、scaling factor、objective、sampler。
