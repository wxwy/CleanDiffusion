# Evaluation 笔记

Diffusion 项目的评估不能只靠 loss，也不能只靠挑几张好看的图。

教学阶段建议同时使用三类评估：

1. 训练信号：loss、gradient、EMA。
2. 自动指标：FID、CLIP score、precision/recall。
3. 人工检查：固定 prompt、失败样例、采样稳定性。

## 1. Loss 不能横向直接比较

不同 objective 的 loss 数值含义不同：

- DDPM：noise MSE。
- FM/RF：velocity MSE。
- Consistency：endpoint consistency + boundary。
- CFM：velocity + endpoint consistency。
- DMD-lite：teacher endpoint distillation。
- DMD2-style：surrogate / GAN proxy。

因此不能说 `DDPM loss=0.1` 一定优于 `FM loss=0.5`。loss 更适合看同一个 objective 内的训练趋势。

## 2. FID

FID 比较生成图片和真实图片在特征空间的均值/协方差差异。

优点：

- 常见，便于和论文或其他项目对齐。
- 对整体分布有一定参考价值。

限制：

- 小样本 FID 不稳定。
- 对数据预处理和 Inception 实现敏感。
- 不能直接解释单个 prompt 是否对齐。

教学建议：

- 小规模实验先不要过度依赖 FID。
- 真要记录 FID，固定 sample 数、真实集、resize、seed。

## 3. CLIP score

CLIP score 衡量图文对齐：

```text
score = cosine(CLIP_image(image), CLIP_text(prompt))
```

优点：

- 对 text-to-image 条件生成有直观意义。
- 适合比较 CFG scale。

限制：

- 可能偏向 CLIP 喜欢的图像，而不是真实质量。
- 不能替代人工检查。

## 4. Precision / Recall

生成模型中的 precision / recall 可以粗略理解为：

- precision：生成样本像不像真实数据。
- recall：真实数据模式覆盖得够不够。

常见失败：

- 高 precision 低 recall：样本好看但模式少。
- 低 precision 高 recall：覆盖广但质量差。

这类指标对理解 mode collapse 很有用。

## 5. 人工检查表

每次保存 sample grid 时，至少人工检查：

```text
prompt 是否对齐：
主体是否清楚：
颜色是否符合：
构图是否稳定：
是否过饱和：
是否重复：
是否出现明显伪影：
低步数是否崩：
高 CFG 是否崩：
```

建议固定一小组 prompt，所有 objective 都用同一组。

## 6. 采样成本

记录质量时也要记录成本：

- sample steps。
- 每张图片耗时。
- batch size。
- GPU 型号。
- image size。
- CFG scale。
- 是否使用 EMA。

少步模型的价值必须和质量一起看，否则无法比较 Consistency / CFM / DMD。

## 7. CleanDiffusion 推荐记录格式

建议每个实验补一份短记录：

```text
run_name:
commit:
objective:
sampler:
checkpoint:
image_size:
sample_steps:
guidance_scale:
metrics:
人工观察:
失败样例:
下一步:
```

这比只保存图片更适合系统学习。
