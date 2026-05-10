# 失败样例库

失败样例库不是 bug 垃圾桶，而是学习 diffusion 最重要的材料之一。

每次出现坏图、loss 异常、采样发散或 resume 失败，都应该用固定格式记录，避免下次重复定位。

## 1. 记录模板

```text
日期：
commit：
文件：
命令：
checkpoint：
objective：
sampler：
image_size：
sample_steps：
guidance_scale：

现象：
关键日志：
判断依据：
根因：
处理：
验证：
下一步：
```

如果还不能确定根因，`根因` 写“未确认”，但 `判断依据` 必须具体。

## 2. 常见失败：loss 正常但图片发灰

可能检查项：

- 图片 normalize / denormalize 是否一致。
- sampler 输出是否 clamp 到正确范围。
- DDPM 训练 schedule 和采样 schedule 是否一致。
- 是否误用 EMA 或加载了未训练 checkpoint。
- 保存图片时是否把 `[-1, 1]` 当成 `[0, 1]`。

判断依据：

- 打印 sample tensor min/max/mean/std。
- 保存中间 timestep。
- 用同一个 checkpoint 跑不同 sampler。

## 3. 常见失败：FM/RF 采样发散

可能检查项：

- `t=0 -> noise, t=1 -> data` 方向是否写反。
- `v_target = x0 - x1` 是否写反。
- Euler `dt` 是否为正。
- CFG 是否过大。
- velocity 输出 scale 是否异常。

判断依据：

- 记录每一步 `x.norm()` 和 `v.norm()`。
- 用 `guidance_scale=0/1` 对比。
- 用更大 `steps` 检查是否数值误差导致。

## 4. 常见失败：Consistency 输出过平滑

可能检查项：

- endpoint consistency 权重是否过大。
- boundary/reconstruction loss 是否太弱。
- EMA decay 是否太大。
- 不同时间点是否采样得太近。

判断依据：

- 分开记录 consistency loss 和 boundary loss。
- 查看 `x0_hat` 的 min/max/std。
- 对比 one-step 和 multi-step 变体。

## 5. 常见失败：CFG 变大后崩坏

可能检查项：

- 条件和无条件 branch 差异是否过大。
- text dropout 是否太低。
- guidance scale 是否超出当前 checkpoint 能承受范围。
- 对 velocity / endpoint 做 CFG 时是否使用了正确输出类型。

判断依据：

- 固定 seed，扫 `scale=0,1,3,5,7.5`。
- 分别保存 cond / uncond 预测 norm。
- 检查 prompt 是否过长或 tokenizer 异常。

## 6. 常见失败：DMD student 坍缩

可能检查项：

- teacher checkpoint 是否足够好。
- teacher target 是否固定或噪声过大。
- one-step student 容量是否不足。
- surrogate / discriminator loss 是否压过 distillation loss。

判断依据：

- 单独关闭 GAN proxy，只跑 teacher endpoint distillation。
- 保存 teacher target 和 student output 对比。
- 检查 student 输出方差是否快速变小。

## 7. 示例记录

```text
日期：2026-05-10
commit：example
文件：clean_diffusion/fm.py
命令：python clean_diffusion/fm.py sample ...
checkpoint：checkpoints/fm_smoke/step=000000002
objective：fm
sampler：euler
image_size：32
sample_steps：4
guidance_scale：7.5

现象：图片颜色过饱和，局部纹理块状。
关键日志：sample tensor std 明显高于 guidance_scale=1.0。
判断依据：固定 seed 下 scale=1 正常，scale=7.5 崩坏。
根因：当前 checkpoint 训练步数极少，conditional/unconditional 差值不稳定，高 CFG 放大误差。
处理：降低 smoke test guidance_scale；正式实验增加训练步数并扫 scale。
验证：scale=3.0 可正常保存图片。
下一步：记录 CFG sweep。
```

失败样例要和成功样例一起提交到实验记录里。只保留成功图片会让项目失去研究价值。
