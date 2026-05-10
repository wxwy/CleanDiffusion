# TODO

## 当前阶段：项目初始化

- [x] 创建 CleanDiffusion 项目目录结构。
- [x] 创建 Codex / Claude 协作配套文件。
- [x] 创建项目记忆文件。

## 阶段 0：DDPM baseline

- [ ] 从本地 CFM 原型迁移最小 DDPM baseline。
- [ ] 接入 `--objective ddpm`，默认只支持 DDPM。
- [ ] 保留 DDPM / DDIM sampler。
- [ ] 支持 TensorBoard、`metrics.jsonl`、sample images、checkpoint metadata。
- [ ] 完成 1 step 训练 smoke test。
- [ ] 完成 resume smoke test。
- [ ] 完成 DDIM 低步数采样 smoke test。

## 后续路线

- [ ] 阶段 1：Flow Matching。
- [ ] 阶段 2：Rectified Flow。
- [ ] 阶段 3：Teacher-free Consistency Training。
- [ ] 阶段 4：Consistency Flow Matching。
- [ ] 阶段 5：DMD-lite。
- [ ] 阶段 6：DMD2-style experiments。
