# SESSION

## 2026-05-10

- 初始化 CleanDiffusion 项目目录。
- 创建 CleanDiffusion 推荐架构目录：`clean_diffusion/configs`、`data`、`models`、`objectives`、`samplers`、`training`、`utils`、`scripts`。
- 创建协作配套文件：`AGENTS.md`、`CLAUDE.md`、`SESSION.md`、`TODO.md`、`MEMORY/project.md`、`.codex/README.md`、`.claude/README.md`。
- 当前只做项目骨架与规则沉淀，未实现 DDPM / FM / CFM / DMD 代码。
- 第一阶段建议先做 DDPM baseline 与 `--objective ddpm` 参数化，再逐个加入 FM、Rectified Flow、Consistency、CFM、DMD。
- 已初始化本地 git 仓库，默认分支为 `main`。
- 已初始化本地 git 仓库，默认分支为 `main`。
- 公开仓库前已执行信息暴露检查，清理本地绝对路径、账号名和认证状态记录。

## 下次会话入口

1. 先读取 `AGENTS.md`、`SESSION.md`、`TODO.md`。
2. 长期项目事实查看 `MEMORY/project.md`。
3. 代码改动前检查当前文件结构与 git 状态。
4. 第一阶段不要直接进入 FM/CFM，实现范围控制在 DDPM baseline 与 objective 参数化。
5. 若要继续上传 GitHub，需要先确认本机 GitHub CLI 已认证，或提供已有远端仓库 URL。
