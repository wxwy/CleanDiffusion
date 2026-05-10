#!/usr/bin/env python3
"""2D Flow Matching 的端到端教学脚本。

这个文件用于系统学习 diffusion / flow 的第一步：先不要看图像、UNet、CLIP，
只在二维平面上观察噪声点如何沿速度场移动到数据分布。

文件仍然保持 CleanRL-style 单文件结构：

1. 配置
2. 2D 数据分布
3. 小 MLP 模型
4. Flow Matching objective
5. Euler sampler
6. train / sample / checkpoint / 可视化

核心公式与图像版 `clean_diffusion/fm.py` 完全一致：

  x0 = data point
  x1 = Gaussian noise
  t=0 -> noise, t=1 -> data
  x_t = (1 - t) * x1 + t * x0
  v_target = x0 - x1
  loss = MSE(v_theta(x_t, t), v_target)

训练示例：

python clean_diffusion/toy_fm_2d.py train \
  --run-name toy_fm_2d_smoke \
  --max-steps 200 \
  --batch-size 512

采样示例：

python clean_diffusion/toy_fm_2d.py sample \
  --ckpt outputs/toy_fm_2d_smoke/last.pt \
  --steps 32
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. 配置：只保留影响教学实验行为的参数。
# =============================================================================


@dataclass
class TrainConfig:
    output_root: str = "outputs"
    run_name: str = ""
    resume: str = ""
    device: str = "auto"

    max_steps: int = 2000
    batch_size: int = 512
    lr: float = 2e-3
    hidden_dim: int = 128
    depth: int = 3
    seed: int = 42

    print_every: int = 100
    save_every: int = 500
    plot_every: int = 500

    def resolved_run_name(self) -> str:
        if self.run_name:
            return sanitize_name(self.run_name)
        return f"toy_fm_2d_{time.strftime('%Y%m%d_%H%M%S')}"


@dataclass
class SampleConfig:
    ckpt: str
    output_dir: str = ""
    device: str = "auto"
    steps: int = 32
    num_points: int = 2048
    seed: int = 123


# =============================================================================
# 2. 少量工具：保持短小，避免干扰下面的算法主线。
# =============================================================================


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")


def mkdir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_name(name: str) -> str:
    keep = []
    for ch in name.strip().replace("/", "_"):
        keep.append(ch if ch.isalnum() or ch in "_.=-" else "_")
    cleaned = "".join(keep).strip("_")
    return cleaned or "run"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


# =============================================================================
# 3. 2D 数据：八个高斯团，便于观察生成分布是否覆盖所有 mode。
# =============================================================================


def sample_data(batch_size: int, device: torch.device) -> torch.Tensor:
    angles = torch.linspace(0, 2 * math.pi, 9, device=device)[:-1]
    centers = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * 2.0
    index = torch.randint(0, centers.shape[0], (batch_size,), device=device)
    return centers[index] + 0.12 * torch.randn(batch_size, 2, device=device)


def sample_noise(batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.randn(batch_size, 2, device=device)


# =============================================================================
# 4. 模型：MLP 输入二维点和时间 t，输出二维速度。
# =============================================================================


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / max(half - 1, 1)
        )
        args = t[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class VelocityMLP(nn.Module):
    def __init__(self, hidden_dim: int = 128, depth: int = 3) -> None:
        super().__init__()
        self.time_embed = TimeEmbedding(hidden_dim)
        layers: list[nn.Module] = []
        in_dim = 2 + hidden_dim
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_t, self.time_embed(t)], dim=1))


# =============================================================================
# 5. Flow Matching objective：本文件最核心的训练逻辑。
# =============================================================================


def flow_matching_loss(model: nn.Module, batch_size: int, device: torch.device) -> torch.Tensor:
    x0 = sample_data(batch_size, device)
    x1 = sample_noise(batch_size, device)
    t = torch.rand(batch_size, device=device)

    x_t = (1.0 - t[:, None]) * x1 + t[:, None] * x0
    v_target = x0 - x1
    v_pred = model(x_t, t)
    return F.mse_loss(v_pred, v_target)


# =============================================================================
# 6. Euler sampler：从高斯噪声出发，沿速度场积分到数据分布。
# =============================================================================


@torch.no_grad()
def euler_sample(
    model: nn.Module,
    num_points: int,
    steps: int,
    device: torch.device,
    keep_trajectory: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    x = sample_noise(num_points, device)
    trajectory = [x.detach().cpu()] if keep_trajectory else []
    dt = 1.0 / steps

    for i in range(steps):
        t_value = torch.full((num_points,), i / steps, device=device)
        v = model(x, t_value)
        x = x + dt * v
        if keep_trajectory:
            trajectory.append(x.detach().cpu())

    return x.detach().cpu(), trajectory


# =============================================================================
# 7. 可视化与 checkpoint：服务学习记录，不形成框架层。
# =============================================================================


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: TrainConfig,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": asdict(cfg),
        },
        path,
    )


def load_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer | None = None) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("step", 0))


def plot_state(
    model: nn.Module,
    output_path: Path,
    device: torch.device,
    steps: int = 32,
    num_points: int = 2048,
) -> None:
    model.eval()
    data = sample_data(num_points, device).detach().cpu()
    samples, trajectory = euler_sample(
        model, num_points=num_points, steps=steps, device=device, keep_trajectory=True
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].scatter(data[:, 0], data[:, 1], s=4, alpha=0.5)
    axes[0].set_title("data x0")

    axes[1].scatter(samples[:, 0], samples[:, 1], s=4, alpha=0.5)
    axes[1].set_title("Euler samples")

    for point_id in range(0, min(64, samples.shape[0]), 2):
        line = torch.stack([state[point_id] for state in trajectory])
        axes[2].plot(line[:, 0], line[:, 1], linewidth=0.8, alpha=0.7)
    axes[2].set_title("noise -> data trajectories")

    for axis in axes:
        axis.set_xlim(-3.2, 3.2)
        axis.set_ylim(-3.2, 3.2)
        axis.set_aspect("equal")
        axis.grid(True, linewidth=0.3, alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    model.train()


# =============================================================================
# 8. train / sample 主流程：从这里能完整读到一个实验如何运行。
# =============================================================================


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    run_dir = mkdir(Path(cfg.output_root) / cfg.resolved_run_name())
    plot_dir = mkdir(run_dir / "plots")
    metrics_path = run_dir / "metrics.jsonl"

    model = VelocityMLP(hidden_dim=cfg.hidden_dim, depth=cfg.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    start_step = 0
    if cfg.resume:
        start_step = load_checkpoint(cfg.resume, model, optimizer)
        log(f"resume from {cfg.resume}, step={start_step}")

    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    log(f"train on {device}, outputs -> {run_dir}")

    for step in range(start_step + 1, cfg.max_steps + 1):
        loss = flow_matching_loss(model, cfg.batch_size, device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % cfg.print_every == 0:
            log(f"step={step} loss={loss.item():.6f}")
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, "loss": float(loss.item())}) + "\n")

        if step == 1 or step % cfg.save_every == 0 or step == cfg.max_steps:
            save_checkpoint(run_dir / f"step={step:06d}.pt", model, optimizer, step, cfg)
            save_checkpoint(run_dir / "last.pt", model, optimizer, step, cfg)
        if step == 1 or step % cfg.plot_every == 0 or step == cfg.max_steps:
            plot_state(model, plot_dir / f"step={step:06d}.png", device)


def sample(cfg: SampleConfig) -> None:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    ckpt = torch.load(cfg.ckpt, map_location="cpu")
    train_cfg = TrainConfig(**ckpt.get("config", {}))

    model = VelocityMLP(hidden_dim=train_cfg.hidden_dim, depth=train_cfg.depth).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    output_dir = mkdir(cfg.output_dir or Path(cfg.ckpt).resolve().parent / "samples")
    samples, _ = euler_sample(model, cfg.num_points, cfg.steps, device, keep_trajectory=False)
    torch.save(samples, output_dir / f"samples_steps{cfg.steps}.pt")
    plot_state(
        model,
        output_dir / f"samples_steps{cfg.steps}.png",
        device,
        steps=cfg.steps,
        num_points=cfg.num_points,
    )
    log(f"saved samples to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2D Flow Matching teaching script")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    for field_name, field_def in TrainConfig.__dataclass_fields__.items():
        default = field_def.default
        arg = "--" + field_name.replace("_", "-")
        if isinstance(default, bool):
            train_parser.add_argument(arg, action=argparse.BooleanOptionalAction, default=default)
        else:
            train_parser.add_argument(arg, type=type(default), default=default)

    sample_parser = subparsers.add_parser("sample")
    for field_name, field_def in SampleConfig.__dataclass_fields__.items():
        default = field_def.default
        arg = "--" + field_name.replace("_", "-")
        if field_name == "ckpt":
            sample_parser.add_argument(arg, type=str, required=True)
        elif isinstance(default, bool):
            sample_parser.add_argument(arg, action=argparse.BooleanOptionalAction, default=default)
        else:
            sample_parser.add_argument(arg, type=type(default), default=default)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        values = vars(args).copy()
        values.pop("command")
        train(TrainConfig(**values))
    elif args.command == "sample":
        values = vars(args).copy()
        values.pop("command")
        sample(SampleConfig(**values))
    else:
        raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
