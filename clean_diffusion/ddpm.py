#!/usr/bin/env python3
"""DDPM + DDIM + CFG 的端到端教学实现。

这个文件故意采用 CleanRL-style 单文件结构：读者打开一个文件，就能顺着
配置 -> 数据 -> 模型 -> DDPM objective -> sampler -> train/sample 主流程读完。

当前文件只实现阶段 0：

- DDPM noise prediction objective
- DDPM ancestral sampler
- DDIM sampler
- classifier-free guidance
- COCO captions + CLIP text conditioning
- checkpoint / resume / TensorBoard / metrics.jsonl / sample images

训练示例：

python clean_diffusion/ddpm.py train \
  --coco-img-root /path/to/coco/images/train \
  --coco-ann-file /path/to/captions_train2017.json \
  --clip-root /path/to/clip \
  --output-root checkpoints \
  --run-name ddpm_smoke \
  --image-size 32 \
  --batch-size 2 \
  --base-channels 16 \
  --channel-mults 1,2,4 \
  --max-steps 1 \
  --no-amp

采样示例：

python clean_diffusion/ddpm.py sample \
  --ckpt-root checkpoints/ddpm_smoke \
  --clip-root /path/to/clip \
  --output-dir samples/ddpm_smoke \
  --image-size 32 \
  --base-channels 16 \
  --channel-mults 1,2,4 \
  --scheduler ddim \
  --steps 4 \
  --prompts "a small red car"

DDPM 的核心训练公式：

  x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
  loss = MSE(eps_theta(x_t, t, text), eps)

注意：后续 FM / CFM 文件会统一使用 x0=data、x1=noise、t=0->noise、
t=1->data。本文件是 DDPM baseline，保留 DDPM 标准离散噪声时间。
"""

from __future__ import annotations

import sys


def _early_help_if_requested() -> None:
    """不导入 Torch/Pillow 也能查看入口，避免 CUDA 环境问题阻塞 help。"""
    if len(sys.argv) == 1 or sys.argv[1] in {"-h", "--help"}:
        print(
            "CleanDiffusion DDPM single-file baseline\n\n"
            "Usage:\n"
            "  python clean_diffusion/ddpm.py train [options]\n"
            "  python clean_diffusion/ddpm.py sample [options]\n\n"
            "Run subcommand help:\n"
            "  python clean_diffusion/ddpm.py train --help\n"
            "  python clean_diffusion/ddpm.py sample --help"
        )
        raise SystemExit(0)
    if len(sys.argv) >= 3 and sys.argv[2] in {"-h", "--help"}:
        if sys.argv[1] == "train":
            print(
                "Train DDPM\n\n"
                "Required:\n"
                "  --coco-img-root PATH\n"
                "  --coco-ann-file PATH\n"
                "  --clip-root PATH\n\n"
                "Common:\n"
                "  --output-root checkpoints --ckpt-root checkpoints --run-name NAME\n"
                "  --image-size 32 --batch-size 2 --base-channels 16 --channel-mults 1,2,4\n"
                "  --max-steps 1 --sample-scheduler ddim --sample-steps 4 --no-amp"
            )
            raise SystemExit(0)
        if sys.argv[1] == "sample":
            print(
                "Sample from DDPM checkpoint\n\n"
                "Required:\n"
                "  --ckpt-root PATH\n"
                "  --clip-root PATH\n\n"
                "Common:\n"
                "  --output-dir samples --image-size 32 --base-channels 16 --channel-mults 1,2,4\n"
                "  --scheduler ddim --steps 4 --guidance-scales 3.0 --prompts \"a small red car\""
            )
            raise SystemExit(0)


_early_help_if_requested()

import argparse
import copy
import json
import math
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import torchvision.transforms as T


# =============================================================================
# 1. 配置：教学代码先把会影响算法行为的参数摆在文件前面。
# =============================================================================


@dataclass
class TrainConfig:
    coco_img_root: str
    coco_ann_file: str
    clip_root: str
    output_root: str = "checkpoints"
    ckpt_root: str = "checkpoints"
    run_name: str = ""
    init_from: str = ""
    log_root: str = ""
    device: str = "auto"

    image_size: int = 128
    batch_size: int = 24
    num_workers: int = 8
    text_dropout: float = 0.1

    timesteps: int = 1000
    schedule: str = "quadratic"
    sample_scheduler: str = "ddim"

    epochs: int = 20
    max_steps: int = 0
    lr: float = 2e-4
    ema_decay: float = 0.9995
    grad_clip: float = 1.0

    sample_every: int = 2500
    save_every: int = 5000
    print_every: int = 100
    guidance_scale: float = 5.0
    sample_steps: int = 50
    sample_eta: float = 0.0
    sample_size: int = 4

    time_emb_dim: int = 256
    text_dim: int = 512
    base_channels: int = 128
    channel_mults: str = "1,2,4,8"
    groups: int = 8

    amp: bool = True
    seed: int = 42

    def resolved_run_name(self) -> str:
        if self.run_name:
            return sanitize_name(self.run_name)
        suffix = "_smoke" if self.max_steps > 0 else ""
        return sanitize_name(f"ddpm_unet_{time.strftime('%Y%m%d_%H%M%S')}{suffix}")

    def channel_mult_tuple(self) -> tuple[int, ...]:
        return parse_int_tuple(self.channel_mults)


@dataclass
class SampleConfig:
    ckpt_root: str
    clip_root: str
    output_dir: str = "samples"
    step: str = "last"
    device: str = "auto"

    image_size: int = 128
    batch_size: int = 4
    timesteps: int = 1000
    schedule: str = "quadratic"
    scheduler: str = "ddim"
    steps: int = 50
    eta: float = 0.0
    guidance_scales: str = "5.0"
    prompts: list[str] | None = None

    time_emb_dim: int = 256
    text_dim: int = 512
    base_channels: int = 128
    channel_mults: str = "1,2,4,8"
    groups: int = 8

    def channel_mult_tuple(self) -> tuple[int, ...]:
        return parse_int_tuple(self.channel_mults)

    def guidance_scale_list(self) -> list[float]:
        return [float(x) for x in self.guidance_scales.split(",") if x.strip()]


# =============================================================================
# 2. 少量通用工具：保持短小，不打断下面的算法主线。
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


def parse_int_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(",") if x.strip())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def chunks(items: Sequence, n: int) -> Iterable[Sequence]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def import_clip():
    try:
        import clip
    except ImportError as exc:
        raise ImportError("未找到 OpenAI CLIP。请先安装项目依赖。") from exc
    return clip


# =============================================================================
# 3. 数据：COCO caption -> image tensor + CLIP token。
# =============================================================================


class CocoCaptions(Dataset):
    def __init__(self, img_root: str, ann_file: str, image_size: int, text_dropout: float):
        self.img_root = Path(img_root)
        self.text_dropout = text_dropout
        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )
        with Path(ann_file).open("r", encoding="utf-8") as f:
            data = json.load(f)
        id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}
        self.samples = [(id_to_filename[ann["image_id"]], ann["caption"]) for ann in data["annotations"]]
        log(f"Loaded {len(self.samples)} COCO caption samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        filename, caption = self.samples[idx]
        image = Image.open(self.img_root / filename).convert("RGB")
        image = self.transform(image)
        dropped = torch.rand(1).item() < self.text_dropout
        return {
            "image": image,
            "caption": caption,
            "cond_caption": "" if dropped else caption,
            "is_dropped": dropped,
        }


def build_dataloader(cfg: TrainConfig) -> DataLoader:
    clip = import_clip()
    dataset = CocoCaptions(cfg.coco_img_root, cfg.coco_ann_file, cfg.image_size, cfg.text_dropout)

    def collate_fn(batch: list[dict]):
        images = torch.stack([item["image"] for item in batch], dim=0)
        cond_captions = [item["cond_caption"] for item in batch]
        text_tokens = clip.tokenize(cond_captions, truncate=True)
        captions = [item["caption"] for item in batch]
        is_dropped = torch.tensor([item["is_dropped"] for item in batch], dtype=torch.bool)
        return images, text_tokens, captions, cond_captions, is_dropped

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )


@torch.no_grad()
def encode_text_from_tokens(clip_model: nn.Module, text_tokens: torch.Tensor, device: torch.device) -> torch.Tensor:
    text_tokens = text_tokens.to(device)
    emb = clip_model.encode_text(text_tokens).to(device=device, dtype=torch.float32)
    return emb.unsqueeze(1).repeat(1, 77, 1)


@torch.no_grad()
def encode_text_from_prompts(clip_model: nn.Module, prompts: list[str], device: torch.device) -> torch.Tensor:
    clip = import_clip()
    tokens = clip.tokenize(prompts, truncate=True).to(device)
    emb = clip_model.encode_text(tokens).to(device=device, dtype=torch.float32)
    return emb.unsqueeze(1).repeat(1, 77, 1)


def load_clip_model(clip_root: str, device: torch.device):
    clip = import_clip()
    clip_model, _ = clip.load("ViT-B/16", device=device, download_root=clip_root)
    clip_model.eval()
    return clip_model


# =============================================================================
# 4. 模型：一个足够小的 text-conditioned UNet。
# =============================================================================


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time_tensor: torch.Tensor) -> torch.Tensor:
        device = time_tensor.device
        half_dim = self.dim // 2
        freqs = torch.exp(-math.log(10000) / (half_dim - 1) * torch.arange(half_dim, device=device))
        emb = time_tensor[:, None] * freqs[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat((emb, torch.zeros_like(emb[:, :1])), dim=-1)
        return emb


class CrossAttention(nn.Module):
    def __init__(self, dim: int, context_dim: int, num_heads: int = 4):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} 必须能被 num_heads={num_heads} 整除")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(context_dim, dim)
        self.to_v = nn.Linear(context_dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, n, dim = x.shape
        _, m, _ = context.shape
        q = self.to_q(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).view(b, m, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(b, m, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim**0.5)
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, dim)
        return self.out(out)


class ResBlockWithText(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, text_dim: int, groups: int):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.cross_attn = CrossAttention(dim=out_channels, context_dim=text_dim, num_heads=4)
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, text_context: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]

        b, c, hgt, wdt = h.shape
        h_flat = h.view(b, c, hgt * wdt).transpose(1, 2)
        h = h + self.cross_attn(h_flat, text_context).transpose(1, 2).view(b, c, hgt, wdt)

        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.res_conv(x)


class TextConditionedUNet(nn.Module):
    def __init__(
        self,
        time_emb_dim: int,
        text_dim: int,
        base_channels: int,
        channel_mults: tuple[int, ...],
        groups: int,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        self.init_conv = nn.Conv2d(3, base_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        in_ch = base_channels
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            self.down_blocks.append(
                nn.ModuleList(
                    [
                        ResBlockWithText(in_ch, out_ch, time_emb_dim, text_dim, groups),
                        ResBlockWithText(out_ch, out_ch, time_emb_dim, text_dim, groups),
                    ]
                )
            )
            in_ch = out_ch
            if i != len(channel_mults) - 1:
                self.down_samples.append(nn.AvgPool2d(2))

        self.mid_block1 = ResBlockWithText(in_ch, in_ch, time_emb_dim, text_dim, groups)
        self.mid_block2 = ResBlockWithText(in_ch, in_ch, time_emb_dim, text_dim, groups)

        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for i, mult in enumerate(reversed(channel_mults)):
            skip_ch = base_channels * mult
            self.up_blocks.append(
                nn.ModuleList(
                    [
                        ResBlockWithText(in_ch + skip_ch, skip_ch, time_emb_dim, text_dim, groups),
                        ResBlockWithText(skip_ch, skip_ch, time_emb_dim, text_dim, groups),
                    ]
                )
            )
            in_ch = skip_ch
            if i != len(channel_mults) - 1:
                self.up_samples.append(nn.Upsample(scale_factor=2, mode="nearest"))

        self.final_conv = nn.Sequential(
            nn.GroupNorm(groups, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, text_context: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2 and t.size(1) == 1:
            t = t.squeeze(-1)
        t_emb = self.time_mlp(t.float())

        x = self.init_conv(x)
        skips = [x]
        for i, (block1, block2) in enumerate(self.down_blocks):
            x = block1(x, t_emb, text_context)
            x = block2(x, t_emb, text_context)
            skips.append(x)
            if i < len(self.down_samples):
                x = self.down_samples[i](x)

        x = self.mid_block1(x, t_emb, text_context)
        x = self.mid_block2(x, t_emb, text_context)

        for i, (block1, block2) in enumerate(self.up_blocks):
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = block1(x, t_emb, text_context)
            x = block2(x, t_emb, text_context)
            if i < len(self.up_samples):
                x = self.up_samples[i](x)

        return self.final_conv(x)


def build_model_from_config(cfg: TrainConfig | SampleConfig, device: torch.device) -> nn.Module:
    return TextConditionedUNet(
        time_emb_dim=cfg.time_emb_dim,
        text_dim=cfg.text_dim,
        base_channels=cfg.base_channels,
        channel_mults=cfg.channel_mult_tuple(),
        groups=cfg.groups,
    ).to(device)


# =============================================================================
# 5. DDPM objective：这是训练时最核心的算法部分。
# =============================================================================


def make_beta_schedule(timesteps: int, schedule: str, device: torch.device) -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(1e-4, 0.02, timesteps, device=device)
    if schedule == "quadratic":
        return torch.linspace(1e-4**0.5, 0.02**0.5, timesteps, device=device) ** 2
    if schedule == "cosine":
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, device=device)
        alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 1e-8, 0.999)
    raise ValueError(f'schedule should be one of ["linear", "quadratic", "cosine"], got {schedule}')


@dataclass
class DiffusionSchedule:
    timesteps: int
    beta: torch.Tensor
    alpha: torch.Tensor
    alpha_bar: torch.Tensor
    device: torch.device


def build_diffusion_schedule(timesteps: int, schedule: str, device: torch.device) -> DiffusionSchedule:
    beta = make_beta_schedule(timesteps, schedule, device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return DiffusionSchedule(timesteps=timesteps, beta=beta, alpha=alpha, alpha_bar=alpha_bar, device=device)


def expand_like(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return a.view(-1, *([1] * (x.dim() - 1)))


def q_sample(schedule: DiffusionSchedule, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """DDPM 前向扩散 q(x_t | x0)。"""
    a_bar_t = schedule.alpha_bar[t]
    return expand_like(torch.sqrt(a_bar_t), x0) * x0 + expand_like(torch.sqrt(1 - a_bar_t), x0) * noise


def ddpm_noise_prediction_loss(
    model: nn.Module,
    schedule: DiffusionSchedule,
    x0: torch.Tensor,
    text_context: torch.Tensor,
) -> torch.Tensor:
    """训练目标：让模型从 x_t、t、text 中预测加进去的 Gaussian noise。"""
    b = x0.size(0)
    t = torch.randint(0, schedule.timesteps, (b,), device=x0.device)
    noise = torch.randn_like(x0)
    x_t = q_sample(schedule, x0, t, noise)
    pred_noise = model(x_t, t, text_context)
    return F.mse_loss(pred_noise, noise)


# =============================================================================
# 6. CFG + sampler：训练学 eps_theta，采样把 eps_theta 反向用起来。
# =============================================================================


@torch.no_grad()
def cfg_predict_eps(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    cond_context: torch.Tensor,
    uncond_context: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    eps_uncond = model(x, t, uncond_context)
    eps_cond = model(x, t, cond_context)
    return eps_uncond + guidance_scale * (eps_cond - eps_uncond)


@torch.no_grad()
def ddpm_sample(
    model: nn.Module,
    schedule: DiffusionSchedule,
    cond_context: torch.Tensor,
    uncond_context: torch.Tensor,
    guidance_scale: float,
    shape: tuple[int, int, int, int],
) -> torch.Tensor:
    """DDPM ancestral sampling：从 pure noise 逐步采样到 x0。"""
    b = shape[0]
    x = torch.randn(shape, device=schedule.device)
    for t_int in reversed(range(schedule.timesteps)):
        t = torch.full((b,), t_int, device=schedule.device, dtype=torch.long)
        eps = cfg_predict_eps(model, x, t, cond_context, uncond_context, guidance_scale)

        alpha_t = expand_like(schedule.alpha[t], x)
        beta_t = expand_like(schedule.beta[t], x)
        alpha_bar_t = expand_like(schedule.alpha_bar[t], x)
        mean = (1.0 / torch.sqrt(alpha_t)) * (x - beta_t / torch.sqrt(1 - alpha_bar_t) * eps)
        if t_int > 0:
            x = mean + torch.sqrt(beta_t) * torch.randn_like(x)
        else:
            x = mean
    return x


def ddim_timestep_schedule(timesteps: int, steps: int, device: torch.device) -> torch.Tensor:
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if steps > timesteps:
        raise ValueError(f"steps must be <= timesteps, got steps={steps}, timesteps={timesteps}")
    return torch.linspace(0, timesteps - 1, steps, device=device).long()


@torch.no_grad()
def ddim_sample(
    model: nn.Module,
    schedule: DiffusionSchedule,
    cond_context: torch.Tensor,
    uncond_context: torch.Tensor,
    guidance_scale: float,
    steps: int,
    eta: float,
    shape: tuple[int, int, int, int],
) -> torch.Tensor:
    """DDIM sampling：用更少步数走确定性或半随机的反向轨迹。"""
    b = shape[0]
    x = torch.randn(shape, device=schedule.device)
    ts = ddim_timestep_schedule(schedule.timesteps, steps, schedule.device)

    for i in reversed(range(steps)):
        t = ts[i].repeat(b)
        eps = cfg_predict_eps(model, x, t, cond_context, uncond_context, guidance_scale)
        a_bar_t = expand_like(schedule.alpha_bar[t], x)
        x0_hat = (x - torch.sqrt(1 - a_bar_t) * eps) / torch.sqrt(a_bar_t)

        if i > 0:
            t_prev = ts[i - 1].repeat(b)
            a_bar_prev = expand_like(schedule.alpha_bar[t_prev], x)
            sigma = eta * torch.sqrt((1 - a_bar_prev) / (1 - a_bar_t) * (1 - a_bar_t / a_bar_prev))
            x = torch.sqrt(a_bar_prev) * x0_hat + torch.sqrt(1 - a_bar_prev - sigma**2) * eps
            x = x + sigma * torch.randn_like(x)
        else:
            x = x0_hat
    return x


def sample_with_scheduler(
    model: nn.Module,
    schedule: DiffusionSchedule,
    cond_context: torch.Tensor,
    uncond_context: torch.Tensor,
    scheduler: str,
    guidance_scale: float,
    steps: int,
    eta: float,
    shape: tuple[int, int, int, int],
) -> torch.Tensor:
    model.eval()
    if scheduler == "ddpm":
        return ddpm_sample(model, schedule, cond_context, uncond_context, guidance_scale, shape)
    if scheduler == "ddim":
        return ddim_sample(model, schedule, cond_context, uncond_context, guidance_scale, steps, eta, shape)
    raise ValueError(f'scheduler should be one of ["ddpm", "ddim"], got {scheduler}')


# =============================================================================
# 7. checkpoint / metrics / image：服务训练流程，但不改变算法。
# =============================================================================


def step_folder_name(step: int) -> str:
    return f"step={step:09d}"


def is_checkpoint_dir(path: str | Path) -> bool:
    root = Path(path)
    return root.is_dir() and any(p.name.startswith("step=") and p.is_dir() for p in root.iterdir())


def resolve_run_paths(cfg: TrainConfig) -> tuple[str, Path, Path]:
    run_name = cfg.resolved_run_name()
    if cfg.init_from:
        return run_name, Path(cfg.init_from), Path(cfg.output_root) / run_name
    ckpt_root = Path(cfg.ckpt_root)
    run_root = ckpt_root / run_name
    resolved_ckpt_root = run_root if run_root.is_dir() else ckpt_root
    output_root = resolved_ckpt_root if is_checkpoint_dir(resolved_ckpt_root) else Path(cfg.output_root) / run_name
    return run_name, resolved_ckpt_root, output_root


def latest_step(root: Path) -> int | None:
    latest_path = root / "latest"
    if not latest_path.exists():
        return None
    latest = latest_path.read_text(encoding="utf-8").strip()
    if latest.startswith("step="):
        return int(latest.split("=")[1])
    return int(latest)


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    ema_model: nn.Module | None,
    ckpt_root: str | Path,
    step: str,
):
    root = Path(ckpt_root)
    if not root.exists():
        mkdir(root)
        return [], 0

    subfolders = [p for p in root.iterdir() if p.name.startswith("step=") and p.is_dir()]
    if not subfolders:
        return [], 0

    take_step = latest_step(root) if step == "last" else int(step)
    if take_step is None:
        take_step = max(int(p.name.split("=")[1]) for p in subfolders)
    folder = root / step_folder_name(take_step)
    if not folder.is_dir():
        folder = root / f"step={take_step}"
    log(f"加载 checkpoint: {folder}")

    model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu"))
    if optimizer is not None and (folder / "optimizer.pt").exists():
        optimizer.load_state_dict(torch.load(folder / "optimizer.pt", map_location="cpu"))
        log("已加载 optimizer")
    if scaler is not None and (folder / "scaler.pt").exists():
        scaler.load_state_dict(torch.load(folder / "scaler.pt", map_location="cpu"))
        log("已加载 AMP scaler")
    if ema_model is not None:
        ema_path = folder / "ema.pt"
        if ema_path.exists():
            ema_model.load_state_dict(torch.load(ema_path, map_location="cpu"))
            log("已加载 EMA")
        else:
            ema_model.load_state_dict(copy.deepcopy(model.state_dict()))
            log("未找到 EMA，使用 model 初始化")
    loss_path = folder / "loss.npy"
    loss_list = list(np.load(loss_path)) if loss_path.exists() else []
    return loss_list, take_step


def save_checkpoint(
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    loss_list: list[float],
    step: int,
    output_root: Path,
    cfg: TrainConfig,
) -> None:
    folder = mkdir(output_root / step_folder_name(step))
    torch.save(model.state_dict(), folder / "model.pt")
    torch.save(ema_model.state_dict(), folder / "ema.pt")
    torch.save(optimizer.state_dict(), folder / "optimizer.pt")
    torch.save(scaler.state_dict(), folder / "scaler.pt")
    np.save(folder / "loss.npy", np.array(loss_list, dtype=np.float32))

    plt.figure(figsize=(12, 8))
    plt.plot(loss_list)
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(folder / "loss.png")
    plt.close()

    metadata = {
        "objective": "ddpm",
        "sampler": cfg.sample_scheduler,
        "global_step": step,
        "image_size": cfg.image_size,
        "guidance_scale": cfg.guidance_scale,
        "lambda_cons": 0.0,
        "git_commit_hash": git_commit_hash(),
    }
    (folder / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "latest").write_text(step_folder_name(step), encoding="utf-8")
    log(f"已保存 checkpoint 到 {folder}")


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_for_pil(x: torch.Tensor) -> Image.Image:
    x = (x.clamp(-1, 1) + 1) / 2
    x = (x * 255).byte().detach().cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(x)


@torch.no_grad()
def save_original_vs_sampled_grid(originals: torch.Tensor, samples: torch.Tensor, step: int, save_dir: Path) -> Path:
    mkdir(save_dir)
    sample_size = min(originals.size(0), samples.size(0))
    original_imgs = [normalize_for_pil(x) for x in originals[:sample_size]]
    sample_imgs = [normalize_for_pil(x) for x in samples[:sample_size]]
    width, height = original_imgs[0].size
    label_h = 24
    canvas = Image.new("RGB", (width * sample_size, height * 2 + label_h * 2), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    for i in range(sample_size):
        x0 = i * width
        draw.text((x0 + 5, 2), "ORIGINAL", fill=(0, 0, 0), font=font)
        draw.text((x0 + 5, height + label_h + 2), "SAMPLED", fill=(0, 0, 0), font=font)
        canvas.paste(original_imgs[i], (x0, label_h))
        canvas.paste(sample_imgs[i], (x0, height + label_h * 2))
    save_path = save_dir / f"step={step:09d}.jpg"
    canvas.save(save_path)
    log(f"saved visualization to {save_path}")
    return save_path


@torch.no_grad()
def save_samples(samples: torch.Tensor, prompts: list[str], output_dir: Path, prefix: str) -> None:
    mkdir(output_dir)
    imgs = (samples.clamp(-1, 1) + 1) / 2
    for i, img_tensor in enumerate(imgs.cpu()):
        safe_prompt = sanitize_name(prompts[i])[:80] or f"sample_{i}"
        img = img_tensor.permute(1, 2, 0).numpy()
        path = output_dir / f"{prefix}_{i:03d}_{safe_prompt}.png"
        plt.imsave(path, img)
        log(f"saved: {path}")


def update_ema(model: nn.Module, ema_model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for p, p_ema in zip(model.parameters(), ema_model.parameters()):
            p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


# =============================================================================
# 8. 训练主流程：真实端到端路径，数据 -> text -> loss -> EMA -> log/save/sample。
# =============================================================================


@torch.no_grad()
def sample_for_training_log(
    ema_model: nn.Module,
    schedule: DiffusionSchedule,
    images: torch.Tensor,
    cond_context: torch.Tensor,
    clip_model: nn.Module,
    cfg: TrainConfig,
    step: int,
    writer: SummaryWriter,
) -> None:
    sample_size = min(cfg.sample_size, images.size(0))
    uncond_context = encode_text_from_prompts(clip_model, [""] * sample_size, schedule.device)
    samples = sample_with_scheduler(
        model=ema_model,
        schedule=schedule,
        cond_context=cond_context[:sample_size],
        uncond_context=uncond_context,
        scheduler=cfg.sample_scheduler,
        guidance_scale=cfg.guidance_scale,
        steps=cfg.sample_steps,
        eta=cfg.sample_eta,
        shape=(sample_size, 3, cfg.image_size, cfg.image_size),
    )
    originals = (images[:sample_size].clamp(-1, 1) + 1) / 2
    sampled = (samples.clamp(-1, 1) + 1) / 2
    writer.add_image("samples/original", make_grid(originals.cpu(), nrow=sample_size), step)
    writer.add_image("samples/generated", make_grid(sampled.cpu(), nrow=sample_size), step)
    save_original_vs_sampled_grid(images[:sample_size], samples, step, Path(cfg.output_root) / cfg.run_name / "output")


def train(cfg: TrainConfig) -> None:
    if cfg.sample_scheduler not in ("ddpm", "ddim"):
        raise ValueError(f'当前 DDPM 文件只支持 sample_scheduler="ddpm" 或 "ddim"，got {cfg.sample_scheduler}')

    device = get_device(cfg.device)
    set_seed(cfg.seed)
    run_name, ckpt_root, output_root = resolve_run_paths(cfg)
    if cfg.init_from and not is_checkpoint_dir(ckpt_root):
        raise FileNotFoundError(f"--init-from 必须指向包含 step= 子目录的实验 checkpoint 目录: {ckpt_root}")

    cfg.output_root = str(output_root.parent)
    cfg.run_name = output_root.name
    mkdir(output_root)
    log_root = Path(cfg.log_root) / cfg.run_name if cfg.log_root else output_root / "tensorboard"
    mkdir(log_root)
    write_json(output_root / "config.json", asdict(cfg))
    append_jsonl(
        output_root / "run_history.jsonl",
        {
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_name": cfg.run_name,
            "start_from": str(ckpt_root),
            "config": asdict(cfg),
        },
    )

    log(f"device: {device}")
    log(f"run_name: {cfg.run_name}")
    log(f"resume checkpoint 目录: {ckpt_root}")
    log(f"checkpoint 输出目录: {output_root}")
    log(f"TensorBoard 日志目录: {log_root}")

    clip_model = load_clip_model(cfg.clip_root, device)
    model = build_model_from_config(cfg, device)
    ema_model = copy.deepcopy(model).to(device)
    for p in ema_model.parameters():
        p.requires_grad = False
    schedule = build_diffusion_schedule(cfg.timesteps, cfg.schedule, device)
    dataloader = build_dataloader(cfg)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scaler = torch.amp.GradScaler(device.type, enabled=cfg.amp and device.type == "cuda")
    loss_list, global_step = load_checkpoint(model, optimizer, scaler, ema_model, ckpt_root, step="last")
    writer = SummaryWriter(log_dir=str(log_root))
    last_saved_step = 0

    try:
        log(f"从 step={global_step} 开始训练")
        model.train()
        for epoch in range(cfg.epochs):
            for images, text_tokens, captions, cond_captions, is_dropped in dataloader:
                images = images.to(device, non_blocking=True)
                text_context = encode_text_from_tokens(clip_model, text_tokens, device)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=cfg.amp and device.type == "cuda"):
                    loss = ddpm_noise_prediction_loss(model, schedule, images, text_context)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                update_ema(model, ema_model, cfg.ema_decay)

                global_step += 1
                loss_value = float(loss.detach().cpu().item())
                grad_norm_value = float(grad_norm.detach().cpu().item())
                loss_list.append(loss_value)
                writer.add_scalar("train/loss", loss_value, global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("train/grad_norm", grad_norm_value, global_step)
                append_jsonl(
                    output_root / "metrics.jsonl",
                    {
                        "step": global_step,
                        "epoch": epoch + 1,
                        "objective": "ddpm",
                        "loss": loss_value,
                        "lr": optimizer.param_groups[0]["lr"],
                        "grad_norm": grad_norm_value,
                    },
                )

                if global_step % cfg.print_every == 0:
                    log(f"[Epoch {epoch + 1}] Step {global_step} Loss: {loss_value:.4f}")

                if global_step % cfg.sample_every == 0:
                    log(f"CFG-{cfg.sample_scheduler.upper()} sampling at step {global_step} ...")
                    sample_for_training_log(ema_model, schedule, images, text_context, clip_model, cfg, global_step, writer)
                    model.train()

                if global_step % cfg.save_every == 0:
                    save_checkpoint(model, ema_model, optimizer, scaler, loss_list, global_step, output_root, cfg)
                    last_saved_step = global_step

                if cfg.max_steps > 0 and global_step >= cfg.max_steps:
                    if last_saved_step != global_step:
                        save_checkpoint(model, ema_model, optimizer, scaler, loss_list, global_step, output_root, cfg)
                    log(f"已达到 max_steps={cfg.max_steps}，结束训练")
                    return

        save_checkpoint(model, ema_model, optimizer, scaler, loss_list, global_step, output_root, cfg)
    finally:
        writer.flush()
        writer.close()


# =============================================================================
# 9. 采样主流程：加载 EMA -> 编码 prompt -> CFG sampler -> 图片落盘。
# =============================================================================


def sample(cfg: SampleConfig) -> None:
    if cfg.scheduler not in ("ddpm", "ddim"):
        raise ValueError(f'当前 DDPM 文件只支持 scheduler="ddpm" 或 "ddim"，got {cfg.scheduler}')
    device = get_device(cfg.device)
    output_dir = mkdir(cfg.output_dir)
    ckpt_root = Path(cfg.ckpt_root)
    prompts = cfg.prompts or ["a small red car"]

    log(f"device: {device}")
    clip_model = load_clip_model(cfg.clip_root, device)
    model = build_model_from_config(cfg, device)
    ema_model = copy.deepcopy(model)
    load_checkpoint(model, optimizer=None, scaler=None, ema_model=ema_model, ckpt_root=ckpt_root, step=cfg.step)
    ema_model.to(device).eval()
    schedule = build_diffusion_schedule(cfg.timesteps, cfg.schedule, device)

    for guidance_scale in cfg.guidance_scale_list():
        log(f"sampling {cfg.scheduler.upper()} CFG scale = {guidance_scale}")
        for prompt_batch in chunks(prompts, cfg.batch_size):
            prompt_batch = list(prompt_batch)
            cond_context = encode_text_from_prompts(clip_model, prompt_batch, device)
            uncond_context = encode_text_from_prompts(clip_model, [""] * len(prompt_batch), device)
            samples = sample_with_scheduler(
                model=ema_model,
                schedule=schedule,
                cond_context=cond_context,
                uncond_context=uncond_context,
                scheduler=cfg.scheduler,
                guidance_scale=guidance_scale,
                steps=cfg.steps,
                eta=cfg.eta,
                shape=(len(prompt_batch), 3, cfg.image_size, cfg.image_size),
            )
            prefix = f"{cfg.scheduler}_cfg{guidance_scale}_steps{cfg.steps}"
            save_samples(samples, prompt_batch, output_dir, prefix)


# =============================================================================
# 10. CLI：一个文件两个子命令，保持运行入口清楚。
# =============================================================================


def add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--coco-img-root", required=True)
    parser.add_argument("--coco-ann-file", required=True)
    parser.add_argument("--clip-root", required=True)
    parser.add_argument("--output-root", default="checkpoints")
    parser.add_argument("--ckpt-root", default="checkpoints")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--init-from", default="")
    parser.add_argument("--log-root", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--text-dropout", type=float, default=0.1)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--schedule", choices=["linear", "quadratic", "cosine"], default="quadratic")
    parser.add_argument("--sample-scheduler", choices=["ddpm", "ddim"], default="ddim")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ema-decay", type=float, default=0.9995)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--sample-every", type=int, default=2500)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--sample-eta", type=float, default=0.0)
    parser.add_argument("--sample-size", type=int, default=4)
    parser.add_argument("--time-emb-dim", type=int, default=256)
    parser.add_argument("--text-dim", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=128)
    parser.add_argument("--channel-mults", default="1,2,4,8")
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--amp", dest="amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--seed", type=int, default=42)


def add_sample_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ckpt-root", required=True)
    parser.add_argument("--clip-root", required=True)
    parser.add_argument("--output-dir", default="samples")
    parser.add_argument("--step", default="last")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--schedule", choices=["linear", "quadratic", "cosine"], default="quadratic")
    parser.add_argument("--scheduler", choices=["ddpm", "ddim"], default="ddim")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--guidance-scales", default="5.0")
    parser.add_argument("--prompts", nargs="*", default=None)
    parser.add_argument("--time-emb-dim", type=int, default=256)
    parser.add_argument("--text-dim", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=128)
    parser.add_argument("--channel-mults", default="1,2,4,8")
    parser.add_argument("--groups", type=int, default=8)


def main() -> None:
    parser = argparse.ArgumentParser(description="CleanDiffusion DDPM single-file baseline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train", help="train DDPM")
    add_train_args(train_parser)
    sample_parser = subparsers.add_parser("sample", help="sample from a DDPM checkpoint")
    add_sample_args(sample_parser)
    args = parser.parse_args()

    if args.command == "train":
        kwargs = vars(args)
        kwargs.pop("command")
        train(TrainConfig(**kwargs))
    elif args.command == "sample":
        kwargs = vars(args)
        kwargs.pop("command")
        sample(SampleConfig(**kwargs))
    else:
        raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
