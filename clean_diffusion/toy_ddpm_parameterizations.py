#!/usr/bin/env python3
"""DDPM 参数化转换的最小可执行校验。

这个脚本不训练模型，只验证 epsilon / x0 / v / score 四种输出在同一个
DDPM 加噪公式下如何互相转换。它适合作为阅读 `clean_diffusion/ddpm.py`
之前或之后的公式检查。

DDPM 加噪公式：

  x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps

记：

  a = sqrt(alpha_bar_t)
  s = sqrt(1 - alpha_bar_t)

四种参数化：

  epsilon: eps
  x0:      x0
  v:       a * eps - s * x0
  score:  -eps / s

运行：

python clean_diffusion/toy_ddpm_parameterizations.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch


@dataclass
class Config:
    batch_size: int = 4
    channels: int = 3
    image_size: int = 8
    timesteps: int = 1000
    seed: int = 42
    atol: float = 1e-5


def make_linear_alpha_bar(timesteps: int) -> torch.Tensor:
    beta = torch.linspace(1e-4, 0.02, timesteps)
    alpha = 1.0 - beta
    return torch.cumprod(alpha, dim=0)


def expand_like(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return value.view(-1, *([1] * (target.ndim - 1)))


def q_sample(x0: torch.Tensor, eps: torch.Tensor, alpha_bar_t: torch.Tensor) -> torch.Tensor:
    a = expand_like(torch.sqrt(alpha_bar_t), x0)
    s = expand_like(torch.sqrt(1.0 - alpha_bar_t), x0)
    return a * x0 + s * eps


def eps_to_x0(x_t: torch.Tensor, eps_hat: torch.Tensor, alpha_bar_t: torch.Tensor) -> torch.Tensor:
    a = expand_like(torch.sqrt(alpha_bar_t), x_t)
    s = expand_like(torch.sqrt(1.0 - alpha_bar_t), x_t)
    return (x_t - s * eps_hat) / a.clamp_min(1e-12)


def x0_to_eps(x_t: torch.Tensor, x0_hat: torch.Tensor, alpha_bar_t: torch.Tensor) -> torch.Tensor:
    a = expand_like(torch.sqrt(alpha_bar_t), x_t)
    s = expand_like(torch.sqrt(1.0 - alpha_bar_t), x_t)
    return (x_t - a * x0_hat) / s.clamp_min(1e-12)


def eps_x0_to_v(x0: torch.Tensor, eps: torch.Tensor, alpha_bar_t: torch.Tensor) -> torch.Tensor:
    a = expand_like(torch.sqrt(alpha_bar_t), x0)
    s = expand_like(torch.sqrt(1.0 - alpha_bar_t), x0)
    return a * eps - s * x0


def v_to_x0(x_t: torch.Tensor, v_hat: torch.Tensor, alpha_bar_t: torch.Tensor) -> torch.Tensor:
    a = expand_like(torch.sqrt(alpha_bar_t), x_t)
    s = expand_like(torch.sqrt(1.0 - alpha_bar_t), x_t)
    return a * x_t - s * v_hat


def v_to_eps(x_t: torch.Tensor, v_hat: torch.Tensor, alpha_bar_t: torch.Tensor) -> torch.Tensor:
    a = expand_like(torch.sqrt(alpha_bar_t), x_t)
    s = expand_like(torch.sqrt(1.0 - alpha_bar_t), x_t)
    return s * x_t + a * v_hat


def eps_to_score(eps_hat: torch.Tensor, alpha_bar_t: torch.Tensor) -> torch.Tensor:
    s = expand_like(torch.sqrt(1.0 - alpha_bar_t), eps_hat)
    return -eps_hat / s.clamp_min(1e-12)


def score_to_eps(score_hat: torch.Tensor, alpha_bar_t: torch.Tensor) -> torch.Tensor:
    s = expand_like(torch.sqrt(1.0 - alpha_bar_t), score_hat)
    return -s * score_hat


def score_to_x0(x_t: torch.Tensor, score_hat: torch.Tensor, alpha_bar_t: torch.Tensor) -> torch.Tensor:
    a = expand_like(torch.sqrt(alpha_bar_t), x_t)
    s2 = expand_like(1.0 - alpha_bar_t, x_t)
    return (x_t + s2 * score_hat) / a.clamp_min(1e-12)


def max_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max().item())


def run_check(cfg: Config) -> None:
    torch.manual_seed(cfg.seed)
    shape = (cfg.batch_size, cfg.channels, cfg.image_size, cfg.image_size)

    alpha_bar = make_linear_alpha_bar(cfg.timesteps)
    t = torch.randint(1, cfg.timesteps, (cfg.batch_size,))
    alpha_bar_t = alpha_bar[t]

    x0 = torch.randn(shape)
    eps = torch.randn(shape)
    x_t = q_sample(x0, eps, alpha_bar_t)

    v = eps_x0_to_v(x0, eps, alpha_bar_t)
    score = eps_to_score(eps, alpha_bar_t)

    checks = {
        "epsilon -> x0": max_error(eps_to_x0(x_t, eps, alpha_bar_t), x0),
        "x0 -> epsilon": max_error(x0_to_eps(x_t, x0, alpha_bar_t), eps),
        "v -> x0": max_error(v_to_x0(x_t, v, alpha_bar_t), x0),
        "v -> epsilon": max_error(v_to_eps(x_t, v, alpha_bar_t), eps),
        "score -> epsilon": max_error(score_to_eps(score, alpha_bar_t), eps),
        "score -> x0": max_error(score_to_x0(x_t, score, alpha_bar_t), x0),
    }

    for name, error in checks.items():
        print(f"{name:18s} max_error={error:.8f}")
        if error > cfg.atol:
            raise AssertionError(f"{name} error {error:.8f} > atol {cfg.atol}")

    print("all DDPM parameterization conversions passed")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Check DDPM parameterization conversions")
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--channels", type=int, default=Config.channels)
    parser.add_argument("--image-size", type=int, default=Config.image_size)
    parser.add_argument("--timesteps", type=int, default=Config.timesteps)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--atol", type=float, default=Config.atol)
    return Config(**vars(parser.parse_args()))


def main() -> None:
    run_check(parse_args())


if __name__ == "__main__":
    main()
