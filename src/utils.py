from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torchvision.utils import make_grid, save_image


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def resolve_device(requested_device: str) -> torch.device:
    normalized = requested_device.lower()
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Use --device cpu, --device mps, or install a CUDA-enabled PyTorch build.")
        return torch.device("cuda")
    if normalized == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available. On Apple Silicon, install an MPS-enabled PyTorch build or use --device cpu.")
        return torch.device("mps")
    raise ValueError("device must be one of: cpu, cuda, mps")


def make_noise(batch_size: int, latent_dim: int, device: torch.device) -> torch.Tensor:
    return torch.randn(batch_size, latent_dim, 1, 1, device=device)


def build_checkpoint(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    args: Any,
    dataset_name: str,
    image_channels: int,
) -> dict[str, Any]:
    return {
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "config": {
            "latent_dim": args.latent_dim,
            "image_size": args.image_size,
            "image_channels": image_channels,
            "base_channels": args.base_channels,
            "dataset": dataset_name,
            "gan_mode": args.gan_mode,
        },
    }


@torch.no_grad()
def save_sample_grid(
    generator: torch.nn.Module,
    latent_dim: int,
    device: torch.device,
    output_path: str | Path,
    num_images: int = 16,
) -> None:
    generator.eval()
    noise = make_noise(num_images, latent_dim, device)
    generated_images = generator(noise).cpu()
    generated_images = (generated_images + 1) / 2
    nrow = max(1, math.isqrt(num_images))
    grid = make_grid(generated_images, nrow=nrow, padding=2)
    save_image(grid, output_path)
    generator.train()
