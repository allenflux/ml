from __future__ import annotations

import argparse

import torch

from src.models import Generator
from src.utils import ensure_dir, resolve_device, save_sample_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images from a trained DCGAN generator.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--image-channels", type=int, default=None)
    parser.add_argument("--num-images", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config", {})

    latent_dim = args.latent_dim or config.get("latent_dim", 128)
    base_channels = args.base_channels or config.get("base_channels", 64)
    image_size = args.image_size or config.get("image_size", 64)
    image_channels = args.image_channels or config.get("image_channels", 3)

    generator = Generator(
        latent_dim=latent_dim,
        image_channels=image_channels,
        base_channels=base_channels,
        image_size=image_size,
    ).to(device)

    state_dict = checkpoint.get("generator_state_dict", checkpoint)
    generator.load_state_dict(state_dict)

    output_dir = ensure_dir("outputs/generated")
    output_path = output_dir / "generated_grid.png"
    save_sample_grid(
        generator=generator,
        latent_dim=latent_dim,
        device=device,
        output_path=output_path,
        num_images=args.num_images,
    )
    print(f"Generated images saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
