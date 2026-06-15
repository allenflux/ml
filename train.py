from __future__ import annotations

import argparse
import os

import torch
from torch import nn, optim

from src.data import build_dataloader
from src.models import Discriminator, Generator, weights_init
from src.utils import build_checkpoint, ensure_dir, make_noise, resolve_device, save_sample_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a heavy image generator with DCGAN or WGAN-GP.")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--gan-mode", type=str, default="wgan-gp", choices=("dcgan", "wgan-gp"))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--image-channels", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--critic-steps", type=int, default=5)
    parser.add_argument("--gp-lambda", type=float, default=10.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=max(1, (os.cpu_count() or 8) - 1))
    parser.add_argument("--profile", type=str, default="standard", choices=("standard", "heavy-cpu", "overnight-cpu"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def apply_profile(args: argparse.Namespace) -> argparse.Namespace:
    if args.profile == "heavy-cpu":
        args.gan_mode = "wgan-gp"
        args.epochs = max(args.epochs, 220)
        args.batch_size = max(args.batch_size, 128)
        args.latent_dim = max(args.latent_dim, 160)
        args.base_channels = max(args.base_channels, 96)
        args.critic_steps = max(args.critic_steps, 5)
        args.num_workers = max(args.num_workers, 6)
        args.cpu_threads = max(args.cpu_threads, 8)
        args.sample_every = max(args.sample_every, 10)
        args.save_every = max(args.save_every, 25)
    elif args.profile == "overnight-cpu":
        args.gan_mode = "wgan-gp"
        args.epochs = max(args.epochs, 320)
        args.batch_size = max(args.batch_size, 144)
        args.latent_dim = max(args.latent_dim, 192)
        args.base_channels = max(args.base_channels, 128)
        args.critic_steps = max(args.critic_steps, 6)
        args.num_workers = max(args.num_workers, 8)
        args.cpu_threads = max(args.cpu_threads, 10)
        args.sample_every = max(args.sample_every, 12)
        args.save_every = max(args.save_every, 30)
    return args


def compute_gradient_penalty(
    critic: nn.Module,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    alpha = torch.rand(real_images.size(0), 1, 1, 1, device=device)
    interpolated = (alpha * real_images + (1.0 - alpha) * fake_images).requires_grad_(True)
    mixed_scores = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=mixed_scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    return ((gradient_norm - 1.0) ** 2).mean()


def train() -> None:
    args = apply_profile(parse_args())
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(max(1, min(4, args.cpu_threads // 2 or 1)))
    device = resolve_device(args.device)

    if args.image_size != 64:
        raise ValueError("This upgraded project currently supports --image-size 64.")

    samples_dir = ensure_dir("outputs/samples")
    checkpoints_dir = ensure_dir("outputs/checkpoints")

    dataloader, dataset_config = build_dataloader(
        dataset_name=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        image_channels=args.image_channels,
        num_workers=args.num_workers,
    )

    generator = Generator(
        latent_dim=args.latent_dim,
        image_channels=dataset_config.image_channels,
        base_channels=args.base_channels,
        image_size=args.image_size,
    ).to(device)
    discriminator = Discriminator(
        image_channels=dataset_config.image_channels,
        base_channels=args.base_channels,
        image_size=args.image_size,
        use_sigmoid=args.gan_mode == "dcgan",
    ).to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    criterion = nn.BCELoss() if args.gan_mode == "dcgan" else None
    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    print(
        "Training setup:",
        f"mode={args.gan_mode}",
        f"profile={args.profile}",
        f"dataset={dataset_config.dataset_name}",
        f"device={device}",
        f"batch_size={args.batch_size}",
        f"epochs={args.epochs}",
        f"cpu_threads={args.cpu_threads}",
        f"workers={args.num_workers}",
    )

    for epoch in range(1, args.epochs + 1):
        g_running_loss = 0.0
        d_running_loss = 0.0
        gp_running = 0.0

        for real_images, _ in dataloader:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            if args.gan_mode == "dcgan":
                real_labels = torch.ones(batch_size, 1, device=device)
                fake_labels = torch.zeros(batch_size, 1, device=device)

                optimizer_d.zero_grad(set_to_none=True)
                real_outputs = discriminator(real_images)
                d_loss_real = criterion(real_outputs, real_labels)

                noise = make_noise(batch_size, args.latent_dim, device)
                fake_images = generator(noise)
                fake_outputs = discriminator(fake_images.detach())
                d_loss_fake = criterion(fake_outputs, fake_labels)
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                optimizer_d.step()

                optimizer_g.zero_grad(set_to_none=True)
                fake_outputs = discriminator(fake_images)
                g_loss = criterion(fake_outputs, real_labels)
                g_loss.backward()
                optimizer_g.step()
            else:
                critic_loss_value = 0.0
                gp_value = 0.0
                for _ in range(args.critic_steps):
                    optimizer_d.zero_grad(set_to_none=True)
                    noise = make_noise(batch_size, args.latent_dim, device)
                    fake_images = generator(noise)
                    critic_real = discriminator(real_images)
                    critic_fake = discriminator(fake_images.detach())
                    gradient_penalty = compute_gradient_penalty(discriminator, real_images, fake_images.detach(), device)
                    d_loss = -(critic_real.mean() - critic_fake.mean()) + args.gp_lambda * gradient_penalty
                    d_loss.backward()
                    optimizer_d.step()
                    critic_loss_value = d_loss.item()
                    gp_value = gradient_penalty.item()

                optimizer_g.zero_grad(set_to_none=True)
                noise = make_noise(batch_size, args.latent_dim, device)
                fake_images = generator(noise)
                g_loss = -discriminator(fake_images).mean()
                g_loss.backward()
                optimizer_g.step()
                d_loss = torch.tensor(critic_loss_value, device=device)
                gp_running += gp_value

            g_running_loss += g_loss.item()
            d_running_loss += d_loss.item()

        avg_g_loss = g_running_loss / len(dataloader)
        avg_d_loss = d_running_loss / len(dataloader)
        if args.gan_mode == "wgan-gp":
            avg_gp = gp_running / len(dataloader)
            print(
                f"Epoch [{epoch}/{args.epochs}]  Critic Loss: {avg_d_loss:.4f}  "
                f"Generator Loss: {avg_g_loss:.4f}  GP: {avg_gp:.4f}"
            )
        else:
            print(f"Epoch [{epoch}/{args.epochs}]  D Loss: {avg_d_loss:.4f}  G Loss: {avg_g_loss:.4f}")

        if epoch % args.sample_every == 0 or epoch == 1 or epoch == args.epochs:
            sample_path = samples_dir / f"{dataset_config.dataset_name}_epoch_{epoch:03d}.png"
            save_sample_grid(generator, args.latent_dim, device, sample_path)

        if epoch % args.save_every == 0 or epoch == args.epochs:
            checkpoint = build_checkpoint(
                generator=generator,
                discriminator=discriminator,
                args=args,
                dataset_name=dataset_config.dataset_name,
                image_channels=dataset_config.image_channels,
            )
            torch.save(checkpoint, checkpoints_dir / f"{dataset_config.dataset_name}_epoch_{epoch:03d}.pt")

    final_checkpoint = build_checkpoint(
        generator=generator,
        discriminator=discriminator,
        args=args,
        dataset_name=dataset_config.dataset_name,
        image_channels=dataset_config.image_channels,
    )
    torch.save(final_checkpoint, checkpoints_dir / "generator_last.pt")
    print(f"Training finished. Checkpoints saved to: {checkpoints_dir.resolve()}")
    print(f"Sample images saved to: {samples_dir.resolve()}")


if __name__ == "__main__":
    train()
