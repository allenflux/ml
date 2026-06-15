from __future__ import annotations

import torch
from torch import nn


def weights_init(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        image_channels: int = 3,
        base_channels: int = 64,
        image_size: int = 64,
    ) -> None:
        super().__init__()
        if image_size != 64:
            raise ValueError("This DCGAN generator currently supports image_size=64.")

        self.latent_dim = latent_dim
        self.image_channels = image_channels
        self.base_channels = base_channels
        self.image_size = image_size

        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base_channels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels, image_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        if noise.ndim == 2:
            noise = noise.view(noise.size(0), noise.size(1), 1, 1)
        return self.model(noise)


class Discriminator(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        base_channels: int = 64,
        image_size: int = 64,
        use_sigmoid: bool = True,
    ) -> None:
        super().__init__()
        if image_size != 64:
            raise ValueError("This DCGAN discriminator currently supports image_size=64.")

        layers: list[nn.Module] = [
            nn.Conv2d(image_channels, base_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 8, 1, 4, 1, 0, bias=False),
            nn.Flatten(),
        ]
        if use_sigmoid:
            layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)
