from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DatasetConfig:
    image_channels: int
    image_size: int
    dataset_name: str


def ensure_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def _build_transform(image_size: int, image_channels: int) -> transforms.Compose:
    transform_steps = [transforms.Resize((image_size, image_size))]
    if image_channels == 1:
        transform_steps.append(transforms.Grayscale(num_output_channels=1))
    elif image_channels == 3:
        transform_steps.append(transforms.Lambda(ensure_rgb))
    else:
        raise ValueError("image_channels must be 1 or 3.")

    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * image_channels, (0.5,) * image_channels),
        ]
    )
    return transforms.Compose(transform_steps)


def build_dataloader(
    dataset_name: str,
    data_root: str,
    batch_size: int,
    image_size: int,
    image_channels: int,
    num_workers: int,
) -> tuple[DataLoader, DatasetConfig]:
    normalized_name = dataset_name.lower()
    transform = _build_transform(image_size=image_size, image_channels=image_channels)
    root_path = Path(data_root)

    if normalized_name == "mnist":
        dataset = datasets.MNIST(root=root_path, train=True, download=True, transform=transform)
        config = DatasetConfig(image_channels=1, image_size=image_size, dataset_name="mnist")
    elif normalized_name == "fashionmnist":
        dataset = datasets.FashionMNIST(root=root_path, train=True, download=True, transform=transform)
        config = DatasetConfig(image_channels=1, image_size=image_size, dataset_name="fashionmnist")
    elif normalized_name == "cifar10":
        dataset = datasets.CIFAR10(root=root_path, train=True, download=True, transform=transform)
        config = DatasetConfig(image_channels=3, image_size=image_size, dataset_name="cifar10")
    elif normalized_name == "imagefolder":
        dataset = datasets.ImageFolder(root=root_path, transform=transform)
        config = DatasetConfig(
            image_channels=image_channels,
            image_size=image_size,
            dataset_name="imagefolder",
        )
    else:
        raise ValueError("dataset must be one of: mnist, fashionmnist, cifar10, imagefolder")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    return dataloader, config
