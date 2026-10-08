"""Preprocessing, reproduced exactly from upstream.

Source: `dataloader.get_transform(mode="test")` and
`dataloader.determine_resize_crop_sizes` in
https://github.com/JeongsooP/Community-Forensics

Do not "improve" this. The score is far more sensitive to resize and crop than
to anything else, and drift here makes two deployments disagree.

We reimplement it rather than importing upstream's `dataprocessor_hf`, which
pulls in the entire training stack (wandb, torchrun helpers, augmentation ops)
to build four torchvision transforms.
"""

from __future__ import annotations

from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int, resize_to: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(resize_to),  # shorter side, aspect preserved
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),  # -> [0, 1]
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def as_rgb(img: Image.Image) -> Image.Image:
    """Palette and LA/RGBA images would otherwise yield 1- or 4-channel tensors."""
    return img if img.mode == "RGB" else img.convert("RGB")
