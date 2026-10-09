"""Preprocessing in numpy, for the runtimes that do not import torch.

Byte-identical to the torch transform in `preprocess.py` -- a test asserts
`max|diff| == 0` across aspect ratios. Both exist because the torch backend
reuses torchvision's transform while the ONNX and Triton backends must not
import torch at all; they share this module so they cannot drift from each
other.

If you change one, change the other, and expect the parity test to catch you.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)


def preprocess(img: Image.Image, *, input_size: int, resize_to: int) -> np.ndarray:
    """PIL image -> CHW float32 tensor, normalized. One image, no batch axis."""
    if img.mode != "RGB":
        img = img.convert("RGB")

    # torchvision's Resize(int) scales the SHORTER side and truncates with
    # int(); reproducing that exactly is what makes the two paths agree.
    w, h = img.size
    if w <= h:
        nw, nh = resize_to, int(resize_to * h / w)
    else:
        nw, nh = int(resize_to * w / h), resize_to
    img = img.resize((nw, nh), Image.BILINEAR)

    # torchvision's center_crop computes `int(round((dim - crop) / 2.0))`.
    # Python's round() is banker's rounding and already returns an int, which
    # is what torchvision gets too -- so this matches it exactly.
    left = round((nw - input_size) / 2.0)
    top = round((nh - input_size) / 2.0)
    img = img.crop((left, top, left + input_size, top + input_size))

    arr = np.asarray(img, np.float32).transpose(2, 0, 1) / 255.0
    return (arr - IMAGENET_MEAN) / IMAGENET_STD


def batch(images, *, input_size: int, resize_to: int) -> np.ndarray:
    """NCHW float32 batch."""
    return np.stack([preprocess(i, input_size=input_size, resize_to=resize_to) for i in images])


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable, and always in float64->float32 to match the torch
    backend, which deliberately runs the sigmoid in fp32."""
    x = x.astype(np.float32, copy=False)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x))).astype(
        np.float32
    )
