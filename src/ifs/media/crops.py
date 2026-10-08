"""Crop policies.

Detectors are typically evaluated with a resize-to-short-side plus a fixed
center crop, so on a large photograph the model never sees most of the frame.
That is right for reproducing a paper's numbers and wrong for a service whose
callers upload whole photographs.

`five` and `tiled` trade eval parity for coverage. Whichever you pick is part of
the pipeline fingerprint -- the service applies it, backends know nothing of it.
"""

from __future__ import annotations

from typing import Literal

from PIL import Image

Policy = Literal["center", "five", "tiled"]
Aggregation = Literal["max", "mean"]


def expand(img: Image.Image, policy: Policy, input_size: int) -> list[Image.Image]:
    """Return the crops to score. `center` returns the image untouched --
    the detector's own transform does the resize+crop in that case."""
    if policy == "center":
        return [img]
    if policy == "five":
        return _five_crop(img, input_size)
    if policy == "tiled":
        return _tiles(img, input_size)
    raise ValueError(f"unknown crop policy {policy!r}")


def aggregate(scores: list[float], how: Aggregation) -> float:
    if not scores:
        raise ValueError("no scores to aggregate")
    return max(scores) if how == "max" else sum(scores) / len(scores)


def _five_crop(img: Image.Image, size: int) -> list[Image.Image]:
    w, h = img.size
    # Below 2x the input there is nothing to gain: the crops would overlap
    # almost entirely and we'd pay 5x compute for the same answer.
    if min(w, h) < size * 2:
        return [img]
    s = min(w, h) // 2
    boxes = [
        (0, 0, s, s),
        (w - s, 0, w, s),
        (0, h - s, s, h),
        (w - s, h - s, w, h),
        ((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2),
    ]
    return [img.crop(b) for b in boxes]


def _tiles(img: Image.Image, size: int, max_tiles: int = 9) -> list[Image.Image]:
    w, h = img.size
    if min(w, h) < size * 2:
        return [img]
    cols = min(3, max(1, w // size))
    rows = min(3, max(1, h // size))
    tw, th = w // cols, h // rows
    out = [
        img.crop((c * tw, r * th, (c + 1) * tw, (r + 1) * th))
        for r in range(rows)
        for c in range(cols)
    ]
    return out[:max_tiles]
