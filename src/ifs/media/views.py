"""Rendered views of one image.

`crops.py` decides what a *detector* sees. This decides what a *reader* sees --
a reader being whatever downstream model will eventually look at the image.
Prompt injections live in the gap between the two: text that is illegible at
native resolution but sharp once the victim's preprocessing has resized it,
text buried in an alpha channel that flattening reveals, text a contrast
stretch pulls out of a near-white background.

Which views are rendered is forensic policy, not an implementation detail. It
decides what this service can and cannot see, so it is part of the pipeline
fingerprint -- exactly as crop policy is.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Literal

from PIL import Image, ImageOps

log = logging.getLogger(__name__)

Normalize = Literal["alpha_black", "alpha_white", "contrast_stretch", "lsb"]
Resample = Literal["nearest", "bilinear", "bicubic", "lanczos", "area"]

# `area` is OpenCV's INTER_AREA, whose PIL equivalent is BOX. The names here
# are the ones attackers and victims use, not PIL's.
_FILTERS: dict[str, Image.Resampling] = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
    "area": Image.Resampling.BOX,
}


@dataclass(frozen=True)
class View:
    name: str
    """Stable identifier, e.g. `native`, `normalize:lsb`, `rescale:384:bicubic`."""

    kind: Literal["native", "normalize", "rescale"]
    image: Image.Image

    @property
    def is_native(self) -> bool:
        return self.kind == "native"


def expand(img: Image.Image, cfg) -> list[View]:
    """Render the configured views. `native` is always first.

    A pass that cannot apply to this image returns nothing rather than a copy
    of the original: a duplicate view would cost a full OCR call to re-read
    text we already have, and would then look like a *differential* finding
    against itself.
    """
    views = [View("native", "native", _rgb(img))]

    for name in cfg.normalize:
        rendered = _normalize(img, name)
        if rendered is not None:
            views.append(View(f"normalize:{name}", "normalize", rendered))

    long_edge = max(img.size)
    for size in sorted(set(cfg.rescale.sizes), reverse=True):
        # Upscaling reveals nothing. These attacks are built around the
        # victim *shrinking* the image, so a target above the native size is
        # not a view any victim would ever produce.
        if size >= long_edge:
            continue
        for name in cfg.rescale.filters:
            views.append(View(f"rescale:{size}:{name}", "rescale", _rescale(img, size, name)))

    if len(views) > cfg.max_views:
        log.warning(
            "view policy produced %d views, truncating to max_views=%d -- "
            "the dropped views are not inspected",
            len(views),
            cfg.max_views,
        )
        views = views[: cfg.max_views]
    return views


def policy_fingerprint(cfg) -> str:
    """Short stable digest of the view policy, for the pipeline fingerprint.

    Hashed rather than spelled out because the full policy is long and the
    fingerprint is meant to be compared, not read. `/v1/inspect` returns the
    policy in full alongside it.
    """
    parts = [
        "native",
        *sorted(cfg.normalize),
        *(f"{s}:{f}" for s in sorted(set(cfg.rescale.sizes)) for f in sorted(cfg.rescale.filters)),
    ]
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]
    return f"views@{digest}"


def describe(cfg) -> dict:
    """The policy in full, for `/v1/inspect` and `/v1/models`."""
    return {
        "normalize": list(cfg.normalize),
        "rescale": {"sizes": list(cfg.rescale.sizes), "filters": list(cfg.rescale.filters)},
        "max_views": cfg.max_views,
        "fingerprint": policy_fingerprint(cfg),
    }


def _rgb(img: Image.Image) -> Image.Image:
    return img if img.mode == "RGB" else img.convert("RGB")


def _rescale(img: Image.Image, size: int, filter_name: str) -> Image.Image:
    """Fit the long edge to `size`, preserving aspect.

    Long-edge fit is the policy here because it is what the common vision
    preprocessors do. A victim resizing differently sees a different image,
    which is precisely why the filter and size list are configurable and
    fingerprinted rather than fixed.
    """
    w, h = img.size
    scale = size / max(w, h)
    target = (max(1, round(w * scale)), max(1, round(h * scale)))
    return _rgb(img).resize(target, _FILTERS[filter_name])


def _normalize(img: Image.Image, name: str) -> Image.Image | None:
    if name == "alpha_black":
        return _flatten(img, (0, 0, 0, 255))
    if name == "alpha_white":
        return _flatten(img, (255, 255, 255, 255))
    if name == "contrast_stretch":
        return _stretch(img)
    if name == "lsb":
        return _lsb(img)
    raise ValueError(f"unknown normalization {name!r}")


def _flatten(img: Image.Image, color: tuple[int, int, int, int]) -> Image.Image | None:
    """Composite onto a flat background. Reveals text hidden in the alpha channel."""
    if img.mode not in ("RGBA", "LA", "PA") and "transparency" not in img.info:
        return None
    rgba = img.convert("RGBA")
    background = Image.new("RGBA", rgba.size, color)
    return Image.alpha_composite(background, rgba).convert("RGB")


def _stretch(img: Image.Image) -> Image.Image | None:
    """Per-channel autocontrast. Reveals near-white-on-white and near-black-on-black text."""
    rgb = _rgb(img)
    extrema = rgb.getextrema()
    # Already spanning the full range in every channel: autocontrast would be
    # the identity, and we would pay an OCR call to re-read the native view.
    if all(lo == 0 and hi == 255 for lo, hi in extrema):
        return None
    return ImageOps.autocontrast(rgb, cutoff=0)


def _lsb(img: Image.Image) -> Image.Image:
    """The least-significant bit plane, as hard black and white.

    Text written into the low bit is invisible to a human and survives into a
    model's tensor. Rendering the plane at full contrast is what makes it
    legible to OCR.
    """
    return img.convert("L").point(lambda v: 255 if v & 1 else 0)
