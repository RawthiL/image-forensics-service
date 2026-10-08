"""Authoritative catalog of what this backend can load.

Two separate concepts, deliberately not conflated:

* a **variant** is an architecture + its preprocessing. It decides how a
  checkpoint is instantiated and how pixels reach it.
* a **model** is a named, published checkpoint, pinned to a variant and a
  revision.

Selecting a model by `name` resolves through MODELS. Selecting by `path`
requires naming the `variant` explicitly, because a bare file on disk carries no
statement of which architecture it is. Nothing here is inferred from image size
or any other runtime property.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Variant:
    """An architecture and the preprocessing that belongs to it."""

    timm_model: str
    input_size: int
    resize_to: int
    """Shorter-side resize before the center crop, from upstream's
    `determine_resize_crop_sizes`."""


@dataclass(frozen=True)
class Model:
    """A published checkpoint."""

    variant: str
    repo_id: str
    filename: str
    revision: str
    """Pinned commit sha. A published model is never 'whatever main holds today'."""

    description: str = ""


VARIANTS: dict[str, Variant] = {
    "vit-s16-224": Variant(
        timm_model="vit_small_patch16_224.augreg_in21k_ft_in1k", input_size=224, resize_to=256
    ),
    "vit-s16-384": Variant(
        timm_model="vit_small_patch16_384.augreg_in21k_ft_in1k", input_size=384, resize_to=440
    ),
}

MODELS: dict[str, Model] = {
    "commfor-384": Model(
        variant="vit-s16-384",
        repo_id="OwensLab/commfor-model-384",
        filename="model.safetensors",
        revision="6076002bf0d9dd37537f965ee2f06f826c333b61",
        description="Community Forensics ViT-S/16 @384. The stronger checkpoint.",
    ),
    "commfor-224": Model(
        variant="vit-s16-224",
        repo_id="OwensLab/commfor-model-224",
        filename="model.safetensors",
        revision="26afc31e6b40c312c3fd42c05a758be62446215b",
        description="Community Forensics ViT-S/16 @224. Faster, measurably weaker.",
    ),
}


class CatalogError(ValueError):
    """A configuration naming something this backend cannot load."""


def resolve_model(name: str) -> Model:
    try:
        return MODELS[name]
    except KeyError:
        raise CatalogError(
            f"unknown model {name!r} for backend 'community-forensics'. "
            f"Known models: {', '.join(sorted(MODELS))}. "
            f"To load a checkpoint not in this catalog, set detector.path and "
            f"detector.variant instead."
        ) from None


def resolve_variant(name: str) -> Variant:
    try:
        return VARIANTS[name]
    except KeyError:
        raise CatalogError(
            f"unknown variant {name!r} for backend 'community-forensics'. "
            f"Known variants: {', '.join(sorted(VARIANTS))}."
        ) from None
