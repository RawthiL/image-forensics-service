from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image

from ....runtime.device import resolve_device, resolve_dtype
from ...base import DetectionResult, DetectorInfo
from ...weights import resolve_catalog, resolve_path
from .catalog import VARIANTS, CatalogError, Variant, resolve_model, resolve_variant
from .preprocess import as_rgb, build_transform

log = logging.getLogger(__name__)


class CommunityForensicsDetector:
    """Single ViT-S/16 with a one-logit head.

    Exactly one of `model` (a catalog name) or `path` (a checkpoint file) must
    be given. With `path`, `variant` is required -- a file on disk does not say
    which architecture it is, and guessing is how you silently load the wrong
    thing.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        path: str | None = None,
        variant: str | None = None,
        revision: str | None = None,
        weights_source: str = "auto",
        weights_dir: str | None = None,
        allow_download: bool = True,
        device: str | None = None,
        dtype: str | None = None,
        **_: object,
    ) -> None:
        self._model_name, self._path, self._variant_name, self._revision_override = _select(
            model=model, path=path, variant=variant, revision=revision
        )
        self._variant: Variant = resolve_variant(self._variant_name)
        self._weights_source = weights_source
        self._weights_dir = weights_dir
        self._allow_download = allow_download
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(dtype, self.device)
        self._resolved_revision: str | None = None
        self._weights_origin: str | None = None
        self._net: torch.nn.Module | None = None
        self._tf = build_transform(self._variant.input_size, self._variant.resize_to)

    # -- Detector protocol -------------------------------------------------

    def info(self) -> DetectorInfo:
        return DetectorInfo(
            name="community-forensics",
            version=self._model_name or f"path:{self._variant_name}",
            revision=self._resolved_revision,
            device=str(self.device),
            dtype=str(self.dtype).removeprefix("torch."),
            input_size=self._variant.input_size,
        )

    def warmup(self) -> None:
        if self._net is not None:
            return
        import timm

        if self._path is not None:
            found = resolve_path(self._path)
        else:
            entry = resolve_model(self._model_name)  # type: ignore[arg-type]
            found = resolve_catalog(
                repo_id=entry.repo_id,
                filename=entry.filename,
                revision=self._revision_override or entry.revision,
                source=self._weights_source,  # type: ignore[arg-type]
                weights_dir=self._weights_dir,
                allow_download=self._allow_download,
            )
        self._resolved_revision = found.revision
        self._weights_origin = found.source

        # pretrained=False: every weight is immediately overwritten by the
        # checkpoint, so fetching ImageNet weights first is pure startup latency
        # (and a network call on an air-gapped box).
        net = timm.create_model(self._variant.timm_model, pretrained=False, num_classes=1)
        net.load_state_dict(_load_state_dict(found.path, self._variant), strict=True)
        net.eval().to(device=self.device, dtype=self.dtype)
        self._net = net

        # One synthetic forward pass so the first real request isn't paying for
        # lazy CUDA kernel / cuDNN autotune init.
        size = self._variant.input_size
        with torch.inference_mode():
            net(torch.zeros(1, 3, size, size, device=self.device, dtype=self.dtype))
        log.info(
            "community-forensics ready: %s (weights from %s)",
            self.info().fingerprint,
            self._weights_origin,
        )

    def predict(self, images: Sequence[Image.Image]) -> list[DetectionResult]:
        if self._net is None:
            self.warmup()
        assert self._net is not None

        batch = torch.stack([self._tf(as_rgb(img)) for img in images])
        batch = batch.to(device=self.device, dtype=self.dtype)
        with torch.inference_mode():
            logits = self._net(batch)
        # Sigmoid in fp32: at fp16 the tails round to exactly 0.0/1.0, throwing
        # away the margin callers threshold on.
        scores = torch.sigmoid(logits.float()).flatten().tolist()
        return [DetectionResult(ai_generated=s).clamped() for s in scores]

    def close(self) -> None:
        self._net = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def _select(
    *, model: str | None, path: str | None, variant: str | None, revision: str | None
) -> tuple[str | None, str | None, str, str | None]:
    """Validate the model/path choice up front, before anything is loaded."""
    if model and path:
        raise CatalogError(
            "set exactly one of detector.model or detector.path, not both "
            f"(got model={model!r}, path={path!r})"
        )
    if not model and not path:
        raise CatalogError(
            "backend 'community-forensics' needs detector.model (a catalog name) "
            "or detector.path (a checkpoint file)"
        )

    if path:
        if not variant:
            raise CatalogError(
                "detector.path requires detector.variant: a checkpoint file does not "
                "state which architecture it is. "
                f"Known variants: {', '.join(sorted(VARIANTS))}."
            )
        if revision:
            raise CatalogError("detector.revision is meaningless with detector.path")
        return None, path, variant, None

    entry = resolve_model(model)  # type: ignore[arg-type]
    if variant and variant != entry.variant:
        raise CatalogError(
            f"detector.variant={variant!r} contradicts model {model!r}, which is "
            f"{entry.variant!r}. Omit variant when selecting a catalog model."
        )
    return model, None, entry.variant, revision


def _load_state_dict(path: Path, variant: Variant) -> dict[str, torch.Tensor]:
    """Read the checkpoint and strip the upstream wrapper's `vit.` prefix.

    Upstream's ViTClassifier holds the backbone as `self.vit`, so its keys are
    `vit.blocks.0...`. We load into a bare timm model, which wants `blocks.0...`.
    """
    from safetensors.torch import load_file

    try:
        state = load_file(str(path))
    except Exception as exc:
        raise CatalogError(f"could not read checkpoint {path}: {exc}") from exc

    if any(k.startswith("vit.") for k in state):
        state = {k.removeprefix("vit."): v for k, v in state.items()}

    # Fail here with something actionable rather than deep inside timm: a
    # size mismatch almost always means the wrong variant was named.
    pos = state.get("pos_embed")
    if pos is not None:
        expected = (variant.input_size // 16) ** 2 + 1
        if pos.shape[1] != expected:
            raise CatalogError(
                f"checkpoint at {path} has {pos.shape[1]} position embeddings, but variant "
                f"{variant.timm_model!r} expects {expected}. The checkpoint is for a different "
                f"input size -- name the matching detector.variant."
            )
    return state
