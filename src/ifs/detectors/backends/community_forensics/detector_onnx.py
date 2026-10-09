"""Community Forensics on the ONNX runtime.

Model-specific only: catalog lookup, artifact location and preprocessing.
Session management and provider selection live in `ifs.runtime.onnx`, which
knows nothing about this model and is reusable by the next one.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from ....runtime.onnx import OnnxRunner, providers_for
from ...base import DetectionResult, DetectorInfo
from ...weights import WeightsError, resolve_path
from .catalog import MODELS, CatalogError, Variant, resolve_variant
from .preprocess_np import batch, sigmoid

log = logging.getLogger(__name__)

INPUT_NAME = "pixel_values"


class OnnxCommunityForensicsDetector:
    """Runs an exported `.onnx` of a Community Forensics checkpoint.

    Select by `path` (an explicit `.onnx`) or by `model`, which resolves to
    `<weights.dir>/onnx/<model>.onnx`. Nothing is downloaded: upstream publishes
    safetensors, not ONNX, so the artifact comes from `export.py`. A missing
    file names the export command rather than falling back to another runtime.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        path: str | None = None,
        variant: str | None = None,
        weights_dir: str | None = None,
        device: str | None = None,
        intra_op_threads: int = 0,
        **_: object,
    ) -> None:
        self._model_name = model
        self._explicit_path = path
        self._weights_dir = weights_dir
        self._variant_name = _variant_for(model=model, path=path, variant=variant)
        self._variant: Variant = resolve_variant(self._variant_name)
        self._runner = OnnxRunner(
            "",  # path resolved at warmup, so a config error surfaces before IO
            providers=providers_for(device),
            intra_op_threads=intra_op_threads,
        )

    # -- Detector protocol -------------------------------------------------

    def info(self) -> DetectorInfo:
        return DetectorInfo(
            name="community-forensics-onnx",
            version=self._model_name or f"path:{self._variant_name}",
            revision=self._runner.path.name if str(self._runner.path) != "." else None,
            device=self._runner.providers[0],
            dtype="float32",
            input_size=self._variant.input_size,
        )

    def warmup(self) -> None:
        import numpy as np

        self._runner.path = Path(resolve_path(self._resolve_artifact()).path)
        self._runner.load()
        size = self._variant.input_size
        self._runner.run(np.zeros((1, 3, size, size), np.float32))
        log.info("community-forensics-onnx ready: %s", self.info().fingerprint)

    def predict(self, images: Sequence[Image.Image]) -> list[DetectionResult]:
        x = batch(images, input_size=self._variant.input_size, resize_to=self._variant.resize_to)
        logits = self._runner.run(x).reshape(-1)
        return [DetectionResult(ai_generated=float(s)).clamped() for s in sigmoid(logits)]

    def close(self) -> None:
        self._runner.close()

    # -- internals ---------------------------------------------------------

    def _resolve_artifact(self) -> str:
        if self._explicit_path:
            return self._explicit_path
        if not self._weights_dir:
            raise CatalogError(
                "backend 'community-forensics-onnx' with detector.model needs weights.dir, "
                "or use detector.path to name the .onnx file directly"
            )
        candidate = Path(self._weights_dir).expanduser() / "onnx" / f"{self._model_name}.onnx"
        if not candidate.is_file():
            raise WeightsError(
                f"no ONNX artifact at {candidate}. Upstream publishes safetensors, not "
                f"ONNX, so export it once:\n"
                f"    python -m ifs.detectors.backends.community_forensics.export "
                f"--model {self._model_name} --out-dir {self._weights_dir}\n"
                f"Nothing is downloaded or substituted for a missing export."
            )
        return str(candidate)


def _variant_for(*, model: str | None, path: str | None, variant: str | None) -> str:
    if model and path:
        raise CatalogError("set exactly one of detector.model or detector.path, not both")
    if path:
        if not variant:
            raise CatalogError(
                "detector.path requires detector.variant: an .onnx file does not state "
                "which preprocessing belongs to it"
            )
        return variant
    if not model:
        raise CatalogError(
            "backend 'community-forensics-onnx' needs detector.model or detector.path"
        )
    if model not in MODELS:
        raise CatalogError(f"unknown model {model!r}; known: {', '.join(sorted(MODELS))}")
    entry = MODELS[model]
    if variant and variant != entry.variant:
        raise CatalogError(
            f"detector.variant={variant!r} contradicts model {model!r} ({entry.variant!r})"
        )
    return entry.variant
