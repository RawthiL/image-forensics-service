"""Community Forensics on a remote Triton server.

Model-specific only: the variant that fixes preprocessing, and the tensor
names this model's export uses. The KServe v2 transport lives in
`ifs.runtime.triton`, which knows nothing about this model.

Preprocessing stays client-side on purpose. It moves the score more than the
model does (see the crop-policy measurements in the README), so keeping it here
means the pipeline fingerprint describes what actually happened rather than
trusting a server-side ensemble this project does not version.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from PIL import Image

from ....runtime.triton import DEFAULT_URL, TritonClient, TritonError
from ...base import DetectionResult, DetectorInfo
from .catalog import CatalogError, Variant, resolve_variant
from .preprocess_np import batch, sigmoid

log = logging.getLogger(__name__)

INPUT_NAME = "pixel_values"
OUTPUT_NAME = "logit"

__all__ = ["TritonCommunityForensicsDetector", "TritonError"]


class TritonCommunityForensicsDetector:
    def __init__(
        self,
        *,
        url: str = DEFAULT_URL,
        model_name: str = "commfor-384",
        model_version: str = "",
        variant: str | None = None,
        timeout_s: float = 30.0,
        verify_at_startup: bool = True,
        **_: object,
    ) -> None:
        if not variant:
            raise CatalogError(
                "backend 'community-forensics-triton' needs detector.options.variant: "
                "preprocessing happens here, and a remote model name does not state "
                "which input size it expects"
            )
        self._variant_name = variant
        self._variant: Variant = resolve_variant(variant)
        self._verify = verify_at_startup
        self._client = TritonClient(
            url=url, model_name=model_name, model_version=model_version, timeout_s=timeout_s
        )

    # -- Detector protocol -------------------------------------------------

    def info(self) -> DetectorInfo:
        return DetectorInfo(
            name="community-forensics-triton",
            version=f"{self._client.model_name}:{self._client.model_version or 'latest'}",
            revision=self._variant_name,
            device=f"triton@{self._client.url}",
            dtype="float32",
            input_size=self._variant.input_size,
        )

    def warmup(self) -> None:
        self._client.connect()
        if not self._verify:
            return
        self._client.assert_ready()
        size = self._variant.input_size
        self._client.assert_input_hw(size, size)
        log.info("community-forensics-triton ready: %s", self.info().fingerprint)

    def predict(self, images: Sequence[Image.Image]) -> list[DetectionResult]:
        x = batch(
            images, input_size=self._variant.input_size, resize_to=self._variant.resize_to
        ).astype(np.float32, copy=False)
        logits = self._client.infer(x, input_name=INPUT_NAME, output_name=OUTPUT_NAME).reshape(-1)
        return [DetectionResult(ai_generated=float(s)).clamped() for s in sigmoid(logits)]

    def close(self) -> None:
        self._client.close()
