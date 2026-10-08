from __future__ import annotations

import hashlib
from collections.abc import Sequence

from PIL import Image

from ...base import DetectionResult, DetectorInfo


class StubDetector:
    """Hashes pixel content into a stable, meaningless score."""

    def __init__(self, *, fixed_score: float | None = None, **_: object) -> None:
        # Accepts and ignores the generic kwargs (model/path/device/...): this
        # backend has no weights to select.
        self.fixed_score = fixed_score

    def info(self) -> DetectorInfo:
        return DetectorInfo(name="stub", version="1", device="cpu")

    def warmup(self) -> None:
        return None

    def predict(self, images: Sequence[Image.Image]) -> list[DetectionResult]:
        out = []
        for img in images:
            if self.fixed_score is not None:
                score = self.fixed_score
            else:
                digest = hashlib.sha256(img.convert("RGB").tobytes()).digest()
                score = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
            out.append(DetectionResult(ai_generated=score).clamped())
        return out

    def close(self) -> None:
        return None
