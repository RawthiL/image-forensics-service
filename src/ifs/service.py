"""The seam between the API dialects and the detector backends.

Dialects call `DetectionService.score()` and get a `DetectionResult`. They never
import a concrete backend, never see a tensor, and never know how crops work --
which is what makes the backend swappable without touching any route.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from PIL import Image

from .config import Settings
from .detectors.base import DetectionResult, Detector, DetectorInfo
from .detectors.registry import build
from .media import crops

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScoredImage:
    result: DetectionResult
    elapsed_ms: float
    crops_scored: int


class DetectionService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._detector: Detector | None = None
        # Torch is not thread-safe for concurrent forward passes on one module,
        # and uvicorn runs sync work in a thread pool. One lock, one GPU queue.
        self._lock = asyncio.Lock()

    @property
    def detector(self) -> Detector:
        if self._detector is None:
            raise RuntimeError("service not started")
        return self._detector

    def info(self) -> DetectorInfo:
        """Identity of the model alone."""
        return self.detector.info()

    @property
    def pipeline_fingerprint(self) -> str:
        """Model identity *plus* the service-level scoring policy.

        Crop policy is applied here, not by the backend, so it belongs in this
        fingerprint rather than in DetectorInfo.
        """
        s = self.settings
        return f"{self.info().fingerprint}/{s.scoring.crop_policy}+{s.scoring.crop_aggregation}"

    @property
    def _crop_size(self) -> int:
        """Crop sizing hint: the backend's native input size when it declares one."""
        return self.info().input_size or self.settings.scoring.crop_size_fallback

    async def start(self) -> None:
        s = self.settings
        log.info("loading backend %r", s.detector.backend)
        detector = build(s.detector.backend, **s.detector_kwargs())
        # Blocking: weight download + a forward pass. Keep it off the event loop
        # so health checks on a cold container still answer while it loads.
        await asyncio.to_thread(detector.warmup)
        self._detector = detector

    async def stop(self) -> None:
        if self._detector is not None:
            await asyncio.to_thread(self._detector.close)
            self._detector = None

    async def score(self, image: Image.Image) -> ScoredImage:
        """Score one image under the configured crop policy."""
        s = self.settings
        started = time.perf_counter()

        tiles = crops.expand(image, s.scoring.crop_policy, self._crop_size)
        async with self._lock:
            results = await asyncio.to_thread(self.detector.predict, tiles)

        merged = _merge(results, s.scoring.crop_aggregation)
        rounded = DetectionResult(
            ai_generated=round(merged.ai_generated, s.scoring.score_decimals),
            per_class={k: round(v, s.scoring.score_decimals) for k, v in merged.per_class.items()},
            deepfake=None if merged.deepfake is None else round(merged.deepfake, s.scoring.score_decimals),
        )
        return ScoredImage(
            result=rounded,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            crops_scored=len(tiles),
        )


def _merge(results: list[DetectionResult], how: str) -> DetectionResult:
    if len(results) == 1:
        return results[0]
    ai = crops.aggregate([r.ai_generated for r in results], how)  # type: ignore[arg-type]
    deepfakes = [r.deepfake for r in results if r.deepfake is not None]
    per_class: dict[str, float] = {}
    for key in {k for r in results for k in r.per_class}:
        vals = [r.per_class[k] for r in results if k in r.per_class]
        per_class[key] = crops.aggregate(vals, how)  # type: ignore[arg-type]
    return DetectionResult(
        ai_generated=ai,
        per_class=per_class,
        deepfake=crops.aggregate(deepfakes, how) if deepfakes else None,  # type: ignore[arg-type]
    )
