"""The detector contract.

Everything under `backends/` is swappable. The API layer imports *only* the
names defined here -- never a concrete backend. Adding a model means adding a
folder under `backends/`; it means changing nothing else.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from PIL import Image


@dataclass(frozen=True)
class DetectionResult:
    """One backend's verdict on one image.

    `ai_generated` is the only field the API dialects require. The rest is
    optional enrichment a dialect surfaces when a backend provides it.
    """

    ai_generated: float
    """P(image was produced or edited by a generative model), in [0, 1]."""

    per_class: dict[str, float] = field(default_factory=dict)
    """Optional secondary head, e.g. {"midjourney": 0.4, "flux": 0.1}."""

    deepfake: float | None = None
    """Optional face-manipulation score, when the backend has such a head."""

    def clamped(self) -> DetectionResult:
        return DetectionResult(
            ai_generated=_clamp(self.ai_generated),
            per_class={k: _clamp(v) for k, v in self.per_class.items()},
            deepfake=None if self.deepfake is None else _clamp(self.deepfake),
        )


@dataclass(frozen=True)
class DetectorInfo:
    """Identity of a loaded backend: everything *the model* contributes to a score.

    Deliberately excludes service-level choices such as crop policy. Those are
    applied by `service.py`, not by the backend, and are composed into the
    pipeline fingerprint there.
    """

    name: str
    version: str
    revision: str | None = None
    device: str = "cpu"
    dtype: str = "float32"
    input_size: int | None = None
    """Native input resolution, if the backend has one. Used to size crops."""

    @property
    def fingerprint(self) -> str:
        return f"{self.name}@{self.version}+{self.revision or 'local'}/{self.dtype}"


@runtime_checkable
class Detector(Protocol):
    """Implement this to add a backend. Four members, no base class needed."""

    def info(self) -> DetectorInfo:
        """Identity and the knobs that affect the score."""
        ...

    def warmup(self) -> None:
        """Load weights and run one forward pass. Called once at startup."""
        ...

    def predict(self, images: Sequence[Image.Image]) -> list[DetectionResult]:
        """Score a batch. Must return one result per input, in order."""
        ...

    def close(self) -> None:
        """Release resources. Called on shutdown."""
        ...


def _clamp(v: float) -> float:
    return min(1.0, max(0.0, v))
