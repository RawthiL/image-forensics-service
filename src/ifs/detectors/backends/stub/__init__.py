"""Deterministic no-ML backend. See MODEL.md."""

from ...registry import register
from .detector import StubDetector


@register("stub")
def _factory(**kwargs) -> StubDetector:
    return StubDetector(**kwargs)
