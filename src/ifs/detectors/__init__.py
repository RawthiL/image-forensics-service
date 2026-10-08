"""Detector contract, registry, and shared weight resolution."""

from .base import DetectionResult, Detector, DetectorInfo
from .registry import available, build, register, unavailable

__all__ = [
    "DetectionResult",
    "Detector",
    "DetectorInfo",
    "available",
    "build",
    "register",
    "unavailable",
]
