"""Community Forensics (Park & Owens, CVPR 2025). See MODEL.md."""

from ...registry import register
from .detector import CommunityForensicsDetector


@register("community-forensics")
def _factory(**kwargs) -> CommunityForensicsDetector:
    return CommunityForensicsDetector(**kwargs)
