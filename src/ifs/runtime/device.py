"""Device and dtype selection -- the only place that knows about CUDA/MPS.

torch is imported lazily inside each function, not at module scope: the API
layer imports this module for /v1/models, and must stay installable and
testable without a 2GB torch wheel.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)


def resolve_device(requested: str | None) -> torch.device:
    """`auto` picks the best available; an explicit value is honored or fails loudly."""
    import torch

    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(requested: str | None, device: torch.device) -> torch.dtype:
    """Default to fp16 on CUDA, fp32 everywhere else.

    fp16 on CPU is emulated and slower than fp32, and MPS half support is
    uneven -- so `auto` only turns it on where it actually wins.
    """
    import torch

    if requested and requested != "auto":
        return getattr(torch, requested)
    return torch.float16 if device.type == "cuda" else torch.float32


def describe() -> dict[str, Any]:
    """Runtime facts for /v1/models, so operators can see what they actually got."""
    try:
        import torch
    except ImportError:
        return {"torch": None, "note": "torch not installed; only torch-free backends available"}

    info: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "threads": torch.get_num_threads(),
    }
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
    return info
