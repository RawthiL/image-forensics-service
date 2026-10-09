"""Community Forensics (Park & Owens, CVPR 2025). See MODEL.md.

Three runtimes over one model. They share `catalog.py` and produce identical
scores; they differ only in what has to be installed and where inference runs.

| name                          | needs            | inference |
|-------------------------------|------------------|-----------|
| community-forensics           | torch, timm      | in-process|
| community-forensics-onnx      | onnxruntime      | in-process|
| community-forensics-triton    | nothing extra    | remote    |
"""

from ...registry import register


@register("community-forensics")
def _torch_factory(**kwargs):
    try:
        from .detector import CommunityForensicsDetector
    except ImportError as exc:  # torch/timm missing
        raise RuntimeError(
            "backend 'community-forensics' needs torch and timm: install a torch extra "
            "(`cpu` or `cu124`) plus `community-forensics`. For a lean deployment use "
            "the 'community-forensics-onnx' or 'community-forensics-triton' backend instead."
        ) from exc

    return CommunityForensicsDetector(**kwargs)


@register("community-forensics-onnx")
def _onnx_factory(**kwargs):
    try:
        from .detector_onnx import OnnxCommunityForensicsDetector
    except ImportError as exc:  # onnxruntime missing
        raise RuntimeError(
            "backend 'community-forensics-onnx' needs onnxruntime: "
            "install the `onnx-cpu` or `onnx-gpu` extra"
        ) from exc

    return OnnxCommunityForensicsDetector(**kwargs)


@register("community-forensics-triton")
def _triton_factory(**kwargs):
    from .detector_triton import TritonCommunityForensicsDetector

    return TritonCommunityForensicsDetector(**kwargs)
