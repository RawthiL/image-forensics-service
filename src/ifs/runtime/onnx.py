"""ONNX Runtime execution. Model-agnostic.

Session lifecycle, execution-provider selection and a single-input/single-output
run. Knows nothing about any particular model: no catalog, no preprocessing, no
variant. A backend supplies the artifact path and the tensor.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger(__name__)


class OnnxError(RuntimeError):
    """The session could not be created, or the graph is not as expected."""


def providers_for(device: str | None) -> list[str]:
    """Map a device string to an explicit provider list.

    onnxruntime's default is to try every registered provider and quietly use
    CPU when CUDA is unavailable, which turns a misconfigured GPU node into a
    slow CPU node with no signal. The choice is explicit here instead.
    """
    if device in (None, "auto", "cpu"):
        return ["CPUExecutionProvider"]
    if device.startswith("cuda"):
        return ["CUDAExecutionProvider"]
    raise OnnxError(f"device {device!r} is not supported by the onnx runtime; use 'cpu' or 'cuda'")


class OnnxRunner:
    """One loaded ONNX graph with exactly one input and one output."""

    def __init__(
        self,
        path: str | Path,
        *,
        providers: list[str],
        intra_op_threads: int = 0,
    ) -> None:
        self.path = Path(path)
        self.providers = providers
        self._intra_op_threads = intra_op_threads
        self._session: Any = None
        self._input_name: str | None = None

    @property
    def input_name(self) -> str:
        if self._input_name is None:
            raise OnnxError("runner is not loaded")
        return self._input_name

    @property
    def providers_in_use(self) -> list[str]:
        return list(self._session.get_providers()) if self._session else []

    def load(self) -> None:
        if self._session is not None:
            return
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise OnnxError(
                "onnxruntime is not installed; add the `onnx-cpu` or `onnx-gpu` extra"
            ) from exc

        opts = ort.SessionOptions()
        if self._intra_op_threads:
            opts.intra_op_num_threads = self._intra_op_threads
        try:
            self._session = ort.InferenceSession(str(self.path), opts, providers=self.providers)
        except Exception as exc:
            raise OnnxError(f"could not load {self.path}: {exc}") from exc

        inputs = self._session.get_inputs()
        if len(inputs) != 1:
            raise OnnxError(f"{self.path} has {len(inputs)} inputs; this runner expects exactly 1")
        self._input_name = inputs[0].name
        log.info("onnx runner ready: %s (providers=%s)", self.path.name, self.providers_in_use)

    def input_shape(self) -> list[int | str]:
        """Declared input shape, dynamic axes included as strings."""
        if self._session is None:
            raise OnnxError("runner is not loaded")
        return list(self._session.get_inputs()[0].shape)

    def run(self, x: np.ndarray) -> np.ndarray:
        if self._session is None:
            self.load()
        assert self._session is not None
        return self._session.run(None, {self.input_name: x})[0]

    def close(self) -> None:
        self._session = None
        self._input_name = None
