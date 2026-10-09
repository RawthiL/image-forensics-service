"""Triton Inference Server client. Model-agnostic.

Speaks the KServe v2 REST protocol with the binary tensor extension, directly
over httpx. `tritonclient` would add aiohttp and geventhttpclient to an image
whose point is to be small, for a protocol that is a JSON header followed by
raw tensor bytes.

Knows nothing about any particular model: a backend supplies the model name,
the tensor, and the input/output tensor names.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger(__name__)

DEFAULT_URL = "http://localhost:8000"


class TritonError(RuntimeError):
    """Triton refused, could not be reached, or serves an unexpected signature."""


class TritonClient:
    def __init__(
        self,
        *,
        url: str = DEFAULT_URL,
        model_name: str,
        model_version: str = "",
        timeout_s: float = 30.0,
    ) -> None:
        self.url = url.rstrip("/")
        self.model_name = model_name
        self.model_version = model_version
        self._timeout = timeout_s
        self._client: httpx.Client | None = None

    # -- lifecycle ---------------------------------------------------------

    def connect(self) -> None:
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def _require(self) -> httpx.Client:
        if self._client is None:
            self.connect()
        assert self._client is not None
        return self._client

    # -- introspection -----------------------------------------------------

    def assert_ready(self) -> None:
        """Fail at startup with a clear message rather than on the first relay.

        Checks the server first, then the model, so "Triton is not there" and
        "the model is not loaded" are different messages.
        """
        client = self._require()

        try:
            health = client.get(f"{self.url}/v2/health/ready")
        except httpx.HTTPError as exc:
            raise TritonError(self._unreachable_hint(exc)) from exc
        if health.status_code != 200:
            raise TritonError(
                f"Triton at {self.url} is not ready (HTTP {health.status_code} from "
                f"/v2/health/ready). The server is reachable but not serving."
            )

        try:
            resp = client.get(f"{self.url}/v2/models/{self.model_name}/ready")
        except httpx.HTTPError as exc:
            raise TritonError(self._unreachable_hint(exc)) from exc
        if resp.status_code != 200:
            raise TritonError(
                f"Triton at {self.url} is up, but model {self.model_name!r} is not ready "
                f"(HTTP {resp.status_code}). Check the name matches a directory in the "
                f"model repository, and that it loaded without error."
            )

    def _unreachable_hint(self, exc: Exception) -> str:
        """Turn a bare connection failure into something actionable.

        The commonest cause is aiming at Triton's gRPC or metrics port: this
        client speaks the KServe v2 **REST** protocol, which Triton serves on
        8000 by default. An HTTP request to the gRPC port does not fail with a
        helpful message, it just fails.
        """
        base = f"cannot reach Triton at {self.url}: {exc}"
        port = httpx.URL(self.url).port
        if port == 8001:
            return (
                f"{base}\n"
                f"Note: 8001 is Triton's gRPC port *by default*, and this client speaks "
                f"KServe v2 over HTTP/REST. If your server uses the defaults, try "
                f"{httpx.URL(self.url).copy_with(port=8000)}. If you remapped Triton's "
                f"HTTP port to 8001 (common, to avoid colliding with this service), the "
                f"port is right and the problem is reachability."
            )
        if port == 8002:
            return (
                f"{base}\n"
                f"Port 8002 is Triton's Prometheus metrics port. The inference API is "
                f"on port 8000 by default."
            )
        return (
            f"{base}\n"
            f"Check that Triton's HTTP port (8000 by default, not 8001/gRPC) is "
            f"reachable from inside this container. From a container, 127.0.0.1 is "
            f"the container itself -- use the host address or join Triton's network."
        )

    def metadata(self) -> dict[str, Any]:
        client = self._require()
        try:
            return client.get(f"{self.url}/v2/models/{self.model_name}").json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise TritonError(f"could not read Triton model metadata: {exc}") from exc

    def assert_input_hw(self, height: int, width: int) -> None:
        """Check the served model takes the spatial dims we are about to send.

        Triton reports dims without the batch axis when `max_batch_size > 0`,
        and with it otherwise, so both layouts are accepted.
        """
        inputs = self.metadata().get("inputs") or []
        if not inputs:
            raise TritonError(f"Triton model {self.model_name!r} declares no inputs")
        shape = [d for d in inputs[0].get("shape", [])]
        hw = tuple(shape[-2:]) if len(shape) >= 2 else ()
        if hw and all(isinstance(d, int) and d > 0 for d in hw) and hw != (height, width):
            raise TritonError(
                f"Triton model {self.model_name!r} expects input {shape}, but this "
                f"configuration produces [..., {height}, {width}]. "
                f"Name the matching variant, or load the matching model."
            )

    # -- inference ---------------------------------------------------------

    def infer(
        self, x: np.ndarray, *, input_name: str, output_name: str, datatype: str = "FP32"
    ) -> np.ndarray:
        import numpy as np

        client = self._require()
        payload = x.tobytes()
        header = {
            "inputs": [
                {
                    "name": input_name,
                    "shape": list(x.shape),
                    "datatype": datatype,
                    "parameters": {"binary_data_size": len(payload)},
                }
            ],
            "outputs": [{"name": output_name, "parameters": {"binary_data": True}}],
        }
        header_bytes = json.dumps(header).encode()
        version = f"/versions/{self.model_version}" if self.model_version else ""
        try:
            resp = client.post(
                f"{self.url}/v2/models/{self.model_name}{version}/infer",
                content=header_bytes + payload,
                headers={
                    "Inference-Header-Content-Length": str(len(header_bytes)),
                    "Content-Type": "application/octet-stream",
                },
            )
        except httpx.HTTPError as exc:
            raise TritonError(f"Triton request failed: {exc}") from exc
        if resp.status_code != 200:
            raise TritonError(f"Triton returned HTTP {resp.status_code}: {resp.text[:300]}")

        head_len = int(resp.headers.get("Inference-Header-Content-Length", len(resp.content)))
        meta = json.loads(resp.content[:head_len])
        out = meta["outputs"][0]
        if out.get("datatype") != datatype:
            raise TritonError(f"unexpected output datatype {out.get('datatype')!r}")
        return np.frombuffer(resp.content[head_len:], dtype=np.float32).reshape(out["shape"])
