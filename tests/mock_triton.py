"""A protocol-faithful stand-in for Triton, backed by onnxruntime.

Implements just enough of the KServe v2 REST API for the Triton backend to be
tested without a 17 GB container: server health, model readiness, model
metadata, and binary-tensor inference. If this and the real server ever
disagree, the bug is here -- verified against a live Triton 26.03.

Run standalone for manual testing:
    python tests/mock_triton.py --model weights/onnx/commfor-384.onnx --port 8001
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np


def make_server(onnx_path: str, model_name: str = "commfor-384", port: int = 0):
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out_name = sess.get_outputs()[0].name
    # Triton reports dims without the batch axis when max_batch_size > 0.
    meta_shape = [-1] + [d if isinstance(d, int) else -1 for d in inp.shape[1:]]

    class Handler(BaseHTTPRequestHandler):
        def _empty(self, code):
            self.send_response(code)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _json(self, code, body):
            raw = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            ready_paths = (
                "/v2/health/ready",
                "/v2/health/live",
                f"/v2/models/{model_name}/ready",
            )
            if self.path in ready_paths:
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()
            elif self.path == f"/v2/models/{model_name}":
                self._json(200, {
                    "name": model_name, "platform": "onnxruntime_onnx",
                    "inputs": [{"name": inp.name, "datatype": "FP32", "shape": meta_shape}],
                    "outputs": [{"name": out_name, "datatype": "FP32", "shape": [-1, 1]}],
                })
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if not self.path.startswith(f"/v2/models/{model_name}"):
                self._json(404, {"error": "unknown model"})
                return
            total = int(self.headers["Content-Length"])
            head_len = int(self.headers["Inference-Header-Content-Length"])
            raw = self.rfile.read(total)
            header = json.loads(raw[:head_len])
            spec = header["inputs"][0]
            x = np.frombuffer(raw[head_len:], dtype=np.float32).reshape(spec["shape"])
            y = sess.run(None, {inp.name: x})[0].astype(np.float32)

            meta = json.dumps({
                "model_name": model_name,
                "outputs": [{
                    "name": out_name, "datatype": "FP32", "shape": list(y.shape),
                    "parameters": {"binary_data_size": y.nbytes},
                }],
            }).encode()
            body = meta + y.tobytes()
            self.send_response(200)
            self.send_header("Inference-Header-Content-Length", str(len(meta)))
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", default="commfor-384")
    ap.add_argument("--port", type=int, default=8001)
    a = ap.parse_args()
    _, url = make_server(a.model, a.name, a.port)
    print(f"mock triton serving {a.name} at {url}")
    while True:
        time.sleep(3600)
