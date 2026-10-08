"""`ifs-serve` entrypoint."""

from __future__ import annotations

import uvicorn

from .config import get_settings


def main() -> None:
    s = get_settings()
    uvicorn.run("ifs.main:app", host=s.server.host, port=s.server.port, workers=1)
