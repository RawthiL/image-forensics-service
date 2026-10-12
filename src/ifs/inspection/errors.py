"""Failures of a composed service."""

from __future__ import annotations


class UpstreamError(RuntimeError):
    """A service this one composes did not answer.

    Carries `service` so the handler can say *which* hop failed. Whether this
    reaches the caller as a 502 or is folded into the report as a degraded
    upstream is the operator's choice, via `required`.
    """

    def __init__(self, service: str, message: str) -> None:
        super().__init__(message)
        self.service = service
