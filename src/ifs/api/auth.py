"""Service-level authorization.

A single shared key that gates the whole deployment. This is a *deployment*
boundary -- "only our relay miner may reach this container" -- not user
authentication. There are no accounts, no per-caller identity, and no rate
limiting tied to it.

Two headers are accepted:

* ``X-API-Key: <key>``          (default; configurable via IFS_API_KEY_HEADER)
* ``Authorization: Bearer <key>``

Only the ``Bearer`` scheme is read from ``Authorization``. That is deliberate:
the Hive compatibility layer receives ``Authorization: token <vendor-key>`` from
clients that were written against Hive, and those vendor keys must keep being
ignored. Restricting to ``Bearer`` keeps the two from colliding.
"""

from __future__ import annotations

import logging
import secrets

from fastapi import Depends, Request

from ..config import Settings, get_settings

log = logging.getLogger(__name__)

_BEARER_PREFIX = "bearer "


class AuthError(Exception):
    """Rejected at the deployment boundary. Mapped to 401 by the app."""

    def __init__(self, message: str, *, code: str = "unauthorized") -> None:
        super().__init__(message)
        self.code = code


def require_api_key(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> None:
    """Gate a route. No-op when no key is configured."""
    expected = settings.auth.api_key
    if not expected:
        return  # auth disabled; startup logs a warning

    presented = _extract(request, settings.auth.api_key_header)
    if presented is None:
        raise AuthError(
            f"missing service credentials: send {settings.auth.api_key_header} "
            "or Authorization: Bearer <key>",
            code="missing_api_key",
        )
    # Constant-time: a plain == leaks the key prefix through response timing.
    if not secrets.compare_digest(presented, expected):
        raise AuthError("invalid service credentials", code="invalid_api_key")


def _extract(request: Request, header_name: str) -> str | None:
    direct = request.headers.get(header_name)
    if direct:
        return direct.strip()

    authorization = request.headers.get("authorization")
    if authorization and authorization.lower().startswith(_BEARER_PREFIX):
        return authorization[len(_BEARER_PREFIX) :].strip() or None

    # An `Authorization: token ...` here is a vendor key on a compatibility
    # route. Ignored on purpose -- see this module's docstring.
    return None


def warn_if_open(settings: Settings) -> None:
    """Called at startup so an unauthenticated deployment is never silent."""
    if settings.auth.api_key:
        log.info("service API key is ENABLED (header: %s)", settings.auth.api_key_header)
    else:
        log.warning(
            "service API key is NOT set -- every caller that can reach this port "
            "can use it. Set IFS_API_KEY to gate the deployment."
        )
