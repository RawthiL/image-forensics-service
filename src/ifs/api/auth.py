"""Service-level authorization.

A *deployment* boundary -- "only our relay miner may reach this container" --
not user authentication. There are no accounts, no per-caller identity, and no
rate limiting tied to it.

This deployment serves up to three independently-stakeable services, so keys
are per endpoint: `endpoints.<name>.api_keys`. A key opens the endpoints that
list it and no others, so a supplier staked for screening cannot be handed a
credential that also opens detection. Repeat a key, or use a YAML anchor, to
share one across endpoints.

`api_keys` is a list so a key can be rotated without downtime. An **empty list
leaves that endpoint open** -- per endpoint, not per deployment, which is why
startup names each open one. Only *mounted* endpoints count: a key for a route
this deployment does not serve grants nothing, and such a route is not reported
as a hole.

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
from collections.abc import Callable, Sequence

from fastapi import Depends, Request

from ..config import Endpoint, Settings, get_settings

log = logging.getLogger(__name__)

_BEARER_PREFIX = "bearer "


class AuthError(Exception):
    """Rejected at the deployment boundary. Mapped to 401 by the app."""

    def __init__(self, message: str, *, code: str = "unauthorized") -> None:
        super().__init__(message)
        self.code = code


def require_api_key(endpoint: Endpoint) -> Callable[..., None]:
    """Build the dependency gating one endpoint. No-op when it lists no keys.

    A factory rather than a plain dependency because the endpoint is fixed
    when the router is mounted, not when the request arrives -- the route
    decides which keys it accepts, and the caller never gets to choose.
    """

    def dependency(request: Request, settings: Settings = Depends(get_settings)) -> None:
        keys = settings.endpoints.get(endpoint).api_keys
        if not keys:
            request.state.auth_endpoints = frozenset(settings.endpoints.live())
            return  # this endpoint is open; startup logs a warning

        presented = _presented(request, settings)
        if not _matches(presented, keys):
            raise AuthError("invalid service credentials", code="invalid_api_key")
        request.state.auth_endpoints = _matching_endpoints(presented, settings)

    return dependency


def require_any_api_key(request: Request, settings: Settings = Depends(get_settings)) -> None:
    """Gate a route that describes the whole deployment, e.g. /v1/models.

    Any mounted endpoint's key opens it. What that key is entitled to is
    recorded on the request, so the route answers with those sections only --
    a detection key must not learn the inspection service's upstream URLs just
    because both happen to live in one container.
    """
    live = settings.endpoints.live()
    open_endpoints = frozenset(e for e in live if not settings.endpoints.get(e).api_keys)

    if not any(settings.endpoints.get(e).api_keys for e in live):
        request.state.auth_endpoints = frozenset(live)
        return

    presented = _extract(request, settings.auth.api_key_header)
    if presented is None:
        # Not an error while *some* endpoint is open: a caller with no key can
        # already reach that endpoint's routes, so it may see them described.
        if open_endpoints:
            request.state.auth_endpoints = open_endpoints
            return
        raise AuthError(_missing(settings), code="missing_api_key")

    allowed = _matching_endpoints(presented, settings) | open_endpoints
    if not allowed:
        raise AuthError("invalid service credentials", code="invalid_api_key")
    request.state.auth_endpoints = allowed


def authorized_endpoints(request: Request) -> frozenset[str]:
    """What the presented credential is entitled to. Open endpoints included."""
    return getattr(request.state, "auth_endpoints", frozenset())


def _presented(request: Request, settings: Settings) -> str:
    value = _extract(request, settings.auth.api_key_header)
    if value is None:
        raise AuthError(_missing(settings), code="missing_api_key")
    return value


def _missing(settings: Settings) -> str:
    return (
        f"missing service credentials: send {settings.auth.api_key_header} "
        "or Authorization: Bearer <key>"
    )


def _matching_endpoints(presented: str, settings: Settings) -> frozenset[Endpoint]:
    """Every mounted endpoint this credential opens.

    A set, not a lookup: one key may legitimately be listed by several
    endpoints, which is how an operator shares one credential across services.
    """
    return frozenset(
        endpoint
        for endpoint in settings.endpoints.live()
        if _matches(presented, settings.endpoints.get(endpoint).api_keys)
    )


def _matches(presented: str, accepted: Sequence[str]) -> bool:
    """Constant-time against every candidate.

    Every key is compared, and the results are OR-ed without short-circuiting:
    returning early on the first match would leak, through response timing,
    which key in the list was presented.
    """
    found = False
    candidate = presented.encode("utf-8")
    for key in accepted:
        # Compared as bytes because compare_digest rejects non-ASCII str, and
        # Starlette decodes header bytes as latin-1 -- so a header carrying one
        # would otherwise raise rather than return 401.
        found |= secrets.compare_digest(candidate, key.encode("utf-8"))
    return found


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
    """Called at startup so an unauthenticated endpoint is never silent."""
    live = settings.endpoints.live()
    header = settings.auth.api_key_header
    open_endpoints = [e for e in live if not settings.endpoints.get(e).api_keys]
    gated = [e for e in live if settings.endpoints.get(e).api_keys]

    if not open_endpoints:
        log.info("API keys are SET for %s (header: %s)", ", ".join(gated), header)
        return

    if not gated:
        log.warning(
            "NO API keys are set on %s -- every caller that can reach this port can "
            "use %s. Set endpoints.<name>.api_keys to gate the deployment.",
            ", ".join(open_endpoints),
            "it" if len(open_endpoints) == 1 else "them",
        )
        return

    # The footgun this exists for: an operator who keys one service and
    # believes the container is closed.
    log.warning(
        "API keys gate %s but %s %s OPEN -- keys are per endpoint, so gating one "
        "does not gate the others. Set %s.",
        ", ".join(gated),
        ", ".join(open_endpoints),
        "is" if len(open_endpoints) == 1 else "are",
        ", ".join(f"endpoints.{name}.api_keys" for name in open_endpoints),
    )
