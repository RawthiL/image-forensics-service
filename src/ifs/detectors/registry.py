"""Backend discovery.

Every sub-package of `ifs.detectors.backends` is imported at startup, so its
`@register(...)` call runs. Adding a model is adding a folder -- there is no
list here to keep in sync.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from collections.abc import Callable

from .base import Detector

log = logging.getLogger(__name__)

_REGISTRY: dict[str, Callable[..., Detector]] = {}
_FAILED: dict[str, str] = {}
_discovered = False


def register(name: str) -> Callable[[Callable[..., Detector]], Callable[..., Detector]]:
    """Decorator for a backend factory: `@register("my-model")`."""

    def deco(factory: Callable[..., Detector]) -> Callable[..., Detector]:
        if name in _REGISTRY:
            raise ValueError(f"detector {name!r} is already registered")
        _REGISTRY[name] = factory
        return factory

    return deco


def available() -> list[str]:
    """Backends that imported successfully."""
    discover()
    return sorted(_REGISTRY)


def unavailable() -> dict[str, str]:
    """Backend folders that failed to import, and why.

    Surfaced on /v1/models so a missing optional dependency is visible rather
    than silently absent.
    """
    discover()
    return dict(_FAILED)


def build(name: str, **kwargs) -> Detector:
    discover()
    if name not in _REGISTRY:
        if name in _FAILED:
            raise ValueError(f"detector {name!r} failed to load: {_FAILED[name]}")
        raise ValueError(
            f"unknown detector {name!r}; available: {', '.join(sorted(_REGISTRY)) or '(none)'}"
        )
    return _REGISTRY[name](**kwargs)


def discover() -> None:
    """Import every backend sub-package exactly once.

    A backend whose heavy optional dependency is missing is recorded in
    `_FAILED` rather than breaking startup for the ones that do work -- that is
    what lets the API layer run without torch.
    """
    global _discovered
    if _discovered:
        return
    _discovered = True

    from . import backends

    for module in pkgutil.iter_modules(backends.__path__):
        if not module.ispkg:
            continue  # backends are folders, not single modules
        dotted = f"{backends.__name__}.{module.name}"
        try:
            importlib.import_module(dotted)
        except Exception as exc:
            _FAILED[module.name] = f"{type(exc).__name__}: {exc}"
            log.warning("backend %r unavailable: %s", module.name, exc)
