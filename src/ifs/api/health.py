"""Operational endpoints. Not part of any vendor dialect."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from ..detectors.registry import available, unavailable
from ..runtime import device
from .auth import authorized_endpoints, require_any_api_key
from .deps import InspectorDep, ServiceDep, SettingsDep

router = APIRouter(tags=["ops"])


@router.get("/healthz", summary="Liveness probe")
async def healthz() -> dict:
    return {"status": "ok"}


@router.get("/readyz", summary="Readiness probe (every served endpoint is up)")
async def readyz(
    service: ServiceDep, inspector: InspectorDep, settings: SettingsDep
) -> dict:
    """One line per endpoint this deployment serves.

    A screening-only deployment has no model to wait for and is ready as soon
    as it is listening, which is most of the point of serving it alone.
    """
    endpoints = {}
    if settings.endpoints.detect.enabled:
        endpoints["detect"] = {"pipeline": service.pipeline_fingerprint}
    if settings.endpoints.screen.enabled:
        endpoints["screen"] = {"pipeline": inspector.pipeline_fingerprint("container")}
    if settings.endpoints.inspect.enabled:
        endpoints["inspect"] = {"pipeline": inspector.pipeline_fingerprint("image")}
    return {"status": "ready", "endpoints": endpoints}


@router.get(
    "/v1/models",
    summary="What this deployment serves, and with what",
    # Reveals internals, so it is not a probe. Any served endpoint's key opens
    # it, and the answer is scoped to what that key is entitled to.
    dependencies=[Depends(require_any_api_key)],
)
async def models(
    request: Request,
    service: ServiceDep,
    settings: SettingsDep,
    inspector: InspectorDep,
) -> dict:
    live = settings.endpoints.live()
    # Scoped per credential: a screening key must not learn the inspection
    # service's upstream URLs, nor which checkpoint the detector runs, just
    # because they happen to share a container.
    allowed = authorized_endpoints(request)
    body: dict = {"endpoints": list(live), "runtime": device.describe()}

    if "detect" in live and "detect" in allowed:
        info = service.info()
        body["detect"] = {
            "active": {
                "name": info.name,
                "version": info.version,
                "revision": info.revision,
                "fingerprint": info.fingerprint,
                "device": info.device,
                "dtype": info.dtype,
                "input_size": info.input_size,
            },
            "scoring": {
                "crop_policy": settings.scoring.crop_policy,
                "crop_aggregation": settings.scoring.crop_aggregation,
                "score_decimals": settings.scoring.score_decimals,
            },
            "pipeline_fingerprint": service.pipeline_fingerprint,
            "available": available(),
            # Backend folders present but not importable, e.g. a missing
            # optional dependency. Surfaced so the cause is visible rather
            # than silent.
            "unavailable": unavailable(),
        }
    if "screen" in live and "screen" in allowed:
        body["screen"] = inspector.describe_screen()
    if "inspect" in live and "inspect" in allowed:
        body["inspect"] = inspector.describe()
    return body
