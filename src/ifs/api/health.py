"""Operational endpoints. Not part of any vendor dialect."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..detectors.registry import available, unavailable
from ..runtime import device
from .auth import require_api_key
from .deps import ServiceDep, SettingsDep

router = APIRouter(tags=["ops"])


@router.get("/healthz", summary="Liveness probe")
async def healthz() -> dict:
    return {"status": "ok"}


@router.get("/readyz", summary="Readiness probe (model loaded)")
async def readyz(service: ServiceDep) -> dict:
    return {"status": "ready", "pipeline": service.pipeline_fingerprint}


@router.get(
    "/v1/models",
    summary="Active backend and what else is installed",
    dependencies=[Depends(require_api_key)],  # reveals internals; not a probe
)
async def models(service: ServiceDep, settings: SettingsDep) -> dict:
    info = service.info()
    return {
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
        # Backend folders present but not importable, e.g. a missing optional
        # dependency. Surfaced so the cause is visible rather than silent.
        "unavailable": unavailable(),
        "runtime": device.describe(),
    }
