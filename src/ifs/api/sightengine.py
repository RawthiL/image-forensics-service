"""Sightengine-compatible dialect.

Mimics:  POST/GET https://api.sightengine.com/1.0/check.json?models=genai
Docs:    https://sightengine.com/docs/ai-generated-image-detection

Response shape is reproduced field-for-field from the published sample so that a
client only has to change its base URL:

    {"status":"success",
     "request":{"id":"req_...","timestamp":1491402308.4762,"operations":5},
     "type":{"ai_generated":0.001},
     "media":{"id":"med_...","uri":"https://..."}}

`api_user` / `api_secret` are accepted and ignored on purpose. On Pocket the
relay is already paid and authenticated at the gateway, so rejecting a missing
key would break the one-line swap this dialect exists to enable.
"""

from __future__ import annotations

import secrets
import time

from fastapi import APIRouter, File, Form, Query, UploadFile

from ..schemas.sightengine import SightengineResponse
from .deps import ServiceDep, SettingsDep, load_one

router = APIRouter(tags=["compat: sightengine"])

# Sightengine bills genai at 5 operations per call. Echoed for clients that
# parse it; it is not a charge on this deployment.
_OPERATIONS_PER_GENAI_CALL = 5


def _ids() -> tuple[str, str]:
    return f"req_{secrets.token_urlsafe(16)[:21]}", f"med_{secrets.token_urlsafe(16)[:21]}"


async def _check(service, settings, *, image, uri: str | None) -> SightengineResponse:
    scored = await service.score(image)
    req_id, med_id = _ids()
    return SightengineResponse.build(
        request_id=req_id,
        timestamp=time.time(),
        operations=_OPERATIONS_PER_GENAI_CALL,
        ai_generated=scored.result.ai_generated,
        deepfake=scored.result.deepfake,
        media_id=med_id,
        media_uri=uri,
    )


@router.post("/1.0/check.json", response_model=SightengineResponse, summary="Sightengine check (upload)")
async def check_post(
    service: ServiceDep,
    settings: SettingsDep,
    media: UploadFile | None = File(None),
    url: str | None = Form(None),
    models: str = Form("genai"),
    api_user: str | None = Form(None),    # accepted, ignored
    api_secret: str | None = Form(None),  # accepted, ignored
) -> SightengineResponse:
    image = await load_one(settings=settings, upload=media, url=url)
    return await _check(service, settings, image=image, uri=url)


@router.get("/1.0/check.json", response_model=SightengineResponse, summary="Sightengine check (by URL)")
async def check_get(
    service: ServiceDep,
    settings: SettingsDep,
    url: str = Query(..., description="URL of the image to analyze"),
    models: str = Query("genai"),
    api_user: str | None = Query(None),    # accepted, ignored
    api_secret: str | None = Query(None),  # accepted, ignored
) -> SightengineResponse:
    image = await load_one(settings=settings, url=url)
    return await _check(service, settings, image=image, uri=url)
