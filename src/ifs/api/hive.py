"""Hive-compatible dialect.

Mimics:  POST https://api.thehive.ai/api/v2/task/sync
Docs:    https://docs.thehive.ai/docs/ai-image-and-video-detection
         https://docs.thehive.ai/reference/authentication

Like the Sightengine dialect, the `Authorization: token ...` header is accepted
and ignored -- the swap is the whole point -- and like it, this is a superset:
the vendor's own fields plus `media_base64`, so a caller without a multipart
body builder is not pushed onto a different dialect.
"""

from __future__ import annotations

import secrets

from fastapi import APIRouter, File, Form, Header, UploadFile

from ..schemas.hive import HiveTaskResponse
from .deps import ServiceDep, SettingsDep, load_one

router = APIRouter(prefix="/api/v2", tags=["compat: hive"])


@router.post("/task/sync", response_model=HiveTaskResponse, summary="Hive sync task")
async def task_sync(
    service: ServiceDep,
    settings: SettingsDep,
    media: UploadFile | None = File(None),
    url: str | None = Form(None),
    media_base64: str | None = Form(None),  # extension: not a vendor field
    authorization: str | None = Header(None),  # accepted, ignored
) -> HiveTaskResponse:
    loaded = await load_one(settings=settings, upload=media, url=url, b64=media_base64)
    scored = await service.score(loaded.image)
    return HiveTaskResponse.build(
        task_id=secrets.token_hex(16),
        ai_generated=scored.result.ai_generated,
        per_class=scored.result.per_class,
        deepfake=scored.result.deepfake,
        media_uri=url,
        elapsed_s=round(scored.elapsed_ms / 1000, 4),
    )
