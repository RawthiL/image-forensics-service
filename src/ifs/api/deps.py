"""Shared request plumbing for every dialect."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request, UploadFile
from PIL import Image

from ..config import Settings, get_settings
from ..media.loader import MediaError, decode, from_base64, from_url
from ..service import DetectionService


def get_service(request: Request) -> DetectionService:
    return request.app.state.service


ServiceDep = Annotated[DetectionService, Depends(get_service)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


async def load_one(
    *,
    settings: Settings,
    upload: UploadFile | None = None,
    url: str | None = None,
    b64: str | None = None,
) -> Image.Image:
    """Resolve whichever of the three input forms the caller used."""
    provided = [x for x in (upload, url, b64) if x]
    if not provided:
        raise MediaError("provide one of: media (file), url, or media_base64", code="missing_media")
    if len(provided) > 1:
        raise MediaError("provide exactly one of: media, url, media_base64", code="ambiguous_media")

    if upload is not None:
        return decode(await upload.read(), settings)
    if url is not None:
        return await from_url(url, settings)
    assert b64 is not None
    return from_base64(b64, settings)
