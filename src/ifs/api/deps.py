"""Shared request plumbing for every dialect."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request, UploadFile

from ..config import Settings, get_settings
from ..inspection.service import InspectionService
from ..media.loader import LoadedMedia, MediaError, decode, from_base64, from_url
from ..service import DetectionService


def get_service(request: Request) -> DetectionService:
    return request.app.state.service


def get_inspector(request: Request) -> InspectionService:
    return request.app.state.inspector


ServiceDep = Annotated[DetectionService, Depends(get_service)]
InspectorDep = Annotated[InspectionService, Depends(get_inspector)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


async def load_one(
    *,
    settings: Settings,
    upload: UploadFile | None = None,
    url: str | None = None,
    b64: str | None = None,
    keep_raw: bool = False,
) -> LoadedMedia:
    """Resolve whichever of the three input forms the caller used.

    `keep_raw` retains the original container bytes, which only `/v1/inspect`
    needs -- see `LoadedMedia`.
    """
    provided = [x for x in (upload, url, b64) if x]
    if not provided:
        raise MediaError("provide one of: media (file), url, or media_base64", code="missing_media")
    if len(provided) > 1:
        raise MediaError("provide exactly one of: media, url, media_base64", code="ambiguous_media")

    if upload is not None:
        return decode(await upload.read(), settings, keep_raw=keep_raw)
    if url is not None:
        return await from_url(url, settings, keep_raw=keep_raw)
    assert b64 is not None
    return from_base64(b64, settings, keep_raw=keep_raw)
