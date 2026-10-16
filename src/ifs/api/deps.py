"""Shared request plumbing for every dialect."""

from __future__ import annotations

import json
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
    accepts_file: bool = True,
) -> LoadedMedia:
    """Resolve whichever of the three input forms the caller used.

    `keep_raw` retains the original container bytes, which only `/v1/inspect`
    needs -- see `LoadedMedia`.

    `accepts_file` is False when the body was JSON, which carries no file. The
    error then lists only the fields that body could have held, because an
    error advertising a field the caller cannot use is how this started.
    """
    fields = "media (file), url, or media_base64" if accepts_file else "url or media_base64"
    provided = [x for x in (upload, url, b64) if x]
    if not provided:
        raise MediaError(f"provide one of: {fields}", code="missing_media")
    if len(provided) > 1:
        raise MediaError(f"provide exactly one of: {fields}", code="ambiguous_media")

    if upload is not None:
        return decode(await upload.read(), settings, keep_raw=keep_raw)
    if url is not None:
        return await from_url(url, settings, keep_raw=keep_raw)
    assert b64 is not None
    return from_base64(b64, settings, keep_raw=keep_raw)


# Room for the JSON envelope around a base64 payload at the size limit.
_JSON_SLACK = 64 * 1024

# The JSON form of the shared body, for routes that accept one. `media` is
# absent on purpose: JSON carries no file, so the bytes go in `media_base64`.
JSON_BODY_DOC = {
    "requestBody": {
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL of an image to fetch"},
                        "media_base64": {
                            "type": "string",
                            "description": "Base64 image, data: URL accepted",
                        },
                    },
                }
            }
        }
    }
}


_FORM_TYPES = ("multipart/form-data", "application/x-www-form-urlencoded")


def _body_kind(request: Request) -> str:
    """`form`, `json`, or `unknown` -- and `unknown` is the one that matters.

    `Content-Type` is a hint here, never the decision. A request that crossed a
    relay or a proxy arrives with whatever headers that hop chose to carry: a
    RelayMiner takes them from the serialized inner request and, for the ones
    the gateway left unset, backfills from the relay envelope -- which is not
    the client's content type. A body is cheap to recognise and the header is
    not ours to trust, so we look at the bytes.
    """
    ctype = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if ctype in _FORM_TYPES:
        return "form"
    if ctype == "application/json" or ctype.endswith("+json"):
        return "json"
    return "unknown"


async def _read_capped(request: Request, settings: Settings) -> bytes:
    """The body, refused as soon as it passes what the largest image could need.

    The cap is base64 of `max_image_bytes` plus slack for the envelope, so an
    oversized request dies mid-stream rather than being buffered whole.
    """
    cap = settings.limits.max_image_bytes * 4 // 3 + _JSON_SLACK
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > cap:
            raise MediaError(f"request body exceeds {cap} bytes", code="media_too_large")
    return bytes(body)


def json_media_fields(body: bytes, *, declared: bool) -> tuple[str | None, str | None]:
    """`(url, media_base64)` from a JSON object body.

    `declared` says whether the caller announced JSON. When they did not -- a
    proxied request, or curl without a header -- the parse error has to explain
    both bodies, because we cannot know which one they meant to send.
    """
    try:
        data = json.loads(body)
    except (ValueError, UnicodeDecodeError) as exc:
        detail = (
            "request body is not valid JSON"
            if declared
            else "request body is neither a form nor valid JSON"
        )
        raise MediaError(detail, code="invalid_json") from exc
    if not isinstance(data, dict):
        raise MediaError("JSON body must be an object", code="invalid_json")
    if "media" in data:
        raise MediaError(
            "a JSON body carries no file: send the bytes as media_base64",
            code="invalid_json",
        )
    fields = []
    for name in ("url", "media_base64"):
        value = data.get(name)
        if value is not None and not isinstance(value, str):
            raise MediaError(f"{name} must be a string", code="invalid_json")
        fields.append(value)
    return fields[0], fields[1]


async def load_request(
    request: Request,
    settings: Settings,
    upload: UploadFile | None,
    url: str | None,
    b64: str | None,
    *,
    keep_raw: bool = False,
) -> LoadedMedia:
    """Load the image from a form body or a JSON one, whichever arrived.

    The form is recognised by its content type, which is the one case where
    the header must be right: multipart cannot be parsed without the boundary
    it carries. Anything else is tried as JSON -- see `_body_kind`.

    Only the native routes use this. The vendor dialects stay form-only on
    purpose: they exist to be byte-compatible with Sightengine and Hive, and
    neither accepts JSON, so accepting it there would invent a contract.
    """
    kind = _body_kind(request)
    if kind == "form":
        return await load_one(
            settings=settings, upload=upload, url=url, b64=b64, keep_raw=keep_raw
        )

    body = await _read_capped(request, settings)
    if not body.strip():
        # Nothing to parse. A caller who announced JSON gets the JSON fields;
        # one who sent no body at all could have meant either, so name all three.
        return await load_one(settings=settings, keep_raw=keep_raw, accepts_file=kind != "json")

    url, b64 = json_media_fields(body, declared=kind == "json")
    return await load_one(
        settings=settings, url=url, b64=b64, keep_raw=keep_raw, accepts_file=False
    )
