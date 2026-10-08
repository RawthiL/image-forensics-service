"""Getting bytes into a PIL image, safely.

Three input paths, mirroring the vendors we emulate: multipart upload, a URL to
fetch, and base64 in a JSON body. All three land here so the size, pixel-count,
and format guards live in exactly one place.
"""

from __future__ import annotations

import base64
import binascii
import io
import logging

import httpx
from PIL import Image, UnidentifiedImageError

from ..config import Settings

log = logging.getLogger(__name__)

# Pillow's own bomb guard raises a warning at ~89M pixels; we set a hard limit
# from config and disable the soft one so the behavior is ours, not Pillow's.
Image.MAX_IMAGE_PIXELS = None


class MediaError(ValueError):
    """Bad input from the caller. The API layer maps this to a 4xx."""

    def __init__(self, message: str, *, code: str = "invalid_media") -> None:
        super().__init__(message)
        self.code = code


async def from_url(url: str, settings: Settings) -> Image.Image:
    if not settings.limits.allow_url_fetch:
        raise MediaError("URL fetching is disabled on this deployment", code="url_fetch_disabled")
    if not url.lower().startswith(("http://", "https://")):
        raise MediaError("url must be http(s)", code="invalid_url")

    try:
        async with httpx.AsyncClient(
            timeout=settings.limits.url_fetch_timeout_s, follow_redirects=True
        ) as client:
            # Stream so an oversized body is cut off mid-flight rather than
            # fully buffered before we check its length.
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                buf = bytearray()
                async for chunk in resp.aiter_bytes():
                    buf.extend(chunk)
                    if len(buf) > settings.limits.max_image_bytes:
                        raise MediaError(
                            f"image exceeds {settings.limits.max_image_bytes} bytes", code="media_too_large"
                        )
    except httpx.HTTPError as exc:
        raise MediaError(f"could not fetch url: {exc}", code="url_fetch_failed") from exc

    return decode(bytes(buf), settings)


def from_base64(payload: str, settings: Settings) -> Image.Image:
    # Tolerate data: URLs, which browsers produce and callers paste.
    if payload.startswith("data:"):
        _, _, payload = payload.partition(",")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise MediaError("media_base64 is not valid base64", code="invalid_base64") from exc
    return decode(raw, settings)


def decode(raw: bytes, settings: Settings) -> Image.Image:
    if not raw:
        raise MediaError("empty image payload", code="empty_media")
    if len(raw) > settings.limits.max_image_bytes:
        raise MediaError(f"image exceeds {settings.limits.max_image_bytes} bytes", code="media_too_large")

    try:
        img = Image.open(io.BytesIO(raw))
        # Read the header first so we can reject a bomb before allocating it.
        width, height = img.size
        if width * height > settings.limits.max_pixels:
            raise MediaError(
                f"image is {width}x{height}, over the {settings.limits.max_pixels} pixel limit",
                code="media_too_large",
            )
        img.load()
    except UnidentifiedImageError as exc:
        raise MediaError("unsupported or corrupt image format", code="invalid_media") from exc
    except OSError as exc:
        raise MediaError(f"could not decode image: {exc}", code="invalid_media") from exc

    return img
