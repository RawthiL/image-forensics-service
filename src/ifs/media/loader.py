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
from dataclasses import dataclass

import httpx
from PIL import Image, UnidentifiedImageError

from ..config import Settings
from .ssrf import MAX_REDIRECTS, UnsafeUrlError, validate_url

log = logging.getLogger(__name__)

# Pillow's own bomb guard raises a warning at ~89M pixels; we set a hard limit
# from config and disable the soft one so the behavior is ours, not Pillow's.
Image.MAX_IMAGE_PIXELS = None


@dataclass(frozen=True)
class LoadedMedia:
    """A decoded image, and optionally the bytes it was decoded from.

    `/v1/inspect` needs the original container: EXIF, XMP, the JPEG COM
    marker and any trailing polyglot data are all gone once Pillow has handed
    back pixels. `/v1/detect` does not, and a 15 MiB buffer held for the life
    of every in-flight detection request would show up in this service's
    resident set, which is a number the project measures. So the bytes are
    kept only when a route asks for them.
    """

    image: Image.Image
    raw: bytes | None = None
    format: str | None = None

    @property
    def container(self) -> bytes:
        if self.raw is None:
            raise RuntimeError("this media was loaded without keep_raw")
        return self.raw


class MediaError(ValueError):
    """Bad input from the caller. The API layer maps this to a 4xx."""

    def __init__(self, message: str, *, code: str = "invalid_media") -> None:
        super().__init__(message)
        self.code = code


async def from_url(url: str, settings: Settings, *, keep_raw: bool = False) -> LoadedMedia:
    """Fetch an image by URL, refusing anything that is not publicly routable.

    Redirects are followed manually so every hop is validated. httpx's own
    `follow_redirects` would check only the URL the caller handed us, and a
    redirect to `http://169.254.169.254/` would sail through.
    """
    if not settings.limits.allow_url_fetch:
        raise MediaError("URL fetching is disabled on this deployment", code="url_fetch_disabled")

    allow_private = settings.limits.allow_private_urls
    raw = bytearray()

    # trust_env=False: an HTTP(S)_PROXY in the environment would route the
    # request through a host the guard never checked.
    async with httpx.AsyncClient(
        timeout=settings.limits.url_fetch_timeout_s, follow_redirects=False, trust_env=False
    ) as client:
        for _ in range(MAX_REDIRECTS + 1):
            try:
                validate_url(url, allow_private=allow_private)
            except UnsafeUrlError as exc:
                raise MediaError(str(exc), code="url_not_allowed") from exc

            try:
                async with client.stream("GET", url) as resp:
                    if resp.is_redirect:
                        location = resp.headers.get("location")
                        if not location:
                            raise MediaError(
                                "redirect without a location header", code="url_fetch_failed"
                            )
                        url = str(resp.url.join(location))
                        continue
                    resp.raise_for_status()
                    # Stream so an oversized body is cut off mid-flight rather
                    # than fully buffered before we check its length.
                    async for chunk in resp.aiter_bytes():
                        raw.extend(chunk)
                        if len(raw) > settings.limits.max_image_bytes:
                            raise MediaError(
                                f"image exceeds {settings.limits.max_image_bytes} bytes",
                                code="media_too_large",
                            )
                    break
            except httpx.HTTPError as exc:
                raise MediaError(f"could not fetch url: {exc}", code="url_fetch_failed") from exc
        else:
            raise MediaError("too many redirects", code="url_fetch_failed")

    return decode(bytes(raw), settings, keep_raw=keep_raw)


def from_base64(payload: str, settings: Settings, *, keep_raw: bool = False) -> LoadedMedia:
    # Tolerate data: URLs, which browsers produce and callers paste.
    if payload.startswith("data:"):
        _, _, payload = payload.partition(",")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise MediaError("media_base64 is not valid base64", code="invalid_base64") from exc
    return decode(raw, settings, keep_raw=keep_raw)


def decode(raw: bytes, settings: Settings, *, keep_raw: bool = False) -> LoadedMedia:
    if not raw:
        raise MediaError("empty image payload", code="empty_media")
    if len(raw) > settings.limits.max_image_bytes:
        raise MediaError(
            f"image exceeds {settings.limits.max_image_bytes} bytes", code="media_too_large"
        )

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

    return LoadedMedia(image=img, raw=raw if keep_raw else None, format=img.format)
