"""Text carried alongside the pixels.

Container metadata is a weak carrier on its own: most pixel pipelines decode
to RGB and drop EXIF on the floor, so a payload there usually never reaches a
model at all. It becomes a real vector the moment application code lifts it
into a prompt -- a captioning tool reading EXIF to "describe the photo", a
document parser forwarding fields, an agent that shells out to exiftool, an
indexer treating metadata as searchable text.

It is also nearly free to check: no model, no network, no new dependency. So
it is here for coverage, not because it is where the attacks are.

XMP is scanned as text rather than parsed as XML. We are looking for injected
prose, not structure, and not running an XML parser over attacker-controlled
input is a feature.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator

from PIL import ExifTags, Image

from .report import ExtractedText, Finding

log = logging.getLogger(__name__)

# Below this, a "string" in a binary blob is noise rather than a message.
_MIN_RUN = 8
_PRINTABLE = re.compile(rb"[\x20-\x7e]{%d,}" % _MIN_RUN)

# The XMP packet wrapper, which ISO 16684-1 defines as
# `<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>` -- the `begin`
# attribute value *is* U+FEFF. Every Adobe-written file carries it, so scanning
# the wrapper would report a zero-width character on a huge share of ordinary
# images. We want the packet's contents, not its envelope.
_XPACKET = re.compile(r"<\?xpacket[^?]*\?>")

# EXIF charset prefixes on UserComment (EXIF 2.3, 4.6.5).
_COMMENT_CHARSETS = {
    b"ASCII\x00\x00\x00": "ascii",
    b"UNICODE\x00": "utf-16",
    b"JIS\x00\x00\x00\x00\x00": "shift_jis",
    b"\x00" * 8: "latin-1",
}

# Windows XP* tags are UTF-16LE, unlike every other EXIF string.
_XP_TAGS = {40091, 40092, 40093, 40094, 40095}


def extract(img: Image.Image, raw: bytes) -> tuple[list[ExtractedText], list[Finding]]:
    """Every string the container carries, plus structural findings.

    Structural findings (trailing data) are returned separately because they
    are not text: nothing downstream can classify them, and they stand on
    their own as evidence.
    """
    texts: list[ExtractedText] = []
    findings: list[Finding] = []

    texts.extend(_exif(img))
    texts.extend(_png_text(img))
    texts.extend(_info_strings(img))
    texts.extend(_icc(img))
    texts.extend(_jpeg_comments(raw))

    findings.extend(_trailing_data(img.format, raw))
    return texts, findings


def _exif(img: Image.Image) -> Iterator[ExtractedText]:
    try:
        exif = img.getexif()
    except Exception as exc:  # a malformed IFD must not fail the request
        log.debug("exif parse failed: %s", exc)
        return
    if not exif:
        return

    ifds = [("exif", exif)]
    for name, tag in (("exif.Exif", 0x8769), ("exif.GPS", 0x8825)):
        try:
            sub = exif.get_ifd(tag)
        except Exception:
            continue
        if sub:
            ifds.append((name, sub))

    for prefix, ifd in ifds:
        for tag, value in ifd.items():
            text = _decode(tag, value)
            if text:
                name = ExifTags.TAGS.get(tag, ExifTags.GPSTAGS.get(tag, f"0x{tag:04x}"))
                yield ExtractedText(f"{prefix}:{name}", text, "metadata")


def _decode(tag: int, value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, bytes):
        return ""
    if tag in _XP_TAGS:
        return value.decode("utf-16-le", "replace").rstrip("\x00").strip()
    for prefix, encoding in _COMMENT_CHARSETS.items():
        if value.startswith(prefix):
            return value[len(prefix) :].decode(encoding, "replace").rstrip("\x00").strip()
    return value.decode("utf-8", "replace").rstrip("\x00").strip()


def _png_text(img: Image.Image) -> Iterator[ExtractedText]:
    """tEXt, iTXt and zTXt chunks. Pillow decompresses zTXt for us."""
    for key, value in getattr(img, "text", {}).items():
        if isinstance(value, str):
            cleaned = _unwrap(key, value)
            if cleaned:
                yield ExtractedText(f"png:text:{key}", cleaned, "metadata")


def _info_strings(img: Image.Image) -> Iterator[ExtractedText]:
    """XMP packets, GIF/PNG comments, and anything else Pillow surfaced as text."""
    skip = {"icc_profile", "exif", "photoshop", "transparency", "background", "palette"}
    for key, value in img.info.items():
        if key in skip:
            continue
        if isinstance(value, bytes):
            value = value.decode("utf-8", "replace")
        if isinstance(value, str):
            cleaned = _unwrap(key, value)
            if cleaned:
                yield ExtractedText(f"info:{key}", cleaned, "metadata")


def _unwrap(key: str, value: str) -> str:
    """Strip a carrier's own envelope, so only its contents are scanned.

    XMP is the case that matters: an XMP packet arrives through either the PNG
    `tEXt` path or `img.info`, depending on the container, so the strip lives
    here rather than in one of them.
    """
    if "xmp" in key.lower():
        value = _XPACKET.sub(" ", value)
    return value.strip()


def _icc(img: Image.Image) -> Iterator[ExtractedText]:
    """Printable runs from the ICC profile.

    The profile is a binary structure with free-text description and copyright
    tags inside it. Pulling runs out is cruder than parsing, and catches the
    same payloads without an ICC parser.
    """
    profile = img.info.get("icc_profile")
    if isinstance(profile, bytes):
        joined = " ".join(_runs(profile))
        if joined:
            yield ExtractedText("icc:profile", joined, "metadata")


def _jpeg_comments(raw: bytes) -> Iterator[ExtractedText]:
    """JPEG COM (0xFFFE) segments, which Pillow does not expose."""
    if not raw.startswith(b"\xff\xd8"):
        return
    pos = 2
    end = len(raw)
    while pos + 4 <= end:
        if raw[pos] != 0xFF:
            break
        marker = raw[pos + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            pos += 2
            continue
        if marker == 0xDA:  # start of scan: entropy-coded data follows
            break
        length = int.from_bytes(raw[pos + 2 : pos + 4], "big")
        if length < 2:
            break
        if marker == 0xFE:
            body = raw[pos + 4 : pos + 2 + length]
            text = body.decode("utf-8", "replace").strip("\x00").strip()
            if text:
                yield ExtractedText("jpeg:COM", text, "metadata")
        pos += 2 + length


def _trailing_data(fmt: str | None, raw: bytes) -> list[Finding]:
    """Bytes after the image's logical end.

    A file that is a valid image *and* something else is a polyglot. The
    appended half is invisible to anything that only decodes pixels, and is
    exactly what a second parser downstream would read.
    """
    end = _logical_end(fmt, raw)
    if end is None:
        return []
    extra = len(raw) - end
    # A few bytes of padding are common enough in the wild to be noise.
    if extra <= 16:
        return []
    return [
        Finding(
            kind="trailing_data",
            severity="medium",
            score=0.6,
            where=f"container:{(fmt or 'unknown').lower()}",
            detail=(
                f"{extra} bytes follow the end of the image data; the file is a valid "
                "image and also carries something else"
            ),
        )
    ]


def _logical_end(fmt: str | None, raw: bytes) -> int | None:
    if fmt == "JPEG":
        marker = raw.rfind(b"\xff\xd9")
        return marker + 2 if marker != -1 else None
    if fmt == "PNG":
        marker = raw.rfind(b"IEND")
        return marker + 8 if marker != -1 else None  # IEND + 4-byte CRC
    if fmt == "GIF":
        return len(raw) if raw.endswith(b"\x3b") else None
    return None


def _runs(blob: bytes) -> list[str]:
    return [m.group().decode("ascii") for m in _PRINTABLE.finditer(blob)]
