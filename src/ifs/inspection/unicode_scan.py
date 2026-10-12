"""Codepoint-level smuggling.

Applies to *any* text this service extracts -- a metadata field, an OCR
result, a filename. A model reads a string; a human reads a rendering of it.
Several Unicode ranges exist in the gap, and one of them, the Tags block,
carries a full ASCII alphabet that renders as nothing at all.

These checks are deterministic: no model, no network, no false positives worth
the name. A Tags-block codepoint in an image has no benign explanation.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

# U+E0000..U+E007F. U+E0020..U+E007E map 1:1 onto printable ASCII by
# subtracting U+E0000, which is what makes the block a carrier rather than a
# curiosity: an entire instruction can be written in characters that no
# renderer draws.
TAG_START, TAG_END = 0xE0000, 0xE007F

ZERO_WIDTH = {
    0x200B,  # ZERO WIDTH SPACE
    0x200C,  # ZERO WIDTH NON-JOINER
    0x200D,  # ZERO WIDTH JOINER
    0x2060,  # WORD JOINER
    0xFEFF,  # ZERO WIDTH NO-BREAK SPACE / BOM
}

# Overrides, not marks: these reorder what is displayed without changing what
# is parsed, which is the "Trojan Source" shape applied to prose.
BIDI_OVERRIDE = {0x202A, 0x202B, 0x202C, 0x202D, 0x202E, 0x2066, 0x2067, 0x2068, 0x2069}

VARIATION_SELECTORS = set(range(0xFE00, 0xFE10)) | set(range(0xE0100, 0xE01F0))

PRIVATE_USE = [(0xE000, 0xF8FF), (0xF0000, 0xFFFFD), (0x100000, 0x10FFFD)]


@dataclass(frozen=True)
class Hit:
    kind: str
    severity: str
    count: int
    decoded: str | None = None
    """For the Tags block: the payload, read back as plain text."""


def scan(text: str) -> list[Hit]:
    """Every smuggling class present in `text`, with counts."""
    if not text:
        return []

    tags = [c for c in text if TAG_START <= ord(c) <= TAG_END]
    zero_width = [c for c in text if ord(c) in ZERO_WIDTH]
    bidi = [c for c in text if ord(c) in BIDI_OVERRIDE]
    selectors = [c for c in text if ord(c) in VARIATION_SELECTORS]
    private = [c for c in text if _in_private_use(ord(c))]

    hits: list[Hit] = []
    if tags:
        hits.append(
            Hit("unicode_tag_block", "high", len(tags), decoded=decode_tags("".join(tags)))
        )
    if bidi:
        hits.append(Hit("unicode_bidi_override", "high", len(bidi)))
    if zero_width:
        # Legitimate in Indic and Arabic shaping, and in a few emoji
        # sequences, so this is a signal rather than a verdict.
        hits.append(Hit("unicode_zero_width", "medium", len(zero_width)))
    if selectors:
        hits.append(Hit("unicode_variation_selector", "medium", len(selectors)))
    if private:
        hits.append(Hit("unicode_private_use", "low", len(private)))
    return hits


def decode_tags(text: str) -> str:
    """Read a Tags-block payload back as the ASCII it encodes."""
    return "".join(
        chr(ord(c) - TAG_START) for c in text if TAG_START + 0x20 <= ord(c) <= TAG_START + 0x7E
    )


def strip(text: str) -> str:
    """Remove the invisible carriers, leaving what a human would read.

    Used before the rules pass so a payload cannot evade a pattern by hiding
    a zero-width space inside `ig<ZWSP>nore previous instructions`.
    """
    return "".join(
        c
        for c in text
        if not (
            TAG_START <= ord(c) <= TAG_END
            or ord(c) in ZERO_WIDTH
            or ord(c) in BIDI_OVERRIDE
            or ord(c) in VARIATION_SELECTORS
        )
    )


def normalize(text: str) -> str:
    """What the rules and the classifier see: tags decoded, carriers removed, NFKC.

    The Tags payload is *appended* rather than dropped. It is the attacker's
    actual instruction, so it belongs in the text that gets classified, not
    only in a finding that says some odd codepoints were present.
    """
    decoded = decode_tags(text)
    visible = unicodedata.normalize("NFKC", strip(text))
    return f"{visible} {decoded}".strip() if decoded else visible


def _in_private_use(cp: int) -> bool:
    return any(lo <= cp <= hi for lo, hi in PRIVATE_USE)
