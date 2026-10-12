"""Client for the OCR service.

Speaks Google Cloud Vision's `images:annotate`, which is the API
`image-inference-server` exposes for its PP-OCR backend. That service owns the
models, the Triton deployment and the batching; this file owns nothing but the
wire format and the decision of what to send.

Views are sent as PNG. The payloads this looks for are faint by construction
-- a single quantisation step away from the background, or living in the low
bit -- and JPEG would destroy exactly the signal that matters.
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass

import httpx

from ..media.views import View
from .errors import UpstreamError
from .report import ExtractedText

log = logging.getLogger(__name__)

ANNOTATE_PATH = "/v1/images:annotate"
HEALTH_PATH = "/health"

# Full recognition: the detection model, then the recognition model once per
# detected box. The dense-text variant is the honest description of what an
# injection payload looks like.
RECOGNIZE = "DOCUMENT_TEXT_DETECTION"

# Detection only -- an extension this upstream adds to Cloud Vision. Answers
# "is there text here, and where" without running the recognition model or its
# per-box loop, which is the expensive half.
REGIONS = "TEXT_REGION_DETECTION"


@dataclass(frozen=True)
class ViewRegions:
    """Where text was found in one view, without reading it."""

    where: str
    count: int
    area_fraction: float
    """Share of the view covered by text boxes. Normalised, so a 224px view and
    a 4000px one are comparable."""


@dataclass(frozen=True)
class OcrOutcome:
    texts: list[ExtractedText]
    regions: list[ViewRegions]
    views_read: int
    errors: list[str]


async def fingerprint(client: httpx.AsyncClient, cfg) -> str | None:
    """Identify the served OCR model, for the composed pipeline fingerprint.

    Cloud Vision's response envelope has nowhere to put this, so it comes from
    the service's own health route instead. Best-effort: an unreachable health
    route is not a reason to refuse to start.
    """
    try:
        resp = await client.get(_url(cfg, HEALTH_PATH), timeout=cfg.timeout_s)
        resp.raise_for_status()
        body = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("could not read OCR service health: %s", exc)
        return None
    endpoint = (body.get("endpoints") or {}).get("ocr") or {}
    model = endpoint.get("model")
    return f"ocr:{model}" if model else None


async def read(
    client: httpx.AsyncClient, cfg, views: list[View], *, recognize: bool
) -> OcrOutcome:
    """Read every view, in as few requests as the upstream limits allow.

    `recognize=False` asks for text *regions* only. The upstream then runs the
    detection model alone, which is what makes the screening pass cheap.
    Regions are requested either way: recognition needs detection anyway, so
    in full mode they cost nothing beyond response size, and they answer the
    case where a view has visible text boxes that recognition could not read.
    """
    features = [REGIONS] + ([RECOGNIZE] if recognize else [])
    encoded = [(v, _png(v.image)) for v in views]
    texts: list[ExtractedText] = []
    regions: list[ViewRegions] = []
    errors: list[str] = []
    read_count = 0

    for chunk in _chunks(encoded, cfg.batch, cfg.max_request_bytes):
        responses = await _annotate(
            client, cfg, [payload for _, payload in chunk], features
        )
        if len(responses) != len(chunk):
            raise UpstreamError(
                "ocr",
                f"OCR service returned {len(responses)} responses for {len(chunk)} images",
            )
        for (view, _), response in zip(chunk, responses, strict=True):
            # Cloud Vision reports per-image failures inside a 200 response.
            # One bad view must not discard the rest of the batch.
            if error := response.get("error"):
                errors.append(f"{view.name}: {error.get('message', 'unknown error')}")
                continue
            read_count += 1
            where = f"view:{view.name}"

            annotation = response.get("fullTextAnnotation") or {}
            text = (annotation.get("text") or "").strip()
            if text:
                texts.append(
                    ExtractedText(
                        where=where,
                        text=text,
                        source="view",
                        confidence=_confidence(annotation),
                    )
                )

            found = _regions(response, view, cfg.min_region_score)
            if found is not None:
                regions.append(found)

    return OcrOutcome(texts=texts, regions=regions, views_read=read_count, errors=errors)


async def _annotate(
    client: httpx.AsyncClient, cfg, payloads: list[str], features: list[str]
) -> list[dict]:
    body = {
        "requests": [
            {
                "image": {"content": payload},
                "features": [{"type": name} for name in features],
            }
            for payload in payloads
        ]
    }
    headers = {"x-goog-api-key": cfg.api_key} if cfg.api_key else {}
    try:
        resp = await client.post(
            _url(cfg, ANNOTATE_PATH), json=body, headers=headers, timeout=cfg.timeout_s
        )
        resp.raise_for_status()
        parsed = resp.json()
    except httpx.HTTPStatusError as exc:
        raise UpstreamError(
            "ocr", f"OCR service returned {exc.response.status_code}: {_detail(exc.response)}"
        ) from exc
    except httpx.HTTPError as exc:
        raise UpstreamError("ocr", f"could not reach the OCR service: {exc}") from exc
    except ValueError as exc:
        raise UpstreamError("ocr", f"OCR service returned invalid JSON: {exc}") from exc

    responses = parsed.get("responses")
    if not isinstance(responses, list):
        raise UpstreamError("ocr", "OCR service response has no 'responses' array")
    return responses


def _chunks(
    encoded: list[tuple[View, str]], batch: int, max_bytes: int
) -> list[list[tuple[View, str]]]:
    """Group views into requests, bounded by both count and payload size.

    The upstream caps images per request *and* body size, and the native view
    of a large photograph can approach that cap on its own. Counting bytes
    here turns what would be a 413 into one more request.
    """
    chunks: list[list[tuple[View, str]]] = []
    current: list[tuple[View, str]] = []
    size = 0
    for item in encoded:
        item_size = len(item[1])
        if current and (len(current) >= batch or size + item_size > max_bytes):
            chunks.append(current)
            current, size = [], 0
        current.append(item)
        size += item_size
    if current:
        chunks.append(current)
    return chunks


def _png(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=False, compress_level=1)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _regions(response: dict, view: View, min_score: float) -> ViewRegions | None:
    """Text boxes in one view, as a count and a normalised area.

    Returns None when the upstream did not answer with regions at all, which
    is different from answering with none: one means the feature was not
    served, the other means this view has no text in it.
    """
    annotations = response.get("textRegionAnnotations")
    if annotations is None:
        return None
    kept = [a for a in annotations if a.get("score", 1.0) >= min_score]
    width, height = view.image.size
    frame = float(width * height) or 1.0
    covered = sum(_quad_area(a.get("boundingPoly", {}).get("vertices", [])) for a in kept)
    return ViewRegions(
        where=f"view:{view.name}",
        count=len(kept),
        area_fraction=round(min(covered / frame, 1.0), 6),
    )


def _quad_area(vertices: list[dict]) -> float:
    """Shoelace. The boxes are quadrilaterals, not necessarily axis-aligned."""
    if len(vertices) < 3:
        return 0.0
    points = [(float(v.get("x", 0)), float(v.get("y", 0))) for v in vertices]
    total = 0.0
    for i, (x0, y0) in enumerate(points):
        x1, y1 = points[(i + 1) % len(points)]
        total += x0 * y1 - x1 * y0
    return abs(total) / 2.0


def _confidence(annotation: dict) -> float:
    pages = annotation.get("pages") or []
    if not pages:
        return 1.0
    values = [p.get("confidence", 1.0) for p in pages]
    return round(sum(values) / len(values), 4)


def _url(cfg, path: str) -> str:
    return f"{cfg.url.rstrip('/')}{path}"


def _detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    error = body.get("error")
    if isinstance(error, dict):
        return str(error.get("message", ""))[:200]
    return str(body)[:200]
