"""Client for the prompt-injection classifier.

The text half of this problem is already solved next door, by
`language-forensics-service`: it owns the tokenizer pinning, the window
policy, the batching and -- not least -- the weights licensing. Reimplementing
any of that here would mean maintaining a second, diverging copy of the subtle
part, and dragging a model licence into a repository that does not otherwise
need one.

So this file is a wire format and nothing else. Every text extracted from the
image is sent with role `document`, which is what it is: content that arrived
with the data, not from the user.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from .errors import UpstreamError
from .report import ExtractedText

log = logging.getLogger(__name__)

DETECT_PATH = "/v1/detect"
MODELS_PATH = "/v1/models"

# Text from an image is indirect injection by definition: it came in with the
# content. The classifier echoes the role back rather than scoring on it.
ROLE = "document"


async def fingerprint(client: httpx.AsyncClient, cfg) -> str | None:
    """Identify the served classifier, once, at startup.

    Read here rather than taken from each `/v1/detect` response so the
    fingerprint this service reports cannot depend on which requests happened
    to classify text. Best-effort: an upstream that is not ready yet is not a
    reason to refuse to start.
    """
    headers = {"x-api-key": cfg.api_key} if cfg.api_key else {}
    try:
        resp = await client.get(
            f"{cfg.url.rstrip('/')}{MODELS_PATH}", headers=headers, timeout=cfg.timeout_s
        )
        resp.raise_for_status()
        body = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("could not read classifier identity: %s", exc)
        return None
    return body.get("pipeline_fingerprint") or body.get("model_fingerprint")


@dataclass(frozen=True)
class Judgement:
    text: ExtractedText
    score: float
    flagged: bool


@dataclass(frozen=True)
class ClassifierOutcome:
    judgements: list[Judgement]
    fingerprint: str | None
    threshold: float


async def classify(
    client: httpx.AsyncClient, cfg, texts: list[ExtractedText]
) -> ClassifierOutcome:
    """Score every extracted text in one call."""
    inputs = [
        {"text": _truncate(t.text, cfg.max_text_chars), "role": ROLE, "id": t.where}
        for t in texts
    ]
    body: dict = {"inputs": inputs}
    if cfg.threshold is not None:
        body["threshold"] = cfg.threshold
    headers = {"x-api-key": cfg.api_key} if cfg.api_key else {}

    try:
        resp = await client.post(
            _url(cfg), json=body, headers=headers, timeout=cfg.timeout_s
        )
        resp.raise_for_status()
        parsed = resp.json()
    except httpx.HTTPStatusError as exc:
        raise UpstreamError(
            "classifier",
            f"classifier returned {exc.response.status_code}: {_detail(exc.response)}",
        ) from exc
    except httpx.HTTPError as exc:
        raise UpstreamError("classifier", f"could not reach the classifier: {exc}") from exc
    except ValueError as exc:
        raise UpstreamError("classifier", f"classifier returned invalid JSON: {exc}") from exc

    results = parsed.get("results")
    if not isinstance(results, list):
        raise UpstreamError("classifier", "classifier response has no 'results' array")

    judgements = []
    for result in results:
        index = result.get("index")
        if not isinstance(index, int) or not 0 <= index < len(texts):
            # Do not guess which input a result belongs to: a misattributed
            # score would name the wrong location in the report.
            log.warning("classifier returned an out-of-range index %r, ignoring", index)
            continue
        judgements.append(
            Judgement(
                text=texts[index],
                score=float(result.get("score", 0.0)),
                flagged=bool(result.get("flagged", False)),
            )
        )

    return ClassifierOutcome(
        judgements=judgements,
        fingerprint=parsed.get("pipeline_fingerprint") or parsed.get("model_fingerprint"),
        threshold=float(parsed.get("threshold", cfg.threshold or 0.5)),
    )


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit]


def _url(cfg) -> str:
    return f"{cfg.url.rstrip('/')}{DETECT_PATH}"


def _detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(body, dict):
        for key in ("message", "detail", "error"):
            if key in body:
                return str(body[key])[:200]
    return str(body)[:200]
