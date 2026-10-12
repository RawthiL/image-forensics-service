"""The two endpoints that look for things hidden *in* an image.

`POST /v1/screen` reads the file but never the picture: container metadata,
smuggled codepoints, bytes past the end of the image data. It needs no model,
no sibling service and no GPU, which makes it the one service here a supplier
can run on a small box.

`POST /v1/inspect` reads the picture too. It renders the views a victim's
preprocessing would produce, asks the OCR service for the text boxes and their
contents, and asks the classifier whether what it found is an injection.

They are separate routes, separately enabled and separately keyed, because
they are separate Pocket Network services with entirely different running
costs. Both run the container checks -- the same code -- so the two never
disagree about a file.

Deliberately not part of any vendor dialect: Sightengine and Hive have no
field for this, and bending one of their envelopes around it would break the
promise that a client swaps a base URL and nothing else.

Unlike `/v1/detect`, the answer is not a number. An operator deciding whether
to feed an image to an agent needs to know *where* something was found,
because the location carries most of the meaning -- the same sentence is a
photograph of a sign at native resolution and an attack when it appears only
after a downscale.
"""

from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel, Field

from ..inspection.report import InspectionReport
from .deps import InspectorDep, SettingsDep, load_one

screen_router = APIRouter(prefix="/v1", tags=["screening"])
router = APIRouter(prefix="/v1", tags=["inspection"])


class FindingOut(BaseModel):
    kind: str = Field(..., description="`scaling_attack`, `unicode_tag_block`, ...")
    severity: str = Field(..., description="none | low | medium | high")
    score: float = Field(..., ge=0, le=1)
    where: str = Field(..., description="`exif:UserComment`, `view:rescale:384:bicubic`, ...")
    detail: str | None = None
    excerpt: str | None = Field(
        None, description="Present only when that endpoint's `include_text` is enabled"
    )


class UpstreamOut(BaseModel):
    status: str = Field(..., description="ok | disabled | skipped | unreachable")
    fingerprint: str | None = None
    detail: str | None = None


class AnalysisResponse(BaseModel):
    scope: str = Field(
        ...,
        description=(
            "How much of the file was examined. `container` means the pixels "
            "were never looked at, so a `clean` is the far weaker statement."
        ),
    )
    verdict: str = Field(..., description="clean | suspicious | malicious")
    injection: float = Field(..., ge=0, le=1, description="Highest finding score")
    severity: str = Field(..., description="Highest finding severity")
    findings: list[FindingOut]
    upstreams: dict[str, UpstreamOut] = Field(
        default_factory=dict,
        description=(
            "Which composed services actually ran. A check that was skipped or "
            "unreachable says so here rather than being silently absent. Empty "
            "at container scope, which composes nothing."
        ),
    )
    views_rendered: int
    pipeline_fingerprint: str = Field(
        ..., description="Everything that decides the answer at this scope"
    )
    elapsed_ms: float


@screen_router.post(
    "/screen",
    response_model=AnalysisResponse,
    summary="Screen the image container (no model, no upstream)",
)
async def screen(
    inspector: InspectorDep,
    settings: SettingsDep,
    media: UploadFile | None = File(None, description="Image file upload"),
    url: str | None = Form(None, description="URL of an image to fetch"),
    media_base64: str | None = Form(None, description="Base64 image, data: URL accepted"),
) -> AnalysisResponse:
    loaded = await _load(settings, media, url, media_base64)
    report = await inspector.screen(loaded.image, loaded.container)
    return _render(report, inspector, settings.endpoints.screen.flag_threshold)


@router.post(
    "/inspect",
    response_model=AnalysisResponse,
    summary="Inspect an image for an embedded prompt injection",
)
async def inspect(
    inspector: InspectorDep,
    settings: SettingsDep,
    media: UploadFile | None = File(None, description="Image file upload"),
    url: str | None = Form(None, description="URL of an image to fetch"),
    media_base64: str | None = Form(None, description="Base64 image, data: URL accepted"),
) -> AnalysisResponse:
    loaded = await _load(settings, media, url, media_base64)
    report = await inspector.inspect(loaded.image, loaded.container)
    return _render(report, inspector, settings.endpoints.inspect.flag_threshold)


async def _load(settings, upload, url, b64):
    # keep_raw: the container *is* the evidence for the screening endpoint,
    # and Pillow has thrown it away by the time it hands back pixels.
    return await load_one(settings=settings, upload=upload, url=url, b64=b64, keep_raw=True)


def _render(report: InspectionReport, inspector, threshold: float) -> AnalysisResponse:
    return AnalysisResponse(
        scope=report.scope,
        verdict=report.verdict(threshold),
        injection=round(report.score, 4),
        severity=report.severity,
        findings=[FindingOut(**f.as_dict()) for f in report.sorted_findings()],
        upstreams={k: UpstreamOut(**v.as_dict()) for k, v in report.upstreams.items()},
        views_rendered=report.views_rendered,
        pipeline_fingerprint=inspector.pipeline_fingerprint(report.scope),
        elapsed_ms=round(report.elapsed_ms, 2),
    )
