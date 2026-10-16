"""Our own API. Honest about what it returns; no vendor baggage.

Use this one for new integrations. The vendor dialects exist to capture callers
who already wrote against Sightengine or Hive.
"""

from __future__ import annotations

from fastapi import APIRouter, File, Form, Request, UploadFile
from pydantic import BaseModel, Field

from .deps import JSON_BODY_DOC, ServiceDep, SettingsDep, load_request

router = APIRouter(prefix="/v1", tags=["detection"])


class DetectResponse(BaseModel):
    ai_generated: float = Field(..., ge=0, le=1, description="P(image is AI-generated)")
    deepfake: float | None = Field(None, ge=0, le=1)
    per_class: dict[str, float] = Field(default_factory=dict)
    model_fingerprint: str = Field(..., description="Identity of the model alone")
    pipeline_fingerprint: str = Field(
        ..., description="Model identity plus the service-level scoring policy"
    )
    crops_scored: int
    elapsed_ms: float


@router.post(
    "/detect",
    response_model=DetectResponse,
    summary="Score one image",
    openapi_extra=JSON_BODY_DOC,
)
async def detect(
    request: Request,
    service: ServiceDep,
    settings: SettingsDep,
    media: UploadFile | None = File(None, description="Image file upload"),
    url: str | None = Form(None, description="URL of an image to fetch"),
    media_base64: str | None = Form(None, description="Base64 image, data: URL accepted"),
) -> DetectResponse:
    # Same body rules as the other two native routes: a form, or a JSON object
    # carrying `url` or `media_base64`. The vendor dialects stay form-only.
    loaded = await load_request(request, settings, media, url, media_base64)
    scored = await service.score(loaded.image)
    return DetectResponse(
        ai_generated=scored.result.ai_generated,
        deepfake=scored.result.deepfake,
        per_class=scored.result.per_class,
        model_fingerprint=service.info().fingerprint,
        pipeline_fingerprint=service.pipeline_fingerprint,
        crops_scored=scored.crops_scored,
        elapsed_ms=round(scored.elapsed_ms, 2),
    )
