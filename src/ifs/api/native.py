"""Our own API. Honest about what it returns; no vendor baggage.

Use this one for new integrations. The vendor dialects exist to capture callers
who already wrote against Sightengine or Hive.
"""

from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel, Field

from .deps import ServiceDep, SettingsDep, load_one

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


@router.post("/detect", response_model=DetectResponse, summary="Score one image")
async def detect(
    service: ServiceDep,
    settings: SettingsDep,
    media: UploadFile | None = File(None, description="Image file upload"),
    url: str | None = Form(None, description="URL of an image to fetch"),
    media_base64: str | None = Form(None, description="Base64 image, data: URL accepted"),
) -> DetectResponse:
    image = await load_one(settings=settings, upload=media, url=url, b64=media_base64)
    scored = await service.score(image)
    return DetectResponse(
        ai_generated=scored.result.ai_generated,
        deepfake=scored.result.deepfake,
        per_class=scored.result.per_class,
        model_fingerprint=service.info().fingerprint,
        pipeline_fingerprint=service.pipeline_fingerprint,
        crops_scored=scored.crops_scored,
        elapsed_ms=round(scored.elapsed_ms, 2),
    )
