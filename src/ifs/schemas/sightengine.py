"""Response models for the Sightengine dialect."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RequestBlock(BaseModel):
    id: str
    timestamp: float
    operations: int


class MediaBlock(BaseModel):
    id: str
    uri: str | None = None


class TypeBlock(BaseModel):
    ai_generated: float = Field(..., ge=0, le=1)
    deepfake: float | None = Field(None, ge=0, le=1)


class SightengineResponse(BaseModel):
    status: str = "success"
    request: RequestBlock
    type: TypeBlock
    media: MediaBlock

    @classmethod
    def build(
        cls,
        *,
        request_id: str,
        timestamp: float,
        operations: int,
        ai_generated: float,
        deepfake: float | None,
        media_id: str,
        media_uri: str | None,
    ) -> "SightengineResponse":
        return cls(
            request=RequestBlock(id=request_id, timestamp=timestamp, operations=operations),
            type=TypeBlock(ai_generated=ai_generated, deepfake=deepfake),
            media=MediaBlock(id=media_id, uri=media_uri),
        )


class SightengineError(BaseModel):
    """Sightengine signals failure with status=failure plus an error block."""

    status: str = "failure"
    request: RequestBlock
    error: dict
