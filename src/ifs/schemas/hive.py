"""Response models for the Hive dialect.

Hive returns two classifier heads whose scores each sum to 1:
  * generation head: ai_generated / not_ai_generated
  * source head: ~80 generator names + none / inconclusive / deepfake

We only have a real signal for the generation head, so the source head reports
`inconclusive` carrying the full generated-probability mass. That is honest and
still parses for clients expecting the shape. A backend that exposes a genuine
per-generator head fills `per_class` and it flows through here instead.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class HiveClass(BaseModel):
    class_: str = Field(..., alias="class")
    score: float

    model_config = {"populate_by_name": True}


class HiveOutput(BaseModel):
    time: float = 0.0
    classes: list[HiveClass]


class HiveInput(BaseModel):
    id: str | None = None
    model: str = "ai-generated-image-detection"
    media: dict = Field(default_factory=dict)


class HiveStatusCode(BaseModel):
    code: str = "0"
    message: str = "SUCCESS"


class HiveResponseBody(BaseModel):
    input: HiveInput
    output: list[HiveOutput]


class HiveStatusItem(BaseModel):
    status: HiveStatusCode = Field(default_factory=HiveStatusCode)
    response: HiveResponseBody


class HiveTaskResponse(BaseModel):
    id: str
    code: int = 200
    project_id: int = 0
    status: list[HiveStatusItem]

    @classmethod
    def build(
        cls,
        *,
        task_id: str,
        ai_generated: float,
        per_class: dict[str, float],
        deepfake: float | None,
        media_uri: str | None,
        elapsed_s: float,
    ) -> "HiveTaskResponse":
        generation = [
            HiveClass(class_="ai_generated", score=ai_generated),
            HiveClass(class_="not_ai_generated", score=round(1.0 - ai_generated, 6)),
        ]

        if per_class:
            total = sum(per_class.values())
            # Hive's source head sums to 1; normalize whatever the backend gave
            # us and park the remainder on `none`.
            source = [HiveClass(class_=k, score=v) for k, v in sorted(per_class.items())]
            source.append(HiveClass(class_="none", score=round(max(0.0, 1.0 - total), 6)))
        else:
            source = [
                HiveClass(class_="inconclusive", score=ai_generated),
                HiveClass(class_="none", score=round(1.0 - ai_generated, 6)),
            ]

        if deepfake is not None:
            source.append(HiveClass(class_="deepfake", score=deepfake))

        return cls(
            id=task_id,
            status=[
                HiveStatusItem(
                    response=HiveResponseBody(
                        input=HiveInput(id=task_id, media={"url": media_uri} if media_uri else {}),
                        output=[HiveOutput(time=elapsed_s, classes=generation + source)],
                    )
                )
            ],
        )
