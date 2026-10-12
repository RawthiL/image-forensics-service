"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

from .api import health, hive, inspect, native, sightengine
from .api.auth import AuthError, require_api_key, warn_if_open
from .config import get_settings
from .inspection.errors import UpstreamError
from .inspection.service import InspectionService
from .media.loader import MediaError
from .service import DetectionService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("ifs")

DESCRIPTION = """
Three image-forensics services in one deployment. Each is mounted, keyed and
staked independently, and they differ in what an operator must supply:

**`POST /v1/detect` — AI-generated image detection.** Needs a model. Which one,
how it is loaded and how its score is composed is this project's own subject:
backends are folders under `detectors/backends/`, served in-process or by
Triton, and the response carries the model and pipeline fingerprints a caller
needs to reason about reproducibility. Two vendor-shaped compatibility layers
ride along with it.

**`POST /v1/screen` — container screening.** Needs nothing at all. It reads the
file but never the picture: metadata text, smuggled Unicode, and bytes past the
end of the image data. No model, no sibling service, no GPU.

**`POST /v1/inspect` — prompt-injection inspection.** Needs two sibling
services over HTTP: an OCR service for the text boxes and their contents, and a
prompt-injection classifier for the verdict. It loads no model of its own. It
renders the views a victim's preprocessing would produce, so it can see text
that exists only after a downscale — and runs the container checks too.

**The vendor routes are compatibility layers** on the detection service. They
exist so an application already written against a commercial detection API can
point at this service by changing its base URL and nothing else. They reshape
the same inference result into someone else's envelope, and they inherit that
envelope's limits:

* `POST|GET /1.0/check.json` -- [Sightengine][se]-shaped. The better-supported of the two.
* `POST /api/v2/task/sync` -- [Hive][hive]-shaped. Its per-generator head is approximated.

[se]: https://sightengine.com/docs/ai-generated-image-detection
[hive]: https://docs.thehive.ai/docs/ai-image-and-video-detection

Vendor API keys on the compatibility routes are accepted and ignored: swapping
the base URL is all a client should have to do.

Neither vendor publishes an OpenAPI specification. This one is authoritative for
this service only; the compatibility routes are documented here as a convenience
and are described normatively by `docs/API_COMPAT.md`.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    warn_if_open(settings)
    live = settings.endpoints.live()

    service = DetectionService(settings)
    app.state.service = service
    inspector = InspectionService(settings)
    app.state.inspector = inspector

    # A model is loaded only by the endpoint that needs one. A screening-only
    # deployment downloads nothing, warms nothing and needs no GPU -- which is
    # the whole reason that endpoint exists.
    if "detect" in live:
        await service.start()
        log.info(
            "detect ready: pipeline=%s weights=%s",
            service.pipeline_fingerprint,
            settings.detector.path or f"catalog:{settings.detector.model}",
        )
    if "inspect" in live:
        await inspector.start()
    log.info("serving: %s", ", ".join(live))

    yield

    await inspector.stop()
    await service.stop()


TAGS_METADATA = [
    {
        "name": "detection",
        "description": "**AI-generated image detection.** Needs a model.",
    },
    {
        "name": "screening",
        "description": (
            "**Container screening.** Reads the file, never the picture. "
            "Needs no model and no sibling service."
        ),
    },
    {
        "name": "inspection",
        "description": (
            "**Prompt-injection inspection.** Needs an OCR service and a "
            "prompt-injection classifier; loads no model of its own."
        ),
    },
    {
        "name": "ops",
        "description": "Health, readiness, and what this deployment serves.",
    },
    {
        "name": "compat: sightengine",
        "description": (
            "Compatibility layer on the detection service. Mimics "
            "`api.sightengine.com/1.0/check.json?models=genai`. Not a specification "
            "of this service -- see `docs/API_COMPAT.md` for fidelity."
        ),
    },
    {
        "name": "compat: hive",
        "description": (
            "Compatibility layer on the detection service. Mimics "
            "`api.thehive.ai/api/v2/task/sync`. Not a specification of this "
            "service -- see `docs/API_COMPAT.md` for fidelity."
        ),
    },
]


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Image Forensics Service",
        version="0.1.0",
        description=DESCRIPTION,
        openapi_tags=TAGS_METADATA,
        lifespan=lifespan,
    )

    # Liveness and readiness stay open: a container orchestrator's probe has
    # no business holding the service key, and neither route reveals anything.
    app.include_router(health.router)

    # One dependency per endpoint, fixed at mount time. Anything not enabled
    # is never mounted, and answers 404 rather than 401 -- a route this
    # deployment does not serve is absent, not forbidden.
    endpoints = settings.endpoints
    if endpoints.detect.enabled:
        guard = [Depends(require_api_key("detect"))]
        app.include_router(native.router, dependencies=guard)
        # The compatibility layers are the detection service in a different
        # envelope, so they are mounted with it and take its keys.
        if endpoints.detect.compat.sightengine:
            app.include_router(sightengine.router, dependencies=guard)
        if endpoints.detect.compat.hive:
            app.include_router(hive.router, dependencies=guard)
    if endpoints.screen.enabled:
        app.include_router(
            inspect.screen_router, dependencies=[Depends(require_api_key("screen"))]
        )
    if endpoints.inspect.enabled:
        app.include_router(inspect.router, dependencies=[Depends(require_api_key("inspect"))])

    @app.exception_handler(AuthError)
    async def _auth_error(request: Request, exc: AuthError) -> JSONResponse:
        return JSONResponse(
            status_code=401,
            content={"status": "failure", "error": {"type": exc.code, "message": str(exc)}},
            headers={"WWW-Authenticate": "Bearer"},
        )

    @app.exception_handler(UpstreamError)
    async def _upstream_error(request: Request, exc: UpstreamError) -> JSONResponse:
        # 502, not 500: this service is fine, a service it composes is not.
        # The body names which hop failed so an operator is not left guessing.
        return JSONResponse(
            status_code=502,
            content={
                "status": "failure",
                "error": {
                    "type": "upstream_unavailable",
                    "service": exc.service,
                    "message": str(exc),
                },
            },
        )

    @app.exception_handler(MediaError)
    async def _media_error(request: Request, exc: MediaError) -> JSONResponse:
        status = 413 if exc.code == "media_too_large" else 400
        return JSONResponse(
            status_code=status,
            content={"status": "failure", "error": {"type": exc.code, "message": str(exc)}},
        )

    _document_security(app, settings)
    return app


# Routes are guarded by a plain dependency rather than fastapi.security, because
# the key may arrive in either of two headers. That keeps runtime behavior
# dynamic but leaves the scheme out of the generated spec, so we add it here --
# otherwise /docs has no Authorize button and the contract is not machine-readable.
_UNGUARDED = {("/healthz", "get"), ("/readyz", "get")}


def _document_security(app: FastAPI, settings) -> None:
    def openapi() -> dict:
        if app.openapi_schema:
            return app.openapi_schema

        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
            tags=app.openapi_tags,
        )
        schema["components"] = schema.get("components", {})
        schema["components"]["securitySchemes"] = {
            "ServiceApiKey": {
                "type": "apiKey",
                "in": "header",
                "name": settings.auth.api_key_header,
                "description": (
                    "Key gating one endpoint. This is a deployment boundary, not "
                    "user authentication -- it carries no caller identity. "
                    "Distinct from the vendor keys on the compatibility routes, "
                    "which are accepted and ignored.\n\n"
                    "Keys are listed per endpoint under "
                    "`endpoints.<name>.api_keys`, and a key opens only the "
                    "endpoints that list it. An empty list leaves that endpoint "
                    "open. `/v1/models` accepts any served endpoint's key and "
                    "answers with the sections that key is entitled to."
                ),
            },
            "ServiceBearer": {
                "type": "http",
                "scheme": "bearer",
                "description": (
                    "The same key as ServiceApiKey, sent as "
                    "`Authorization: Bearer <key>`. Only the Bearer scheme is read: "
                    "`Authorization: token ...` on a compatibility route is a vendor "
                    "credential and stays ignored."
                ),
            },
        }
        requirement = [{"ServiceApiKey": []}, {"ServiceBearer": []}]
        for path, operations in schema["paths"].items():
            for method, operation in operations.items():
                if (path, method) in _UNGUARDED:
                    continue
                operation["security"] = requirement
                operation.setdefault("responses", {})["401"] = {
                    "description": "Missing or invalid service credentials"
                }

        app.openapi_schema = schema
        return schema

    app.openapi = openapi  # type: ignore[method-assign]


app = create_app()
