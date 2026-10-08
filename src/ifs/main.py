"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from fastapi.openapi.utils import get_openapi

from fastapi import Depends

from .api import health, hive, native, sightengine
from .api.auth import AuthError, require_api_key, warn_if_open
from .config import get_settings
from .media.loader import MediaError
from .service import DetectionService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("ifs")

DESCRIPTION = """
AI-generated image detection.

**`POST /v1/detect` is this service's API.** It is the surface this project
specifies, documents and versions, and the one to write new integrations
against. It is the only route without vendor baggage, and the only one that
returns the model and pipeline fingerprints a caller needs to reason about
reproducibility.

**The vendor routes are compatibility layers.** They exist so an application
already written against a commercial detection API can point at this service by
changing its base URL and nothing else. They reshape the same inference result
into someone else's envelope, and they inherit that envelope's limits:

* `POST|GET /1.0/check.json` -- [Sightengine](https://sightengine.com/docs/ai-generated-image-detection)-shaped. The better-supported of the two.
* `POST /api/v2/task/sync` -- [Hive](https://docs.thehive.ai/docs/ai-image-and-video-detection)-shaped. Its per-generator head is approximated.

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
    service = DetectionService(settings)
    app.state.service = service
    await service.start()
    info = service.info()
    log.info(
        "ready: pipeline=%s weights=%s",
        service.pipeline_fingerprint,
        settings.detector.path or f"catalog:{settings.detector.model}",
    )
    yield
    await service.stop()


TAGS_METADATA = [
    {
        "name": "detection",
        "description": "**Primary API.** Write new integrations against this.",
    },
    {
        "name": "ops",
        "description": "Health, readiness, and the active backend.",
    },
    {
        "name": "compat: sightengine",
        "description": (
            "Compatibility layer. Mimics `api.sightengine.com/1.0/check.json?models=genai`. "
            "Not a specification of this service -- see `docs/API_COMPAT.md` for fidelity."
        ),
    },
    {
        "name": "compat: hive",
        "description": (
            "Compatibility layer. Mimics `api.thehive.ai/api/v2/task/sync`. "
            "Not a specification of this service -- see `docs/API_COMPAT.md` for fidelity."
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

    guarded = [Depends(require_api_key)]
    app.include_router(native.router, dependencies=guarded)
    if settings.api.enable_sightengine_dialect:
        app.include_router(sightengine.router, dependencies=guarded)
    if settings.api.enable_hive_dialect:
        app.include_router(hive.router, dependencies=guarded)

    @app.exception_handler(AuthError)
    async def _auth_error(request: Request, exc: AuthError) -> JSONResponse:
        return JSONResponse(
            status_code=401,
            content={"status": "failure", "error": {"type": exc.code, "message": str(exc)}},
            headers={"WWW-Authenticate": "Bearer"},
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
                    "Single shared key gating the deployment. Set server-side via "
                    "IFS_API_KEY; when unset, every route is open. This is a "
                    "deployment boundary, not user authentication -- it carries no "
                    "caller identity. Distinct from the vendor keys on the "
                    "compatibility routes, which are accepted and ignored."
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
