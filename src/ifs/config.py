"""Configuration.

`config.yaml` is authoritative; environment variables override individual keys
for container deployment. Both are validated at startup and a bad configuration
is a hard error -- nothing is guessed, defaulted around, or substituted.

Env override uses a double underscore for nesting:

    IFS_DETECTOR__MODEL=commfor-224
    IFS_SCORING__CROP_POLICY=center
    IFS_AUTH__API_KEY=...
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    NoDecode,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from .media import views as view_types

log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "config.yaml"


class _Section(BaseModel):
    """Base for every config section.

    `extra="forbid"` must be on each section, not only on Settings: without it a
    typo nested inside a section (`scoring.crop_polcy`) is silently discarded and
    the service runs with a default nobody chose.
    """

    model_config = ConfigDict(extra="forbid")


class DetectorConfig(_Section):
    """Which backend, and which checkpoint it loads.

    Nothing here is inferred. `model` names an entry in the backend's catalog;
    `path` points at a file and then `variant` must say what architecture it is.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    backend: str = "community-forensics"
    model: str | None = None
    path: str | None = None
    variant: str | None = None
    revision: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _coherent_source(self) -> DetectorConfig:
        """Only the rules that hold for every backend.

        Whether a checkpoint is needed *at all* is the backend's business: a
        remote runtime (Triton) names its model on the server and takes neither
        `model` nor `path`, while the stub takes nothing. Each backend raises
        its own clear error at startup, so demanding a local checkpoint here
        would reject valid configurations.
        """
        if self.model and self.path:
            raise ValueError(
                "detector: set at most one of `model` or `path`, not both "
                f"(model={self.model!r}, path={self.path!r}). Note that an "
                "IFS_DETECTOR__* variable merges into the config file's detector "
                "section rather than replacing it, so setting IFS_DETECTOR__MODEL "
                "while the file sets `path` produces both."
            )
        if self.path and not self.variant:
            raise ValueError(
                "detector: `path` requires `variant` -- a checkpoint file does not state "
                "which architecture it is"
            )
        return self


class WeightsConfig(_Section):
    source: Literal["auto", "local", "hub"] = "auto"
    """`local` and `hub` restrict to one source and raise if it cannot serve the
    configured checkpoint. `auto` checks the local directory first, then the hub
    -- a lookup order for the *same* file, never a substitution."""

    dir: str | None = None
    allow_download: bool = True

    @model_validator(mode="after")
    def _local_needs_dir(self) -> WeightsConfig:
        if self.source == "local" and not self.dir:
            raise ValueError("weights: source 'local' requires `dir`")
        return self


class RuntimeConfig(_Section):
    device: str = "auto"  # auto | cpu | cuda | cuda:0 | mps
    dtype: str = "auto"   # auto | float32 | float16 | bfloat16


class ScoringConfig(_Section):
    crop_policy: Literal["center", "five", "tiled"] = "five"
    crop_aggregation: Literal["max", "mean"] = "max"
    crop_size_fallback: int = 384
    score_decimals: int = 4


class ServerConfig(_Section):
    host: str = "0.0.0.0"
    port: int = 8000


#: The services this deployment can serve. Each is mounted, keyed and staked
#: independently, and they differ in what they *need*:
#:
#:   detect   an AI-generation model, loaded in-process or served by Triton
#:   screen   nothing at all
#:   inspect  an OCR service and a prompt-injection classifier, over HTTP
Endpoint = Literal["detect", "screen", "inspect"]
ENDPOINTS: tuple[Endpoint, ...] = ("detect", "screen", "inspect")


class AuthConfig(_Section):
    """Only the header name is global. The keys live on each endpoint."""

    api_key_header: str = "X-API-Key"


class LimitsConfig(_Section):
    max_image_bytes: int = 15 * 1024 * 1024
    max_pixels: int = 50_000_000
    allow_url_fetch: bool = False
    """Fetch caller-supplied image URLs.

    **Off by default.** Fetching a URL makes this node issue requests from its
    own network position, which is server-side request forgery unless guarded;
    it also costs egress bandwidth and lets answers differ between suppliers
    when the content behind a URL changes. Callers can always send the image
    itself via `media` or `media_base64`.
    """

    url_fetch_timeout_s: float = 10.0

    allow_private_urls: bool = False
    """Disable the SSRF guard. Development only.

    With this on, a caller can make the node reach loopback, RFC1918,
    link-local and cloud-metadata addresses. Never enable it on a host that
    can see anything worth reaching.
    """


class RescaleConfig(_Section):
    """The downscales to replay.

    These are the shapes a victim's preprocessing produces. An attack is built
    against a specific size and resampler, so what is not listed here is not
    looked for.
    """

    sizes: list[int] = Field(default_factory=lambda: [224, 336, 384, 512, 768])
    filters: list[view_types.Resample] = Field(
        default_factory=lambda: ["bicubic", "bilinear", "area"]
    )

    @model_validator(mode="after")
    def _non_empty(self) -> RescaleConfig:
        if not self.sizes or not self.filters:
            raise ValueError(
                "inspect.views.rescale: `sizes` and `filters` must both be non-empty; "
                "to disable rescaling, set one of them to [] is not supported -- "
                "remove sizes you do not want instead"
            )
        if any(s < 16 for s in self.sizes):
            raise ValueError("inspect.views.rescale.sizes: every size must be at least 16")
        return self


class ViewsConfig(_Section):
    """What this service looks at. Forensic policy, and part of the fingerprint."""

    normalize: list[view_types.Normalize] = Field(
        default_factory=lambda: ["alpha_black", "alpha_white", "contrast_stretch", "lsb"]
    )
    rescale: RescaleConfig = Field(default_factory=RescaleConfig)
    max_views: int = 32
    """Hard ceiling, so a long `sizes` list cannot turn one image into hundreds
    of OCR calls. Views past it are dropped, with a warning."""


class _UpstreamConfig(_Section):
    """A service this one composes.

    `url` absent means the hop is not configured and is reported as `disabled`
    -- `required` then has nothing to govern. These URLs are operator
    configuration and are never caller-supplied, which is why they are exempt
    from the SSRF guard that `limits.allow_url_fetch` turns on.
    """

    url: str | None = None
    api_key: str | None = None
    timeout_s: float = 10.0
    required: bool = True
    """On failure: raise (502) rather than answer. Default on, because a report
    that silently skipped a check reads exactly like a clean one."""


class OcrUpstreamConfig(_UpstreamConfig):
    """`image-inference-server`, speaking Google Cloud Vision's images:annotate."""

    batch: int = 8
    """Views per request. The upstream caps this too (`max_images_per_request`);
    set 1 if it is configured without batching."""

    max_request_bytes: int = 12 * 1024 * 1024
    """Second bound on a request, in base64 characters. The native view of a
    large photograph can approach the upstream body limit on its own."""

    min_region_score: float = 0.6
    """Detector score floor applied to text regions here, on top of whatever
    floor the upstream already applies."""


class ClassifierUpstreamConfig(_UpstreamConfig):
    """`language-forensics-service`, on its native /v1/detect."""

    timeout_s: float = 5.0
    threshold: float | None = None
    """Override the classifier's own flagging threshold. None = its default."""

    max_text_chars: int = 20000
    skip_when_no_text: bool = True
    """Most images carry no text. Skipping the call then keeps the common case
    to one local pass plus one OCR call."""


class RegionsConfig(_Section):
    """When text *boxes* alone are evidence.

    The screening pass never reads the text, so everything it concludes comes
    from where boxes appear and where they do not.
    """

    min_regions: int = 1
    """Boxes a derived view needs before its disagreement with the native view
    counts. 1 is right for the attacks this looks for -- one line of
    instructions is one box."""

    min_area_fraction: float = 0.0005
    """...and how much of the frame those boxes must cover. Filters a stray
    box on a noisy rescale without filtering a single line of text."""


class _EndpointConfig(_Section):
    """What every endpoint has: whether it is mounted, and who may call it.

    `api_keys` is a list so a key can be rotated without downtime -- publish
    the new one, let callers move, drop the old. An empty list leaves the
    endpoint OPEN; that is per endpoint, not per deployment, and startup says
    so in as many words.
    """

    enabled: bool = False

    api_keys: Annotated[list[str], NoDecode] = Field(default_factory=list)

    @field_validator("api_keys", mode="before")
    @classmethod
    def _split(cls, value: object) -> object:
        """Accept a comma-separated string as well as a list.

        `NoDecode` turns off pydantic-settings' JSON parsing for this field so
        that `IFS_ENDPOINTS__DETECT__API_KEYS=k1,k2` works from a compose file
        or a systemd unit, where writing a JSON array is awkward and an
        unquoted empty value would be a parse error rather than "no keys".
        Blank entries are dropped, so an unset variable means open, not a key
        that is the empty string.
        """
        if isinstance(value, str):
            text = value.strip()
            # A JSON array is what pydantic-settings would have accepted, so
            # keep taking it -- splitting `["a","b"]` on commas would silently
            # produce two keys made of punctuation.
            if text.startswith("["):
                try:
                    value = json.loads(text)
                except ValueError as exc:
                    raise ValueError(
                        "api_keys: looks like a JSON array but does not parse; "
                        "use a JSON array or a comma-separated list"
                    ) from exc
            else:
                return [part.strip() for part in text.split(",") if part.strip()]
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        return value


class _ReportingConfig(_EndpointConfig):
    """Shared by the two endpoints that return findings rather than a score."""

    flag_threshold: float = 0.5
    score_decimals: int = 4

    include_text: bool = False
    """Quote offending text in findings.

    Off by default: two suppliers running different OCR builds return
    different strings for the same image, so excerpts are the least
    reproducible part of a report. Useful for a human triaging a hit; noise in
    anything comparing answers across suppliers.
    """


class CompatConfig(_Section):
    """The vendor-shaped routes. They are the detection service in someone
    else's envelope, so they are mounted with it and take its keys."""

    sightengine: bool = True
    hive: bool = True


class DetectEndpoint(_EndpointConfig):
    """AI-generated image detection: POST /v1/detect.

    The only endpoint that needs a model of its own. What model, and how it is
    loaded and scored, is the `detector`, `weights`, `runtime` and `scoring`
    sections -- this repository's original subject, and the part no sibling
    project provides.
    """

    enabled: bool = True
    compat: CompatConfig = Field(default_factory=CompatConfig)


class ScreenEndpoint(_ReportingConfig):
    """Container screening: POST /v1/screen.

    Needs nothing. No model, no sibling service, no GPU -- it reads the file,
    not the picture. A supplier can serve this from a small box.
    """


class InspectEndpoint(_ReportingConfig):
    """Prompt-injection inspection: POST /v1/inspect.

    Needs both sibling services: an OCR service for pixels-to-text, and a
    prompt-injection classifier for text-to-verdict. Loads no model itself.
    """

    views: ViewsConfig = Field(default_factory=ViewsConfig)
    regions: RegionsConfig = Field(default_factory=RegionsConfig)
    ocr: OcrUpstreamConfig = Field(default_factory=OcrUpstreamConfig)
    classifier: ClassifierUpstreamConfig = Field(default_factory=ClassifierUpstreamConfig)

    @model_validator(mode="after")
    def _warn_on_inert_upstreams(self) -> InspectEndpoint:
        if self.enabled and self.ocr.url is None and self.classifier.url is not None:
            log.warning(
                "endpoints.inspect.classifier.url is set but ocr.url is not: the "
                "classifier will only ever see container metadata, never text rendered "
                "into pixels, which is where these payloads usually are. For metadata "
                "alone, endpoints.screen needs no upstream at all."
            )
        return self


class EndpointsConfig(_Section):
    """The services this deployment serves. Anything not enabled answers 404."""

    detect: DetectEndpoint = Field(default_factory=DetectEndpoint)
    screen: ScreenEndpoint = Field(default_factory=ScreenEndpoint)
    inspect: InspectEndpoint = Field(default_factory=InspectEndpoint)

    def get(self, endpoint: Endpoint) -> _EndpointConfig:
        return getattr(self, endpoint)

    def live(self) -> tuple[Endpoint, ...]:
        """Endpoints actually mounted.

        An unmounted route cannot be reached, so it is neither a hole to warn
        about nor an entitlement to grant.
        """
        return tuple(name for name in ENDPOINTS if self.get(name).enabled)

    @model_validator(mode="after")
    def _at_least_one(self) -> EndpointsConfig:
        if not self.live():
            raise ValueError(
                "endpoints: every endpoint is disabled, so this deployment would "
                "serve nothing but its health probes. Enable at least one of: "
                + ", ".join(ENDPOINTS)
            )
        return self


class _YamlSource(PydanticBaseSettingsSource):
    """Reads the YAML file named by IFS_CONFIG_FILE, or ./config.yaml.

    Reads are cached by (path, mtime): pydantic-settings invokes a source more
    than once per Settings construction, and an operator reading startup logs
    should see the config line once, not once per internal pass.
    """

    _cache: ClassVar[dict[tuple[str, float], dict[str, Any]]] = {}

    def get_field_value(self, field, field_name):  # pragma: no cover - unused hook
        raise NotImplementedError

    def __call__(self) -> dict[str, Any]:
        named = os.environ.get("IFS_CONFIG_FILE")
        path = Path(named or DEFAULT_CONFIG_PATH)
        if not path.is_file():
            if named:
                # An explicitly named file that is missing is an error, not a
                # reason to fall back to defaults.
                raise FileNotFoundError(f"IFS_CONFIG_FILE={path} does not exist")
            # No config file. Say so at WARNING with the absolute path searched:
            # a silently-missing config used to mean "load whatever the defaults
            # name", which downloaded a model nobody asked for.
            log.warning(
                "no config file at %s -- configuration comes from environment "
                "variables and defaults only",
                path.resolve(),
            )
            return {}
        key = (str(path.resolve()), path.stat().st_mtime)
        if key in self._cache:
            return self._cache[key]

        with path.open() as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            # ValueError, not TypeError: this is a configuration fault that the
            # caller surfaces alongside pydantic validation errors.
            raise ValueError(  # noqa: TRY004
                f"{path}: expected a mapping at the top level"
            )
        log.info("configuration loaded from %s", path.resolve())
        self._cache[key] = data
        return data


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="IFS_",
        env_nested_delimiter="__",
        extra="forbid",  # a typo in config.yaml is an error, not a silent no-op
        protected_namespaces=(),
    )

    endpoints: EndpointsConfig = Field(default_factory=EndpointsConfig)
    """Which services this deployment serves."""

    detector: DetectorConfig | None = None
    """Required when `endpoints.detect` is on, and meaningless otherwise.

    There is deliberately no default: a detection deployment whose config is
    absent or unreadable must fail loudly, not silently resolve to some model
    and fetch it. A screening-only deployment needs no model at all, so it
    needs no `detector` section either."""

    weights: WeightsConfig = Field(default_factory=WeightsConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ):
        # Precedence: constructor > environment > config.yaml > defaults.
        return (init_settings, env_settings, _YamlSource(settings_cls), file_secret_settings)

    @model_validator(mode="after")
    def _warn_on_inert_sections(self) -> Settings:
        """Say so when a section cannot affect the chosen backend.

        A remote runtime does no local inference, so `weights` and `runtime`
        are dead config there. Silently ignoring them makes a misconfiguration
        look like it was applied.
        """
        if self.detector and self.detector.backend.endswith("-triton"):
            inert = []
            if self.weights.dir is not None or self.weights.source != "auto":
                inert.append("weights")
            if self.runtime.device != "auto" or self.runtime.dtype != "auto":
                inert.append("runtime")
            if inert:
                log.warning(
                    "backend %r runs inference on a remote Triton server, so the "
                    "%s section(s) have no effect here -- device, dtype and weights "
                    "are decided by Triton.",
                    self.detector.backend,
                    " and ".join(inert),
                )
        return self

    @model_validator(mode="after")
    def _warn_on_ignored_weights(self) -> Settings:
        if self.detector and self.detector.path and (
            self.weights.dir is not None or self.weights.source != "auto"
        ):
            log.warning(
                "detector.path is set, so the `weights` section is ignored: the file at "
                "%s is loaded directly, with no catalog lookup and no download.",
                self.detector.path,
            )
        return self

    @model_validator(mode="after")
    def _detect_needs_a_model(self) -> Settings:
        """The one cross-section rule: detection needs something to detect with.

        Screening needs no model and inspection borrows two from sibling
        services, so `detector` is required only for this one endpoint -- and
        is required *loudly*, rather than defaulted into fetching something
        nobody chose.
        """
        if self.endpoints.detect.enabled and self.detector is None:
            raise ValueError(
                "endpoints.detect is enabled but there is no `detector` section. "
                "Detection is the only endpoint that needs a model of its own; "
                "name one, or set endpoints.detect.enabled: false."
            )
        return self

    def detector_kwargs(self) -> dict[str, Any]:
        """Everything the backend factory needs, assembled in one place."""
        if self.detector is None:
            raise RuntimeError("no detector configured")
        return {
            "model": self.detector.model,
            "path": self.detector.path,
            "variant": self.detector.variant,
            "revision": self.detector.revision,
            "weights_source": self.weights.source,
            "weights_dir": self.weights.dir,
            "allow_download": self.weights.allow_download,
            "device": self.runtime.device,
            "dtype": self.runtime.dtype,
            **self.detector.options,
        }


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Drop cached settings and the parsed config file. For tests and reloads."""
    global _settings
    _settings = None
    _YamlSource._cache.clear()
