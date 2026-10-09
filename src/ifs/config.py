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

import logging
import os
from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

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


class AuthConfig(_Section):
    api_key: str | None = None
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


class ApiConfig(_Section):
    enable_sightengine_dialect: bool = True
    enable_hive_dialect: bool = True


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

    detector: DetectorConfig
    """Required. There is deliberately no default: an absent or unreadable
    config must fail loudly, not silently resolve to some model and fetch it."""
    weights: WeightsConfig = Field(default_factory=WeightsConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)

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
        if self.detector.backend.endswith("-triton"):
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
        if self.detector.path and (
            self.weights.dir is not None or self.weights.source != "auto"
        ):
            log.warning(
                "detector.path is set, so the `weights` section is ignored: the file at "
                "%s is loaded directly, with no catalog lookup and no download.",
                self.detector.path,
            )
        return self

    def detector_kwargs(self) -> dict[str, Any]:
        """Everything the backend factory needs, assembled in one place."""
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
