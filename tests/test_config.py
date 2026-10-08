"""Configuration is authoritative: a wrong config is an error, never a fallback."""

import os
import pathlib

import pytest
import yaml
from pydantic import ValidationError

from ifs.config import DetectorConfig, Settings, WeightsConfig


def _settings(tmp_path, monkeypatch, doc: dict, env: dict | None = None) -> Settings:
    # Start from a clean slate so the repo's own config.yaml and any ambient
    # IFS_* variables cannot leak into a config test.
    for var in [v for v in os.environ if v.startswith("IFS_")]:
        monkeypatch.delenv(var, raising=False)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(doc))
    monkeypatch.setenv("IFS_CONFIG_FILE", str(path))
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    return Settings()


# -- detector selection ---------------------------------------------------


def test_model_and_path_together_is_an_error():
    with pytest.raises(ValidationError, match="exactly one"):
        DetectorConfig(model="commfor-384", path="/w/x.safetensors")


def test_neither_model_nor_path_is_an_error():
    with pytest.raises(ValidationError, match="either `model`"):
        DetectorConfig()


def test_path_without_variant_is_an_error():
    """A checkpoint file does not state which architecture it is."""
    with pytest.raises(ValidationError, match="requires `variant`"):
        DetectorConfig(path="/w/x.safetensors")


def test_path_with_variant_is_accepted():
    cfg = DetectorConfig(path="/w/x.safetensors", variant="vit-s16-384")
    assert cfg.model is None and cfg.variant == "vit-s16-384"


# -- catalog --------------------------------------------------------------


def test_unknown_model_names_the_known_ones():
    from ifs.detectors.backends.community_forensics.catalog import CatalogError, resolve_model

    with pytest.raises(CatalogError) as exc:
        resolve_model("commfor-512")
    assert "commfor-384" in str(exc.value)


def test_variant_contradicting_the_model_is_an_error():
    from ifs.detectors.backends.community_forensics.catalog import CatalogError
    from ifs.detectors.backends.community_forensics.detector import _select

    with pytest.raises(CatalogError, match="contradicts"):
        _select(model="commfor-384", path=None, variant="vit-s16-224", revision=None)


def test_catalog_models_are_pinned_to_a_revision():
    """A published model is never 'whatever main holds today'."""
    from ifs.detectors.backends.community_forensics.catalog import MODELS, VARIANTS

    for name, entry in MODELS.items():
        assert len(entry.revision) == 40, f"{name} is not pinned to a commit sha"
        assert entry.variant in VARIANTS


# -- weights --------------------------------------------------------------


def test_local_source_requires_a_dir():
    with pytest.raises(ValidationError, match="requires `dir`"):
        WeightsConfig(source="local")


def test_explicit_path_never_falls_back(tmp_path):
    from ifs.detectors.weights import WeightsError, resolve_path

    with pytest.raises(WeightsError, match="no download or catalog lookup"):
        resolve_path(tmp_path / "absent.safetensors")


def test_local_source_does_not_reach_the_hub(tmp_path):
    from ifs.detectors.weights import WeightsError, resolve_catalog

    with pytest.raises(WeightsError, match="does not exist"):
        resolve_catalog(
            repo_id="Org/Repo",
            filename="model.safetensors",
            revision="a" * 40,
            source="local",
            weights_dir=str(tmp_path),
        )


def test_download_disabled_is_an_error_not_a_fallback(tmp_path):
    from ifs.detectors.weights import WeightsError, resolve_catalog

    with pytest.raises(WeightsError, match="allow_download is false"):
        resolve_catalog(
            repo_id="Org/Repo",
            filename="model.safetensors",
            revision="a" * 40,
            source="auto",
            weights_dir=str(tmp_path),
            allow_download=False,
        )


# -- file loading and overrides -------------------------------------------


def test_yaml_is_read(tmp_path, monkeypatch):
    s = _settings(tmp_path, monkeypatch, {"detector": {"backend": "stub", "model": "none"},
                                          "scoring": {"crop_policy": "center"}})
    assert s.detector.backend == "stub"
    assert s.scoring.crop_policy == "center"


def test_env_overrides_yaml(tmp_path, monkeypatch):
    s = _settings(
        tmp_path,
        monkeypatch,
        {"detector": {"backend": "stub", "model": "none"}, "scoring": {"crop_policy": "center"}},
        env={"IFS_SCORING__CROP_POLICY": "tiled"},
    )
    assert s.scoring.crop_policy == "tiled"


def test_unknown_key_is_rejected(tmp_path, monkeypatch):
    """A typo must not be a silent no-op."""
    with pytest.raises(ValidationError):
        _settings(tmp_path, monkeypatch, {"detector": {"backend": "stub", "model": "none"},
                                          "scoring": {"crop_polcy": "center"}})


def test_missing_named_config_file_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("IFS_CONFIG_FILE", str(tmp_path / "nope.yaml"))
    with pytest.raises(FileNotFoundError):
        Settings()


def test_local_config_yaml_is_valid_if_present(monkeypatch):
    """Sanity-check the operator's own config.yaml when there is one.

    `config.yaml` is gitignored -- it holds local paths and the deployment key
    -- so it does not exist on a clean checkout or a CI runner. Skip there
    rather than fail: `config.example.yaml` is the tracked file CI must guard.
    """
    if not pathlib.Path("config.yaml").is_file():
        pytest.skip("no local config.yaml (expected on a clean checkout)")
    for var in [v for v in os.environ if v.startswith("IFS_")]:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("IFS_CONFIG_FILE", "config.yaml")
    s = Settings()
    assert s.detector.backend
    assert bool(s.detector.model) ^ bool(s.detector.path)


def test_example_config_is_valid(monkeypatch):
    """The tracked reference config must parse -- it is what operators copy."""
    for var in [v for v in os.environ if v.startswith("IFS_")]:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("IFS_CONFIG_FILE", "config.example.yaml")
    s = Settings()
    assert s.detector.backend == "community-forensics"
    assert bool(s.detector.model) ^ bool(s.detector.path)


def test_example_config_ships_the_safe_defaults(monkeypatch):
    """The file people copy must not hand them a URL-fetching deployment."""
    for var in [v for v in os.environ if v.startswith("IFS_")]:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("IFS_CONFIG_FILE", "config.example.yaml")
    s = Settings()
    assert s.limits.allow_url_fetch is False, "example config enables SSRF-prone URL fetching"
    assert s.limits.allow_private_urls is False, "example config disables the SSRF guard"


# -- the absent-config trap ----------------------------------------------


def test_absent_config_is_an_error_not_a_default_model(tmp_path, monkeypatch):
    """A missing config.yaml used to resolve to `model: commfor-384` and download it.

    Absent configuration must fail, not pick a model nobody asked for.
    """
    monkeypatch.delenv("IFS_CONFIG_FILE", raising=False)
    for var in [v for v in os.environ if v.startswith("IFS_")]:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)  # no config.yaml here
    with pytest.raises(ValidationError, match="detector"):
        Settings()


def test_path_config_never_consults_the_catalog_or_hub(tmp_path, monkeypatch):
    """With `path`, resolution must not touch huggingface_hub at all."""
    import huggingface_hub

    ckpt = tmp_path / "w.safetensors"
    ckpt.write_bytes(b"")
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda *a, **k: pytest.fail("hf_hub_download called despite detector.path"),
    )
    from ifs.detectors.weights import resolve_path

    assert resolve_path(ckpt).source == "path"


def test_spec_generation_is_independent_of_local_config(tmp_path, monkeypatch):
    """docs/openapi.json must not vary with the operator's config.yaml.

    Routes and the security-scheme name are configurable, so generating the
    spec from a local file could silently drop documented routes. The exporter
    pins a canonical config; this asserts it stays pinned.
    """
    import subprocess
    import sys

    narrow = tmp_path / "config.yaml"
    narrow.write_text(
        "detector:\n  backend: stub\n  model: s\n"
        "api:\n  enable_hive_dialect: false\n  enable_sightengine_dialect: false\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("IFS_")}
    env["IFS_CONFIG_FILE"] = str(narrow)

    spec = pathlib.Path("docs/openapi.json")
    before = spec.read_text()
    subprocess.run(
        [sys.executable, "scripts/export_openapi.py"], env=env, check=True, capture_output=True
    )
    assert spec.read_text() == before, "spec changed when generated under a narrower config"
