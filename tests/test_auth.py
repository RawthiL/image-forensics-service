"""Service-level authorization: a deployment boundary, not user auth.

Keys are per endpoint, because the endpoints are separately-stakeable Pocket
Network services. A key opens the endpoints that list it and no others.
"""

import io
import os

import pytest
import yaml
from fastapi.testclient import TestClient
from PIL import Image

KEY = "s3cret-relay-miner-key"
DETECT_KEY = "detect-only-key"
SCREEN_KEY = "screen-only-key"
INSPECT_KEY = "inspect-only-key"

ALL_ON = {"detect": {"enabled": True}, "screen": {"enabled": True}, "inspect": {"enabled": True}}


@pytest.fixture
def png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (10, 20, 30)).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    """Build an app from an endpoints block, isolated from the repo's config."""

    def build(endpoints):
        config = {"detector": {"backend": "stub", "model": "stub-v1"}, "endpoints": endpoints}
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(config))
        monkeypatch.setenv("IFS_CONFIG_FILE", str(path))
        for var in [v for v in os.environ if v.startswith("IFS_") and v != "IFS_CONFIG_FILE"]:
            monkeypatch.delenv(var, raising=False)

        from ifs.config import reset_settings
        from ifs.main import create_app

        reset_settings()
        return TestClient(create_app())

    yield build
    from ifs.config import reset_settings

    reset_settings()


def call(client, route, png, key=None):
    return client.post(
        route,
        files={"media": ("a.png", png, "image/png")},
        headers={"X-API-Key": key} if key else {},
    )


# --- the basics --------------------------------------------------------------


def test_open_by_default(client, png_bytes):
    """No keys listed = no gate, so local runs and tests stay frictionless."""
    assert call(client, "/v1/detect", png_bytes).status_code == 200


def test_rejects_missing_and_wrong_keys(deployment, png):
    with deployment({"detect": {"api_keys": [KEY]}}) as c:
        missing = call(c, "/v1/detect", png)
        assert missing.status_code == 401
        assert missing.json()["error"]["type"] == "missing_api_key"
        assert missing.headers["WWW-Authenticate"] == "Bearer"
        assert call(c, "/v1/detect", png, "wrong").json()["error"]["type"] == "invalid_api_key"


@pytest.mark.parametrize(
    "headers",
    [{"X-API-Key": KEY}, {"Authorization": f"Bearer {KEY}"}, {"authorization": f"bearer {KEY}"}],
)
def test_accepts_both_header_forms(deployment, png, headers):
    with deployment({"detect": {"api_keys": [KEY]}}) as c:
        files = {"media": ("a.png", png, "image/png")}
        assert c.post("/v1/detect", files=files, headers=headers).status_code == 200


def test_every_listed_key_works(deployment, png):
    """A list, so a key can be rotated without downtime: publish the new one,
    let callers move, drop the old."""
    with deployment({"detect": {"api_keys": ["old-key", "new-key"]}}) as c:
        assert call(c, "/v1/detect", png, "old-key").status_code == 200
        assert call(c, "/v1/detect", png, "new-key").status_code == 200
        assert call(c, "/v1/detect", png, "retired-key").status_code == 401


def test_compat_layers_follow_the_detection_keys(deployment, png):
    with deployment({"detect": {"api_keys": [KEY]}}) as c:
        files = {"media": ("a.png", png, "image/png")}
        for path, data in (("/1.0/check.json", {"models": "genai"}), ("/api/v2/task/sync", {})):
            assert c.post(path, files=files, data=data).status_code == 401
            assert (
                c.post(path, files=files, data=data, headers={"X-API-Key": KEY}).status_code == 200
            )


def test_vendor_key_does_not_satisfy_the_service_key(deployment, png):
    """`Authorization: token ...` is a Hive client's credential, not ours."""
    with deployment({"detect": {"api_keys": [KEY]}}) as c:
        assert (
            c.post(
                "/api/v2/task/sync",
                files={"media": ("a.png", png, "image/png")},
                headers={"Authorization": f"token {KEY}"},
            ).status_code
            == 401
        )


def test_probes_stay_open(deployment):
    """An orchestrator's liveness check has no business holding a key."""
    with deployment({"detect": {"api_keys": [KEY]}}) as c:
        assert c.get("/healthz").status_code == 200
        assert c.get("/readyz").status_code == 200
        assert c.get("/v1/models").status_code == 401


def test_custom_header_name(tmp_path, monkeypatch, png):
    config = {
        "detector": {"backend": "stub", "model": "stub-v1"},
        "endpoints": {"detect": {"api_keys": [KEY]}},
        "auth": {"api_key_header": "X-Relay-Token"},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setenv("IFS_CONFIG_FILE", str(path))
    from ifs.config import reset_settings
    from ifs.main import create_app

    reset_settings()
    with TestClient(create_app()) as c:
        files = {"media": ("a.png", png, "image/png")}
        assert c.post("/v1/detect", files=files).status_code == 401
        assert c.post("/v1/detect", files=files, headers={"X-Relay-Token": KEY}).status_code == 200
    reset_settings()


def test_non_ascii_credential_is_rejected_not_a_crash(deployment, png):
    """A 401, not a 500.

    Starlette decodes header bytes as latin-1, so a credential can arrive as a
    non-ASCII str -- which `secrets.compare_digest` refuses outright. Comparing
    as bytes keeps that a rejection. Sent as bytes here because the test client
    will not encode a non-ASCII str.
    """
    with deployment({"detect": {"api_keys": [KEY]}}) as c:
        response = c.post(
            "/v1/detect",
            files={"media": ("a.png", png, "image/png")},
            headers={"X-API-Key": "ñøt-the-key".encode("latin-1")},
        )
    assert response.status_code == 401
    assert response.json()["error"]["type"] == "invalid_api_key"


# --- keys are per endpoint ---------------------------------------------------


def test_each_endpoint_takes_only_its_own_keys(deployment, png):
    """The point of the exercise: staking for one service must not hand out
    the others."""
    endpoints = {
        "detect": {"enabled": True, "api_keys": [DETECT_KEY]},
        "screen": {"enabled": True, "api_keys": [SCREEN_KEY]},
        "inspect": {"enabled": True, "api_keys": [INSPECT_KEY]},
    }
    with deployment(endpoints) as c:
        for route, mine in (
            ("/v1/detect", DETECT_KEY),
            ("/v1/screen", SCREEN_KEY),
            ("/v1/inspect", INSPECT_KEY),
        ):
            assert call(c, route, png, mine).status_code == 200
            for theirs in {DETECT_KEY, SCREEN_KEY, INSPECT_KEY} - {mine}:
                assert call(c, route, png, theirs).status_code == 401


def test_a_shared_key_opens_what_lists_it(deployment, png):
    """Repeat a key, or use a YAML anchor, to share one credential."""
    endpoints = {
        "detect": {"enabled": True, "api_keys": [KEY]},
        "screen": {"enabled": True, "api_keys": [KEY]},
        "inspect": {"enabled": True, "api_keys": [INSPECT_KEY]},
    }
    with deployment(endpoints) as c:
        assert call(c, "/v1/detect", png, KEY).status_code == 200
        assert call(c, "/v1/screen", png, KEY).status_code == 200
        assert call(c, "/v1/inspect", png, KEY).status_code == 401


def test_one_endpoint_keyed_leaves_the_others_open(deployment, png):
    """Documented, warned about at startup, and genuinely the behaviour."""
    endpoints = {
        "detect": {"enabled": True},
        "screen": {"enabled": True, "api_keys": [SCREEN_KEY]},
    }
    with deployment(endpoints) as c:
        assert call(c, "/v1/detect", png).status_code == 200
        assert call(c, "/v1/screen", png).status_code == 401


def test_startup_names_the_open_endpoints(deployment, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="ifs.api.auth"), deployment(
        {"detect": {"enabled": True}, "screen": {"enabled": True, "api_keys": [SCREEN_KEY]}}
    ):
        pass
    warning = "\n".join(r.getMessage() for r in caplog.records)
    assert "detect is OPEN" in warning
    assert "endpoints.detect.api_keys" in warning


# --- /v1/models is scoped to the credential ----------------------------------


def test_models_accepts_any_served_endpoints_key(deployment):
    endpoints = {
        "detect": {"enabled": True, "api_keys": [DETECT_KEY]},
        "inspect": {"enabled": True, "api_keys": [INSPECT_KEY]},
    }
    with deployment(endpoints) as c:
        assert c.get("/v1/models").status_code == 401
        assert c.get("/v1/models", headers={"X-API-Key": DETECT_KEY}).status_code == 200
        assert c.get("/v1/models", headers={"X-API-Key": INSPECT_KEY}).status_code == 200
        assert c.get("/v1/models", headers={"X-API-Key": "neither"}).status_code == 401


def test_models_describes_only_what_the_key_is_entitled_to(deployment):
    """A detection key must not learn the inspection service's upstream URLs,
    nor an inspection key which checkpoint the detector runs."""
    endpoints = {
        "detect": {"enabled": True, "api_keys": [DETECT_KEY]},
        "screen": {"enabled": True, "api_keys": [SCREEN_KEY]},
        "inspect": {"enabled": True, "api_keys": [INSPECT_KEY]},
    }
    with deployment(endpoints) as c:
        views = {
            name: c.get("/v1/models", headers={"X-API-Key": key}).json()
            for name, key in (
                ("detect", DETECT_KEY),
                ("screen", SCREEN_KEY),
                ("inspect", INSPECT_KEY),
            )
        }
    for name, body in views.items():
        # Every key sees which services exist; only its own section's detail.
        assert body["endpoints"] == ["detect", "screen", "inspect"]
        assert set(body) & {"detect", "screen", "inspect"} == {name}


def test_an_unmounted_endpoint_is_not_an_open_door(deployment):
    """Regression: an endpoint this deployment does not serve must not leave
    /v1/models reachable without credentials just because it lists no keys."""
    with deployment({"detect": {"enabled": True, "api_keys": [DETECT_KEY]}}) as c:
        assert c.get("/v1/models").status_code == 401
        assert c.get("/v1/models", headers={"X-API-Key": SCREEN_KEY}).status_code == 401


def test_a_disabled_endpoint_answers_404_not_401(deployment, png):
    """Absent, not forbidden."""
    with deployment({"detect": {"enabled": True, "api_keys": [DETECT_KEY]}}) as c:
        assert call(c, "/v1/screen", png, DETECT_KEY).status_code == 404
        assert call(c, "/v1/inspect", png, DETECT_KEY).status_code == 404
