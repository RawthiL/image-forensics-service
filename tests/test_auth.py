"""Service-level authorization: a deployment boundary, not user auth."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

KEY = "s3cret-relay-miner-key"


@pytest.fixture
def png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (10, 20, 30)).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def secured(stub_config, monkeypatch):
    monkeypatch.setenv("IFS_AUTH__API_KEY", KEY)
    from ifs.config import reset_settings
    from ifs.main import create_app

    reset_settings()
    with TestClient(create_app()) as c:
        yield c
    reset_settings()


def test_open_by_default(client, png_bytes):
    """No key configured = no gate, so local runs and tests stay frictionless."""
    r = client.post("/v1/detect", files={"media": ("a.png", png_bytes, "image/png")})
    assert r.status_code == 200


def test_rejects_missing_key(secured, png):
    r = secured.post("/v1/detect", files={"media": ("a.png", png, "image/png")})
    assert r.status_code == 401
    assert r.json()["error"]["type"] == "missing_api_key"
    assert r.headers["WWW-Authenticate"] == "Bearer"


def test_rejects_wrong_key(secured, png):
    r = secured.post(
        "/v1/detect",
        files={"media": ("a.png", png, "image/png")},
        headers={"X-API-Key": "wrong"},
    )
    assert r.status_code == 401
    assert r.json()["error"]["type"] == "invalid_api_key"


@pytest.mark.parametrize(
    "headers",
    [{"X-API-Key": KEY}, {"Authorization": f"Bearer {KEY}"}, {"authorization": f"bearer {KEY}"}],
)
def test_accepts_both_header_forms(secured, png, headers):
    r = secured.post("/v1/detect", files={"media": ("a.png", png, "image/png")}, headers=headers)
    assert r.status_code == 200


def test_compat_layers_are_gated_too(secured, png):
    for path, data in (("/1.0/check.json", {"models": "genai"}), ("/api/v2/task/sync", {})):
        files = {"media": ("a.png", png, "image/png")}
        assert secured.post(path, files=files, data=data).status_code == 401
        assert secured.post(
            path, files=files, data=data, headers={"X-API-Key": KEY}
        ).status_code == 200


def test_vendor_key_does_not_satisfy_the_service_key(secured, png):
    """`Authorization: token ...` is a Hive client's credential, not ours.

    It must stay ignored, and must never be mistaken for the service key.
    """
    r = secured.post(
        "/api/v2/task/sync",
        files={"media": ("a.png", png, "image/png")},
        headers={"Authorization": f"token {KEY}"},
    )
    assert r.status_code == 401


def test_probes_stay_open(secured):
    """An orchestrator's liveness check has no business holding the key."""
    assert secured.get("/healthz").status_code == 200
    assert secured.get("/readyz").status_code == 200


def test_models_endpoint_is_gated(secured):
    assert secured.get("/v1/models").status_code == 401
    assert secured.get("/v1/models", headers={"X-API-Key": KEY}).status_code == 200


def test_custom_header_name(stub_config, monkeypatch, png):
    monkeypatch.setenv("IFS_AUTH__API_KEY", KEY)
    monkeypatch.setenv("IFS_AUTH__API_KEY_HEADER", "X-Relay-Token")
    from ifs.config import reset_settings
    from ifs.main import create_app

    reset_settings()

    with TestClient(create_app()) as c:
        files = {"media": ("a.png", png, "image/png")}
        assert c.post("/v1/detect", files=files).status_code == 401
        assert c.post("/v1/detect", files=files, headers={"X-Relay-Token": KEY}).status_code == 200
    reset_settings()
