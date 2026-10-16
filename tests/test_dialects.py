"""The contract that matters: responses must match the vendors' documented shapes."""

import base64
import json


def test_native_detect(client, png_bytes):
    r = client.post("/v1/detect", files={"media": ("x.png", png_bytes, "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["ai_generated"] <= 1.0
    assert body["model_fingerprint"].startswith("stub@")
    assert body["pipeline_fingerprint"].endswith("/five+max")


def test_sightengine_shape(client, png_bytes):
    """Field-for-field against the sample in Sightengine's docs."""
    r = client.post(
        "/1.0/check.json",
        files={"media": ("x.png", png_bytes, "image/png")},
        data={"models": "genai", "api_user": "ignored", "api_secret": "ignored"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert set(body) >= {"status", "request", "type", "media"}
    assert set(body["request"]) >= {"id", "timestamp", "operations"}
    assert body["request"]["id"].startswith("req_")
    assert body["media"]["id"].startswith("med_")
    assert 0.0 <= body["type"]["ai_generated"] <= 1.0


def test_hive_shape(client, png_bytes):
    r = client.post(
        "/api/v2/task/sync",
        files={"media": ("x.png", png_bytes, "image/png")},
        headers={"Authorization": "token ignored"},
    )
    assert r.status_code == 200
    classes = r.json()["status"][0]["response"]["output"][0]["classes"]
    names = {c["class"]: c["score"] for c in classes}
    assert "ai_generated" in names and "not_ai_generated" in names
    # Hive documents that each head's scores sum to 1.
    assert abs(names["ai_generated"] + names["not_ai_generated"] - 1.0) < 1e-6


def test_vendor_keys_are_not_required(client, png_bytes):
    """The whole point of the dialects: swap the base URL, change nothing else."""
    r = client.post("/1.0/check.json", files={"media": ("x.png", png_bytes, "image/png")})
    assert r.status_code == 200


def test_missing_media_is_a_400(client):
    r = client.post("/v1/detect")
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "missing_media"


def test_native_detect_accepts_a_json_body(client, png_bytes):
    """`/v1/detect` takes the same JSON body as `/v1/screen` and `/v1/inspect`."""
    encoded = base64.b64encode(png_bytes).decode()
    form = client.post("/v1/detect", files={"media": ("x.png", png_bytes, "image/png")}).json()
    sent = client.post("/v1/detect", json={"media_base64": encoded})
    assert sent.status_code == 200, sent.text
    assert sent.json()["ai_generated"] == form["ai_generated"]
    assert sent.json()["pipeline_fingerprint"] == form["pipeline_fingerprint"]


def test_detect_reads_json_under_any_content_type(client, png_bytes):
    """After a relay or proxy hop, the content type is not the client's."""
    body = json.dumps({"media_base64": base64.b64encode(png_bytes).decode()}).encode()
    r = client.post("/v1/detect", content=body, headers={"content-type": "application/x-protobuf"})
    assert r.status_code == 200, r.text
    assert 0.0 <= r.json()["ai_generated"] <= 1.0


def test_vendor_dialects_stay_form_only(client, png_bytes):
    """A JSON body is not part of either vendor's documented contract.

    They must answer the way the vendor would -- with a missing-media error,
    not a success -- or the swap-the-base-URL promise is broken.
    """
    body = {"media_base64": base64.b64encode(png_bytes).decode()}
    assert client.post("/1.0/check.json", json=body).status_code == 400
    assert client.post("/api/v2/task/sync", json=body).status_code == 400


def test_models_endpoint_reports_backend(client):
    body = client.get("/v1/models").json()
    assert body["endpoints"] == ["detect"]
    assert body["detect"]["active"]["name"] == "stub"
    assert "stub" in body["detect"]["available"]
    # Not served here, so not described.
    assert "screen" not in body and "inspect" not in body
