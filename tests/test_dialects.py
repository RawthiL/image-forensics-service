"""The contract that matters: responses must match the vendors' documented shapes."""


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


def test_models_endpoint_reports_backend(client):
    body = client.get("/v1/models").json()
    assert body["endpoints"] == ["detect"]
    assert body["detect"]["active"]["name"] == "stub"
    assert "stub" in body["detect"]["available"]
    # Not served here, so not described.
    assert "screen" not in body and "inspect" not in body
