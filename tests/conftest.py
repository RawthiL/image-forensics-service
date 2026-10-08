import io

import pytest
import yaml
from fastapi.testclient import TestClient
from PIL import Image

STUB_CONFIG = {"detector": {"backend": "stub", "model": "stub-v1"}}


@pytest.fixture
def stub_config(tmp_path, monkeypatch):
    """An isolated config file.

    Tests must not inherit the repo's own config.yaml -- it is the operator's
    file and may select any backend, with `path` or `model`.
    """
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(STUB_CONFIG))
    monkeypatch.setenv("IFS_CONFIG_FILE", str(path))
    return path


@pytest.fixture
def client(stub_config):
    from ifs.config import reset_settings
    from ifs.main import create_app

    reset_settings()
    with TestClient(create_app()) as c:
        yield c
    reset_settings()


@pytest.fixture
def png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (120, 30, 200)).save(buf, format="PNG")
    return buf.getvalue()
