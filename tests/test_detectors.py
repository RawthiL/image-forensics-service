"""The swap seam. If these pass, a new backend folder drops in untouched."""

import pytest
from PIL import Image

from ifs.detectors.base import DetectionResult, Detector, DetectorInfo
from ifs.detectors.registry import available, build, discover, unavailable


def test_stub_satisfies_the_protocol():
    assert isinstance(build("stub"), Detector)


def test_backends_are_discovered_from_folders():
    """No hand-maintained list: every sub-package of backends/ is imported."""
    discover()
    assert "stub" in available()
    # community-forensics needs torch; it is either available or explained.
    assert "community-forensics" in available() or "community_forensics" in unavailable()


def test_model_fingerprint_excludes_service_policy():
    """DetectorInfo covers the model only -- crop policy lives on the service."""
    a = DetectorInfo(name="m", version="1", dtype="float32")
    b = DetectorInfo(name="m", version="1", dtype="float16")
    c = DetectorInfo(name="m", version="1", dtype="float32", revision="abc")
    assert a.fingerprint != b.fingerprint
    assert a.fingerprint != c.fingerprint
    assert "center" not in a.fingerprint


def test_predict_is_order_preserving_and_one_per_input():
    det = build("stub")
    imgs = [Image.new("RGB", (32, 32), c) for c in ((0, 0, 0), (255, 0, 0), (0, 255, 0))]
    out = det.predict(imgs)
    assert len(out) == 3
    assert out == det.predict(imgs)  # deterministic


def test_results_are_clamped():
    assert DetectionResult(ai_generated=1.5).clamped().ai_generated == 1.0
    assert DetectionResult(ai_generated=-0.2).clamped().ai_generated == 0.0


def test_unknown_backend_names_the_alternatives():
    with pytest.raises(ValueError, match="unknown detector"):
        build("does-not-exist")
    assert "stub" in available()


def test_backends_do_not_import_the_api_layer():
    """Isolation, enforced. A backend that reaches for HTTP fails here."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "src/ifs/detectors/backends"
    forbidden = ("ifs.api", "ifs.schemas", "ifs.service", "from ....api", "from ....service")
    for path in root.rglob("*.py"):
        src = path.read_text()
        for needle in forbidden:
            assert needle not in src, f"{path} imports {needle}"


# -- multi-crop merging ---------------------------------------------------


def test_per_class_merge_across_crops():
    """Regression: the per_class merge referenced a comprehension variable that
    does not leak, so any backend filling per_class raised NameError as soon as
    more than one crop was scored. No shipped backend fills it, so nothing
    caught this until ruff did."""
    from ifs.service import _merge

    results = [
        DetectionResult(ai_generated=0.1, per_class={"flux": 0.2, "dalle": 0.9}),
        DetectionResult(ai_generated=0.8, per_class={"flux": 0.7}),
    ]
    merged = _merge(results, "max")
    assert merged.ai_generated == 0.8
    assert merged.per_class == {"flux": 0.7, "dalle": 0.9}

    mean = _merge(results, "mean")
    assert mean.per_class["flux"] == pytest.approx(0.45)
    assert mean.per_class["dalle"] == pytest.approx(0.9)  # present in one crop only


def test_deepfake_merge_across_crops():
    from ifs.service import _merge

    merged = _merge(
        [
            DetectionResult(ai_generated=0.1, deepfake=0.2),
            DetectionResult(ai_generated=0.3, deepfake=0.6),
        ],
        "max",
    )
    assert merged.deepfake == 0.6


def test_merge_leaves_a_single_result_untouched():
    from ifs.service import _merge

    only = DetectionResult(ai_generated=0.42, per_class={"x": 0.1})
    assert _merge([only], "max") is only
