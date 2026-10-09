"""Three runtimes over one model must agree, and must not drift apart.

`community-forensics` (torch), `community-forensics-onnx` and
`community-forensics-triton` share a catalog and a preprocessing
implementation. These tests pin the two things that would silently break a
deployment: preprocessing parity, and score parity.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).parent))

ONNX_ARTIFACT = pathlib.Path("weights/onnx/commfor-384.onnx")
SHAPES = [(1408, 768), (3070, 1896), (500, 390), (390, 500), (1024, 1024), (440, 440)]


def _img(w, h, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))


# -- preprocessing --------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES)
def test_numpy_preprocessing_is_byte_identical_to_torch(shape):
    """The ONNX and Triton runtimes reimplement the torch transform in numpy.

    Anything but an exact match means two deployments of the same model
    disagree, so this asserts zero difference rather than a tolerance.
    """
    torch = pytest.importorskip("torch")
    from ifs.detectors.backends.community_forensics.preprocess import as_rgb, build_transform
    from ifs.detectors.backends.community_forensics.preprocess_np import preprocess

    img = _img(*shape, seed=abs(hash(shape)) % 2**31)
    ref = build_transform(384, 440)(as_rgb(img)).numpy()
    got = preprocess(img, input_size=384, resize_to=440)
    assert np.max(np.abs(ref - got)) == 0.0
    assert got.dtype == np.float32 and got.shape == (3, 384, 384)
    del torch


def test_numpy_sigmoid_matches_torch():
    torch = pytest.importorskip("torch")
    from ifs.detectors.backends.community_forensics.preprocess_np import sigmoid

    x = np.array([-40.0, -1.0, 0.0, 1.0, 40.0], np.float32)
    ref = torch.sigmoid(torch.from_numpy(x)).numpy()
    assert np.allclose(sigmoid(x), ref, atol=1e-7)


# -- the three runtimes agree --------------------------------------------


@pytest.mark.skipif(
    not ONNX_ARTIFACT.is_file(), reason="run the export first: see community_forensics/export.py"
)
def test_torch_and_onnx_scores_agree():
    pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    from ifs.detectors.registry import build

    imgs = [_img(w, h, seed=i) for i, (w, h) in enumerate(SHAPES[:3])]
    t = build("community-forensics", model="commfor-384", device="cpu", dtype="float32")
    o = build(
        "community-forensics-onnx", model="commfor-384", weights_dir="./weights", device="cpu"
    )
    t.warmup(), o.warmup()
    for a, b in zip(t.predict(imgs), o.predict(imgs), strict=True):
        # Rounded at the service boundary to score_decimals=4; assert far tighter.
        assert abs(a.ai_generated - b.ai_generated) < 1e-5
    t.close(), o.close()


@pytest.mark.skipif(not ONNX_ARTIFACT.is_file(), reason="needs the exported artifact")
def test_triton_and_onnx_scores_agree():
    pytest.importorskip("onnxruntime")
    from ifs.detectors.registry import build
    from mock_triton import make_server

    srv, url = make_server(str(ONNX_ARTIFACT), "commfor-384")
    try:
        trt = build(
            "community-forensics-triton", url=url, model_name="commfor-384", variant="vit-s16-384"
        )
        onx = build(
            "community-forensics-onnx", model="commfor-384", weights_dir="./weights", device="cpu"
        )
        trt.warmup(), onx.warmup()
        imgs = [_img(w, h, seed=i) for i, (w, h) in enumerate(SHAPES[:3])]
        for a, b in zip(trt.predict(imgs), onx.predict(imgs), strict=True):
            assert a.ai_generated == b.ai_generated  # same numpy preprocessing, same graph
        trt.close(), onx.close()
    finally:
        srv.shutdown()


# -- configuration errors are loud ---------------------------------------


def test_onnx_backend_rejects_model_and_path_together():
    from ifs.detectors.backends.community_forensics.catalog import CatalogError
    from ifs.detectors.registry import build

    with pytest.raises(CatalogError, match="exactly one"):
        build("community-forensics-onnx", model="commfor-384", path="/x.onnx")


def test_onnx_backend_requires_variant_with_path():
    from ifs.detectors.backends.community_forensics.catalog import CatalogError
    from ifs.detectors.registry import build

    with pytest.raises(CatalogError, match=r"requires detector\.variant"):
        build("community-forensics-onnx", path="/x.onnx")


def test_onnx_missing_artifact_names_the_export_command(tmp_path):
    pytest.importorskip("onnxruntime")
    from ifs.detectors.registry import build
    from ifs.detectors.weights import WeightsError

    det = build("community-forensics-onnx", model="commfor-384", weights_dir=str(tmp_path))
    with pytest.raises(WeightsError, match="export it once"):
        det.warmup()


def test_onnx_cuda_is_never_a_silent_cpu_fallback():
    """onnxruntime's default provider list quietly degrades to CPU."""
    from ifs.runtime.onnx import OnnxError, providers_for

    assert providers_for("cuda") == ["CUDAExecutionProvider"]
    assert providers_for("cuda:1") == ["CUDAExecutionProvider"]
    assert providers_for("cpu") == ["CPUExecutionProvider"]
    assert providers_for("auto") == ["CPUExecutionProvider"]
    with pytest.raises(OnnxError, match="not supported"):
        providers_for("mps")


def test_runtime_modules_are_model_agnostic():
    """`ifs/runtime/` holds execution, not model knowledge.

    If a runtime module ever imports a backend, the next model cannot reuse it
    and the extraction has quietly reverted.
    """
    import ast

    root = pathlib.Path(__file__).resolve().parents[1] / "src/ifs/runtime"
    for path in root.rglob("*.py"):
        src = path.read_text()
        # Parse imports rather than grepping: `torch.backends.mps` is not a
        # reference to this project's backends package.
        tree = ast.parse(src)
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [node.module or "", "." * node.level]
            for m in mods:
                assert "detectors" not in m, f"{path.name} imports {m}"
                assert "backends" not in m, f"{path.name} imports {m}"
        assert "commfor" not in src.lower(), f"{path.name} names a specific model"
        assert "community_forensics" not in src.lower(), f"{path.name} names a specific model"


def test_backends_use_the_shared_runtimes_rather_than_their_own():
    """Guards against a backend re-implementing transport it should import."""
    root = pathlib.Path(__file__).resolve().parents[1] / "src/ifs/detectors/backends"
    for path in root.rglob("detector_onnx.py"):
        assert "runtime.onnx import" in path.read_text()
    for path in root.rglob("detector_triton.py"):
        src = path.read_text()
        assert "runtime.triton import" in src
        # The KServe wire format belongs in the shared client, not here.
        assert "Inference-Header-Content-Length" not in src


def test_triton_backend_requires_a_variant():
    from ifs.detectors.backends.community_forensics.catalog import CatalogError
    from ifs.detectors.registry import build

    with pytest.raises(CatalogError, match=r"needs detector\.options\.variant"):
        build("community-forensics-triton", url="http://x", model_name="m")


def test_triton_unreachable_fails_at_startup_not_on_first_relay():
    from ifs.detectors.backends.community_forensics.detector_triton import TritonError
    from ifs.detectors.registry import build

    det = build(
        "community-forensics-triton",
        url="http://127.0.0.1:9",  # discard port
        model_name="commfor-384",
        variant="vit-s16-384",
        timeout_s=2.0,
    )
    with pytest.raises(TritonError, match="cannot reach Triton"):
        det.warmup()


@pytest.mark.skipif(not ONNX_ARTIFACT.is_file(), reason="needs the exported artifact")
def test_triton_rejects_a_model_whose_input_shape_disagrees():
    from ifs.detectors.backends.community_forensics.detector_triton import TritonError
    from ifs.detectors.registry import build
    from mock_triton import make_server

    srv, url = make_server(str(ONNX_ARTIFACT), "commfor-384")
    try:
        det = build(
            "community-forensics-triton",
            url=url,
            model_name="commfor-384",
            variant="vit-s16-224",  # wrong: the served graph is 384
        )
        with pytest.raises(TritonError, match="expects input"):
            det.warmup()
    finally:
        srv.shutdown()


def test_all_three_runtimes_are_registered():
    from ifs.detectors.registry import available

    for name in (
        "community-forensics",
        "community-forensics-onnx",
        "community-forensics-triton",
    ):
        assert name in available()
