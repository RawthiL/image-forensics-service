# Backends

One folder per model. Nothing outside this directory knows a model exists.

## Adding one

Create `backends/<your_model>/` containing:

| File | Required | Purpose |
|---|---|---|
| `__init__.py` | yes | Calls `@register("<name>")` on a factory |
| `detector.py` | convention | The `Detector` implementation |
| `MODEL.md` | yes | Provenance: paper, weights, license, citation, known limits |
| anything else | — | Preprocessing, helpers — all private to the folder |

Minimum viable backend:

```python
# backends/my_model/__init__.py
from ..registry import register
from .detector import MyDetector

@register("my-model")
def _factory(**kwargs):
    return MyDetector(**kwargs)
```

```python
# backends/my_model/detector.py
from ...base import DetectionResult, DetectorInfo

class MyDetector:
    def __init__(self, *, device=None, dtype=None, weights_dir=None,
                 revision=None, allow_download=True, **model_specific):
        ...

    def info(self):   return DetectorInfo(name="my-model", version="1")
    def warmup(self): ...
    def predict(self, images): return [DetectionResult(ai_generated=0.5) for _ in images]
    def close(self):  ...
```

Then `IFS_DETECTOR=my-model`. There is no registry list to edit and no route,
schema, or dialect to touch — `registry.discover()` imports every sub-package of
this directory at startup.

## The contract

Defined in [`../base.py`](../base.py). Four methods. `Detector` is a `Protocol`,
so you need no base class and no inheritance.

`tests/test_detectors.py` is the contract test every backend must pass.

## What the host passes you

The factory receives these generic kwargs from settings, plus anything in
`IFS_DETECTOR_OPTIONS`:

| kwarg | From `config.yaml` |
|---|---|
| `model` | `detector.model` — a name in *your* catalog, or `None` |
| `path` | `detector.path` — a checkpoint file, or `None` |
| `variant` | `detector.variant` — required alongside `path` |
| `revision` | `detector.revision` |
| `weights_source` | `weights.source` (`auto`/`local`/`hub`) |
| `weights_dir` | `weights.dir` |
| `allow_download` | `weights.allow_download` |
| `device` | `runtime.device` |
| `dtype` | `runtime.dtype` |

Accept `**kwargs` and ignore what you don't use. Backend-specific knobs arrive
through `detector.options`, never as new global settings.

Exactly one of `model` or `path` is set; the config layer guarantees that before
your factory runs. Publish a **catalog** (see
[`community_forensics/catalog.py`](community_forensics/catalog.py)) mapping
names to pinned checkpoints, and raise a clear error for an unknown name rather
than loading something else.

## What you must not do

* Import anything from `ifs.api`, `ifs.schemas`, or `ifs.service`. Backends know
  nothing about HTTP.
* Apply crop policies. The service does that, using `DetectorInfo.input_size`
  as a size hint.
* Read environment variables directly. Take configuration as constructor kwargs.
* Download weights yourself. Use [`../weights.py`](../weights.py), which gives
  you source resolution for free and **never substitutes** one checkpoint for
  another.
* Infer which checkpoint to load from anything other than configuration. If the
  config does not say, raise.
