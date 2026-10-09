# image-forensics-service

An **AI-generated image detection backend for [Pocket Network](https://pocket.network)**.
It exposes its own **OpenAPI-specified `/v1/detect` API**, plus **compatibility layers** shaped like two commercial detection APIs so existing clients can swap a base URL and nothing else.

The service is **model-agnostic**. Clients always see the same request and
response format, whichever detector the operator chooses to run. Each model is a
self-contained plugin under [`src/ifs/detectors/backends/`](src/ifs/detectors/backends/).
Adding a model means adding a folder.


---

## Architecture

```
                   ┌──────────────────── api/ + schemas/ (interface) ────────────────────┐
 client / Pocket   │  /v1/detect (primary)  ·  auth  ·  image fetch + decode  ·  limits  │
 RelayMiner ──────▶│  /1.0/check.json, /api/v2/task/sync (compatibility layers)          │
                   │                              │                     ▲                 │
                   └──────────────────────────────┼─────────────────────┼─────────────────┘
                                                  ▼                     │
                   ┌────────────────────── service.py (seam) ───────────────────────────┐
                   │  crop policy · aggregation · rounding · pipeline fingerprint       │
                   └──────────────────────────────┼─────────────────────┼───────────────┘
                                                  ▼ PIL images          │ DetectionResult
                   ┌──── detectors/backends/<model>/ (one folder per model) ────┐
                   │  community_forensics/  ViT-S/16 · catalog · preprocessing   │
                   │  stub/                 pixel hash, for CI                   │
                   │  <new>/                …                                    │
                   └─────────────────────────────────────────────────────────────┘
```

The code is split into three layers:

| Layer | Path | Responsibility | Knows about models? |
|-------|------|----------------|---------------------|
| **Interface** | [`src/ifs/api/`](src/ifs/api/), [`src/ifs/schemas/`](src/ifs/schemas/) | HTTP routes, auth, input handling, vendor response shapes | **No** |
| **Contract** | [`src/ifs/detectors/base.py`](src/ifs/detectors/base.py) | The `Detector` protocol and shared result types | No |
| **Backends** | [`src/ifs/detectors/backends/<model>/`](src/ifs/detectors/backends/) | Weight catalog, preprocessing, inference, license notes | Only its own model |

At startup the service reads `config.yaml`, builds the named backend, and
warms it. Every sub-package of `backends/` is imported automatically, so a new
folder registers itself. Nothing in `api/` branches on model type — a test
asserts backends never import `ifs.api`, `ifs.schemas` or `ifs.service`.

## Supported backends

| Backend | Folder | Models | Code license | **Weights license** |
|---------|--------|--------|--------------|---------------------|
| Community Forensics | [`backends/community_forensics`](src/ifs/detectors/backends/community_forensics/) | `commfor-384`, `commfor-224` (ViT-S/16, binary head) | MIT | MIT ✅ (training data CC-BY-4.0) |
| Stub | [`backends/stub`](src/ifs/detectors/backends/stub/) | none — hashes pixels, for CI | this repo | — |

### Runtimes

Models can be served by three interchangeable runtimes. They share one
catalog and one preprocessing implementation, and produce **identical scores** —
measured, not assumed. Pick on deployment constraints, not accuracy.

| `detector.backend` | Build target | Needs | Inference | Image | RSS under load |
|---|---|---|---|---|---|
| `community-forensics-triton` | `triton` | `triton` extra (numpy only) | remote Triton | **489 MB** | **~99 MB** |
| `community-forensics-onnx` | `onnx` / `onnx-gpu` | `onnx-cpu` / `onnx-gpu` extra | in-process | 589 MB | ~410 MB |
| `community-forensics` | `cpu` / `gpu` | `cpu`/`cu124` + `community-forensics` | in-process | 1.7 / 9.2 GB | 577 / 759 MB |

See [Run it](#3-run-it) for the matching compose service and published tag.


The ONNX artifact is produced locally, because upstream publishes safetensors
only:

```bash
uv run python -m ifs.detectors.backends.community_forensics.export \
    --model commfor-384 --out-dir ./weights
```

That writes `weights/onnx/commfor-384.onnx` for the ONNX runtime and a complete
Triton model repository at `weights/triton/commfor-384/` (`config.pbtxt` with
dynamic batching, plus `1/model.onnx`). The export verifies itself against the
torch checkpoint and refuses to write a model that disagrees.

### Triton

If you already run Triton for other services, this backend adds almost nothing:
no torch, no onnxruntime, no weights and no GPU on the service side -- it
preprocesses and speaks HTTP. Measured: **99 MB** resident, against 410 MB for
the in-process ONNX runtime. Use the `triton` image and the `ifs-triton` compose
service, which reserves no GPU:

```bash
IFS_TAG=0.1.0 docker compose --profile triton up ifs-triton
```

Point Triton at the exported repository and configure:

```yaml
detector:
  backend: community-forensics-triton
  # No `model` or `path`: the checkpoint lives on the Triton server and is
  # named by options.model_name.
  options:
    url: http://triton:8000
    model_name: commfor-384
    variant: vit-s16-384      # fixes preprocessing, which stays client-side
```


`GET /v1/models` reports what is loaded, what else is installed, and why any
backend folder failed to import.

## Detection quality

**Read this before serving this model on Pocket Network.**

This service returns a probability, not a verdict, and the shipped detector has
measurable limits. Numbers below are from an independent 2026 benchmark
([arXiv:2602.07814](https://arxiv.org/html/2602.07814v1)) and from measurements
in this repository.

- **Modern generators evade it.** Flux Dev, Firefly v4 and Midjourney v7 are
  detected at **18–30%** average accuracy across every open method tested.
  Accuracy decays from ~79% on 2020-era generators to ~38% on 2024-era ones.
- **Headline accuracy depends on who measures.** The paper reports 0.987 mAP /
  89.2% on its own eval set; the independent benchmark puts the same checkpoint
  at **75.0% mean / 82.1% median** in the wild — still the best it tested.
- **Training data is 99.67% latent diffusion** (0.25% GAN, 0.06% pixel
  diffusion). That skew is the direct cause of the weakness above.
- **Recompression costs real signal.** Measured here on one known-AI image:
  pristine PNG 0.9988 → JPEG q=95 0.9991 → **q=85 0.8289** → **q=70 0.3895**.
  Resizing alone barely matters (0.9954).
- **Crop policy dominates everything else.** See [Crop policy](#crop-policy).
- **It is not a deepfake detector.** Face manipulation is a different problem.
  The `deepfake` field exists in the contract; no shipped backend fills it.

**The operator is responsible for whether this accuracy is fit for their
callers' purpose.** This section is guidance, not a warranty.

## Model licensing

All code in this repository is MIT-licensed. **That license does not cover model
weights or third-party software.** A checkpoint keeps its own license whichever
server runs it.

- **Community Forensics weights** ([`OwensLab/commfor-model-384`](https://huggingface.co/OwensLab/commfor-model-384),
  [`-224`](https://huggingface.co/OwensLab/commfor-model-224)) and the
  [reference implementation](https://github.com/JeongsooP/Community-Forensics)
  are **MIT** — commercial service is fine, subject to the usual notice
  requirement.
- **The Community Forensics dataset**
  ([`OwensLab/CommunityForensics`](https://huggingface.co/datasets/OwensLab/CommunityForensics))
  is **CC-BY-4.0**. You do not redistribute it by serving the model, but
  **attribution is required** if you retrain or republish derived weights.
- **Fine-tuned or third-party checkpoints** loaded via `detector.path` inherit
  the license of their base model and training data. Check the source of every
  checkpoint you deploy.

If you use this in research, cite the authors — the BibTeX is in
[`community_forensics/MODEL.md`](src/ifs/detectors/backends/community_forensics/MODEL.md).

"Sightengine" and "Hive" are used only to describe API compatibility. This
project is not affiliated with or endorsed by either.

## Repository layout

| Path | What |
|------|------|
| `src/ifs/api/` | HTTP interface: `/v1/detect`, compatibility layers, auth, probes |
| `src/ifs/schemas/` | Vendor response shapes (Sightengine, Hive) |
| `src/ifs/service.py` | The seam: crop policy, aggregation, fingerprints |
| `src/ifs/detectors/base.py` | The `Detector` protocol every model implements |
| `src/ifs/detectors/registry.py` | Folder auto-discovery |
| `src/ifs/detectors/weights.py` | Checkpoint resolution — local dir, HF hub, or explicit path |
| `src/ifs/detectors/backends/community_forensics/` | Catalog, preprocessing (torch + numpy), three runtimes, ONNX/Triton export, license notes |
| `tests/mock_triton.py` | KServe v2 stand-in backed by onnxruntime, so the Triton runtime is testable without a 17 GB container |
| `src/ifs/detectors/backends/README.md` | **How to add a new backend** |
| `src/ifs/media/` | Image loading, SSRF guard, decompression-bomb guard, crop policies |
| `docs/openapi.json` | Generated OpenAPI spec (neither emulated vendor publishes one) |
| `docs/API_COMPAT.md` | Field-by-field vendor mapping, and what is not reproduced |
| `config.example.yaml` | Configuration reference (copy to `config.yaml`, which is gitignored) |
| `uv.lock` | Pinned dependency set for every supported platform and torch build |
| `tests/` | Dialect contract, backend isolation, auth, and config-validation tests |
| `.github/workflows/release.yml` | Test, then build and push the CPU and GPU images on a published release |
| `docker-compose.yml` | Runs the published images; `build` profile builds locally instead |

## Quick start

### 1. Configure

```bash
cp config.example.yaml config.yaml     # set detector.model or detector.path
```

`config.yaml` is authoritative. **There is no implicit default checkpoint** — if
no config file is found, startup fails with `detector: Field required` rather
than resolving to some model and downloading it.

### 2. Choose a checkpoint

By **name**, from the backend's catalog (pinned to a commit sha, fetched to the
Hugging Face cache on first run):

```yaml
detector:
  backend: community-forensics
  model: commfor-384
```

...or by **path**, in which case you must say which architecture it is:

```yaml
detector:
  backend: community-forensics
  path: /weights/commfor-model-384/model.safetensors
  variant: vit-s16-384      # REQUIRED: a file does not state its own architecture
```

With `path`, the `weights` section is unused and **no network access happens at
all**. Never set both `model` and `path`, and never neither — both fail at
startup naming the valid options.

### 3. Run it

**Pick a runtime, then read across.** Every column on a row goes together.

| Runtime (`detector.backend`) | Build target | Compose service | Published tag | Needs a GPU? |
|---|---|---|---|---|
| `community-forensics-triton` | `triton` | `ifs-triton` | `:X.Y.Z-triton` | **No** — inference is on your Triton |
| `community-forensics-onnx` | `onnx` | `ifs` | `:X.Y.Z` | No |
| `community-forensics-onnx` | `onnx-gpu` | `ifs-gpu` | `:X.Y.Z-onnx-gpu` | Yes |
| `community-forensics` (torch) | `cpu` | `ifs-torch` | `:X.Y.Z-torch` | No |
| `community-forensics` (torch) | `gpu` | `ifs-torch-gpu` | `:X.Y.Z-torch-gpu` | Yes |

**If you run the Triton backend, you want the `triton` image.** It contains no
model runtime at all — no torch, no onnxruntime, no weights, no GPU — because
inference happens on your Triton server. It is the smallest image (489 MB) and
the lightest process (~99 MB resident).

#### Build it yourself

```bash
docker build --target triton -t ifs:triton .     # Triton backend
docker build --target onnx   -t ifs:onnx   .     # ONNX, CPU
docker build --target gpu    -t ifs:gpu    .     # torch, CUDA
```

Or through compose, which tags `:dev` so a local build never shadows a pulled
release:

```bash
docker compose --profile build up ifs-triton-build
docker compose --profile build up ifs-build          # onnx
docker compose --profile build up ifs-gpu-build      # onnx-gpu
```

#### Or pull a published image

```bash
IFS_TAG=0.1.0 docker compose --profile triton up ifs-triton   # Triton backend
IFS_TAG=0.1.0 docker compose up ifs                           # ONNX, CPU
IFS_TAG=0.1.0 docker compose --profile gpu up ifs-gpu         # ONNX, CUDA
IFS_TAG=0.1.0 docker compose --profile torch up ifs-torch     # torch, CPU
IFS_TAG=0.1.0 docker compose --profile torch up ifs-torch-gpu # torch, CUDA
```

| Variable | Default | Notes |
|----------|---------|-------|
| `IFS_TAG` | `latest` | A release (`0.1.0`), a branch build (`main`), or a sha tag. **Pin a release in production.** `latest` exists only once a GitHub release has been published |
| `IFS_IMAGE` | `ghcr.io/rawthil/image-forensics-service` | Override for a fork or a mirror |
| `IFS_PULL_POLICY` | `missing` | Set `always` to re-pull a moving tag on every `up` |

Each service appends its own suffix to `IFS_TAG`; you pass the bare version.
`docker compose pull` refreshes a moving tag.

Compose mounts `./config.yaml:/etc/ifs/config.yaml` for every service, and
`./weights:/weights` for the ones that load a checkpoint locally (`ifs-triton`
needs no weights). The startup line `configuration loaded from <path>` confirms
which file was read; `ready: pipeline=... weights=...` confirms where the
checkpoint came from.

Locally, without containers:

```bash
uv sync --extra cpu --extra community-forensics --extra dev
uv run uvicorn ifs.main:app --reload
```

`uv sync` with no extras gives a working API on the `stub` backend and never
downloads torch — enough to develop against the HTTP surface.

> Python 3.11–3.13 (`.python-version` pins 3.12); PyTorch has no 3.14 wheels
> yet. `cpu` and `cu124` install the same distributions from different PyTorch
> indexes — pick exactly one. On an Intel Mac the newest available build is
> torch 2.2.2, which the lock carries alongside the modern Linux wheels.

### 4. Call it

```bash
curl -s http://localhost:8000/v1/detect \
  -H "X-API-Key: $IFS_API_KEY" \
  -F "media=@photo.jpg"
```

```json
{"ai_generated": 0.9995, "deepfake": null, "per_class": {},
 "model_fingerprint": "community-forensics@commfor-384+6076002b…/float16",
 "pipeline_fingerprint": "…/five+max", "crops_scored": 5, "elapsed_ms": 41.2}
```

Existing Sightengine client code works unchanged. Only the URL changes:

```python
import requests
r = requests.post("https://your-endpoint/1.0/check.json",       # was api.sightengine.com
                  data={"models": "genai", "api_user": "…", "api_secret": "…"},
                  files={"media": open("photo.jpg", "rb")})
print(r.json()["type"]["ai_generated"])
```

## API

`POST /v1/detect` is this service's API — what the project specifies, documents
and versions, and what new integrations should target. It is described by
[`docs/openapi.json`](docs/openapi.json), served live at `/openapi.json` with
Swagger UI at `/docs`.

| Endpoint | Description |
|----------|-------------|
| `POST /v1/detect` | Score one image |
| `GET /v1/models` | Active backend, scoring policy, what else is installed |
| `GET /healthz` | Liveness (no auth) |
| `GET /readyz` | `200` once the model is loaded (no auth) |

The request body is `multipart/form-data`:

| Field | Notes |
|-------|-------|
| `media` | Image upload. JPEG, PNG, WebP, BMP, TIFF, MPO |
| `url` | Alternative to `media`: an `http(s)` URL to fetch. **Off by default** — see [Security and limits](#security-and-limits) |
| `media_base64` | Alternative to both: base64 string or `data:` URI |

The response carries the score plus two fingerprints. Every error is returned as
`{"status": "failure", "error": {"type": "…", "message": "…"}}` with a fitting
HTTP status.

### Compatibility layers

Secondary surfaces, so an application already written against a commercial
detection API can point here by changing its base URL and nothing else.

| Layer | Route | Mimics | Vendor docs |
|-------|-------|--------|-------------|
| Sightengine | `POST`/`GET` `/1.0/check.json` | `api.sightengine.com/1.0/check.json?models=genai` | [docs](https://sightengine.com/docs/ai-generated-image-detection) |
| Hive | `POST /api/v2/task/sync` | `api.thehive.ai/api/v2/task/sync` | [docs](https://docs.thehive.ai/docs/ai-image-and-video-detection) |

Of the two, **Sightengine is the better-supported target**: its users are the
swap-a-base-URL population, and its flat single-float response maps cleanly onto
ours. Hive's ~80-name per-generator head is approximated, since no shipped
backend produces one.

**Vendor API keys are accepted and ignored.** `api_user`, `api_secret` and
`Authorization: token …` are parsed and discarded rather than rejected — a
missing vendor key must not `401`, or the swap does not work. Neither layer is
normative; [`docs/API_COMPAT.md`](docs/API_COMPAT.md) is the field-by-field
mapping. Either can be switched off in `config.yaml`.

**Neither vendor publishes an OpenAPI specification** — verified against
Sightengine's `/openapi.json` and `/swagger.json`, and Hive's ReadMe site and
`llms.txt`. This service ships one.

### Crop policy

Detectors score a fixed-size crop. The upstream evaluation transform resizes the
whole frame to reach it, which on a large image **downsamples away the
high-frequency artifacts the model keys on**. `five` takes crops at close to
native resolution instead. On real-world images that difference dominates
everything else:

| Image | `center` | `five` (max) |
|-------|----------|--------------|
| AI illustration, 1408×768 JPEG | 0.2288 ❌ | **0.9995** ✅ |
| AI video frame, 3070×1896 PNG | 0.0315 ❌ | **0.9974** ✅ |
| Real photo, 5152×3864 JPEG | 0.0001 ✅ | **0.0000** ✅ |

`five` is the shipped default, at roughly 5× the compute. `center` matches how
these models are evaluated in papers; `tiled` scores up to nine tiles.
`crop_aggregation` is `max` (default) or `mean` — use `max` for detection, since
a generated region anywhere in the frame should raise the score and averaging
dilutes it.

The policy is part of the pipeline fingerprint. It is not a free knob.

### Reproducibility

Every response carries two fingerprints:

```
model:    <name>@<version>+<weights-revision>/<dtype>
pipeline: <model fingerprint>/<crop-policy>+<aggregation>
```

The split is deliberate: the model fingerprint is what the backend contributes,
the pipeline fingerprint adds the service-level scoring policy that
`service.py` applies and the backend knows nothing about. Two deployments
reporting the same pipeline fingerprint and disagreeing on a score have a bug.

Catalog models are pinned to a commit sha, so "the latest weights on `main`" is
never what you get by accident. Sigmoid is always computed in fp32 even when the
model runs in fp16, because at half precision the tails saturate to exactly
0.0/1.0 and throw away the margin callers threshold on.

### Security and limits

- **A single shared API key gates the deployment** — a deployment boundary
  ("only our relay miner may reach this container"), not user authentication.
  It carries no caller identity and no quota. Unset means every route is open,
  and startup logs a warning. Compared in constant time.
- The key arrives as `X-API-Key` or `Authorization: Bearer …`. **Only the
  `Bearer` scheme is read** from `Authorization`, so a Hive client's
  `Authorization: token …` is never mistaken for the service key.
- `/healthz` and `/readyz` stay open — an orchestrator's probe has no business
  holding the key. `/v1/models` is gated; it reveals backend internals.
- **Image URLs are off by default.** Only images in the request payload are
  accepted (`media`, or base64 in `media_base64`); a `url` returns `400`.
  Fetching URLs would let any caller make the node issue requests from its own
  network position, costs egress bandwidth, and can make answers differ between
  suppliers if the content behind a URL changes. Opt in with
  `limits.allow_url_fetch: true`.
- When enabled, URL fetching is guarded against server-side request forgery
  (SSRF). Every hop — the submitted URL and each redirect target — is resolved
  and checked by IP, rejecting loopback, RFC1918, link-local (including the
  `169.254.169.254` metadata address), CGNAT, reserved, unspecified and
  multicast addresses, plus IPv6 that merely embeds one of those (v4-mapped,
  6to4, Teredo). Redirects are followed manually for that reason: httpx's own
  `follow_redirects` would validate only the URL the caller supplied. URLs
  carrying credentials and non-http(s) schemes are refused, redirect chains are
  capped, and environment proxies are ignored (`trust_env=False`) so a
  configured proxy cannot route around the guard. Disable only in development
  (`limits.allow_private_urls: true`).
- The guard resolves a name and then connects, so a DNS record that changes in
  between (rebinding) is a residual risk. Closing it needs address pinning in
  the transport; it is a reason to leave URL fetching off, not to rely on the
  guard alone.
- Upload and URL body size are capped (`limits.max_image_bytes`). URL bodies are
  streamed with a running cap rather than buffered first.
- Image dimensions are checked against `limits.max_pixels` **before** decoding
  (decompression-bomb guard).
- Inference runs on a thread pool behind a lock, so large images don't stall the
  async server and concurrent requests don't race one torch module.

## Configuration

See [`config.example.yaml`](config.example.yaml). Key fields:

- `detector.backend`: which backend folder to load.
- `detector.model` **or** `detector.path` + `detector.variant`: the checkpoint.
  Exactly one of the two forms; both or neither is a startup error.
- `detector.options`: backend-specific extras, passed to the factory verbatim.
- `weights.source`: `auto` (local dir, then hub), `local` (only `weights.dir`) or
  `hub`. `allow_download: false` with nothing cached is an error, not a fallback.
  Unused when `detector.path` is set.
- `runtime.device` / `runtime.dtype`: `auto` resolves cuda → mps → cpu, and
  selects fp16 only on CUDA.
- `scoring.*`: crop policy, aggregation, score rounding.
- `auth.api_key` / `auth.api_key_header`: the deployment key.
- `limits.*`: byte cap, pixel cap, URL fetching (`allow_url_fetch`, off by
  default), its timeout, and the SSRF guard (`allow_private_urls`).
- `api.*`: switch either compatibility layer off.

Environment variables override individual keys, nested with a double underscore
(`IFS_DETECTOR__MODEL=commfor-224`, `IFS_AUTH__API_KEY=…`), and `IFS_CONFIG_FILE`
points at a different file. A key `config.yaml` does not recognise — including a
typo nested inside a section — is an error, not a silent default.

> An `IFS_DETECTOR__*` variable merges *into* the file's `detector` section
> rather than replacing it. Setting `IFS_DETECTOR__MODEL` while the file sets
> `path` yields both, which is rejected.

## Adding a model backend

1. Create `src/ifs/detectors/backends/<name>/` with an `__init__.py` calling
   `@register("<name>")` on a factory, and a `detector.py` implementing the
   `Detector` protocol. The protocol has four members: `info`, `warmup`,
   `predict`, `close`.
2. Publish a **catalog** mapping names to pinned checkpoints (see
   [`community_forensics/catalog.py`](src/ifs/detectors/backends/community_forensics/catalog.py)),
   and raise a clear error for an unknown name rather than loading something else.
3. Add a `MODEL.md` **stating the model's license**, and a matching extra in
   `pyproject.toml` so the API still installs without its dependencies. Run
   `uv lock` afterwards.

There is no registry list to edit — every sub-package of `backends/` is imported
at startup. The step-by-step guide is in
[`backends/README.md`](src/ifs/detectors/backends/README.md), and
`tests/test_detectors.py` is the contract a new backend must pass.

## Releases

Images are published to the GitHub Container Registry by
[`.github/workflows/release.yml`](.github/workflows/release.yml) when a release
is published. The lean ONNX/CPU build is the unsuffixed default: it serves the
same model as the torch images at roughly a third of the memory, and most
suppliers will not have a GPU.

All tags are under `ghcr.io/<owner>/<repo>`, and each moving tag (`:X.Y`,
`:latest`) carries the same suffix as its exact version.

| Tag | Build target | Runtime |
|-----|--------------|---------|
| `:X.Y.Z`, `:X.Y`, `:latest` | `onnx` | ONNX Runtime, CPU (the default) |
| `:X.Y.Z-triton` | `triton` | Triton client, no local model runtime |
| `:X.Y.Z-onnx-gpu` | `onnx-gpu` | ONNX Runtime, CUDA |
| `:X.Y.Z-torch` | `cpu` | torch, CPU |
| `:X.Y.Z-torch-gpu` | `gpu` | torch, CUDA |

Pre-releases publish the exact `X.Y.Z` tag only, leaving `:X.Y` and `:latest`
pointing at the last stable build. A manual run from the Actions tab publishes
`:<branch>` and `:sha-<short>` for trying an unreleased build.

To cut a release: bump `version` in `pyproject.toml`, then publish a GitHub
release tagged `vX.Y.Z`. The workflow **fails if the tag and the project
version disagree**, so the two cannot drift.

Before pushing, the workflow checks the lock is current, lints, runs the full
suite, verifies `docs/openapi.json` is not stale, and confirms both deployment
targets still resolve a torch build. After pushing, it boots the published
image against the `stub` backend and waits for `/readyz`, then attaches a
signed build-provenance attestation.

Images contain **code only**. Model weights are never baked in, and neither is
a `config.yaml` — the service requires one to be mounted and refuses to start
without it.

## Tests

```bash
uv run pytest               # 43 tests, torch-free and offline via the stub backend
```

```bash
# Regenerate the OpenAPI spec after changing any route or schema
uv run python scripts/export_openapi.py

# Lint
uv run ruff check src tests

# Verify the lock is current, and that each deployment target resolves
uv lock --check
uv pip compile --python-platform x86_64-unknown-linux-gnu --extra cpu pyproject.toml
```

The suite covers the vendor response shapes field-by-field, backend isolation
(a backend that imports the API layer fails), the auth boundary including the
vendor-key collision, configuration validation — that a wrong config is an error
and never a silent fallback to a different model — and the SSRF guard, including
that a redirect to an internal address is re-validated rather than trusted.

## License

The code in this repository is released under the MIT License. Model weights and
third-party components keep their own licenses. See
[Model licensing](#model-licensing).
