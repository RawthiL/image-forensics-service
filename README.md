# image-forensics-service

**Three image-forensics services for [Pocket Network](https://pocket.network),
in one deployment.** Each is mounted, keyed and staked independently, and they
differ in what the operator has to supply:

| Service | Route | Answers | Needs |
|---------|-------|---------|-------|
| **AI-generation detection** | `POST /v1/detect` | was this image generated? | a model |
| **Container screening** | `POST /v1/screen` | is there anything hidden in this *file*? | **nothing** |
| **Prompt-injection inspection** | `POST /v1/inspect` | is there an injection in this *picture*? | two sibling services |

Detection is this project's own subject, and the reason it exists: it carries
the model mechanics — a catalog, preprocessing, three interchangeable runtimes
and an ONNX/Triton export path — that no sibling project provides. It also
exposes **compatibility layers** shaped like two commercial detection APIs, so
existing clients can swap a base URL and nothing else.

The other two load no model at all. Screening reads the file and never the
picture, so it needs no model, no GPU and no network — a supplier can serve it
from a small box. Inspection borrows two models over HTTP from the sibling
services that already own them:
[`image-inference-server`](../image-inference-server) for OCR and
[`language-forensics-service`](../language-forensics-service) for the
prompt-injection classifier.

Everything not enabled is not mounted, and answers `404` — absent, not
forbidden. Serve one, two, or all three.

The detection service is **model-agnostic**. Clients always see the same request
and response format, whichever detector the operator chooses to run. Each model
is a self-contained plugin under
[`src/ifs/detectors/backends/`](src/ifs/detectors/backends/). Adding a model
means adding a folder.


---

## Architecture

```
                  ┌─────────────────────── api/ (interface) ───────────────────────┐
 client / Pocket  │  auth (per endpoint) · image fetch + decode · limits           │
 RelayMiner ─────▶│  /v1/detect   /1.0/check.json   /api/v2/task/sync              │
                  │  /v1/screen   /v1/inspect                                      │
                  └────────┬──────────────────────────────────┬────────────────────┘
                           ▼                                  ▼
       ┌──── service.py (seam) ─────┐      ┌──── inspection/ (no weights) ────────┐
       │ crop policy · aggregation  │      │ metadata · unicode · rules   local   │
       │ rounding · fingerprint     │      │ media/views.py → N rendered views    │
       └────────────┬───────────────┘      └──────┬─────────────────────┬─────────┘
                    ▼ PIL images                  │ views               │ text
    ┌─ detectors/backends/<model>/ ─┐             ▼                     ▼
    │ community_forensics/  ViT-S/16│   image-inference-server   language-forensics
    │ stub/        pixel hash, CI   │   /v1/images:annotate      /v1/detect
    │ <new>/       …                │   (OCR, Cloud Vision-shaped)  (classifier)
    └───────────────────────────────┘
```

`/v1/screen` is the left-hand column's absence: it runs the `inspection/` local
checks and stops. No seam, no backend, no upstream.

The code is split into three layers:

| Layer | Path | Responsibility | Knows about models? |
|-------|------|----------------|---------------------|
| **Interface** | [`src/ifs/api/`](src/ifs/api/), [`src/ifs/schemas/`](src/ifs/schemas/) | HTTP routes, per-endpoint auth, input handling, vendor response shapes | **No** |
| **Contract** | [`src/ifs/detectors/base.py`](src/ifs/detectors/base.py) | The `Detector` protocol and shared result types | No |
| **Backends** | [`src/ifs/detectors/backends/<model>/`](src/ifs/detectors/backends/) | Weight catalog, preprocessing, inference, license notes | Only its own model |

At startup the service reads `config.yaml`, and — **only if
`endpoints.detect` is enabled** — builds the named backend and warms it. A
screening-only deployment downloads nothing and needs no `detector` section at
all. Every sub-package of `backends/` is imported automatically, so a new
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
| `src/ifs/api/` | HTTP interface: the three native routes, compatibility layers, per-endpoint auth, probes |
| `src/ifs/schemas/` | Vendor response shapes (Sightengine, Hive) |
| `src/ifs/service.py` | The seam: crop policy, aggregation, fingerprints |
| `src/ifs/detectors/base.py` | The `Detector` protocol every model implements |
| `src/ifs/detectors/registry.py` | Folder auto-discovery |
| `src/ifs/detectors/weights.py` | Checkpoint resolution — local dir, HF hub, or explicit path |
| `src/ifs/detectors/backends/community_forensics/` | Catalog, preprocessing (torch + numpy), three runtimes, ONNX/Triton export, license notes |
| `tests/mock_triton.py` | KServe v2 stand-in backed by onnxruntime, so the Triton runtime is testable without a 17 GB container |
| `src/ifs/detectors/backends/README.md` | **How to add a new backend** |
| `src/ifs/media/` | Image loading, SSRF guard, decompression-bomb guard, crop policies, view policies |
| `src/ifs/inspection/` | Container checks (metadata, Unicode, rules) and the two sibling-service clients. No weights — a test enforces it |
| `src/ifs/media/views.py` | What a *reader* sees: normalization passes and the downscales a victim's preprocessing would produce |
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

`config.yaml` is authoritative. Start by choosing which services to serve:

```yaml
endpoints:
  detect:  {enabled: true}     # needs a model, configured below
  screen:  {enabled: true}     # needs nothing
  inspect: {enabled: false}    # needs the two sibling services
```

**There is no implicit default checkpoint.** With `endpoints.detect` enabled
and no `detector` section, startup fails naming that contradiction rather than
resolving to some model and downloading it. With detection *off*, the whole
model stack below is unused and may be deleted from the file — that is what a
screening-only supplier runs.

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

The three native routes are what this project specifies, documents and
versions, and what new integrations should target. They are described by
[`docs/openapi.json`](docs/openapi.json), served live at `/openapi.json` with
Swagger UI at `/docs`.

| Endpoint | Description | Default |
|----------|-------------|---------|
| `POST /v1/detect` | Score one image for AI generation | **on** |
| `POST /v1/screen` | Screen the image container — no model, no upstream | off |
| `POST /v1/inspect` | Inspect the pixels for an embedded prompt injection | off |
| `GET /v1/models` | What this deployment serves, and with what | — |
| `GET /healthz` | Liveness (no auth) | — |
| `GET /readyz` | `200` once every served endpoint is up (no auth) | — |

All three take the same body and reject the same inputs, so a caller that can
reach one can reach the others unchanged.

As `multipart/form-data`:

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

- **API keys gate the deployment** — a deployment boundary ("only our relay
  miner may reach this container"), not user authentication. They carry no
  caller identity and no quota. Compared in constant time.
- **Keys are per endpoint**, because the endpoints are separately-stakeable
  Pocket Network services:

  ```yaml
  endpoints:
    detect:  {api_keys: ["key-for-the-detection-stake"]}   # + both compat layers
    screen:  {api_keys: ["key-for-the-screening-stake"]}
    inspect: {api_keys: ["key-for-the-injection-stake"]}
  ```

  A key opens the endpoints that list it and no others, so a supplier staked
  for screening cannot be handed a credential that also opens detection. Repeat
  a key, or use a YAML anchor, to share one.
- **`api_keys` is a list**, so a key can be rotated without downtime: publish
  the new one, let callers move, drop the old. Every candidate is compared and
  the results OR-ed without short-circuiting, so response timing does not leak
  which key in the list was presented.
- **An empty list leaves that endpoint open — per endpoint, not per
  deployment.** Keying `screen` does not key `detect`. Startup names every open
  endpoint at `WARNING` rather than reporting the deployment "secured".
- Only *mounted* endpoints count. A key for a route this deployment does not
  serve grants nothing, and such a route is not reported as an open hole.
- The key arrives as `X-API-Key` or `Authorization: Bearer …`. **Only the
  `Bearer` scheme is read** from `Authorization`, so a Hive client's
  `Authorization: token …` is never mistaken for the service key.
- `/healthz` and `/readyz` stay open — an orchestrator's probe has no business
  holding a key. `/v1/models` is gated, accepts any served endpoint's key, and
  **answers with the sections that key is entitled to**: a screening key does
  not learn the inspection service's upstream URLs, nor which checkpoint the
  detector runs, just because they share a container.
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

## Container screening

`POST /v1/screen` reads the file and never the picture. It needs **no model, no
sibling service, no GPU and no network** — which is the whole reason it is a
service of its own rather than a mode of another one. A supplier can serve it
from a small box, and it is ready the moment it is listening.

What it finds:

| Class | Carrier |
|-------|---------|
| Metadata payload | EXIF, XMP, IPTC, PNG `tEXt`/`iTXt`/`zTXt`, JPEG `COM`, ICC description |
| Unicode smuggling | Tags block `U+E0000–E007F`, zero-width, bidi overrides, variation selectors |
| Polyglot file | Bytes after the image's logical end |

Metadata is a weak carrier on its own, and this section is the place to say so
plainly: most pixel pipelines decode to RGB and drop EXIF, so a payload there
often never reaches a model at all. It matters when application code lifts
metadata into a prompt — a captioning tool, a document parser, an agent
shelling out to `exiftool`, an indexer treating metadata as searchable text.

**An image with instructions painted across it comes back `clean` here.** That
is the documented limit of container scope, and why every response carries
`scope`. `/v1/inspect` is what looks at the pixels.

XMP is scanned as text rather than parsed as XML. We are looking for injected
prose, not structure, and not running an XML parser over attacker-controlled
input is a feature.

## Prompt-injection inspection

`POST /v1/inspect` asks whether this image carries instructions aimed at the
next model that reads it. It runs the container checks above — the same code,
so the two endpoints never disagree about a file — and then looks at the
pixels.

**It loads no model.** The two it needs are served by sibling projects:

| Hop | Service | API |
|-----|---------|-----|
| pixels → boxes and text | [`image-inference-server`](../image-inference-server) | `POST /v1/images:annotate` (Cloud Vision-shaped) |
| text → verdict | [`language-forensics-service`](../language-forensics-service) | `POST /v1/detect` |

That split is deliberate. Those repositories own their tokenizers, their Triton
deployments and their weights licensing; copying either here would mean a
second diverging implementation of the subtle part, and a second licence to
track. What this service owns is **forensic policy** — which views to render,
what counts as evidence, and how the pieces combine into a verdict.

### What it looks for

| Class | Carrier |
|-------|---------|
| Rendered text | Instructions painted into the pixels |
| Hidden text | Alpha channel, near-white-on-white, the low bit plane |
| **Scaling attack** | Benign at full size; text emerges under the victim's downscale |
| …plus everything `/v1/screen` finds | |

Adversarial-perturbation attacks — optimised noise that steers a specific
vision model — are **not** covered. No reliable detector for them exists, and
this service does not claim one.

### Why view rendering stays local

[`media/views.py`](src/ifs/media/views.py) re-renders the image the way a
victim's preprocessing would: the alpha channel flattened against black and
white, contrast stretched, the least-significant bit plane pulled out, and the
frame downscaled to each configured size with each configured resampler. Those
views are sent to the OCR service; the policy that produced them stays here.

That is what makes the sharpest finding possible. **Text present at a downscale
and absent at native resolution has no benign explanation** — a real sign in a
photograph reads *better* at full resolution, not worse. The
[image-scaling attack](https://blog.trailofbits.com/2025/08/21/weaponizing-image-scaling-against-production-ai-systems/)
is defined by exactly that asymmetry, and the service can only see it because
it holds both views. Keeping the renders here also leaves the OCR service a
dumb, reusable primitive rather than something that has to know about this
attack.

Which views are rendered is policy, so it is part of the pipeline fingerprint,
exactly as crop policy is for detection.

### Boxes, not only text

The OCR service is asked for text **regions** as well as their contents. Both
come from one call — recognition needs detection anyway — and the regions cover
the case recognition cannot: a view with visible text boxes that the recogniser
could not read. **You do not need to know what the text says to know it is
hidden.**

So a view that disagrees with the native frame is reported either way, at the
severity the evidence supports:

| Evidence | Finding | Severity |
|----------|---------|----------|
| Novel text, and it reads as an injection | `scaling_attack` / `hidden_text` | high |
| Novel text, no injection pattern matched | `view_differential` | medium |
| Boxes but nothing readable | `view_differential` | medium |

OCR does hallucinate the odd word on a rescaled photograph, so the middle row
exists to keep that a reported anomaly rather than a false accusation.

### Verdicts

`malicious` is reserved for deterministic attack signatures — a Tags-block
payload, a bidi override, text that appears only after a transformation the
victim would apply. Those have no benign explanation. A classifier score is a
probability however high, so it raises `suspicious` and reports its number
rather than claiming a certainty it does not have.

Findings carry **where** they were found, because the location is most of the
evidence:

```json
{
  "scope": "image",
  "verdict": "malicious",
  "injection": 1.0,
  "severity": "high",
  "findings": [
    {"kind": "scaling_attack", "severity": "high", "score": 1.0,
     "where": "view:rescale:384:bicubic",
     "detail": "text that is absent at native resolution appears in this view, and reads as an injection"},
    {"kind": "instruction_override", "severity": "high", "score": 0.95,
     "where": "view:rescale:384:bicubic", "detail": "classic instruction-override phrasing"}
  ],
  "upstreams": {
    "ocr": {"status": "ok", "fingerprint": "ocr:ppocr-det"},
    "classifier": {"status": "ok", "fingerprint": "horizon-guard-base@e4fbfc6f/window"}
  },
  "views_rendered": 9,
  "pipeline_fingerprint": "inspect/rules@8c21…+views@3f2a…+ocr:ppocr-det+cls:horizon-guard-base@…",
  "elapsed_ms": 412.3
}
```

`upstreams` is not decoration. A report that silently omitted the classifier
because it was unreachable would read exactly like a clean one, and a caller
would act on a check that never happened. Each hop says whether it ran (`ok`),
is not configured (`disabled`), had nothing to do (`skipped`), or failed
(`unreachable`). With `required: true` — the default — a failure is a `502`
naming the hop instead.

### Cost

Most images carry no text at all. When OCR finds none, the classifier is never
called, and the request costs one local pass plus one OCR call. When there is
text, the view fan-out means one call becomes a handful of OCR images — batched
into as few requests as the upstream's limits allow (`inspect.ocr.batch`, and a
second bound on request bytes).

`/v1/screen` costs nothing by comparison, which makes it usable as a
pre-filter in front of `/v1/inspect`.

### Configuring it

```yaml
endpoints:
  screen:
    enabled: true            # needs nothing else
  inspect:
    enabled: true
    ocr:
      url: http://image-inference:8000        # base URL, no path
      api_key: ${OCR_KEY}                     # sent as x-goog-api-key
    classifier:
      url: http://language-forensics:8000     # base URL, no path
      api_key: ${CLASSIFIER_KEY}              # sent as x-api-key
```

Both upstreams are optional even for `/v1/inspect`: with neither configured it
degrades to the container checks and says so in `upstreams`. If that is all you
want, serve `/v1/screen` instead — it is the same answer without the pretence.

These URLs are **operator configuration and never caller-supplied**, which is
why they sit outside the SSRF guard that governs `limits.allow_url_fetch`. A
caller cannot steer them.

### Reproducibility

Both endpoints report a pipeline fingerprint naming everything that decides
their answer, and no more:

```
screen:  screen/rules@<digest>
inspect: inspect/rules@<digest>+views@<digest>+ocr:<upstream>+cls:<upstream>
```

A container screen depends only on the rule table, so that is all it names —
the view policy and the two upstreams are not hops it declined to use, they are
not part of that pipeline at all. The rule table is digested because editing a
pattern changes the answers. Upstream identities are read **once, at startup**
(`/health` for OCR, `/v1/models` for the classifier), so the fingerprint cannot
depend on what other requests were doing; if an upstream is redeployed
underneath the service, the mismatch is logged loudly rather than silently
changing it.

`include_text` is off by default on both. Two suppliers running different OCR
builds return different strings for the same image, so excerpts are the least
reproducible part of a report — useful for a human triaging a hit, noise in
anything comparing answers across suppliers.

**A detector is a filter, not a defense.** Use these alongside system-level
controls — least-privilege tools, confirmation of sensitive actions — never as
the only one.

## Configuration

See [`config.example.yaml`](config.example.yaml). Key fields:

- `endpoints.<detect|screen|inspect>.enabled`: which services this deployment
  serves. Anything not enabled is not mounted and answers `404`. At least one
  must be on.
- `endpoints.<name>.api_keys`: the keys for that endpoint, as a list. Empty
  leaves it open — see [Security and limits](#security-and-limits).
- `endpoints.detect.compat.<sightengine|hive>`: switch either compatibility
  layer off.
- `endpoints.screen.*` / `endpoints.inspect.*`: thresholds, view policy and the
  two upstream services — see
  [Prompt-injection inspection](#prompt-injection-inspection).
- `detector.backend`: which backend folder to load. **Required only when
  `endpoints.detect` is enabled**, and an error to omit if it is.
- `detector.model` **or** `detector.path` + `detector.variant`: the checkpoint.
  Exactly one of the two forms; both or neither is a startup error.
- `detector.options`: backend-specific extras, passed to the factory verbatim.
- `weights.source`: `auto` (local dir, then hub), `local` (only `weights.dir`) or
  `hub`. `allow_download: false` with nothing cached is an error, not a fallback.
  Unused when `detector.path` is set.
- `runtime.device` / `runtime.dtype`: `auto` resolves cuda → mps → cpu, and
  selects fp16 only on CUDA.
- `scoring.*`: crop policy, aggregation, score rounding.
- `auth.api_key_header`: the header credentials arrive in. Only the header
  *name* is global; the keys live on each endpoint.
- `limits.*`: byte cap, pixel cap, URL fetching (`allow_url_fetch`, off by
  default), its timeout, and the SSRF guard (`allow_private_urls`). Shared by
  every endpoint.

Environment variables override individual keys, nested with a double underscore
(`IFS_DETECTOR__MODEL=commfor-224`,
`IFS_ENDPOINTS__INSPECT__OCR__URL=http://image-inference:8000`), and `IFS_CONFIG_FILE`
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
