# API compatibility

**These are compatibility layers, not this service's API.** `POST /v1/detect`
is the API this project specifies and versions; see
[`openapi.json`](openapi.json). The routes below reshape the same inference
result into a commercial vendor's envelope so an existing client can swap its
base URL, and they inherit whatever that envelope cannot express.

This document is normative for the layers — it records what we reproduce, what
we approximate, and what we deliberately omit. The OpenAPI document describes
their request and response schemas, but fidelity to the vendor is defined here.

Neither vendor publishes an OpenAPI spec (verified: Sightengine
`/openapi.json`, `/swagger.json`, docs paths; Hive's ReadMe site and
`llms.txt` — all 404). The mappings below were derived from their published
documentation samples.

---

## Sightengine

**Upstream:** [`https://api.sightengine.com/1.0/check.json`](https://sightengine.com/docs/ai-generated-image-detection)

### Request

| Field | Upstream | Here |
|---|---|---|
| `media` | multipart file | same |
| `url` | GET query param | accepted on both GET and POST |
| `models` | comma list, e.g. `genai` | accepted; only `genai`/`deepfake` are meaningful |
| `api_user`, `api_secret` | required | **accepted and ignored** |

### Response

Upstream's documented sample:

```json
{
  "status": "success",
  "request": { "id": "req_0zrbHDeitGYY7wEGncAne", "timestamp": 1491402308.4762, "operations": 5 },
  "type": { "ai_generated": 0.001 },
  "media": { "id": "med_0zrbk8nlp4vwI5WxIqQ4u", "uri": "https://..." }
}
```

| Field | Fidelity |
|---|---|
| `status` | exact |
| `request.id` | same `req_` prefix, different entropy source |
| `request.timestamp` | exact (unix float) |
| `request.operations` | hardcoded `5`, matching Sightengine's genai billing. Echoed for clients that parse it; nothing is charged here |
| `type.ai_generated` | **our detector's score** |
| `type.deepfake` | present only when a backend fills it; absent otherwise |
| `media.id` | same `med_` prefix |
| `media.uri` | echoed when the caller passed `url`, else `null` |

**Not reproduced:** every other `models=` value (nudity, violence, OCR, …);
`workflow` endpoints; video and audio; the account/usage endpoints.

---

## Hive

**Upstream:** [`https://api.thehive.ai/api/v2/task/sync`](https://docs.thehive.ai/docs/ai-image-and-video-detection)

### Request

| Field | Upstream | Here |
|---|---|---|
| `url` | form field | same |
| `media` | multipart file | same |
| `Authorization: token <key>` | required | **accepted and ignored** |

### Response

Hive nests results under a `status` array and returns two classifier heads whose
scores each sum to 1:

* **generation head** — `ai_generated` / `not_ai_generated`
* **source head** — ~80 generator names plus `none`, `inconclusive`, `deepfake`

```json
{"id":"...","code":200,"project_id":0,
 "status":[{"status":{"code":"0","message":"SUCCESS"},
   "response":{"input":{...},
     "output":[{"time":0.11,"classes":[
       {"class":"ai_generated","score":0.0055},
       {"class":"not_ai_generated","score":0.9945},
       {"class":"inconclusive","score":0.0055},
       {"class":"none","score":0.9945}]}]}}]}
```

| Field | Fidelity |
|---|---|
| nesting, `status`/`code`/`message` | exact shape |
| generation head | **our score**, and its complement; sums to 1 as documented |
| source head | **approximated.** Community Forensics is a binary classifier with no per-generator head, so we report `inconclusive` carrying the generated-probability mass and `none` the rest. A backend that fills `per_class` replaces this with real generator names, normalized, remainder on `none` |
| `deepfake` class | appended only when a backend provides it |
| `output[].time` | real wall-clock seconds for the inference |
| `algorithmic_tags` | **not implemented.** This is Hive's C2PA provenance block; it is on the roadmap |
| `project_id` | always `0` |

**Not reproduced:** the async task API (`/task` + callbacks), task retry and
summary endpoints, video/audio/text submissions, every moderation model.

---

## Native

`POST /v1/detect` is the honest surface. Same three input forms; the response
carries the score, the model fingerprint, how many crops were scored, and the
elapsed time. Prefer it for new integrations — the dialects exist to capture
callers who already wrote against a vendor.

## Service authorization vs vendor keys

Two unrelated credentials can appear on a compatibility request:

| | Purpose | Handling |
|---|---|---|
| `api_user` / `api_secret` (Sightengine), `Authorization: token ...` (Hive) | The caller's credential for the **vendor** | Parsed and **ignored**. Rejecting them would break the base-URL swap |
| `X-API-Key` or `Authorization: Bearer ...` | This **deployment's** key (`IFS_API_KEY`) | Enforced when set; `401` otherwise |

Only the `Bearer` scheme is read from `Authorization`, so a Hive client's
`Authorization: token ...` is never mistaken for the service key — even if the
values match. A test pins this.

## Errors

Both dialects return vendor-ish failure envelopes:

```json
{"status": "failure", "error": {"type": "media_too_large", "message": "..."}}
```

`401` for a missing or invalid service key (`missing_api_key`, `invalid_api_key`), with `WWW-Authenticate: Bearer`.

`400` for bad input (`missing_media`, `ambiguous_media`, `invalid_media`,
`invalid_url`, `invalid_base64`, `url_fetch_failed`, `url_fetch_disabled`),
`413` for `media_too_large`. Upstream error codes are numeric and
vendor-specific; we do not attempt to reproduce their numbering.
