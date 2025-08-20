## Vibe Data Director API

Production-ready FastAPI service for dataset curation with Kumo SDK (KumoRFM) and HuggingFace export.

### Base URL
- Local: `http://localhost:8080`
- Deployed: `<your deployment URL>`

### Authentication
- All endpoints except `GET /health` require Bearer auth.
- Header: `Authorization: Bearer <API_KEY>`
- Errors:
  - 403 when header is missing
  - 401 when token is invalid
  - 503 when auth is not configured on the server

### Rate Limits
- Session init: 10/min
- Uploads (seed): 20/min
- Planning (coverage/specs): 30/min
- Exports: 5/min
- 429 is returned when exceeded. Standard rate-limit headers may be present.

### Environment variables (server)
- `API_KEY` (required): API key your clients will send
- `KUMO_API_KEY` (required for Kumo integration)
- `KUMO_URL` (recommended): set to `https://api.kumo.ai/api`
- `CORS_ORIGINS` (optional): comma-separated list
- `REDIS_URL` (optional): enable Redis sessions; falls back to in-memory
- `SESSION_TTL` (optional): default 1800 seconds
- `ENCRYPTION_KEY` (optional): Fernet key for sensitive fields

### General Limits & Notes
- Max 100 total rows per session
- Max 10 classes per session
- Max 1000 characters per text sample
- Request text fields are sanitized (HTML entity encoding) and truncated
- CSV upload expects headers: `text,class,style,negation`

---

### GET /health
Health check (no auth required).

Response 200:
```json
{ "status": "healthy", "service": "vibe-data-director" }
```

---

### POST /session/init
Initialize a session and generate spec grid.

Auth: Bearer
Rate limit: 10/min

Request body:
```json
{
  "classes": ["positive", "negative"],
  "target_count_per_class": 50,
  "styles": ["formal", "casual"],
  "include_negations": false
}
```

Constraints:
- `classes`: 1–10 unique strings
- `target_count_per_class`: 1–100
- `styles`: optional; default ["none"]
- `include_negations`: optional; default false

Response 200:
```json
{
  "session_id": "session_ab12cd34",
  "specs": [
    {"spec_id": "positive|formal|0", "class": "positive", "style": "formal", "negation": false}
  ]
}
```

Errors: 401/403/422

---

### POST /session/seed_upload
Upload seed data via file (multipart/form-data).

Auth: Bearer
Rate limit: 20/min

Form fields:
- `session_id` (string, required)
- `file` (required): CSV or JSON file

CSV columns:
- `text` (string, required)
- `class` (string, required)
- `style` (string, optional; default "none")
- `negation` (bool, optional; default false)

Response 200:
```json
{
  "rows": [
    {
      "sample_id": "sample_abcdef12",
      "ts": "2025-01-01T00:00:00Z",
      "text": "Great product!",
      "class": "positive",
      "style": "formal",
      "negation": false,
      "source": "seed",
      "spec_id": "positive|formal|0"
    }
  ],
  "total_rows": 1
}
```

Errors: 400 (bad file), 401/403, 404 (unknown session), 422 (limits/validation)

---

### POST /session/seed_upload_json
Upload seed data via JSON body.

Auth: Bearer
Rate limit: 20/min

Request body:
```json
{
  "session_id": "session_ab12cd34",
  "rows": [
    {"text": "This is amazing", "class": "positive", "style": "formal", "negation": false}
  ]
}
```

Notes:
- The field is `class` in the payload (accepted); server maps it internally.
- Same constraints and defaults as file upload.

Response 200: same shape as `/session/seed_upload`.

Errors: 401/403, 404, 422

---

### GET /plan/coverage
Predict per-class coverage for the session.

Auth: Bearer
Rate limit: 30/min

Query params:
- `session_id` (string, required)

Response 200:
```json
{
  "coverage": [
    {"class": "positive", "pred_count": 45},
    {"class": "negative", "pred_count": 38}
  ],
  "details": [
    {
      "class": "positive",
      "spec_contributions": [
        {"spec_id": "positive|formal|0", "pred_count": 2},
        {"spec_id": "positive|casual|0", "pred_count": 0}
      ]
    }
  ]
}
```

Notes:
- Coverage is computed as the sum of per‑spec predictions for each class.
- With sparse or very recent data, short‑horizon counts may be 0. The service returns 200 with fallback values instead of failing.

Errors: 401/403, 404

---

### GET /plan/specs
Recommend next spec IDs for a class.

Auth: Bearer
Rate limit: 30/min

Query params:
- `session_id` (string, required)
- `class_name` (string, required; must be in session classes)

Response 200:
```json
{
  "spec_ids": ["positive|formal|0", "positive|casual|0"],
  "spec_predictions": {
    "positive|formal|0": 2,
    "positive|casual|0": 0
  }
}
```

Notes:
- `spec_predictions` is optional and contains predicted 10‑minute counts per `spec_id` when available. Clients can sort ascending to fill lowest-coverage specs first. If absent, clients should use a safe fallback ordering.

Errors: 401/403, 404, 422

---

### POST /export/hf
Export session samples to a HuggingFace dataset.

Auth: Bearer
Rate limit: 5/min

Request body:
```json
{
  "session_id": "session_ab12cd34",
  "repo_id": "your-org/your-dataset"
}
```

Constraints:
- `repo_id` must match `owner/name` (alphanumeric, `-` and `_` allowed)
- Session must contain at least one sample

Response 200:
```json
{ "repo_url": "https://huggingface.co/datasets/your-org/your-dataset" }
```

Errors: 401/403, 404, 422, 500

---

### Frontend integration tips
- Always include the Bearer token header.
- Respect rate limits; backoff on 429.
- For CSV uploads, send `multipart/form-data` with `file` and `session_id`.
- For JSON uploads, use `/session/seed_upload_json` with `rows` as above.
- Use the `session_id` returned by `/session/init` for all subsequent calls.

### RFM integration details
- Tables and types: `samples(sample_id ID, ts time, class categorical, style categorical, negation categorical, text text, spec_id ID)`, `specs(spec_id ID, class categorical, style categorical, negation categorical)`, `classes(class ID)`.
- Graph links: `samples.spec_id → specs.spec_id`, `specs.class → classes.class`, and `samples.class → classes.class`.
- Coverage: class coverage is computed as the sum of per‑spec `PREDICT COUNT(samples.*, 0, 10, minutes) FOR specs.spec_id = '...'`.
- Specs ranking: `PREDICT LIST_DISTINCT(specs.spec_id, 0, 1, minutes) FOR classes.class = '...'` (PK-compliant FOR).
- Temporal window: server spreads `samples.ts` across the last hour to satisfy short‑horizon predictions; if predictions are unavailable, endpoints return 200 with fallback values.

### Version
- API version: `0.1.0` (service)

## Endpoint Details

### Authentication and Headers
- All endpoints except `GET /health` require `Authorization: Bearer <API_KEY>`
- `Content-Type: application/json` for JSON POSTs
- Multipart form required for file upload endpoint

### Sessions
- POST `/session/init`: create a new session and return the spec grid derived from classes, styles, and negation options.

### Uploads
- POST `/session/seed_upload`: CSV/JSON file upload; requires `session_id` form field
- POST `/session/seed_upload_json`: inline JSON upload; requires `session_id` and `rows`

### Planning
- GET `/plan/coverage`: returns predicted counts per class for the next 10 minutes (sum of per‑spec predictions)
- GET `/plan/specs`: returns ranked `spec_ids` and optional `spec_predictions` per `spec_id`

### Export
- POST `/export/hf`: exports the accumulated samples to a HuggingFace dataset repository


