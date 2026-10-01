# API reference

Base URL: `http://localhost:8000/api/v1`. The machine-readable spec is
[openapi.json](openapi.json), and the interactive version is at `/docs` (Swagger) and
`/redoc`. The consolidated result has its own JSON Schema at `GET /schema/meeting-result`,
and the e2e contract test validates real responses against it.

## Authentication

Send an API key in `X-API-Key` on every request except health, metrics and the schema.

```bash
docker compose -f docker/docker-compose.yml exec api python -m scripts.create_api_key --label me
export KEY=pk_...        # printed once; stored only as a SHA-256 hash
export API=http://localhost:8000/api/v1
```

Keys are tenants. Each meeting belongs to the key that uploaded it, and another key's
meeting answers 404, not 403, so its existence isn't revealed. The web UI at `/` uses the
same keys. Rate limits apply per key: `RATE_LIMIT_DEFAULT` everywhere, plus
`RATE_LIMIT_UPLOAD` on upload, process and regenerate. A 429 carries `Retry-After`.

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health/live`, `/health/ready`, `/health` | Liveness; readiness (DB, Redis, artifact store) |
| `POST` | `/meetings` | Upload a recording (multipart); `202` queued, or `200` for a duplicate |
| `GET` | `/meetings` | List with filters (`status`, `language`, `created_from/to`, `min/max_speakers`), sort and cursor pagination |
| `GET` | `/meetings/{id}` | Metadata, status, audio quality, detected languages, speaker count |
| `DELETE` | `/meetings/{id}` | Delete the meeting, its runs and every artifact (`204`) |
| `POST` | `/meetings/{id}/process` | Enqueue the pipeline (`?force=true` to reprocess, `?from_stage=` to resume) |
| `POST` | `/meetings/{id}/cancel` | Stop after the current stage or chunk |
| `GET` | `/meetings/{id}/status` | Progress snapshot: percent, current stage, per-stage status, ETA |
| `GET` | `/meetings/{id}/status/stream` | Server-Sent Events with the same snapshots; closes when the run ends |
| `GET` | `/meetings/{id}/audio` | Playback audio (processed WAV, else the original); `409` once purged |
| `GET` | `/meetings/{id}/speakers` | Diarization turns, `Person 1..N`, overlap regions |
| `PATCH` | `/meetings/{id}/speakers` | Display names, e.g. `{"names": {"Person 1": "Ravi"}}` (`null` clears one) |
| `GET` | `/meetings/{id}/transcript` | Speaker-attributed transcript: `?format=json\|txt\|srt\|vtt\|md`; raw ASR with `?view=raw` |
| `GET` | `/meetings/{id}/languages` | Language shares, per-speaker languages, switch points, code-mixed segments |
| `GET` | `/meetings/{id}/analytics` | Per-speaker and meeting statistics (`?format=csv` for a spreadsheet) |
| `GET` | `/meetings/{id}/analytics/charts/{speaking-time\|timeline}` | PNG chart (needs the `viz` extra) |
| `GET` | `/meetings/{id}/summary` | Minutes: summary, decisions, action items, open questions (`?format=md`) |
| `POST` | `/meetings/{id}/summary/regenerate` | Re-summarize, optionally in another language or with another model |
| `GET` | `/meetings/{id}/result` | Everything in one document (`?include=transcript,analytics,summary`) |
| `GET` | `/meetings/{id}/export` | Formal minutes as `?format=pdf\|docx\|md\|json` |
| `GET` | `/meetings/{id}/utterances` | Query utterances by speaker, language, time range and text |
| `GET` | `/meetings/{id}/runs`, `/runs/{run_id}` | Processing history, and the result of one run |
| `GET` | `/search?q=` | Full-text search over utterances and summaries (any script) |
| `GET` | `/schema/meeting-result` | JSON Schema of `MeetingResult` |
| `GET` | `/metrics` | Prometheus metrics |

## Walkthrough with curl

```bash
# 1. Upload. All form fields except file are optional.
curl -s -H "X-API-Key: $KEY" -F "file=@standup.m4a" -F "title=Weekly sync" \
     -F "expected_speakers=3" -F "languages=en,hi,or" "$API/meetings"
# {"meeting_id":"3f8b6f0e-...","status":"queued","created_at":"...","duplicate":false}
export ID=3f8b6f0e-...

# 2. Process. Optional: an HMAC-signed webhook when the run finishes.
curl -s -X POST -H "X-API-Key: $KEY" "$API/meetings/$ID/process"
# {"meeting_id":"3f8b6f0e-...","status":"processing"}

# 3. Follow progress: poll, or stream Server-Sent Events.
curl -s -H "X-API-Key: $KEY" "$API/meetings/$ID/status"
curl -N -H "X-API-Key: $KEY" "$API/meetings/$ID/status/stream"
# event: progress
# data: {"status":"processing","percent":42.5,"current_stage":"transcribe","eta_seconds":61.2,...}

# 4. Results.
curl -s -H "X-API-Key: $KEY" "$API/meetings/$ID/result" > result.json
curl -s -H "X-API-Key: $KEY" "$API/meetings/$ID/transcript?format=srt" -o meeting.srt
curl -s -H "X-API-Key: $KEY" "$API/meetings/$ID/summary?format=md"
curl -s -H "X-API-Key: $KEY" "$API/meetings/$ID/analytics?format=csv" -o speakers.csv
curl -s -H "X-API-Key: $KEY" "$API/meetings/$ID/export?format=pdf" -o minutes.pdf

# 5. Name the speakers; every format picks up the names.
curl -s -X PATCH -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
     -d '{"names": {"Person 1": "Ravi", "Person 2": "Sunita"}}' "$API/meetings/$ID/speakers"

# 6. Search across meetings, in any script.
curl -s -G -H "X-API-Key: $KEY" --data-urlencode "q=बजट" "$API/search"

# 7. Rerun only the summary with another model, or delete everything.
curl -s -X POST -H "X-API-Key: $KEY" "$API/meetings/$ID/process?from_stage=summarize&force=true"
curl -s -X DELETE -H "X-API-Key: $KEY" "$API/meetings/$ID"
```

A full example of the consolidated result is in
[samples/code-mixed-meeting.result.json](samples/code-mixed-meeting.result.json).

## Idempotency and duplicates

- `Idempotency-Key` on `POST /meetings` and `POST /process`: a repeated key replays the
  first response (`Idempotent-Replayed: true`). Reusing a key for a different request
  returns 409 `idempotency_key_reused`.
- Uploading the same file twice (SHA-256 match) returns the existing meeting with `200` and
  `"duplicate": true`, unless you pass `?allow_duplicate=true`.

## Webhooks

`callback_url` (on upload, or overridden on process) receives
`POST {"meeting_id": "...", "status": "completed"}` when a run ends. It is signed when
`WEBHOOK_SECRET` is set:

```python
import hashlib, hmac

expected = "sha256=" + hmac.new(secret, f"{ts}.".encode() + body, hashlib.sha256).hexdigest()
assert hmac.compare_digest(expected, request.headers["X-Polymom-Signature"])
# ts = request.headers["X-Polymom-Timestamp"]; reject old timestamps to stop replays
```

Only public `http`/`https` hosts are accepted. Loopback, private, link-local and metadata
addresses are refused, both when the URL is submitted and again at delivery.

## Upload validation

Checks run in this order, and a failure deletes the partial file:

1. The filename is sanitized (basename only; Hindi and Odia characters are kept).
2. The extension must be in `ALLOWED_EXTENSIONS`.
3. The body is streamed to storage and rejected as soon as it exceeds `MAX_UPLOAD_MB`; a
   declared `Content-Length` that is already too large is rejected before any body is read.
4. Empty files are rejected.
5. The magic bytes must match the extension.
6. ffprobe must find a readable audio stream. The duration must be at most
   `MAX_AUDIO_DURATION_MINUTES`.

## Meeting status

`queued` → `processing` → one of `completed`, `completed_with_errors` (an optional stage,
such as the summary, failed; everything else is available), `failed` or `cancelled`.

## Errors

Every error has the same envelope:

```json
{"error": {"code": "unsupported_file_type",
           "message": "File content does not match its extension.",
           "remediation": "Upload one of: wav, mp3, m4a, ...",
           "details": {"extension": "wav", "detected_mime_type": null},
           "request_id": "9d2c..."}}
```

Clients should switch on `code`, which is stable; `message` may change. `request_id`
matches the `X-Request-ID` response header and every log line for that request.

| Status | `code` | When |
| --- | --- | --- |
| 401 | `unauthorized` | Missing or unknown API key |
| 404 | `meeting_not_found`, `not_found` | Unknown id, or owned by another key |
| 409 | `transcript_not_available`, `diarization_not_available`, `language_summary_not_available`, `analytics_not_available`, `summary_not_available` | The stage has not run (yet), or failed |
| 409 | `audio_not_available` | Raw audio purged by retention |
| 409 | `meeting_state_conflict` | `/process` while processing, or on a completed meeting without `force` |
| 409 | `idempotency_key_reused` | Same `Idempotency-Key`, different request |
| 413 | `file_too_large`, `payload_too_large` | Over `MAX_UPLOAD_MB`, or a JSON body over `MAX_JSON_BODY_KB` |
| 415 | `unsupported_file_type` | Extension not allowed, or content does not match it |
| 422 | `empty_file`, `corrupted_media`, `audio_too_long`, `validation_error`, `unsupported_language` | Invalid input |
| 429 | `rate_limited` | Per-key limit; see `Retry-After` |
| 500 | `internal_error`, `media_probe_unavailable` | Server-side problem; details are in the logs under `request_id` |
| 503 | `charts_unavailable`, `llm_not_configured` | Optional component missing |

Errors recorded on a meeting (`meeting.error`, `processing.stages[].error`) use the same
codes, for example `diarization_model_unavailable`, `asr_model_unavailable`,
`audio_processing_failed`, `stage_timeout`, `resource_exhausted`, `llm_error` and
`llm_invalid_output`.
