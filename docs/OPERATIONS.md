# Operations

Running, configuring, scaling and troubleshooting Polymom. For the architecture see
[ARCHITECTURE.md](ARCHITECTURE.md); for the security model see [SECURITY.md](SECURITY.md).

## Deployment topologies

| Setup | Command | Use |
| --- | --- | --- |
| Demo (mock ML) | `make demo` | Clean-clone smoke test and UI walkthrough; no GPU, no tokens |
| Full stack, CPU | `docker compose -f docker/docker-compose.yml up -d --build` | API + cpu/gpu/llm workers + Postgres + Redis |
| + real models on an NVIDIA GPU | `... --profile gpu up -d --build --scale worker-gpu=0` | `worker-gpu-cuda` runs diarization, LID and ASR on CUDA |
| + S3 artifacts | `--profile s3`, `ARTIFACT_STORE=s3`, `S3_ENDPOINT_URL=http://seaweedfs:8333` | Multi-host deployments (workers must share artifacts) |
| + local LLM | `--profile local-llm`, `LLM_PROVIDER=ollama`, `OLLAMA_BASE_URL=http://ollama:11434` | Summaries without sending transcripts to a cloud API |
| + monitoring | `--profile monitoring` | Prometheus :9090, Grafana :3000 (provisioned *Polymom* dashboard) |
| Local dev, no Docker | `make dev && make run`, plus `python -m app.workers.main --queue cpu` (and `gpu`, `llm`) | Needs `REDIS_URL`; or `PIPELINE_EXECUTION=inline` for single-process debugging |

Profiles combine: `--profile gpu --profile s3 --profile monitoring`.

After the stack is up, create an API key and open the UI:

```bash
docker compose -f docker/docker-compose.yml exec api python -m scripts.create_api_key --label me
# paste the printed pk_... key into the box at the top of http://localhost:8000/
```

## Images

`docker/Dockerfile` is multi-stage (a uv builder and a slim runtime). It runs as the non-root
`app` user, pins its base images by digest (`python:3.11.16-slim-trixie`, `uv:0.9.30`) and
has a `HEALTHCHECK` on `/api/v1/health/live`. In compose, workers override it with a check on
their metrics port (`:9101/metrics`).

| Build | Command | Contents |
| --- | --- | --- |
| CPU, light (default) | `make docker-build` | API + workers with mock or remote backends; no torch |
| CPU, real models | `make docker-build-ml` | + torch (CPU), pyannote, faster-whisper, IndicConformer runtime, MMS-LID |
| CUDA | `make docker-build-cuda` | Same, with CUDA 12.8 torch wheels (`TORCH_VARIANT=cu128`, needed for RTX 50xx / Blackwell; `cu124` for older GPUs) |

The CUDA runtime ships inside the torch wheels, so the CUDA image uses the same slim base.
Models download on first use into `HF_HOME=/app/.cache/huggingface`. Compose mounts that as
the `model-cache` volume, so models survive restarts and image upgrades.

Images are published on version tags as `ghcr.io/prayaspanda/polymom:<version>` and
`ghcr.io/prayaspanda/polymom:<version>-cuda`. Both are scanned by Trivy, and the build fails
on any fixable CRITICAL vulnerability.

## Running with a GPU

1. Install the NVIDIA driver and the NVIDIA Container Toolkit. With Docker Desktop on
   Windows, use the WSL2 backend (GPU support is built in). Check with
   `docker run --rm --gpus all polymom:cuda python -c "import torch; print(torch.cuda.is_available())"`.
2. While logged in to Hugging Face, accept the terms of
   [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1),
   [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) and
   [ai4bharat/indic-conformer-600m-multilingual](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual),
   then set `HF_TOKEN` in `.env`.
3. `docker compose -f docker/docker-compose.yml --profile gpu up -d --build --scale worker-gpu=0`.

Memory: the gpu worker keeps Whisper large-v3, MMS-LID (loaded in float16 on CUDA,
about 2 GB) and pyannote resident. On an 8 GB card, float16 Whisper does not fit next to
the others: measured on an RTX 5060 Laptop (8 GB), `WHISPER_COMPUTE_TYPE=float16` ran out
of memory once LID was loaded. Set `WHISPER_COMPUTE_TYPE=int8_float16` (int8 weights,
float16 compute, about half the memory, with negligible accuracy loss) and keep
`QUEUE_CONCURRENCY_GPU=1`. With 12 GB or more, `auto` (float16) is fine. A CUDA out-of-memory
error frees the cache and retries the stage once on CPU (`polymom_gpu_oom_fallbacks_total`).
On CPU-only hosts, use `WHISPER_MODEL_SIZE=small` or `medium` with `WHISPER_COMPUTE_TYPE=int8`.

## Scaling

- **API**: stateless. Run several replicas behind a load balancer. Rate limits and
  idempotency keys live in Redis and Postgres.
- **gpu workers**: one per GPU with `QUEUE_CONCURRENCY_GPU=1`. Scale on
  `polymom_queue_depth{queue="gpu"}` and watch `polymom_stage_real_time_factor`.
- **cpu workers** (preprocess, align, analytics): cheap; scale on `queue_depth{queue="cpu"}`.
- **llm workers**: limited by provider rate limits. Raise `QUEUE_CONCURRENCY_LLM` until
  429 retries appear.
- **Storage**: switch to Postgres (`DATABASE_URL`) and `ARTIFACT_STORE=s3` as soon as more
  than one host runs workers, because checkpoints and processed audio must be visible to all
  of them.

A run moves between queues (cpu → gpu → cpu → llm) through continuation jobs, so each
resource pool scales independently. In a load test (Locust, mock backends, 50 users, one API
container on a laptop), the API handled 24.6 req/s with 0 failures and a p95 of 1.3 s. The
raw numbers are in [load/results_stats.csv](load/results_stats.csv).

## Reliability

| Failure | Behaviour |
| --- | --- |
| Transient error (LLM 429/5xx, network, stage or ffmpeg timeout) | Retried with exponential backoff (`RETRY_BACKOFF_SECONDS` × 2^n, capped at 10 min) up to `MAX_RETRIES`; resumes at the failed stage |
| Permanent error (corrupted media, validation, missing credentials) | Not retried; the meeting is `failed` with a code, message and remediation |
| Summarization fails | The meeting is `completed_with_errors`; transcript, speakers and analytics stay available |
| Stage exceeds `STAGE_TIMEOUTS` | `stage_timeout` (retryable) |
| GPU out of memory | Cache freed, stage retried once on CPU |
| Worker crash or kill | The heartbeat stops; the reaper re-queues runs that are silent for `STUCK_JOB_SECONDS`, and they resume from checkpoints |
| SIGTERM (deploy) | The worker stops at the next chunk or stage boundary, checkpoints and enqueues a continuation (`stop_grace_period` > `SHUTDOWN_GRACE_SECONDS`) |
| Cancel (`POST /cancel`) | The flag is checked between stages and chunks; the meeting becomes `cancelled` and finished work is kept |
| Webhook endpoint down | Retried up to `WEBHOOK_MAX_ATTEMPTS`; never changes the meeting status |

Every stage result is committed with a fingerprint: the input hash, the stage's settings
and model names, and the upstream fingerprints. A resumed run skips completed stages whose
fingerprint still matches. `POST /meetings/{id}/process?from_stage=summarize` reruns only
the tail, for example to try another LLM. The e2e suite exercises cancel, worker
restart/resume and LLM failure against the real compose stack (`make e2e-up && make e2e`).

## Monitoring

- **Metrics** at `/api/v1/metrics` (API) and `:9101/metrics` (each worker): requests and
  latency per route template, jobs by status, queue depth, stage duration and real-time
  factor, audio minutes processed, LLM tokens and cost, retries, OOM fallbacks and webhooks.
  `METRICS_ENABLED=false` hides the API endpoint.
- **Health**: `/api/v1/health/live` checks that the process is up. `/api/v1/health/ready`
  checks the database, Redis and the artifact store; model availability is reported but
  not required.
- **Logs**: JSON (structlog) outside development. Every line carries `request_id` (taken
  from, or echoed in, `X-Request-ID`), and pipeline lines add `meeting_id`, `run_id`,
  `stage` and `queue`.
- **Tracing**: with `OTEL_ENABLED=true` (`uv sync --extra otel`), spans for requests, jobs
  and stages are exported over OTLP/HTTP to `OTEL_EXPORTER_OTLP_ENDPOINT`. LLM calls can
  also be traced to Langfuse (`LANGFUSE_ENABLED=true`).
- **Dashboards**: `--profile monitoring` provisions Grafana with
  `docker/grafana/dashboards/polymom.json`.

## Retention

`scripts/cleanup.py` purges the raw uploads and processed audio of meetings older than
`RETENTION_DAYS` (default 90; `0` disables it). Transcripts, analytics, summaries and exports
are kept. `KEEP_RAW_AUDIO=true` keeps audio forever. It's a dry run by default:

```bash
docker compose -f docker/docker-compose.yml exec api python -m scripts.cleanup           # report
docker compose -f docker/docker-compose.yml exec api python -m scripts.cleanup --apply   # delete
```

Run it daily (cron or a Kubernetes CronJob). Once audio is purged,
`GET /meetings/{id}/audio` returns 409 `audio_not_available`; everything else keeps
working. `DELETE /meetings/{id}` removes a meeting, its runs and all its artifacts
immediately.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| Meeting `failed` with `diarization_model_unavailable` | `HF_TOKEN` is missing, the pyannote terms were not accepted for that account, or the image was built without `INSTALL_ML=true` |
| `unsupported_language`, or empty Odia text | IndicConformer terms not accepted, or `ASR_LANGUAGE_BACKENDS` maps `or` to `whisper` (Whisper has no Odia) |
| `audio_processing_failed` with an ffmpeg filter error | ffmpeg older than 6 on the host. The image ships ffmpeg 7 (Debian trixie); use the image or upgrade ffmpeg |
| Runs stay `queued` | No healthy worker for that queue: check `docker compose ps` and `polymom_queue_depth` |
| Status stuck on one stage after a crash | The reaper re-queues after `STUCK_JOB_SECONDS` (10 min); lower it for faster recovery |
| `completed_with_errors` and no summary | LLM unreachable or its key is invalid. Fix it, then `POST /process?from_stage=summarize` |
| Hindi transcribed in Urdu script | Whisper's language ID confused them. Keep `LID_BACKEND=mms` (routing forces `hi`) or pass `languages=hi` |
| 401 in the UI | Paste an API key into the box at the top. Keys are per tenant, so other keys' meetings are invisible |
| 413 on upload | Over `MAX_UPLOAD_MB` (checked while streaming), or a JSON body above `MAX_JSON_BODY_KB` |
| 429 | Per-key rate limit; `Retry-After` says when to retry. Raise `RATE_LIMIT_*` for batch jobs |
| Very slow on CPU | Use `WHISPER_MODEL_SIZE=small` or `medium` with `WHISPER_COMPUTE_TYPE=int8`, or a GPU worker |
| `make` missing on a Windows host | Run the compose commands from the Makefile directly, or use WSL |

## Configuration reference

Every setting is an environment variable (or a line in `.env`). `.env.example` lists all of
them with the same defaults, and a unit test keeps `.env.example` and `Settings` in sync.
List values are comma-separated; maps use `key:value,key:value`.

### Application

| Variable | Default | Description |
| --- | --- | --- |
| `APP_NAME` | `polymom` | Shown in `/health` and the OpenAPI title. |
| `APP_VERSION` | `1.0.0` | Defaults to the package version. |
| `APP_ENV` | `development` | `development`, `staging`, `production` or `test`. Non-development logs JSON; `production` validates secrets at startup. |
| `LOG_LEVEL` | `INFO` | Root log level. |

### ML providers (never commit real values)

| Variable | Default | Description |
| --- | --- | --- |
| `HF_TOKEN` | `(empty)` | Hugging Face token; required for the gated pyannote and IndicConformer models. |

### LLM summarization

| Variable | Default | Description |
| --- | --- | --- |
| `LLM_PROVIDER` | `openai` | `openai`, `azure`, `anthropic`, `ollama` or `mock`. |
| `LLM_MODEL` | `(empty)` | Empty = provider default: gpt-4o-mini, claude-sonnet-5, qwen2.5:7b-instruct |
| `LLM_API_KEY` | `(empty)` | API key for `openai`, `azure` or `anthropic`. |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint (also vLLM, LM Studio, ...). |
| `ANTHROPIC_BASE_URL` | `https://api.anthropic.com` | Anthropic API endpoint. |
| `AZURE_OPENAI_ENDPOINT` | `(empty)` | Azure OpenAI resource URL. |
| `AZURE_OPENAI_DEPLOYMENT` | `(empty)` | Azure deployment name. |
| `AZURE_OPENAI_API_VERSION` | `2024-10-21` | Azure OpenAI API version. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server (`http://ollama:11434` with `--profile local-llm`). |
| `LLM_TEMPERATURE` | `0.2` | Sampling temperature for summaries. |
| `LLM_MAX_RETRIES` | `2` | Repair attempts for invalid JSON, and retries on 429/5xx (exponential backoff) |
| `LLM_TIMEOUT_SECONDS` | `120` | Per-call timeout. |
| `LLM_MAX_OUTPUT_TOKENS` | `4096` | Output token cap per call. |
| `LLM_INPUT_COST_PER_MTOK` | `0` | USD per 1M tokens, for cost tracking (0 = not tracked) |
| `LLM_OUTPUT_COST_PER_MTOK` | `0` | USD per 1M output tokens (cost tracking). |
| `SUMMARY_OUTPUT_LANGUAGE` | `en` | en \| hi \| or  (evidence quotes always stay in the original language) |
| `SUMMARY_SINGLE_PASS_TOKENS` | `12000` | Transcripts above this estimate use map-reduce with chunks of SUMMARY_CHUNK_TOKENS |
| `SUMMARY_CHUNK_TOKENS` | `6000` | Map-reduce chunk size (estimated tokens). |
| `SUMMARY_CHUNK_OVERLAP_UTTERANCES` | `2` | Utterances repeated between consecutive map chunks |
| `EVIDENCE_MATCH_THRESHOLD` | `80` | rapidfuzz partial_ratio (0-100) an evidence quote needs against its utterance |
| `LANGFUSE_ENABLED` | `false` | Langfuse tracing (uv sync --extra tracing) |
| `LANGFUSE_PUBLIC_KEY` | `(empty)` | Langfuse public key. |
| `LANGFUSE_SECRET_KEY` | `(empty)` | Langfuse secret key. |
| `LANGFUSE_HOST` | `https://cloud.langfuse.com` | Langfuse server. |

### Storage / database

| Variable | Default | Description |
| --- | --- | --- |
| `STORAGE_DIR` | `storage` | Local artifact root and default SQLite location. |
| `DATABASE_URL` | `(empty)` | Leave empty for SQLite at STORAGE_DIR/polymom.db. Postgres example: postgresql+asyncpg://user:pass@localhost:5432/polymom |
| `AUTO_MIGRATE` | `true` | Run Alembic migrations on startup |

### Artifact storage (uploads, processed audio, charts, exports)

| Variable | Default | Description |
| --- | --- | --- |
| `ARTIFACT_STORE` | `local` | local (files under STORAGE_DIR) \| s3 (AWS S3 or MinIO; uv sync --extra s3) |
| `S3_BUCKET` | `polymom` | Bucket for `ARTIFACT_STORE=s3`. |
| `S3_ENDPOINT_URL` | `(empty)` | S3-compatible dev store: http://seaweedfs:8333 (docker compose --profile s3); empty for AWS S3 |
| `S3_ACCESS_KEY` | `(empty)` | S3 access key. |
| `S3_SECRET_KEY` | `(empty)` | S3 secret key. |
| `S3_REGION` | `(empty)` | S3 region. |
| `STAGE_OUTPUT_INLINE_MAX_BYTES` | `524288` | Stage outputs larger than this are stored as artifacts instead of in the DB |

### Retention (scripts/cleanup.py --apply)

| Variable | Default | Description |
| --- | --- | --- |
| `RETENTION_DAYS` | `90` | Purge raw audio of meetings older than this; results are kept. 0 = never |
| `KEEP_RAW_AUDIO` | `false` | true keeps uploads and processed audio forever. |

### Job queue and workers

| Variable | Default | Description |
| --- | --- | --- |
| `REDIS_URL` | `(empty)` | Redis for the arq queues, progress, cancel flags and rate limits |
| `PIPELINE_EXECUTION` | `queue` | queue (production) \| inline (tests and local debugging: runs in the API process) |
| `QUEUE_CONCURRENCY_CPU` | `4` | Concurrent jobs per cpu worker. |
| `QUEUE_CONCURRENCY_GPU` | `1` | Jobs per GPU worker; keep 1 per GPU. |
| `QUEUE_CONCURRENCY_LLM` | `4` | Concurrent jobs per llm worker. |
| `STAGE_TIMEOUTS` | `preprocess:1800,diarize:7200,identify_languages:3600,transcribe:10800,align:600,analytics:600,summarize:1800` | Per-stage limits in seconds; unspecified stages keep their defaults |
| `MAX_RETRIES` | `3` | Retries for transient errors only. |
| `RETRY_BACKOFF_SECONDS` | `10` | Base of the exponential retry backoff. |
| `STUCK_JOB_SECONDS` | `600` | Runs without a heartbeat for this long are re-queued. |
| `HEARTBEAT_SECONDS` | `15` | Workers refresh their job lease this often |
| `SHUTDOWN_GRACE_SECONDS` | `120` | Time a worker gets on SIGTERM to reach a checkpoint. |
| `WORKER_METRICS_PORT` | `9101` | Worker Prometheus port; 0 disables. |
| `MAX_AUDIO_DURATION_MINUTES` | `240` | Longer recordings are rejected (422 `audio_too_long`). |

### Uploads

| Variable | Default | Description |
| --- | --- | --- |
| `MAX_JSON_BODY_KB` | `1024` | Limit for JSON request bodies (413 payload_too_large above it) |

### Webhooks

| Variable | Default | Description |
| --- | --- | --- |
| `WEBHOOK_SECRET` | `(empty)` | HMAC-SHA256 key for X-Polymom-Signature; webhooks are skipped when empty |
| `WEBHOOK_TIMEOUT_SECONDS` | `10` | Per-delivery timeout. |
| `WEBHOOK_MAX_ATTEMPTS` | `5` | Delivery attempts (exponential backoff). |
| `WEBHOOK_ALLOW_PRIVATE_HOSTS` | `false` | Allow callbacks to private/loopback hosts (local testing only; refused in production) |

### Security

| Variable | Default | Description |
| --- | --- | --- |
| `API_KEY_REQUIRED` | `true` | Require `X-API-Key` (forced on in production). |
| `RATE_LIMIT_DEFAULT` | `120/minute` | Per API key, all authenticated routes. |
| `RATE_LIMIT_UPLOAD` | `10/minute` | Per API key: upload, process, regenerate. |
| `CORS_ORIGINS` | `(empty)` | Comma-separated browser origins allowed by CORS (empty = none) |

### Tracing (uv sync --extra otel)

| Variable | Default | Description |
| --- | --- | --- |
| `OTEL_ENABLED` | `false` | OpenTelemetry tracing (`uv sync --extra otel`). |
| `OTEL_SERVICE_NAME` | `polymom` | Service name on spans. |

### Observability

| Variable | Default | Description |
| --- | --- | --- |
| `METRICS_ENABLED` | `true` | Serve Prometheus metrics at /api/v1/metrics |

### Uploads

| Variable | Default | Description |
| --- | --- | --- |
| `MAX_UPLOAD_MB` | `200` | Upload size limit, enforced while streaming. |
| `UPLOAD_CHUNK_BYTES` | `1048576` | Streaming write size for uploads |
| `ALLOWED_EXTENSIONS` | `aac,flac,m4a,mkv,mov,mp3,mp4,ogg,wav,webm` | Accepted file types (magic bytes must match). |
| `FFPROBE_PATH` | `ffprobe` | ffprobe binary. |
| `FFPROBE_TIMEOUT_SECONDS` | `30` | Probe timeout per upload. |

### Preprocessing

| Variable | Default | Description |
| --- | --- | --- |
| `FFMPEG_PATH` | `ffmpeg` | ffmpeg binary. |
| `FFMPEG_TIMEOUT_SECONDS` | `1800` | Per-run ffmpeg timeout. |
| `TARGET_SAMPLE_RATE` | `16000` | Output sample rate of preprocessing. |
| `TARGET_LOUDNESS_LUFS` | `-23` | EBU R128 loudness target. |
| `ENABLE_HIGHPASS` | `true` | High-pass filter against rumble. |
| `HIGHPASS_CUTOFF_HZ` | `80` | High-pass cutoff. |
| `ENABLE_DENOISE` | `false` | Light FFT denoise (`afftdn`). |
| `TRIM_SILENCE` | `false` | Cut leading/trailing silence (always reported). |
| `SILENCE_THRESHOLD_DB` | `-50` | `silencedetect` threshold. |
| `SILENCE_MIN_DURATION_SECONDS` | `0.5` | Minimum silence span. |

### Chunking (long recordings)

| Variable | Default | Description |
| --- | --- | --- |
| `CHUNK_LENGTH_SECONDS` | `1800` | Chunk length for long recordings. |
| `CHUNK_OVERLAP_SECONDS` | `5` | Overlap between chunks. |

### Diarization

| Variable | Default | Description |
| --- | --- | --- |
| `DIARIZATION_BACKEND` | `pyannote` | pyannote (needs `uv sync --extra ml` + HF_TOKEN) \| mock (deterministic fake, no ML) |
| `DIARIZATION_MODEL` | `pyannote/speaker-diarization-3.1` | pyannote pipeline id. |
| `DEVICE` | `auto` | `auto`, `cpu` or `cuda`. |
| `MERGE_GAP_SECONDS` | `0.5` | Join same-speaker turns separated by shorter pauses. |
| `MIN_TURN_SECONDS` | `0.3` | Drop shorter turns (never a speaker's only trace). |
| `DIARIZATION_CHUNK_THRESHOLD_SECONDS` | `3600` | Recordings longer than this are diarized in CHUNK_LENGTH_SECONDS chunks and re-linked |
| `SPEAKER_SIMILARITY_THRESHOLD` | `0.6` | Cosine similarity needed to treat speakers in different chunks as the same person |

### Speech recognition

| Variable | Default | Description |
| --- | --- | --- |
| `ASR_BACKEND` | `real` | real (faster-whisper + IndicConformer) \| mock (scripted en/hi/or lines, no ML) |
| `WHISPER_MODEL_SIZE` | `large-v3` | faster-whisper size: `small`/`medium` on CPU, `large-v3` on GPU. |
| `WHISPER_COMPUTE_TYPE` | `auto` | auto (float16 on GPU, int8 on CPU) \| int8 \| int8_float16 \| float16 \| float32 |
| `ODIA_MODEL_ID` | `ai4bharat/indic-conformer-600m-multilingual` | IndicConformer checkpoint for Odia. |
| `ODIA_DECODING` | `rnnt` | rnnt (more accurate) \| ctc (faster) |
| `ASR_LANGUAGE_BACKENDS` | `en:whisper,hi:whisper,or:indic` | language:backend routing; backends: whisper, indic |
| `ASR_BEAM_SIZE` | `5` | Whisper beam size. |
| `ASR_VAD_FILTER` | `true` | Silero VAD inside Whisper. |
| `ASR_LOW_CONFIDENCE_THRESHOLD` | `0.5` | Segments with average word confidence below this are flagged low_confidence |
| `ASR_COMPRESSION_RATIO_THRESHOLD` | `2.4` | Drop looping segments above this gzip ratio. |
| `ASR_NO_SPEECH_THRESHOLD` | `0.6` | Treat segments as silence above this no-speech probability. |

### Spoken language ID and code-switching

| Variable | Default | Description |
| --- | --- | --- |
| `SUPPORTED_LANGUAGES` | `en,hi,or` | Languages LID may choose. |
| `LANGUAGE_ROUTING_ENABLED` | `true` | Route each language region to its ASR backend (false = one ASR pass over the whole meeting) |
| `LID_BACKEND` | `mms` | mms (en/hi/or, CC-BY-NC-4.0) \| speechbrain (en/hi, Apache-2.0) \| whisper (en/hi) \| mock |
| `LID_MODEL_ID` | `facebook/mms-lid-126` | Model for `LID_BACKEND=mms` (126 languages incl. Odia; CC-BY-NC-4.0 weights). |
| `SPEECHBRAIN_LID_MODEL_ID` | `speechbrain/lang-id-voxlingua107-ecapa` | Model for `LID_BACKEND=speechbrain`. |
| `LID_MIN_CONFIDENCE` | `0.5` | Below this a window is `uncertain` and inherits context. |
| `LID_MIN_WINDOW_SECONDS` | `1.5` | Shortest window scored on its own. |
| `LID_MAX_WINDOW_SECONDS` | `15` | Longer turns are split into windows. |

### Speaker / transcript alignment

| Variable | Default | Description |
| --- | --- | --- |
| `ALIGN_MAX_GAP_SECONDS` | `1` | Words overlapping no turn go to the nearest turn within this gap, else "Unknown" |
| `ALIGN_MERGE_GAP_SECONDS` | `1` | Fragments shorter than UTTERANCE_MIN_WORDS merge into the same speaker within this gap |
| `UTTERANCE_MAX_SECONDS` | `30` | Longer utterances are split at sentence punctuation (. ? ! । ॥) |
| `UTTERANCE_MIN_WORDS` | `2` | Shorter fragments merge into neighbours. |

### Speaker analytics

| Variable | Default | Description |
| --- | --- | --- |
| `INTERRUPTION_MIN_OVERLAP_SECONDS` | `0.5` | A turn starting inside another speaker's turn is an interruption above this overlap |
| `BUCKET_SECONDS` | `60` | Timeline window for per-speaker speaking time |
| `GINI_BALANCED_MAX` | `0.2` | Gini of speaking time: <= balanced max is "balanced", >= dominated min is "dominated" |
| `GINI_DOMINATED_MIN` | `0.4` | Gini at or above this is "dominated". |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `(empty)` | Read by the OpenTelemetry SDK. |
