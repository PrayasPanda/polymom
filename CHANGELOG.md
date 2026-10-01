# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-09-30

First public release. It consolidates the twelve development milestones (PRs #1–#14).

### Added

- **Project scaffold** (milestone 1): FastAPI app factory, typed settings
  (pydantic-settings), structured logging (structlog), error hierarchy with a stable JSON
  envelope, health endpoint, uv project with ruff, mypy `--strict`, pytest, pre-commit,
  multi-stage Dockerfile and CI.
- **Upload API** (2): streamed multipart upload with a size limit, extension allow-list,
  magic-byte and ffprobe validation, Unicode-safe filenames, SQLAlchemy async models with
  Alembic migrations, and meeting list, get and delete.
- **Audio preprocessing** (3): ffmpeg conversion to 16 kHz mono (high-pass, optional
  denoise and silence trim, EBU R128 loudness normalization), quality analysis (loudness,
  RMS, peak, silence, with warnings), overlapping chunker for long recordings, and a
  stage-based pipeline.
- **Speaker diarization** (4): pyannote 3.1 stage with consistent `Person N` labels,
  turn post-processing, overlap regions, and chunked diarization with embedding-based
  speaker re-linking; mock backend for CI.
- **Multilingual ASR** (5): routed ASR, with faster-whisper large-v3 for English and Hindi
  and AI4Bharat IndicConformer for Odia; native script with NFC normalization,
  hallucination guards, low-confidence flags, chunk merging, SRT output and a WER/CER
  evaluation script.
- **Language ID and code-switching** (6): spoken LID with MMS-LID-126 on turn windows,
  smoothing, per-region ASR routing, word-level code-mix tagging, and a `/languages`
  endpoint with switch points.
- **Speaker-attributed transcript** (7): word-to-speaker alignment with overlap and gap
  handling, utterances, txt/srt/vtt/md formats, speaker renaming, and a DER/WDER
  evaluation script.
- **Speaker analytics** (8): per-speaker talk time, turns, interruptions, WPM, questions
  and languages; meeting balance (Gini), overlap, silence, timeline, CSV export and PNG
  charts.
- **LLM summarization** (9): grounded minutes of meeting (executive summary, decisions,
  action items with owner and due date, open questions); OpenAI, Azure, Anthropic, Ollama
  and mock providers; map-reduce for long meetings; evidence verifier; prompt-injection
  flags; Langfuse tracing; and a summary evaluation script.
- **Storage and retrieval** (10): processing runs and normalized results, local and S3
  artifact stores, a consolidated `MeetingResult` with a published JSON Schema, docx, pdf,
  md and json exports with Devanagari and Odia fonts, full-text search, filters and cursor
  pagination, SHA-256 deduplication, and retention cleanup.
- **Workers and hardening** (11): arq queues per resource (cpu, gpu, llm), resumable and
  checkpointed runs, cancel, SSE progress, HMAC-signed webhooks with SSRF protection,
  hashed API keys with tenant isolation, rate limits, idempotency keys, Prometheus
  metrics, OpenTelemetry tracing, a Grafana dashboard and a load test.
- **Release readiness** (12):
  - Web UI at `/` (Jinja2 + HTMX, no frontend build): drag-and-drop upload with title,
    expected speakers and language hints; meetings list with live status; detail page with
    live progress, cancel, audio player and click-to-seek transcript, colour-coded
    speakers, speaker renaming, SVG analytics charts, MoM with decisions and action items,
    and export buttons. Noto fonts for Devanagari and Odia, dark mode, responsive layout,
    keyboard access, strict CSP. It uses the same API-key auth as the API.
  - `GET /meetings/{id}/audio` for playback.
  - End-to-end suite (`tests/e2e`) against the docker compose stack: full flow, SSE, all
    formats and exports, search, delete, failure flows (corrupted file, unsupported type,
    video without audio, over-limit duration, LLM failure producing
    `completed_with_errors`, cancel mid-run, resume after a worker restart) and a JSON
    Schema contract test.
  - Real evaluation: `scripts/prepare_eval_data.py` (FLEURS, MUCS 2021 Hindi-English, AMI),
    `scripts/make_codemixed_meeting.py` (synthetic multi-speaker code-mixed meetings with
    RTTM, transcript and language ground truth, with overlaps) and
    `scripts/run_benchmark.py` (WER/CER per language, code-mixed CER, DER, cpWER, LID and
    speaker-count accuracy, summary precision/recall and grounding, real-time factor),
    with results in `docs/evaluation/`.
  - `make demo`: one command that starts the stack, creates an API key, processes a sample
    meeting and prints the UI URL.
  - Compose profiles `gpu` (CUDA worker), `s3` (SeaweedFS), `local-llm` (Ollama) and
    `monitoring`; an e2e overlay.
  - CUDA image variant (`TORCH_VARIANT=cu128`, Blackwell-ready).
  - Workflows: `e2e.yml`, `docker.yml` (build, Trivy, GHCR push on tags) and `release.yml`;
    Dependabot for uv, Docker, compose and Actions; a coverage gate (85%) with upload.
  - Documentation: README, ARCHITECTURE, TECHNOLOGY_CHOICES, MULTILINGUAL, API (with
    `docs/openapi.json`), OPERATIONS (full configuration reference), PIPELINE, SECURITY,
    CONTRIBUTING and CODE_OF_CONDUCT; sample outputs in `docs/samples/`.

### Changed

- Base images are pinned by digest (Python 3.11.16 slim trixie with ffmpeg 7, uv 0.9.30);
  `libmagic` was removed from the runtime image; `apt-get upgrade` runs at build time.
- Workers have their own health check (metrics port) instead of inheriting the API's.
- The MinIO profile was replaced by SeaweedFS, because MinIO stopped publishing community
  images.
- The evaluation scripts compute WER/CER with rapidfuzz, a base dependency, so they run in
  the production image; `jiwer` was removed.

### Fixed

- The SSE progress stream for an already finished meeting now closes after the final
  snapshot instead of staying open.
- `MAX_JSON_BODY_KB` and `METRICS_ENABLED` were defined but had no effect; both are now
  enforced.
- Summaries from constrained-decoding providers (found with Ollama and
  qwen2.5:7b-instruct during the benchmark) came back without decisions, action items or
  key points. List fields with defaults were optional in the JSON Schema sent to the
  model, so the decoder could stop after the header fields. Every property is now
  required.
- Ollama requests set `num_ctx`. The 2048-token default silently truncated longer
  transcripts.
- MMS-LID loads in float16 on CUDA, so it fits next to Whisper large-v3 on 8 GB GPUs
  (previously CUDA out of memory).
- `/health/ready` and the new startup warning treated an empty `LLM_API_KEY=` or
  `HF_TOKEN=` as configured.

### Removed

- Dead code: unused helpers (`describe`, `speaker_profiles`, `schema_text`, `uploads_dir`,
  `OwnerIdDep`, `DELIVERY_TTL_SECONDS`) and stale references to development milestones in
  code comments.

### Security

- transformers 4.x advisories are accepted, with rationale and a follow-up plan in
  [docs/SECURITY.md](docs/SECURITY.md#known-accepted-risks).

[Unreleased]: https://github.com/PrayasPanda/polymom/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/PrayasPanda/polymom/releases/tag/v1.0.0
