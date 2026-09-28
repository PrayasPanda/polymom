## Summary

Prompt 3 of 12 adds the audio preprocessing stage. Any validated upload (audio, or video that has an audio track) now becomes a model-ready **mono, 16 kHz, 16-bit PCM WAV** at `STORAGE_DIR/processed/{meeting_id}.wav`. The stage normalizes loudness to EBU R128, can apply an optional highpass and denoise, and reports silence and quality. The pipeline is now built from stages, and `POST /api/v1/meetings/{id}/process` runs it in the background. Diarization and ASR are not part of this PR.

## Design decisions

- **Stage pattern:** each `PipelineStage` has a `name`, an async `run(context)` and an `apply(context, meeting)`.
  - `run` does the work and writes its results to the shared `PipelineContext`; `apply` copies what should be saved onto the meeting.
  - Keeping the two apart means stages never touch the database themselves.
  - The orchestrator owns the status changes (`processing` → `completed`/`failed`) and logs each stage's timing.
- **Status is set to `processing` in the request itself, before the background task starts,** so a second click gets a 409 instead of running the pipeline twice.
  - Failed meetings can be retried without `force`.
  - Background work opens its own short-lived database sessions, because the request's session is closed by the time the task runs.
- **Quality is measured on the original audio.** Every normalized file ends up at about −23 LUFS, so RMS, peak and LUFS measured after normalization would say nothing about how the meeting was recorded, and clipping could no longer be detected.
  - One ffmpeg pass (`silencedetect` + `astats` + `ebur128`) gathers all of these; it runs before conversion, and conversion uses its silence spans for optional trimming.
- **Single-pass `loudnorm`, not two-pass:** for speech it's close enough, and it halves the ffmpeg work on long recordings. `loudnorm` upsamples internally, so `aresample` runs after it.
- **Atomic output:** ffmpeg writes to `{id}.wav.part`, which is renamed only after the WAV header checks out.
  - A timeout, ffmpeg error or bad output deletes the partial file.
  - The ffmpeg runner kills the child process on timeout or cancellation.
- **Warnings don't fail processing:** `low_volume`, `clipping`, `too_short` and `mostly_silent` are returned in `audio_quality.warnings`.
- **The chunker uses the stdlib `wave` module rather than ffmpeg:** slices are sample-accurate with no re-encoding, and each chunk records its absolute offset. As the spec asks, it's exposed and tested but not wired in yet.
- **Error mapping:**
  - `AudioProcessingError` → 422 `audio_processing_failed`
  - `FFmpegTimeoutError` → 504 `ffmpeg_timeout`
  - `MeetingStateConflictError` → 409

  In the background path these are stored on the meeting as `"<code>: <message>"`.

## Changes

- `app/services/audio/`:
  - `ffmpeg.py`: async subprocess runner with a timeout.
  - `analysis.py`: silence and level analysis, and the warnings.
  - `preprocessor.py`: the conversion itself.
  - `chunker.py`: `plan_chunks` and `split_wav`.
- `app/schemas/audio.py`: `AudioWarning`, `AudioQuality`, and `PreprocessResult` (which extends `AudioQuality` with `processed_path`). `MeetingRead` gains `audio_quality`; `processed_path` stays internal.
- `app/pipelines/mom_pipeline.py`: `PipelineContext`, `PipelineStage`, `PreprocessStage`, the `MoMPipeline` orchestrator and `build_pipeline`.
- **Endpoint:** `POST /api/v1/meetings/{id}/process?force=`, documented in OpenAPI with 404 and 409 examples. `DELETE` now also removes the processed file.
- **Storage:** migration `0002` adds `processed_path` and `audio_quality` (a JSON column). `MeetingRepository.save` is new. `alembic check` reports no drift.
- **Config** (in `.env.example` too):
  - `TARGET_SAMPLE_RATE`, `TARGET_LOUDNESS_LUFS`
  - `ENABLE_HIGHPASS`, `HIGHPASS_CUTOFF_HZ`, `ENABLE_DENOISE`
  - `TRIM_SILENCE`, `SILENCE_THRESHOLD_DB`, `SILENCE_MIN_DURATION_SECONDS`
  - `FFMPEG_PATH`, `FFMPEG_TIMEOUT_SECONDS`
  - `CHUNK_LENGTH_SECONDS`, `CHUNK_OVERLAP_SECONDS`
- **Docs:** the README covers the pipeline stage pattern (Mermaid diagram), the preprocessing steps, the warnings, `audio_quality` and the config flags. It also fixes a broken curl line continuation.

## Test coverage

90 tests pass, with 95% coverage overall. For the new code:

| Module | Coverage |
| --- | --- |
| `mom_pipeline.py` | 100% |
| `chunker.py` | 100% |
| `meeting_service.py` | 100% |
| `analysis.py` | 99% |
| `preprocessor.py` | 97% |
| `ffmpeg.py` | 86% (the untested lines are the cancellation branch) |

**Fixtures,** generated with ffmpeg `lavfi` at test time (no binaries committed):
- a stereo 44.1 kHz tone
- an mp4 video with an AAC audio track
- a near-silent file (about −70 dBFS)
- a clipped file
- a 1 s file
- a file padded with 1 s of silence at each end
- a 3-minute file

**Unit tests** check that:
- output is mono, 16 kHz and PCM s16, and the original is byte-for-byte unchanged
- audio is extracted from video
- a custom sample rate works and denoise runs
- each warning fires exactly when it should
- leading and trailing silence are reported but only trimmed when the flag is on
- the chunk plan is correct, and 3-minute chunks have sample-exact offsets and overlaps
- the filter chain respects the config flags
- a timeout kills the process and raises `FFmpegTimeoutError`, and non-zero exits and a missing binary are handled
- partial output is cleaned up after a timeout or an invalid WAV
- the orchestrator runs stages in order, records typed and unexpected failures, and leaves later stages unrun after a failure

**Integration tests** check that:
- upload, then process, then GET shows `completed` with `audio_quality` filled in
- a second `process` call gets a 409, and `force=true` reprocesses
- `process` on an unknown meeting gets a 404
- a missing ffmpeg records `failed` with an `audio_processing_failed: ...` error, and the meeting can then be retried
- delete removes the processed file

## Checklist

- [x] `make lint` (ruff check + format) passes
- [x] `make typecheck` (mypy strict) passes
- [x] `make test` passes, with coverage of new code at least 85%
- [x] Docker image builds
- [x] Migration 0002 applies on top of 0001, and `alembic check` is clean
- [x] The original upload is never modified; partial output is removed on failure
- [x] No large binaries or secrets committed

## Next steps

**Prompt 4: speaker diarization.** Add a `DiarizationStage` that runs pyannote (as an optional `diarization` dependency group) on `context.processed_path`, using the chunker for long recordings and `expected_speakers` as a hint. It will persist speaker segments.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
