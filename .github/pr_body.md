## Summary

Prompt 4 of 12 adds speaker diarization as the second pipeline stage, after preprocessing. The stage labels every speaker **Person 1, Person 2, ...** consistently across the whole meeting, marks overlapping speech rather than dropping it, and splits long recordings into chunks whose speakers are re-linked by voice embeddings. Results are served by `GET /api/v1/meetings/{id}/speakers`.

The backend is behind an interface: `pyannote` (real model, optional `ml` extra) or `mock` (deterministic fake, used by tests and on machines without a GPU or HF token). No ASR yet.

## Model choice and rationale

**`pyannote/speaker-diarization-3.1`:**
- **Accurate and widely used:** it's the most widely adopted open diarization pipeline, with strong published results.
- **Pure PyTorch:** 3.0 depended on `onnxruntime`.
- **Language-agnostic:** it separates voices, not words, which suits Hindi, Odia and code-mixed meetings.
- **Handles overlap natively.**
- **Accepts speaker-count hints:** `num_speakers`, `min_speakers` and `max_speakers`, which lets us pass the upload's `expected_speakers`.
- **Returns per-speaker embeddings,** which is what makes cross-chunk re-linking possible without a second embedding model.
- **Hardware:** it runs on CPU and is fast on GPU.
- **MIT licensed:** the model is gated behind accepting its terms, which the README walks through.

**Dependencies:**
- The `ml` extra pins `torch`/`torchaudio` below 2.9, because torchaudio 2.9 removed I/O APIs that pyannote.audio 3.x uses. The lock resolved to pyannote.audio 3.4.0 and torch 2.8.0.
- As a second safeguard, audio is passed to pyannote as an in-memory waveform read with the stdlib `wave` module, so torchaudio's file I/O is never used.
- torch comes from the PyTorch CPU index by default.

## Changes

- **Dependencies:**
  - the optional `ml` extra (`uv sync --extra ml` or `make install-ml`); the base install and CI stay light
  - torch and pyannote are imported lazily, and mypy treats them as untyped, so type checks pass without them installed
- **`app/services/diarization/`:**
  - `base.py`: the `DiarizationBackend` ABC. Backends implement `diarize_raw` (segments plus embeddings); the shared `diarize` applies post-processing.
  - `pyannote_backend.py`:
    - the pipeline is a lazy singleton per (model, device), with thread-safe loading and inference serialized by a lock
    - `DEVICE` can be `auto`, `cpu` or `cuda`, and inference runs in a thread pool
    - errors are typed and say what to fix: missing token, terms not accepted (pyannote returns `None`), download failure, missing extras, CUDA unavailable
  - `mock_backend.py`: deterministic alternating speakers, with embeddings that identify the true speaker (so re-linking can be tested) and a "no speech" path.
  - `postprocess.py`: pure functions `merge_gaps`, `drop_short_turns`, `find_overlaps` (sweep-line), `first_appearance_labels`, `build_result` and `relink_chunks`.
  - `service.py`: picks whole-file or chunked mode by duration, uses the existing chunker, cleans up chunk files, and includes `build_backend` (the factory).
- **Pipeline:** `DiarizationStage` is registered after `PreprocessStage`. `PipelineContext.expected_speakers` comes from the upload.
- **API:** `GET /api/v1/meetings/{id}/speakers` returns the turns, speaker count, speaker list and overlap regions. It returns 409 `diarization_not_available` before diarization and 404 for an unknown meeting.
- **Dependency injection:** `get_diarization_backend` in `deps.py` picks the backend from `DIARIZATION_BACKEND`.
- **Errors:** `DiarizationError` (500 `diarization_failed`), `DiarizationModelLoadError` (503 `diarization_model_unavailable`) and `DiarizationNotAvailableError` (409).
- **Config:**
  - `DIARIZATION_BACKEND`, `DEVICE`, `DIARIZATION_MODEL`
  - `MERGE_GAP_SECONDS`, `MIN_TURN_SECONDS`
  - `DIARIZATION_CHUNK_THRESHOLD_SECONDS`, `SPEAKER_SIMILARITY_THRESHOLD`
- **Docker:**
  - `INSTALL_ML=true` installs the `ml` extra; `TORCH_VARIANT=cu124` (for example) swaps in CUDA wheels of the locked torch version
  - `HF_HOME` and `TORCH_HOME` point into `/app/.cache`, which the compose file mounts as the `model-cache` volume
- **Makefile:** new targets `install-ml`, `test-slow` and `docker-build-ml`.
- **Docs:** a README diarization section covering the model choice, Hugging Face setup, label consistency, overlaps, long recordings, known limitations and config.

### Persistence: a `diarization` JSON column, not a `speaker_turns` table

Migration `0003` adds a JSON column. Diarization is written in one go at the end of the stage, always read whole (by `/speakers` and by the ASR alignment coming in Prompt 6), and replaced whole when a meeting is reprocessed with `force`. A JSON document fits that pattern: saving is one atomic update rather than a delete plus N inserts, there are no joins, and it matches the existing `audio_quality` column. It also works on SQLite and on Postgres, where it could later become JSONB. The trade-off is that you can't query individual turns across meetings in SQL. If analytics later needs that, a `speaker_turns` table can be generated from this column without changing the API.

## Test coverage

137 tests pass, plus 1 slow real-model test that is deselected by default. Overall coverage is 96%; for the new code:

| Module | Coverage |
| --- | --- |
| `postprocess.py` | 99% |
| `service.py` | 100% |
| `mock_backend.py` | 100% |
| `base.py` | 100% |
| `pyannote_backend.py` | 89% (the uncovered lines are the real waveform loader, which needs the `ml` extras) |
| `mom_pipeline.py` | 99% |
| `meeting_service.py` | 100% |
| `routes/meetings.py` | 100% |

- **Post-processing unit tests:**
  - labels follow first appearance, not raw label order
  - gap merging, including that merging is per speaker across interjections
  - short turns are dropped, but a speaker's only turn and "longest when all are short" are kept
  - overlap detection, including touching turns and three-way overlaps
  - timestamps are rounded to 3 decimals
  - single-speaker audio and audio with no speech
- **Re-linking with synthetic embeddings:**
  - speakers swap local labels between chunks but keep their global identity
  - a new speaker below the threshold gets a new label, and person numbers stay stable in later chunks
  - matching is one-to-one within a chunk
  - missing or NaN embeddings are handled
  - overlap-window de-duplication
  - cosine similarity edge cases
- **Backend tests:**
  - the mock backend and the backend factory
  - the chunked service with the mock (consistent labels over 60 s in 20 s chunks, chunk files cleaned up)
  - no chunking below the threshold
  - the pyannote backend with **faked `torch` and `pyannote.audio` modules**: kwargs passed through, loaded once then cached, device selection, missing token, gated model, download failure, missing extras, CUDA unavailable, inference failure, embedding order
- **Integration tests** (mock backend):
  - upload, process, then `/speakers`
  - `expected_speakers=3` produces 3 speakers
  - silent audio completes with 0 speakers
  - 409 before diarization and 404 for an unknown meeting
  - the pyannote backend without `HF_TOKEN` marks the meeting `failed` with an actionable `diarization_model_unavailable: HF_TOKEN is not set...` while keeping `audio_quality`
- **Real model test:** `@pytest.mark.slow` and skipped without `HF_TOKEN` or the `ml` extras. `addopts` excludes `slow`, so CI runs only the fast tests; run it with `make test-slow`.

## Known limitations

- **Real model not run by me:** it hasn't run against pyannote 3.1 in this environment, because I have no `HF_TOKEN` here. The integration code is tested against fakes that follow pyannote's API, and the ML Docker image builds and imports it. Please run `make install-ml && HF_TOKEN=... make test-slow` once.
- **Speed:** CPU inference is slow for long meetings (tens of minutes per hour of audio, depending on the CPU); a GPU is strongly recommended. Inference runs in the API process until the worker queue arrives in Prompt 11.
- **Speaker accuracy:** very brief or quiet speakers may be missed, and similar voices may be merged or split, particularly across chunks (tune `SPEAKER_SIMILARITY_THRESHOLD`). Laughter and music can be attributed to a speaker.
- **Chunk boundaries:** chunks receive `expected_speakers` as a maximum, not an exact count, so the result may contain fewer speakers than expected.
- **Mock vs. default:** the mock backend is only for development and tests. The default is `pyannote`, so a misconfigured deployment fails loudly instead of producing fake speakers.

## Checklist

- [x] `make lint` passes
- [x] `make typecheck` (mypy strict) passes, without the ML extras installed
- [x] `make test` passes, with coverage of new code at least 85%
- [x] Docker build passes without the `ml` extras
- [x] Docker build passes with `INSTALL_ML=true` (CPU torch), and pyannote imports in the image
- [x] Migration 0003 applies on top of 0002, and `alembic check` is clean
- [x] No secrets or model weights committed

## Next steps

**Prompt 5: multilingual ASR.** Add an ASR stage (Whisper / faster-whisper) for English, Hindi, Odia and code-mixed speech on `processed_path`, chunked the same way for long meetings. It produces word-level timestamps that Prompt 6 will align with these speaker turns.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
