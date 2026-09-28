## Summary

Prompt 5 of 12 adds multilingual speech-to-text for **English, Hindi and Odia** as a third pipeline stage (preprocess → diarize → **transcribe**). It produces a standalone timestamped transcript: segments with word timings, confidence, language and backend, in each language's native script, NFC-normalized. The transcript is served by `GET /api/v1/meetings/{id}/transcript` as JSON, plain text or SRT. Aligning words to speakers is out of scope and comes later.

ASR is **routed per language** to interchangeable backends:
- faster-whisper for English and Hindi
- AI4Bharat IndicConformer for Odia
- a deterministic mock for tests

## Model choices and rationale

I checked model availability and licenses on the Hugging Face API before wiring anything in.

| Language | Model | License / access | Why |
| --- | --- | --- | --- |
| en, hi | `Systran/faster-whisper-large-v3` (size configurable) | MIT, not gated | Strongest open model for English and Hindi, with built-in language ID, native word timestamps and Silero VAD. The CTranslate2 build is up to 4× faster and uses less memory (int8 on CPU, float16 on GPU). |
| or | `ai4bharat/indic-conformer-600m-multilingual` (configurable via `ODIA_MODEL_ID`) | MIT, gated (terms auto-approved) | Whisper doesn't support Odia. This is AI4Bharat's current maintained multilingual IndicConformer (updated Feb 2026, about 580k downloads) and covers Odia. It ships as **ONNX with a transformers remote-code loader, so NeMo isn't needed**, and the new `indic` extra is just `transformers` + `onnxruntime`. |

**Alternatives considered:**
- `ai4bharat/indicwav2vec-odia` (wav2vec2 CTC, Apache-2.0): it would give native CTC word offsets, but it's from 2023, has around 40 downloads, and predates IndicConformer's accuracy gains. Since the model id is configurable, a CTC word-offset backend could be added later if word-level Odia timing becomes critical.
- `ai4bharat/IndicConformer-or` (a per-language checkpoint): not available on the Hub (the API returned an auth error).

**Odia timestamps:** IndicConformer returns text only. Segments are cut at energy-based speech regions (accurate to one 30 ms frame, split at 20 s), and each is transcribed separately. **Word timings are approximated** in proportion to word length within each segment, and there are no confidence scores. This trade-off is documented in the README.

**Not verified:** I couldn't read the gated IndicConformer model card without a token. The call signature `model(wav, "or", "ctc"|"rnnt") -> str` is based on AI4Bharat's documented usage and on the remote-code files visible in the repo (`model_onnx.py`, the CTC and RNNT decoders). The slow test confirms it once `HF_TOKEN` is available.

## Changes

- **Dependencies:**
  - the `ml` extra gains `faster-whisper`
  - new `indic` extra (`transformers`, `onnxruntime`)
  - `jiwer` in the dev group
  - all ML imports are lazy, so the base install and CI stay light
- **`app/services/asr/`:**
  - `base.py`: the `ASRBackend` ABC, with `transcribe(audio_path, language, offset)`, `supported_languages` and `supports_auto_detect`.
  - `whisper_backend.py`:
    - model loaded lazily as a thread-safe singleton per (size, device, compute type)
    - `DEVICE=auto|cpu|cuda`, and `compute_type` set to `auto` (float16 on CUDA, int8 on CPU)
    - runs in a worker thread with word timestamps, VAD and `condition_on_previous_text=False`
  - `indic_backend.py`: IndicConformer (lazy, thread-safe), energy-based segmentation and approximate word timings.
  - `mock_backend.py`: scripted en/hi/or lines. Indic lines are emitted in NFD on purpose, so NFC normalization is tested end to end.
  - `router.py`: `ASRRouter`, driven by `ASR_LANGUAGE_BACKENDS`. It's validated at construction, raises `UnsupportedLanguageError` with the fix in the message, and is ready for per-segment routing in Prompt 6.
  - `service.py`:
    - language strategy: one hint → forced; none or several → Whisper auto-detect
    - chunking with offset shifting
    - central post-processing
    - `build_router` (the factory for `ASR_BACKEND=real|mock`)
  - `postprocess.py`, all pure functions:
    - NFC normalization, preserving native script
    - hallucination guards
    - low-confidence flags
    - `merge_chunks`: overlap de-duplication by midpoint ownership, time overlap and fuzzy text match
    - `approximate_words`
  - `segmentation.py`: pure speech-region detection.
- **Schemas** (`app/schemas/asr.py`):
  - `Word`
  - `TranscriptSegment`, with `low_confidence`, `no_speech_prob` and `compression_ratio` diagnostics
  - `LanguageDuration`
  - `ASRResult`, with `detected_languages`, `model_names` and `requested_language`
  - `TranscriptResponse`
- **Pipeline:** `TranscriptionStage` is registered after `DiarizationStage` and receives the upload's `languages_hint` through `PipelineContext`.
- **Storage:** migration `0004` adds a `transcript` JSON column, following the JSON-column choice from Prompt 4. `alembic check` reports no drift.
- **API:** `GET /api/v1/meetings/{id}/transcript?format=json|txt|srt`.
  - `txt` returns `text/plain` lines like `[HH:MM:SS.mmm] (lang) text`.
  - `srt` returns `application/x-subrip` as an attachment.
  - It returns 409 `transcript_not_available` before transcription and 404 for an unknown meeting.
- **Utilities:** `app/utils/subtitles.py` provides `to_srt`, `to_text` and `format_timestamp`.
- **Errors:**
  - `ASRModelLoadError` (503 `asr_model_unavailable`): missing extras, token or terms, a load failure, or CUDA unavailable
  - `UnsupportedLanguageError` (422)
  - `TranscriptionError` (500)
  - `TranscriptNotAvailableError` (409)
  - An ASR failure marks the meeting `failed` but keeps `audio_quality` and `diarization`; a test covers this.
- **Config:**
  - `ASR_BACKEND`, `ASR_LANGUAGE_BACKENDS`
  - `WHISPER_MODEL_SIZE`, `WHISPER_COMPUTE_TYPE`, `ASR_BEAM_SIZE`, `ASR_VAD_FILTER`
  - `ODIA_MODEL_ID`, `ODIA_DECODING`
  - `ASR_LOW_CONFIDENCE_THRESHOLD`, `ASR_COMPRESSION_RATIO_THRESHOLD`, `ASR_NO_SPEECH_THRESHOLD`
- **Evaluation:** `scripts/eval_asr.py` computes WER and CER per language with jiwer on a folder of audio files plus `.txt` references. Text is NFC-normalized, case-folded and stripped of punctuation (including the danda) before scoring. It can write a JSON report. Also added a `make eval-asr DATA=...` target.
- **Docker:** new `INSTALL_INDIC` build arg, alongside `INSTALL_ML`/`TORCH_VARIANT`. Whisper and IndicConformer downloads share the Prompt 4 `model-cache` volume.
- **Docs:** a README section covering the routing design (Mermaid), model choices, native script and NFC handling, quality guards, setup, the eval script and known limitations.

## Test coverage

229 tests pass, plus 3 slow real-model tests that are deselected by default. Overall coverage is 96%; for the new code:

| Module | Coverage |
| --- | --- |
| `service.py` | 100% |
| `mock_backend.py` | 100% |
| `segmentation.py` | 100% |
| `subtitles.py` | 100% |
| `postprocess.py` | 99% |
| `whisper_backend.py` | 99% |
| `router.py` | 97% |
| `indic_backend.py` | 85% (the uncovered lines are the numpy audio loader, which needs the extras) |

- **Unit tests:**
  - **Router:** selection per language (including upper-case codes and auto-detect), unconfigured language, a backend that can't handle the language, auto-detect on an incapable backend, mapping validation, config parsing, and the language strategy.
  - **Chunks:** timestamp shifting, overlap de-duplication (midpoint ownership, fuzzy duplicates, keeping the more complete version, keeping distinct text), and chunked transcription through the service.
  - **Hallucination guards:** empty output, fillers during silence, known phrases, a confident real "umm" being kept, "Thank you." being kept, compression ratio (computed and backend-supplied), and repeats.
  - **NFC normalization:** Devanagari nukta exclusions (precomposed ज़ becomes ज + ़), ऩ composition, Odia two-part vowels (ୋ ୈ ୌ), Odia ଡ଼, an NFD Odia sentence round trip, and a check that nothing is transliterated.
  - **Exports:** SRT and TXT formatting, and speech-region detection.
  - **Eval script:** normalization, per-language scoring, sample discovery, and the end-to-end CLI with the mock.
- **Backends with faked `faster_whisper`, `ctranslate2`, `transformers`, `onnxruntime` and `torch`:**
  - Whisper: kwargs passed through, loaded once then cached, cuda/float16 vs cpu/int8, word to segment mapping, confidence fallback, load and inference failures, missing extras.
  - IndicConformer: segments built from speech regions with offsets, approximate words, the `trust_remote_code` token, caching, missing token or extras, load and inference failures.
- **Integration tests** (mock backend):
  - `/transcript` as JSON (NFC, native script, language durations), `txt` and `srt` (headers and cue layout), an invalid format → 422, before processing → 409, unknown meeting → 404
  - a single-language hint forces that language
  - an ASR failure keeps diarization and audio quality
- **Slow tests,** skipped without the extras:
  - real Whisper on a short public-domain English clip (JFK), fetched at test time
  - real IndicConformer on Odia, which runs only if `HF_TOKEN` and `POLYMOM_ODIA_SAMPLE_URL` are set, because no stable public-domain Odia clip URL was found

No audio is committed.

## Known limitations

- **Odia timing:** Odia word timestamps are approximate (segment boundaries are accurate to 30 ms), and there are no Odia confidence scores.
- **Code-mixed speech:** without a single hint, Whisper decodes everything, including Odia, which it can't transcribe well. Per-segment language ID and routing comes in Prompt 6. Whisper sometimes labels Hindi as Urdu (Perso-Arabic script); passing `languages=hi` avoids that.
- **Compute:** large-v3 needs roughly 5 GB of GPU memory in float16. On CPU, use `small` or `medium`. Inference runs in the API process until the worker queue arrives in Prompt 11.
- **Real models not run by me:** they haven't run in this environment, because I have no `HF_TOKEN` here and the downloads are several GB. The integration code is tested against fakes that follow each library's API, and the ML Docker image builds with all the extras. Please run `make install-ml && HF_TOKEN=... make test-slow` once.

## Checklist

- [x] `make lint` passes
- [x] `make typecheck` (mypy strict) passes, without the ML extras installed
- [x] `make test` passes, with coverage of new code at least 85%
- [x] Docker build passes without extras
- [x] Docker build passes with `INSTALL_ML=true INSTALL_INDIC=true`
- [x] Migration 0004 applies on top of 0003, and `alembic check` is clean
- [x] Native scripts preserved, NFC throughout, nothing transliterated
- [x] No audio, model weights or secrets committed

## Next steps

**Prompt 6: language detection and code-switching.** Run segment-level language ID (Whisper LID and script detection), then use `ASRRouter.select` per segment so that Odia stretches inside a Hindi or English meeting are re-transcribed by IndicConformer, and record code-switch points.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
