## Summary

Prompt 6 of 12 adds spoken language identification and code-switching, so each part of a meeting is transcribed by the right backend, even when speakers switch between English, Hindi and Odia.

There's a new `LanguageIdentificationStage` between diarization and transcription:
- it identifies the language of each diarization turn and smooths the results into language regions
- `TranscriptionStage` then routes each region to its backend (Whisper with the language forced for en/hi, IndicConformer for Odia) and stitches the results in chronological order
- a text-level tagger labels each word's script and language (including romanized Hindi and Odia) and marks code-mixed segments
- `GET /api/v1/meetings/{id}/languages` returns language shares, a per-speaker breakdown and switch points

## Approach and rationale

- **Two levels of language:**
  - *Audio-level* LID decides which ASR backend runs.
  - *Text-level* tagging labels what was actually said: Hinglish, or Odia with English terms. Each fixes what the other can't see.
- **Diarization turns as LID windows:** people usually switch language at turn boundaries, and a turn is long enough for reliable LID while being short enough to be monolingual. Turns over `LID_MAX_WINDOW_SECONDS` (15 s) are split. Without diarization, fixed windows are used.
- **Restrict and renormalize:** LID probabilities are reduced to `SUPPORTED_LANGUAGES`, or to the upload hint if one was given, and renormalized. A 100+ language model splits Odia speech across Odia, Bengali and Assamese; renormalizing asks "which of *our* languages is it".
  - Below `LID_MIN_CONFIDENCE` a window is `uncertain`.
  - A single hinted language skips the model entirely.
- **Smoothing** (pure functions):
  1. A short (< `LID_MIN_WINDOW_SECONDS`), low-confidence window that disagrees with its speaker's dominant language takes that language. The profile is built from the speaker's *other* confident windows.
  2. Uncertain windows take the nearest confident window of the same speaker, then the meeting's dominant language.
  3. Contiguous windows with the same speaker and language are merged.
  4. Overlapping speech is clipped, so no audio is transcribed twice.
- **Routing:**
  - Consecutive same-language regions are batched into one model call. Batches break only at turn boundaries and are capped at `CHUNK_LENGTH_SECONDS`.
  - Each batch is cut into a temporary WAV (deleted afterwards), transcribed with its language forced, and shifted back to meeting time.
  - If the Odia backend fails, the batch is retried with Whisper without a forced language and flagged `fallback_used`.
  - `LANGUAGE_ROUTING_ENABLED=false` restores the Prompt 5 single-pass behaviour exactly.

### LID model: MMS-LID, not VoxLingua107 (verified on Hugging Face)

- **VoxLingua107 has no Odia.** `speechbrain/lang-id-voxlingua107-ecapa/label_encoder.txt` lists 107 languages, including `hi`, `bn`, `as` and `ur`, but not `or`.
- **MMS-LID is the replacement.** The maintained audio LID models that include Odia are Meta's MMS-LID family. `facebook/mms-lid-126` is the smallest checkpoint whose `id2label` has `ory`, `hin` and `eng`. It loads through transformers `Wav2Vec2ForSequenceClassification`, so it adds no new dependency beyond the `indic` extra. It's the default: `LID_BACKEND=mms`.
- ⚠️ **MMS-LID is CC-BY-NC-4.0 (non-commercial).** For commercial deployments, use `LID_BACKEND=speechbrain` (Apache-2.0) or `whisper` (MIT). Both are implemented but can only choose between English and Hindi; Odia meetings would then rely on `languages=or` hints.
- **Other options I checked:** the Hugging Face "indic language identification" audio models are unmaintained community uploads (fewer than 20 downloads each).

## Changes

- **`app/services/language/`:**
  - `base.py`: the `LanguageIdentifier` ABC (`identify(audio_path, start, end, candidates)` → language, confidence, top-k, uncertain) plus the pure `restrict_scores` and `predict` functions.
  - Backends:
    - `mms_lid.py`: the default; maps ISO 639-3 labels to ISO 639-1.
    - `speechbrain_lid.py`: VoxLingua107, en/hi only.
    - `whisper_lid.py`: reuses the loaded ASR model's language detection, en/hi only.
    - `mock_lid.py`: en → hi → or every 6 s.

    All are lazy, thread-safe singletons that run in a worker thread and raise typed errors.
  - `smoothing.py`: `windows_from_turns`, `inherit_short_windows`, `resolve_uncertain`, `merge_adjacent`, `smooth`, `to_regions`.
  - `text_tagger.py`: script from Unicode blocks (Latn, Deva, Orya), and a romanized Hindi/Odia lexicon that leaves out words that are also common English words, favouring precision. Computes the segment-level code-mix stats.
  - `summary.py`: language shares, per-speaker breakdown and switch points.
  - `service.py`: `LanguageIdService`, `build_identifier` and hint handling.
- **ASR:** `ASRRouter.transcribe_regions` plus `batch_regions` (Odia falls back to Whisper). `TranscriptionService.transcribe_routed`. Text tagging now runs on every transcript, routed or not. The mock ASR backends are named per route (`mock-whisper`, `mock-indic`), so tests can see where each segment was routed.
- **Audio:** `app/services/audio/slicing.py` provides sample-accurate WAV slicing (stdlib) and a numpy sample reader for the models.
- **Schemas:**
  - `Word` gains `language` and `script`.
  - `TranscriptSegment` gains `primary_language`, `languages_present`, `is_code_mixed`, `code_mix_ratio`, `lid_confidence` and `fallback_used`. All are defaulted, so transcripts stored by Prompt 5 still load.
  - New: `LanguagePrediction`, `LanguageRegion` and `LanguageSummary` (shares, speakers, switch points, code-mixed count, regions).
- **Pipeline:** `LanguageIdentificationStage` is registered only when `LANGUAGE_ROUTING_ENABLED`. `TranscriptionStage` routes when regions exist and updates `code_mixed_segments` in the summary.
- **Storage and API:**
  - Migration `0005` adds a `language_summary` JSON column, the same JSON-column approach as before. `alembic check` reports no drift.
  - New endpoint: `GET /api/v1/meetings/{id}/languages`. It returns 409 `language_summary_not_available` before language ID and 404 for an unknown meeting.
- **Errors:** `LanguageIdModelLoadError` (503), `LanguageIdError` (500) and `LanguageSummaryNotAvailableError` (409).
- **Config:**
  - `SUPPORTED_LANGUAGES`, `LANGUAGE_ROUTING_ENABLED`
  - `LID_BACKEND`, `LID_MODEL_ID`
  - `LID_MIN_CONFIDENCE`, `LID_MIN_WINDOW_SECONDS`, `LID_MAX_WINDOW_SECONDS`
- **Dependencies:** `speechbrain` joins the `ml` extra. MMS uses transformers from `indic`, so the Dockerfile is unchanged; `INSTALL_ML=true INSTALL_INDIC=true` includes everything.
- **Eval:** `scripts/eval_asr.py --routed` runs LID and routing and prints an **LID confusion matrix in seconds**. The reference comes from `<stem>.lang.json` spans, or the folder language.
- **Docs:** a README section with a Mermaid diagram of LID → smoothing → routing → stitching → tagging, audio-level vs text-level handling, the model choice and license warning, eval usage, known limitations and config. The roadmap now matches the prompt numbering.

## Evaluation results

**No real evaluation ran in this environment.** There's no `HF_TOKEN` here, MMS-LID is a 1B-parameter download, and there's no labelled en/hi/or meeting audio. What I could produce:

- **Pipeline check with mock models,** using an 18 s meeting (speakers alternate every 2 s; the mock LID says en for 0–6 s, hi for 6–12 s, or for 12–18 s):
  - regions are identified and routed as en → `mock-whisper`, hi → `mock-whisper`, or → `mock-indic`
  - segments are stitched in order
  - `/languages` reports 33.33% per language, with 2 switch points at 6 s (en→hi) and 12 s (hi→or)
- **`eval_asr.py --routed --backend mock` on the same audio, without diarization** (15 s fixed windows):

  ```text
  LID accuracy 0.667 over 18.0s (rows: reference, cols: predicted, seconds)
  ref/pred         en       hi       or
  en              6.0      0.0      0.0
  hi              3.0      0.0      3.0
  or              0.0      0.0      6.0
  ```

  This shows the report working. It also shows why the pipeline uses diarization turns and not coarse fixed windows: the 6 s Hindi stretch falls inside two 9 s windows. These are **not** model accuracy numbers.
- **To produce real numbers:** run `make install-ml` and `HF_TOKEN=...`, then `uv run python scripts/eval_asr.py data/ --routed`, with `<stem>.lang.json` references for mixed-language files.

## Test coverage

288 tests pass, plus 4 slow real-model tests that are deselected by default. Overall coverage is 96%; for the new code:

| Module | Coverage |
| --- | --- |
| `language/base.py` | 100% |
| `service.py` | 100% |
| `summary.py` | 100% |
| `text_tagger.py` | 100% |
| `mms_lid.py` | 100% |
| `mock_lid.py` | 100% |
| `whisper_lid.py` | 100% |
| `speechbrain_lid.py` | 98% |
| `smoothing.py` | 95% |
| `asr/router.py` | 99% |
| `audio/slicing.py` | 65% (the uncovered lines are the numpy sample reader, which needs the extras) |

- **Unit tests:**
  - **Scores:** renormalization over allowed languages, uniform fallback, top-k, the uncertain threshold, hint filtering changing the winner, candidate selection.
  - **Windows:** following turns, splitting long turns, covering the recording without turns.
  - **Smoothing:** short-window inheritance (confident windows keep their language; inheritance uses the speaker's own profile only), uncertain resolution (nearest same-speaker window, closer neighbour wins, meeting-dominant and default fallbacks), merging with weighted confidence, the end-to-end `smooth`, clipping of overlapping speech.
  - **Routing:** batching (consecutive same-language regions, span cap), sample-accurate slicing with clamping, routed transcription shifting timestamps and stitching with the right backend per language, temp files cleaned up, Odia failure falling back to Whisper with `fallback_used`, a failing default backend not being retried.
  - **Text tagging:** parametrized script and language cases, *"मीटिंग kal 5 baje hai"* (Devanagari plus romanized Hindi counts as monolingual Hindi), Odia with English terms (ratio 0.33), tie-breaking toward the audio language.
  - **Summary:** switch points, shares and percentages, speaker ordering and dominant language, the empty case.
  - **LID backends with faked `transformers`/`torch`/`speechbrain`/faster-whisper:** label mapping and renormalization (Bengali dropped), load caching, device selection, load/inference/missing-extras errors, SpeechBrain limited to en/hi, the single-hint fast path never loading a model.
- **Integration tests** (mock diarization + mock LID + mock ASR):
  - a full 18 s en/hi/or meeting: routing per language, chronological stitching, word scripts, the `/languages` shares and switch points and speakers
  - a hint restricting the candidates
  - 409 and 404 on `/languages`
  - Prompt 5 transcript tests updated for routed output
- **Slow test:** real SpeechBrain LID, skipped without the extras.

## Known limitations

- **Switching inside a sentence:** a mid-sentence switch stays in one audio region and goes to one backend. Only the text-level tags reveal the mix.
- **Romanized text:** the lexicon is deliberately small and precise. Romanized Hindi or Odia outside it is tagged English, with no transliteration model.
- **Short utterances:** LID below about 1.5 s is unreliable. Smoothing gives such turns the speaker's usual language, which can hide a genuine one-word switch.
- **Hindi and Odia accents:** short or accented Odia can be classified as Hindi (or Bengali, which is discarded). Hint `languages=or` for known Odia meetings.
- **MMS-LID:** it's non-commercial and has 1B parameters (slow on CPU). The commercial alternatives can't detect Odia.
- **Real models not run by me:** nothing ran against the real LID or ASR models here; see Evaluation results.

## Checklist

- [x] `make lint` passes
- [x] `make typecheck` (mypy strict) passes, without the ML extras installed
- [x] `make test` passes, with coverage of new code at least 85%
- [x] Docker build passes without extras, and with `INSTALL_ML=true INSTALL_INDIC=true`
- [x] Migration 0005 applies on top of 0004, and `alembic check` is clean
- [x] `LANGUAGE_ROUTING_ENABLED=false` keeps the Prompt 5 behaviour
- [x] No audio, model weights or secrets committed

## Next steps

**Prompt 7: speaker and transcript alignment.** Assign transcript words to diarization turns using word timestamps (approximate for Odia) and the overlap regions. This produces a speaker-attributed transcript ("Person 1: ...") that keeps the per-word language tags from this PR.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
