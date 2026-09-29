## Summary

Prompt 7 of 12 adds the system's core output: a **speaker-attributed, chronological transcript**. A new `AlignmentStage` runs after transcription.
- It assigns every transcript word to a diarization turn, groups the words into utterances per speaker, and carries over the Prompt 6 language tags.
- `GET /api/v1/meetings/{id}/transcript` now returns this speaker view by default, as JSON, TXT, SRT, VTT or Markdown. The raw ASR output moves to `?view=raw`.
- `PATCH /api/v1/meetings/{id}/speakers` sets display names ("Person 1" → "Ravi") without changing the original labels.

## Alignment algorithm

The engine in `app/services/alignment/` is pure and deterministic.

1. **Word assignment:**
   - A word goes to the speaker whose turns overlap `[start, end]` the most. Ties go to the earlier turn, then the lower label.
   - Other speakers active during the word go into `overlapping_speakers`, so the utterance gets `has_overlap: true`.
   - A word overlapping no turn goes to the nearest turn within `ALIGN_MAX_GAP_SECONDS`, else `Unknown`.
2. **Segments without reliable word times:** this means segments with no words, or with words that have no confidence (the Odia backend's length-based estimates).
   - The segment's tokens are split across the turns it overlaps, in proportion to overlap duration, keeping word order.
   - Each speaker's share of the time is spread evenly over its words.
   - The words are marked `alignment_precision: "segment"`.
3. **Utterances:**
   - Consecutive same-speaker words are grouped.
   - Fragments below `UTTERANCE_MIN_WORDS` merge into the nearest same-speaker utterance within `ALIGN_MERGE_GAP_SECONDS`.
   - Utterances over `UTTERANCE_MAX_SECONDS` split after `. ? ! । ॥`, with a hard cut at 2× that length if no punctuation appears.
   - A final pass merges same-speaker parts that overlap in time, so a speaker's utterances never overlap.
   - Order is by the *rounded* `(start, speaker)` values the API exposes.
4. **Text and language:** text is rebuilt from the words, with punctuation-only tokens attached to the previous word, so "है ।" becomes "है।". `primary_language`, `languages_present` and `is_code_mixed` reuse Prompt 6's `code_mix_stats`.
5. **Label consistency guard:** "Person N" labels are used exactly as diarization produced them.
   - `warnings` lists speakers who have turns but no words, and any words labelled `Unknown`.
   - `alignment_stats` reports the percentage of words assigned, unknown and segment-level.

## Changes

- `app/services/alignment/`:
  - `aligner.py`: `speaker_for_span`, `split_segment`, `align_words`.
  - `utterances.py`: grouping, fragment merging, splitting, the same-speaker overlap merge, `join_words`, `build_transcript` (warnings and stats).
- `app/schemas/transcript.py`: `AlignedWord`, `Utterance`, `AlignmentStats`, `SpeakerTranscript`, `SpeakerTranscriptResponse`, and the rename request and response.
- **Pipeline:** `AlignmentStage` is registered after `TranscriptionStage`.
- **Storage:** migration `0006` adds `speaker_transcript` and `speaker_names` JSON columns, the same JSON-column approach as before. `alembic check` reports no drift.
- **API:**
  - `/transcript` gains `view=speaker|raw`, and the speaker view adds the `vtt` and `md` formats. `view=raw` with `vtt` or `md` returns 422.
  - `PATCH /speakers` validates labels against the diarization and the transcript: unknown labels get 422, and a meeting that hasn't been processed gets 409.
- **Renderers:** `app/utils/subtitles.py` adds speaker-prefixed TXT (`[00:01:23 - 00:01:30] Ravi: ...`), SRT, WebVTT with `<v Speaker>` voice tags, and Markdown with consecutive-speaker blocks. All of them apply display names.
- **Config:** `ALIGN_MAX_GAP_SECONDS`, `ALIGN_MERGE_GAP_SECONDS`, `UTTERANCE_MAX_SECONDS`, `UTTERANCE_MIN_WORDS`.
- **Evaluation:** `scripts/eval_diarization.py` reports:
  - **DER** via `pyannote.metrics` (already a dependency of pyannote.audio in the `ml` extra)
  - **WDER**: the share of time-matched reference words with the wrong speaker, after the best one-to-one speaker mapping. The mapping is greedy and marked with a `ponytail:` note: Hungarian assignment is the upgrade if needed.
- **Dependencies:** `hypothesis` in the dev group. `.hypothesis/` is gitignored.
- **Docs:** a README section with a Mermaid timeline of word assignment, overlap handling, format samples, speaker renaming, eval usage, known limitations and config. The roadmap numbering was fixed.

## Test coverage

337 tests pass, plus 4 slow real-model tests that are deselected by default. Overall coverage is 97%; for the new code:

| Module | Coverage |
| --- | --- |
| `aligner.py` | 99% |
| `utterances.py` | 100% |
| `subtitles.py` | 100% |
| `routes/meetings.py` | 100% |
| `meeting_service.py` | 100% |

This is above the 90% gate.

- **Unit tests:**
  - **Assignment:** overlap-based assignment (parametrized: inside a turn, the overlap going either way, nearest turn before or after, `Unknown`, zero-length words), tie-breaking, no turns, overlap flagging.
  - **Segment splitting:** proportional splitting of an Odia segment across two turns, confidence-less words treated as segment-level, the nearest/`Unknown` fallback for segments, repeated turns of one speaker.
  - **Utterances:** grouping, fragment merging within and beyond the gap, splitting at `. ? ! । ॥` and closing quotes, a hard cut, Hindi and Odia sentences, text rebuilt across Latin, Devanagari and Odia, utterance language, confidence and overlap fields, deterministic ordering (including reversed input).
  - **Transcript:** warnings and stats, the empty case, every renderer.
- **Property-based tests (Hypothesis):** every word is assigned exactly once to a known speaker or `Unknown`, the total word count is preserved, utterances are sorted with sequential ids, and each speaker's utterances never overlap. I also ran this once at 2,000 examples.
  - **These tests found four real bugs, all fixed:**
    1. word sorting ignored the speaker, so the order wasn't deterministic
    2. a zero-length segment without confidence divided by zero
    3. `approximate_words` returned nothing for zero-length spans, silently dropping words
    4. ordering used unrounded times while the API shows rounded ones, and one speaker's words overlapping in time could produce overlapping utterances
- **Integration tests** (full mock pipeline):
  - the speaker JSON: speakers, ordering, word attribution, per-utterance languages, 100% assigned, and total words matching the raw ASR word count
  - every format (`txt`/`srt`/`vtt`/`md`: content types, disposition, speaker prefixes, voice tags, Markdown blocks)
  - renaming, including whitespace trimming, clearing with `null`, validation (422 with the known speakers listed), 404, and 409 before processing
  - the speaker view before processing (409, pointing to `view=raw`) and raw-only formats (422)
  - existing Prompt 5/6 tests moved to `?view=raw`
- **Eval script:** RTTM and word-file parsing, time matching, speaker mapping, WDER, and the end-to-end CLI. The DER test is skipped without the `ml` extra.

## Known limitations

- **Odia:** attribution is only segment-level, so a speaker change inside one Odia segment is placed by duration.
- **Overlapping speech:** each overlapped word goes to one speaker. The other voice is recorded in `overlapping_speakers`, but its words aren't transcribed separately.
- **Diarization errors:** these carry through to attribution. The warnings flag speakers with no words, but alignment can't correct diarization.
- **Backchannels:** a short "hmm" or "yes" inside someone else's turn can be misattributed if the ASR word timing is off by more than its length.
- **Real models not run by me:** DER and WDER haven't been computed on real audio in this environment (no labelled data, no `HF_TOKEN`). The script is ready for `data/<stem>.rttm` + `<stem>.json` (+ `<stem>.words.tsv`).

## Checklist

- [x] `make lint` passes
- [x] `make typecheck` (mypy strict) passes
- [x] `make test` passes, with coverage of new code at least 90%
- [x] Docker build passes, with and without the ML extras
- [x] Migration 0006 applies on top of 0005, and `alembic check` is clean
- [x] "Person N" labels are never renumbered; display names are stored separately
- [x] No audio, model weights or secrets committed

## Next steps

**Prompt 8: speaker statistics.** Compute per-speaker talk time, share of the meeting, turn counts, average turn length, words per minute, interruptions and overlaps, and language mix from the speaker transcript. These will be exposed on the API and feed the summary in Prompt 9.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
