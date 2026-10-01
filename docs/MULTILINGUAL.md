# Multilingual support

Polymom transcribes English, Hindi, Odia and meetings that mix them, between speakers and
inside a single sentence. This page covers the model used per language, how
code-switching is handled, how it was tested, what goes wrong (with real examples from the
benchmark) and what would improve it. The stage mechanics are in
[PIPELINE.md](PIPELINE.md#language-identification-and-code-switching).

## Models per language

| Language | Script | Speech recognition | Language ID | Text tagging |
| --- | --- | --- | --- | --- |
| English (`en`) | Latin | faster-whisper `large-v3` | MMS-LID-126 (`eng`) | Latin script |
| Hindi (`hi`) | Devanagari | faster-whisper `large-v3`, language forced to `hi` | MMS-LID-126 (`hin`) | Devanagari, or romanized Hindi from a small lexicon (`hi-Latn`) |
| Odia (`or`) | Odia | AI4Bharat IndicConformer 600M (RNN-T); Whisper fallback on failure | MMS-LID-126 (`ory`) | Odia script, or romanized Odia lexicon (`or-Latn`) |
| Hindi-English code-mixed | Devanagari + Latin | Whisper with `hi` forced: English words usually stay in Latin script | Labelled `hi` (the matrix language) | Per-word script and language; `is_code_mixed`, `code_mix_ratio` |
| Odia-English code-mixed | Odia + Latin | IndicConformer (English terms come out in Odia script or Latin) | Labelled `or` | As above |

Why these models is covered in [TECHNOLOGY_CHOICES.md](TECHNOLOGY_CHOICES.md). In short,
Whisper has no Odia, and IndicConformer is the maintained open model that has it. MMS-LID
is the smallest maintained spoken-LID model that includes Odia.

All text stays in native script and is never transliterated. It is normalized to Unicode
NFC, and the Devanagari nukta and Odia ଡ଼/ଢ଼ composition exclusions are handled
explicitly. The UI renders it with Noto Sans Devanagari and Noto Sans Oriya, and the PDF
and DOCX exports embed the same fonts. Search uses a trigram (SQLite) or `simple`
(Postgres) index that needs no word segmentation, so it works for every script.

## How code-switching is handled

1. **Diarization first.** People mostly switch language at turn boundaries, so LID runs
   per diarization turn. Turns longer than 15 s are split into windows, and windows
   shorter than 1.5 s are scored but may be overruled by smoothing.
2. **Restricted spoken LID.** MMS-LID scores 126 languages. The scores are cut down to the
   candidates (the upload's `languages` hint, else `SUPPORTED_LANGUAGES`) and
   renormalized, so probability mass on relatives (Bengali, Assamese, Urdu) doesn't win.
   Below `LID_MIN_CONFIDENCE` a window is marked *uncertain*.
3. **Smoothing.** A short, low-confidence window takes its speaker's dominant language.
   An uncertain window takes the nearest confident window of the same speaker, else the
   meeting's dominant language. Consecutive windows of the same speaker and language are
   merged into regions.
4. **Routing.** Consecutive same-language regions are batched and transcribed with the
   language forced by that language's backend (`ASR_LANGUAGE_BACKENDS`). Forcing matters:
   left to auto-detect, Whisper often writes Hindi in Urdu script or translates it.
5. **Word-level tagging.** Every word gets a script and a language. Romanized Hindi and
   Odia function words (*kal, baje, hai / kemiti, achi, heba*) are recognised by a small
   precision-first lexicon. Utterances carry `primary_language`, `languages_present`,
   `is_code_mixed` and `code_mix_ratio`, and the meeting gets switch points and per-speaker
   language shares (`GET /languages`).
6. **Summaries across languages.** The LLM sees `[uN][time][speaker][lang]` lines,
   writes the minutes in `SUMMARY_OUTPUT_LANGUAGE` (en, hi or or), and must quote evidence
   verbatim in the original script. The verifier checks every quote against the cited
   utterance. Relative dates in Hindi and Odia (*kal, agle hafte, ଆସନ୍ତାକାଲି*) are resolved
   against the meeting date.

## Testing performed

| Level | What | Where |
| --- | --- | --- |
| Unit | NFC normalization of nukta and Odia vowel signs, script detection, romanized lexicon, code-mix ratio, LID restriction and renormalization, smoothing rules, routing and batching, Odia fallback | `tests/unit/test_language_logic.py`, `test_language_routing.py`, `test_asr_postprocess.py`, `test_asr_routing.py` |
| Integration | Upload with hints → mock LID and ASR → `/languages`, code-mixed utterances, Devanagari and Odia in txt/srt/vtt/md, PDF and DOCX with Indic fonts, search in Devanagari | `tests/integration/test_languages.py`, `test_transcript.py`, `test_results.py` |
| Summaries | Hand-labelled Hindi, Odia and Hinglish+Odia meetings with gold decisions and action items; injection line included | `tests/fixtures/meetings/*.json`, `scripts/eval_summary.py` |
| Real models | FLEURS en/hi/or, MUCS Hindi-English, synthetic multi-speaker code-mixed meetings, AMI | `scripts/run_benchmark.py` → [evaluation/results.md](evaluation/results.md) |

<!-- MULTILINGUAL-RESULTS -->

## Ideas for improvement

- **Odia word timestamps.** Run CTC forced alignment on IndicConformer's CTC head (or a
  wav2vec2 Odia model) to replace length-based word times. That would make Odia speaker
  attribution word-precise instead of segment-precise.
- **Hindi-specialised decoding for Hindi regions.** Evaluate an Indic fine-tune of Whisper
  or IndicConformer's Hindi head on Hindi-dominant regions. Keep Whisper for Hinglish,
  where English terms matter, and choose per region by the `code_mix_ratio` of a first
  pass.
- **Commercial-friendly LID with Odia.** Fine-tune a small ECAPA or wav2vec2 classifier on
  en/hi/or (plus hi-en) from Kathbath, IndicVoices and FLEURS train splits, to replace
  CC-BY-NC MMS-LID and to add an explicit code-mixed class.
- **Sub-turn switch detection.** Score sliding 3 s windows inside long turns and re-route
  only when two consecutive windows agree, which catches intra-turn switches without
  flapping.
- **Transliteration-aware scoring and search.** Index a romanized form of Devanagari and
  Odia next to the native text, so "budget" finds "बजट", and report a
  transliteration-normalized CER for code-mixed speech, where the same English word may
  be written in either script.
- **Custom vocabulary.** Pass meeting-specific terms (names, products) as Whisper's
  `initial_prompt`, and as hotwords to the RNN-T decoder.
- **Real code-mixed meeting data.** The synthetic meetings are stitched from read or
  lecture speech. A small set of real multi-party Odia-English and Hindi-English
  meetings, annotated with RTTM and transcripts, would be the most valuable addition to
  the evaluation.
