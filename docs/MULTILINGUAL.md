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

### Measured results

From [evaluation/results.md](evaluation/results.md), measured on an RTX 5060 Laptop GPU
with faster-whisper large-v3 `int8_float16` and MMS-LID-126:

| Language / condition | WER | CER | Data |
| --- | --- | --- | --- |
| English (hint `en`) | 5.2% | 2.3% | FLEURS test, 30 clips |
| Hindi (hint `hi`) | 31.0% | 14.6% | FLEURS test, 30 clips |
| Hindi-English code-mixed (no hint, LID-routed) | 53.0% | 38.5% | MUCS 2021 test, 48 segments |
| Odia | pending | pending | FLEURS `or_in`, 30 clips prepared |

Spoken LID put 100% of English and Hindi speech seconds in the right language (60 FLEURS
clips), and 96.8% of the code-mixed Hindi-English speech was labelled Hindi, its matrix
language (48 MUCS segments). Odia LID has unit and integration tests, but it isn't
benchmarked yet.

Odia ASR, and every meeting-level metric (DER, speaker-attributed WER, LID on mixed
meetings), need the gated IndicConformer and pyannote models. No `HF_TOKEN` was available
for this run, so they are reported as pending and not estimated. The data, including the
8 synthetic code-mixed meetings with ground truth, is prepared, and
`make benchmark-docker` produces the numbers.

## Observed limitations

All examples are actual benchmark output (`asr_files` in
[evaluation/results.json](evaluation/results.json)), shown after scoring normalisation.

**1. Script choice for English words inflates Hindi and code-mixed error rates.** Whisper
writes many English-origin words in Latin script where the reference has Devanagari, and
the reverse also happens. Both are correct transcriptions, and every character counts as
an error.

| | Text |
| --- | --- |
| Reference (FLEURS hi) | यूनिवर्सिटी ऑफ़ डंडी के प्रोफ़ेसर पामेला फ़र्ग्यूसन लिखते हैं … |
| Polymom | university of dundee के professor pamela fumerson लिखते हैं … |
| Reference (MUCS) | ubuntu लिनक्स os version 1204 |
| Polymom | … ubuntu linux operating system version 12 0 |
| Reference (MUCS) | jchempaint के बारे में |
| Polymom | जे कैम पेंट के बारे में जे कैम पेंट के बारे में |

The last row also shows a repetition, and the unseen product name "JChemPaint" spelled
phonetically in Devanagari. On MUCS the segment boundaries come from the corpus's own
alignment, so some hypotheses also include a few words from the neighbouring segment (as
in the second MUCS row), which counts against the system.

**2. Hindi spelling variants.** Nukta and anusvara variants count as errors although both
spellings are in common use: `ख़त्म` → `खत्म`, `ज़्यादा` → `जादा`, `हालाँकि` → `हाला कि`,
`सवाल` → `सबाल`. This is why Hindi CER (14.6%) is the more meaningful number than WER
(31.0%). The median Hindi clip has a CER of about 8%.

**3. Very quiet or noisy recordings fail completely.** One FLEURS Hindi clip produced an
empty transcript (100% error) and one noisy clip came out garbled:

| | Text |
| --- | --- |
| Reference | उन्होंने अफवाहों को राजनीतिक बकवास और मूर्खतापूर्ण कहा |
| Polymom | उम्होंने नफ़व को राजनी तिक बगवास उर्मूप तप मौका हाँ |

Two of the 30 clips account for a large share of the Hindi error. Preprocessing reports
such audio (`low_volume`, `mostly_silent` warnings) and segments are flagged
`low_confidence`, but the text itself isn't recovered.

**4. English formatting differences.** English errors are mostly formatting, not
recognition: `15 august` → `august 15`, `twelve` → `12`, `twentieth` → `20th`.

**5. Odia has no word timestamps or confidences.** IndicConformer returns text only. Word
times are estimated within energy-based segments, so a speaker change inside an Odia
segment is placed by duration (`alignment_precision: "segment"`).

**6. Language switches inside a turn are not re-routed.** LID works per diarization turn
(windows of up to 15 s). A speaker who says one Hindi sentence and then one Odia sentence
in the same turn gets one language for the window, so one of the two sentences goes to
the wrong model. Embedded English words inside Hindi are handled, because Whisper
transcribes them in place.

**7. Odia-English mixing is weaker than Hindi-English.** IndicConformer is not trained
for English, so English terms inside Odia speech come out phonetically in Odia script or
are dropped. No public Odia-English code-switched test set was found to quantify this.

**8. Summaries from a small local model miss items.** With `qwen2.5:7b-instruct` on four
labelled meetings: action items 100% precision / 57% recall, decisions 100% / 20%, and 74%
of evidence quotes verified. In the committed
[sample](samples/code-mixed-meeting.md), it found the Hinglish commitment
("Main kal tak fix deploy kar dunga") but:

- it missed the Odia one ("ମୁଁ agle hafte load testing କରିବି।");
- it left `kal` unresolved to a date;
- it mislabelled a decision as superseded because of the injected "ignore all previous
  instructions" line. The instruction itself was not followed, and the line was flagged.

Precision stays high because the verifier drops anything whose quote isn't in the
transcript. Recall is a model-size limit; larger models are one setting away.

**9. The language ID model's licence.** MMS-LID is CC-BY-NC-4.0. The commercial
alternatives in the code (SpeechBrain, Whisper) have no Odia.

**10. 8 GB GPUs need `WHISPER_COMPUTE_TYPE=int8_float16`.** float16 Whisper large-v3
plus MMS-LID ran out of memory on the test GPU; MMS-LID now loads in float16.

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
