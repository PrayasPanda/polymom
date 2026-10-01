# Technology choices

For each area: what Polymom uses, what else was considered, and why. The guiding
constraints were:

1. English, Hindi, Odia and code-mixed speech must all work, and Odia is poorly served by
   most speech models.
2. Everything must run on one workstation GPU and also on CPU.
3. Open weights where possible, so audio doesn't have to leave the machine.
4. Every model sits behind an interface, so it can be swapped by configuration.

Measured results for the chosen models are in [evaluation/results.md](evaluation/results.md).

## Summary

| Area | Choice | Main alternatives |
| --- | --- | --- |
| Preprocessing | ffmpeg (loudnorm EBU R128, high-pass, optional afftdn), 16 kHz mono PCM | sox, librosa/torchaudio in Python, RNNoise/DeepFilterNet denoising |
| Diarization | pyannote/speaker-diarization-3.1 | NeMo MSDD/Sortformer, WhisperX (pyannote inside), SpeechBrain ECAPA + clustering, cloud APIs |
| ASR, English + Hindi | faster-whisper `large-v3` (CTranslate2) | openai-whisper, whisper-large-v3-turbo, IndicWhisper, NeMo Canary/Parakeet, cloud STT |
| ASR, Odia | AI4Bharat IndicConformer 600M multilingual | Whisper (no Odia), IndicWav2Vec, MMS-1B-all, Google/Azure STT |
| Spoken language ID | facebook/mms-lid-126 on diarization-turn windows | SpeechBrain VoxLingua107 ECAPA, Whisper language detection, text-based LID |
| Alignment | Word-timestamp to turn overlap assignment (own code) | WhisperX forced alignment (wav2vec2), NeMo forced aligner |
| LLM summary | Provider-agnostic client: OpenAI, Azure, Anthropic, Ollama (local, default `qwen2.5:7b-instruct`) | LangChain/LlamaIndex, one fixed provider, extractive summarisation |
| Storage | SQLAlchemy 2 async + Alembic; SQLite (dev), Postgres (prod); artifacts on disk or S3 | MongoDB, raw files only, Postgres only |
| Queue | Redis + arq, one queue per resource (cpu, gpu, llm) | Celery, RQ, Dramatiq, FastAPI BackgroundTasks, Temporal |
| API and UI | FastAPI + Jinja2 + HTMX (no frontend build) | Streamlit, Gradio, React/Vite SPA |

## Preprocessing

**Choice:** ffmpeg as a subprocess (restricted with `-protocol_whitelist file,pipe,fd`).
One analysis pass (silencedetect, astats, ebur128) and one conversion pass (first audio
track, optional trim, 80 Hz high-pass, optional `afftdn`, `loudnorm` to −23 LUFS, resample
to 16 kHz mono 16-bit).

**Why:** ffmpeg decodes every container users actually upload (m4a from phones, mp4/mkv
from meeting recorders, webm from browsers) and extracts audio from video in the same
step. Both Whisper and pyannote expect 16 kHz mono. Loudness normalisation matters more
than denoising for these models: quiet laptop-microphone speakers are the most common
failure, while aggressive denoising removes consonants and hurts Hindi and Odia ASR more
than it helps. So denoising is off by default, and quality problems (low volume, clipping,
mostly silent) are reported as warnings instead of blocking.

**Alternatives:** sox has weaker container support. Python decoders (librosa, torchaudio)
would load full recordings into memory and pull torch into the API image. RNNoise and
DeepFilterNet are better denoisers, but they are an extra model per request for a gain we
could not measure on meeting audio.

## Diarization

**Choice:** `pyannote/speaker-diarization-3.1` (MIT; gated with auto-approval). Recordings
longer than an hour are diarized in 30-minute chunks, and speakers are re-linked across
chunks by cosine similarity of pyannote embeddings.

**Why:** it is the most widely used open pipeline, it is language-agnostic (it separates
voices, not words, which matters for code-mixed meetings), it handles overlapped speech,
it accepts speaker-count hints (`expected_speakers`), and it runs on CPU as well as GPU.
Version 3.1 is pure PyTorch, with no onnxruntime conflict with the Odia model.

**Alternatives:**
- **NeMo MSDD / Sortformer** is competitive on benchmarks, but NeMo is a very heavy
  dependency, and Sortformer is limited to 4 speakers.
- **WhisperX** uses pyannote internally anyway and couples diarization to one ASR model.
- **SpeechBrain ECAPA + clustering** needs its own VAD and overlap handling.
- **Cloud APIs** send audio off-machine.

The trade-off is that pyannote needs a Hugging Face token (gated terms) and is slow on CPU.

## ASR for English and Hindi

**Choice:** faster-whisper `large-v3` (MIT, not gated). It uses float16 on GPU and int8 on
CPU, with VAD filtering, word timestamps, `condition_on_previous_text=False` and
hallucination guards: compression-ratio filtering, no-speech with filler detection, and
repeat removal.

**Why:** Whisper large-v3 is the strongest open multilingual model for English and Hindi,
it writes Hindi in Devanagari, and it copes with Hinglish, the English words inside Hindi
sentences, better than monolingual Indic models. The CTranslate2 build is several times
faster than the reference implementation and uses less memory, so it fits an 8 GB GPU
next to pyannote. Word timestamps are what make speaker attribution possible.

**Alternatives:**
- **whisper-large-v3-turbo** is about 4× faster but less accurate on Hindi; set
  `WHISPER_MODEL_SIZE` to use it.
- **IndicWhisper** and other Hindi fine-tunes are better on pure Hindi, but they lose
  English and code-mixing robustness, and their availability is uneven.
- **NeMo Canary/Parakeet** don't cover Hindi and Odia in the open release.
- **Cloud STT** (Google, Azure) is strong on Hindi, but it is paid, sends audio off-machine,
  and is harder to evaluate reproducibly.

## ASR for Odia

**Choice:** `ai4bharat/indic-conformer-600m-multilingual` (MIT, gated with auto-approval).
It ships as ONNX with a transformers loader and uses RNN-T decoding by default (`ctc` is
faster).

**Why:** **Whisper does not support Odia at all.** IndicConformer is trained specifically
for the 22 scheduled Indian languages by AI4Bharat and is actively maintained. It loads
through transformers remote code, so there is no NeMo dependency.

**Alternatives:**
- **MMS-1B-all** covers Odia, but its accuracy on Odia is much lower, and it needs
  per-language adapters.
- **IndicWav2Vec** is an older generation, and its checkpoints are harder to obtain.
- **Cloud STT** has limited Odia support.

The trade-off is that this model produces no timestamps or confidences. Segments are cut
at energy-based speech regions, and word times within them are estimated by word length,
so Odia utterances are marked `alignment_precision: "segment"`.

## Spoken language identification

**Choice:** `facebook/mms-lid-126`, run on diarization-turn windows (1.5–15 s). Scores are
restricted to the supported languages. The results are smoothed so that short uncertain
windows inherit their speaker's language and brief flips are merged. Each language region
is then routed to its ASR backend.

**Why:** routing per region is the only way to use the best model per language inside one
meeting. MMS-LID-126 is the smallest maintained audio LID model that includes **Odia**, next
to Hindi and English.

**Alternatives:**
- **SpeechBrain VoxLingua107** (Apache-2.0) has no Odia in its label set.
- **Whisper's language detection** has no Odia, and often reports Hindi as Urdu.
- **Text-based LID** (after ASR) is circular: you need the right ASR to get the text.

⚠️ The trade-off is licensing: MMS-LID weights are CC-BY-NC-4.0 (non-commercial). For
commercial use, set `LID_BACKEND=speechbrain` or `whisper` (English and Hindi only), and
pass `languages=or` for Odia meetings so they are routed without LID.

## Alignment

**Choice:** own code. Each ASR word is assigned to the diarization turn it overlaps most,
and other speakers active during the word are recorded. Words in gaps go to the nearest
turn within 1 s. Segment-level words (Odia) are split proportionally across turns. Words
are then grouped into utterances, split at sentence punctuation including the danda, with
per-word language tags for code-mix flags.

**Why:** both inputs already carry timestamps, so a deterministic, testable assignment is
enough. It works identically for every language, including those without a forced-alignment
model.

**Alternatives:** WhisperX forced alignment needs a wav2vec2 alignment model per language,
and none exists for Odia. The NeMo forced aligner has the same language gap and brings in
NeMo.

## LLM summarization

**Choice:** a small provider-agnostic client (OpenAI-compatible, Azure, Anthropic, Ollama,
mock) over httpx. Output uses JSON-schema-constrained extraction (summary, decisions,
action items, open questions), and every item must cite utterance ids plus a verbatim
quote. Transcripts that are too long for a single pass go through map-reduce. A verifier
drops or downgrades items whose quotes are not found in the cited utterances. Prompt
injection lines in the transcript are flagged and ignored.

**Why:**
- Grounding is the property that matters for minutes: an action item nobody said is worse
  than a missing one.
- Quotes stay in the original language and script, while the summary language is
  configurable (`SUMMARY_OUTPUT_LANGUAGE`).
- Keeping the client small (about 200 lines) avoids framework lock-in.
- Ollama is the default for fully local runs (`qwen2.5:7b-instruct` handles Hindi well for
  its size), and the cloud providers are one setting away.

**Alternatives:**
- **LangChain/LlamaIndex** add a large dependency surface for three HTTP calls.
- **A single fixed provider** fails the offline requirement.
- **Extractive summaries** cannot produce owners, due dates or decisions.

## Storage

**Choice:**
- SQLAlchemy 2 (async) with Alembic migrations.
- SQLite by default, so the quickstart has no dependencies, and Postgres (asyncpg) in
  compose and production.
- Normalized tables for runs, stage results, speakers, utterances and summaries, plus
  full-text search: FTS5 with a trigram tokenizer on SQLite, and `simple` `tsvector` +
  `ILIKE` on Postgres. Both are script-agnostic, so Devanagari and Odia work.
- Blobs (uploads, processed audio, checkpoints, exports) live in an artifact store: local
  disk, or S3/MinIO/SeaweedFS.

**Why:** relational data (meetings → runs → stages → utterances) with transactional
checkpoints fits SQL. The unit of work commits a stage's outputs and status atomically,
which is what makes resume safe. Keeping blobs out of the database keeps it small and lets
workers on different hosts share files through S3.

**Alternatives:**
- **MongoDB** gives up transactions across collections, which checkpoints need.
- **Files only** means no querying, search or tenant scoping.
- **Postgres-only** would lose the zero-setup local run.

## Queue

**Choice:** Redis + arq, with one queue each for `cpu`, `gpu` and `llm`. A run hands itself
to the next queue as a continuation job. Redis also holds progress (pub/sub for SSE),
cancel flags, heartbeats and rate-limit windows.

**Why:** arq is asyncio-native like the rest of the code (no sync/async bridging), small,
and has retries and job ids. Splitting by resource means a GPU runs exactly one job at a
time while CPU stages and LLM calls continue in parallel. Redis is already needed for rate
limits and progress, so it adds no new infrastructure.

**Alternatives:**
- **Celery** is sync-first, with heavier configuration and less natural async task code.
- **RQ** is sync-only.
- **Dramatiq** is fine, but its async support is limited.
- **FastAPI BackgroundTasks** loses jobs on restart; it survives only as
  `PIPELINE_EXECUTION=inline` for tests.
- **Temporal** is a better fit at larger scale, but it is a whole platform to operate.

## API and web interface

**Choice:** FastAPI, plus a UI of Jinja2 templates with HTMX (vendored, 50 KB), about 200
lines of plain JavaScript and one CSS file, all served by the same process at `/`.

**Why:**
- No separate frontend build, runtime or deployment.
- The UI reuses the API's services and its `X-API-Key` auth: HTMX partials call the same
  dependency, and actions call the public JSON API.
- It works under a strict CSP with no inline scripts or styles; charts are SVG rendered on
  the server.
- The Noto fonts that the PDF export embeds are served to the browser too, so Devanagari
  and Odia render identically everywhere.

**Alternatives:**
- **Streamlit or Gradio** would be quicker for a demo, but they need a second server and
  their own auth, and they give little control over accessibility and layout.
- **A React SPA** would need a Node build and a second artifact for the same five pages.
