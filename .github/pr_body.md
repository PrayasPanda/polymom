## Summary

Prompt 2 of 12 adds the meeting upload API. Clients can upload a recording, fetch a meeting, list meetings (paginated, newest first) and delete a meeting. Each upload is validated before it is stored: the file is streamed to disk under a size cap, its real type is checked against its magic bytes, and ffprobe confirms it contains readable audio. Meetings are saved in SQLite through SQLAlchemy 2.0 async with Alembic migrations; setting `DATABASE_URL` switches to Postgres. No audio processing or ML yet.

## Changes

- **Endpoints** (`app/api/v1/routes/meetings.py`):
  - `POST /api/v1/meetings` returns 202. Form fields: `file`, plus optional `title`, `expected_speakers` (1-20) and `languages` (`en`/`hi`/`or`, repeated or comma-separated).
  - `GET /api/v1/meetings/{id}`
  - `GET /api/v1/meetings` with `limit`/`offset`
  - `DELETE /api/v1/meetings/{id}` returns 204 and deletes the stored file too.
- **Validator** (`app/services/audio/validator.py`), checks in order:
  1. Sanitize the filename, keeping Unicode and Indic vowel signs.
  2. Check the extension against the allow-list.
  3. Stream to a `.part` file in chunks, with the size limit enforced as it streams.
  4. Reject empty files.
  5. Check magic bytes with `filetype`; m4a/mp4/mov and mkv/webm are each treated as one family because they share a container format.
  6. Run ffprobe with a timeout, rejecting unreadable files, files with no audio stream and zero-duration audio.
  7. Atomically rename the file to `uploads/{meeting_id}.{ext}`.

  Partial files are deleted on any failure, including cancellation.
- **Early 413:** a middleware rejects requests whose `Content-Length` is already over the limit before the body is read.
- **Model and schemas:** a `Meeting` ORM entity with every field from the spec, a `MeetingStatus` enum (`queued`/`processing`/`completed`/`failed`), and `AudioMetadata`. Timestamps are stored as timezone-aware UTC on every backend.
- **Storage:**
  - an abstract `MeetingRepository` with a `SqlAlchemyMeetingRepository` implementation
  - `app/db/` holds the engine and session factory plus Alembic (`alembic.ini`, migration `0001`)
  - migrations run automatically at startup (`AUTO_MIGRATE`)
  - the session, repository, validator and service are wired up in `deps.py`
- **Errors:**
  - typed exceptions: `UnsupportedFileTypeError` (415), `FileTooLargeError` (413), `CorruptedMediaError` and `EmptyFileError` (422), `MeetingNotFoundError` (404), `MediaProbeUnavailableError` (500)
  - every error, including FastAPI's own validation errors and HTTP errors, uses the envelope `{"error": {"code", "message", "details"}}`
- **Logging:** `upload_started`, `upload_validated`, `upload_rejected`, `upload_persist_failed`, `meeting_queued` and `meeting_deleted` are logged with `meeting_id` bound through contextvars. File contents are never logged.
- **OpenAPI:** response models include examples, and a documented `ErrorResponse` covers 404/413/415/422.
- **Docker and CI:** the image now installs `libmagic1` alongside `ffmpeg`, and CI installs `ffmpeg`.
- **Docs:** the README covers API usage with curl examples, the validation pipeline, the error codes, the new config variables and migrations.

## API examples

```bash
curl -F "file=@standup.m4a" -F "title=Weekly sync" -F "expected_speakers=4" \
     -F "languages=en,hi" http://localhost:8000/api/v1/meetings
# 202 {"meeting_id":"3715...","status":"queued","created_at":"2026-09-28T15:52:16.503966Z"}

curl http://localhost:8000/api/v1/meetings/3715...
# 200 {..., "mime_type":"audio/x-wav", "audio_metadata":{"codec":"pcm_s16le","sample_rate":16000,"channels":1,...}}

curl "http://localhost:8000/api/v1/meetings?limit=10&offset=0"
curl -X DELETE http://localhost:8000/api/v1/meetings/3715...   # 204

printf 'hello' > fake.wav && curl -F "file=@fake.wav" http://localhost:8000/api/v1/meetings
# 415 {"error":{"code":"unsupported_file_type","message":"File content does not match its extension.","details":{"extension":"wav","detected_mime_type":null}}}
```

## Test coverage

45 tests pass. Overall coverage is 90%; for the new code it is:

| Module | Coverage |
| --- | --- |
| `validator.py` | 94% |
| `meeting_service.py` | 100% |
| `meeting_repository.py` | 100% |
| `routes/meetings.py` | 100% |
| `main.py` | 100% |

- **Unit tests** cover:
  - valid wav and mp3
  - a disallowed extension and a missing extension
  - a text file renamed `.wav`, and mp3 content named `.wav`
  - an oversized file, checking that it is read in bounded chunks
  - empty, corrupted and zero-duration files, and a video with no audio
  - ffprobe missing and ffprobe timing out
  - filename sanitization, including path traversal and Hindi/Odia names
  - cleanup of the stored file when the database write fails
- **Integration tests** run against an isolated temp database and storage directory for each test, and cover:
  - all four endpoints
  - pagination order
  - that delete removes the file
  - 404/413/415/422 envelopes, including malformed UUIDs and invalid form fields
  - that the OpenAPI spec documents the error responses
- Media fixtures are generated at test time with Python's `wave` module and ffmpeg's `lavfi` sources; no binaries are committed. Tests that need ffmpeg are skipped when it isn't installed.
- Coverage now uses `concurrency = ["greenlet", "thread"]`. Without it, code that runs after SQLAlchemy async calls was wrongly reported as uncovered.

## Checklist

- [x] `make lint` (ruff check + format) passes
- [x] `make typecheck` (mypy strict) passes
- [x] `make test` passes, with coverage of new code at least 85%
- [x] Docker image builds with ffmpeg and libmagic
- [x] `alembic check` reports that the models match the migrations
- [x] Manual smoke test with uvicorn: upload, list and a 415 rejection
- [x] No secrets, database files or media committed

## Notes

- Starlette buffers the multipart body into a `SpooledTemporaryFile`, which moves to disk once it passes 1 MB, before the handler runs. Memory use stays bounded, and the `Content-Length` middleware rejects oversized uploads that declare their size up front. A streaming multipart parser that never spools could come later if it's needed.
- Magic-byte detection uses the pure-Python `filetype` library, so it needs no system libmagic on Windows or macOS. The Docker image still installs `libmagic1` as the spec asks.

## Next steps

**Prompt 3: audio preprocessing.** Use ffmpeg to decode, resample to 16 kHz mono and normalize loudness, with optional voice activity detection (VAD). Status moves from `queued` to `processing`, and the processing job is enqueued from `MeetingService.create`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
