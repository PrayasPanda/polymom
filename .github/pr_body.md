## Summary

Prompt 1 of 12: sets up the project structure and tooling for Polymom. There's no business logic yet. The app follows a layered structure, and every pipeline stage is a stub with docstrings and TODOs. The one fully implemented endpoint is `GET /api/v1/health`.

## What's included

- **Tooling:** `pyproject.toml` managed with `uv` (lockfile committed); Ruff for linting and formatting; mypy `--strict` on `app/`; pytest with pytest-asyncio and pytest-cov; pre-commit hooks (ruff, ruff-format, mypy, end-of-file-fixer, trailing-whitespace, check-added-large-files); `.editorconfig`; `.gitattributes` to force LF line endings.
- **App skeleton:**
  - `create_app()` factory with a lifespan hook
  - settings via `pydantic-settings`, with secrets typed as `SecretStr`
  - structlog JSON logging
  - `PolymomError` exception hierarchy with a JSON error envelope
  - DI providers
  - a repository `Protocol` with an in-memory implementation
  - service stubs for audio, diarization, ASR, alignment, analytics and summarization
  - the `MoMPipeline` orchestrator stub and a worker stub
- **API:** `GET /api/v1/health` returns the status, app name, version and env. The meetings routes are stubs.
- **Tests:** a health integration test, a test for the 404 error envelope, and config unit tests.
- **Docker:** multi-stage build, non-root user, `ffmpeg` installed, a `HEALTHCHECK`, and a compose file with a storage volume.
- **CI:** GitHub Actions runs lint, type-check and tests on pushes to `main` and on PRs.
- **Repo hygiene:**
  - PR and issue templates
  - a `.gitignore` covering env files, audio/video files and model caches
  - MIT license
  - README with a Mermaid architecture diagram and the 12-step roadmap

Heavy ML dependencies (torch, whisper, pyannote) are left out on purpose. They'll arrive later as optional dependency groups.

## How to run locally

```bash
cp .env.example .env
make dev
make lint typecheck test
make run
curl http://localhost:8000/api/v1/health
```

Docker: `make docker-build && make docker-up`

## Checklist

- [x] `make lint` passes
- [x] `make typecheck` passes (mypy strict)
- [x] `make test` passes
- [x] `uvicorn app.main:app` starts and `/api/v1/health` returns 200
- [x] No secrets committed (`.env` is gitignored; only `.env.example` with empty values)

## Next steps

**Prompt 2: Upload API.** Implement `POST /api/v1/meetings` as a multipart upload. It should validate the file against `MAX_UPLOAD_MB` and `ALLOWED_EXTENSIONS`, save it under `STORAGE_DIR`, create the meeting record, and return `202` with the `meeting_id`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
