# Contributing

Thanks for helping improve Polymom. This guide covers setup, conventions and what a pull
request needs.

## Setup

Prerequisites: Python 3.11+, [uv](https://docs.astral.sh/uv/), ffmpeg 6 or newer, make,
and Docker (for the e2e suite and images).

```bash
git clone https://github.com/PrayasPanda/polymom.git && cd polymom
make dev            # uv sync --all-groups + pre-commit hooks
make install-ml     # optional: real models (torch CPU, pyannote, faster-whisper, IndicConformer)
```

## Checks

| Command | What it runs |
| --- | --- |
| `make lint` | ruff check + ruff format --check |
| `make typecheck` | mypy --strict on `app/` |
| `make test` | unit + integration tests with coverage (fails under 85% in CI); slow and e2e excluded |
| `make e2e-up && make e2e` | end-to-end suite against the docker compose stack with mock backends |
| `make test-slow` | real-model tests (needs `make install-ml` and `HF_TOKEN`) |
| `make openapi` | regenerates `docs/openapi.json` (a test fails when it is stale) |

Test markers: `slow` (real models), `llm` (real LLM provider), `postgres` and `redis`
(testcontainers, skipped without Docker), `e2e` (compose stack).

## Conventions

- **Branches**: `feat/...`, `fix/...`, `docs/...`, `release/vX.Y.Z` off `main`.
- **Commits**: [Conventional Commits](https://www.conventionalcommits.org/) (`feat(ui): ...`,
  `fix: ...`, `test: ...`, `docs: ...`, `build: ...`, `ci: ...`, `chore(release): ...`).
  Keep them small and focused.
- **Code**: type hints everywhere (`mypy --strict`), with docstrings on public functions
  that explain *why*. Errors are `PolymomError` subclasses with a stable `code` and a
  `remediation`. Logs are structlog events with snake_case names and keyword fields (never
  f-strings in the event name).
- **New stage**: subclass `PipelineStage` (set `name`, `queue`, `config_keys`), register it
  in `build_pipeline`, and add a mock backend if it uses a model, so CI runs without GPUs.
- **New model backend**: implement the stage's interface (`ASRBackend`,
  `DiarizationBackend`, `LanguageIdentifier`, `LLMClient`), make it selectable by setting,
  and document it in `docs/TECHNOLOGY_CHOICES.md`.
- **Settings**: every new setting goes in `app/core/config.py` and `.env.example` (a test
  checks this) and in `docs/OPERATIONS.md`.
- **Never commit** audio, model weights, `.env` or tokens. Test media is generated at test
  time; evaluation data is downloaded by `scripts/prepare_eval_data.py`.

## Pull requests

Fill in the template. CI must be green (lint, types, tests with coverage ≥ 85%,
pip-audit, gitleaks, e2e, Docker build + Trivy). Include:

- tests for new behaviour and for bug fixes (a test that fails before the fix);
- docs updates for user-visible changes, and a CHANGELOG entry under *Unreleased*;
- updated benchmark numbers (`docs/evaluation/`) if you change a model or its settings.

## Releases

Maintainers merge `release/vX.Y.Z`, then tag it on `main`:

```bash
git checkout main && git pull && git tag -a vX.Y.Z -m "vX.Y.Z" && git push origin vX.Y.Z
```

The Release workflow builds and scans the CPU and CUDA images, pushes them to GHCR, and
creates the GitHub release.

By contributing, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md) and to
license your work under the [MIT License](LICENSE).
