.PHONY: install dev install-ml eval-asr lint format typecheck test test-slow run docker-build docker-build-ml docker-up

install:
	uv sync

install-ml:
	uv sync --all-groups --extra ml --extra indic

dev:
	uv sync --all-groups
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy

test:
	uv run pytest

test-slow:
	uv run pytest -m slow --no-cov

run:
	uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -f docker/Dockerfile -t polymom:latest .

docker-build-ml:
	docker build -f docker/Dockerfile --build-arg INSTALL_ML=true --build-arg INSTALL_INDIC=true -t polymom:ml .

docker-up:
	docker compose -f docker/docker-compose.yml up --build

eval-asr:
	uv run python scripts/eval_asr.py $(DATA)
