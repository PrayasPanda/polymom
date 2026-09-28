.PHONY: install dev lint format typecheck test run docker-build docker-up

install:
	uv sync

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

run:
	uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -f docker/Dockerfile -t polymom:latest .

docker-up:
	docker compose -f docker/docker-compose.yml up --build
