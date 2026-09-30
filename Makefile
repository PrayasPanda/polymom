.PHONY: install dev install-ml eval-asr lint format typecheck test test-slow run docker-build docker-build-ml docker-build-cuda docker-up \
	demo demo-down e2e-up e2e e2e-down eval-data eval-meetings benchmark benchmark-docker openapi

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

COMPOSE = docker compose -f docker/docker-compose.yml
E2E_COMPOSE = $(COMPOSE) -f docker/docker-compose.e2e.yml

docker-build-ml:
	docker build -f docker/Dockerfile --build-arg INSTALL_ML=true --build-arg INSTALL_INDIC=true -t polymom:ml .

docker-up:
	docker compose -f docker/docker-compose.yml up --build

eval-asr:
	uv run python scripts/eval_asr.py $(DATA)

docker-build-cuda:
	docker build -f docker/Dockerfile --build-arg INSTALL_ML=true --build-arg INSTALL_INDIC=true --build-arg TORCH_VARIANT=cu128 -t polymom:cuda .

# One command from a clean clone: stack with mock backends, API key, processed sample meeting.
demo:
	$(COMPOSE) up -d --build --wait
	$(COMPOSE) exec -T api python -m scripts.demo

demo-down:
	$(COMPOSE) down

e2e-up:
	$(E2E_COMPOSE) up -d --build --wait

e2e:
	uv run pytest tests/e2e -m e2e --no-cov

e2e-down:
	$(E2E_COMPOSE) down -v

# Evaluation (docs/evaluation). Real models: `make install-ml` and HF_TOKEN first.
eval-data:
	uv run python scripts/prepare_eval_data.py

eval-meetings:
	uv run python scripts/make_codemixed_meeting.py

benchmark:
	uv run python scripts/run_benchmark.py

openapi:
	uv run python -c "import json; from app.main import create_app; print(json.dumps(create_app().openapi(), indent=2, ensure_ascii=False))" > docs/openapi.json

# Benchmark inside the CUDA image (Linux, NVIDIA runtime). HF_TOKEN unlocks pyannote + Odia ASR;
# summaries use Ollama on the same Docker network (see docs/evaluation/README.md).
BENCH_ENV = -e APP_ENV=development -e HF_TOKEN -e DEVICE=cuda -e ASR_BACKEND=real \
	-e DIARIZATION_BACKEND=pyannote -e LID_BACKEND=mms -e WHISPER_MODEL_SIZE=large-v3 \
	-e WHISPER_COMPUTE_TYPE=$${WHISPER_COMPUTE_TYPE:-int8_float16} \
	-e LLM_PROVIDER=$${LLM_PROVIDER:-ollama} -e LLM_MODEL -e LLM_API_KEY \
	-e OLLAMA_BASE_URL=$${OLLAMA_BASE_URL:-http://polymom-ollama:11434} -e STORAGE_DIR=/tmp/bench
benchmark-docker:
	docker run --rm --gpus all --network polymom-bench $(BENCH_ENV) \
		-v "$(CURDIR)/data:/app/data" -v "$(CURDIR)/docs/evaluation:/app/docs/evaluation" \
		-v "$(CURDIR)/app:/app/app" -v "$(CURDIR)/scripts:/app/scripts" -v "$(CURDIR)/tests/fixtures:/app/tests/fixtures" \
		-v polymom-model-cache:/app/.cache polymom:cuda python -m scripts.run_benchmark $(ARGS)
