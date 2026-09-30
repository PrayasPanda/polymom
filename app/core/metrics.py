"""Prometheus metrics shared by the API and the workers.

The API serves them at ``/metrics``; each worker process serves its own on
``WORKER_METRICS_PORT`` (queue name as a label), so Prometheus scrapes both.
"""

from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("polymom_http_requests_total", "HTTP requests.", ["method", "route", "status"])
REQUEST_LATENCY = Histogram(
    "polymom_http_request_duration_seconds",
    "HTTP request latency.",
    ["method", "route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)
JOBS = Counter("polymom_jobs_total", "Pipeline runs by final status.", ["status"])
STAGE_DURATION = Histogram(
    "polymom_stage_duration_seconds",
    "Wall time per pipeline stage.",
    ["stage", "status"],
    buckets=(0.1, 0.5, 1, 5, 15, 30, 60, 120, 300, 600, 1800, 3600, 7200),
)
STAGE_RTF = Histogram(
    "polymom_stage_real_time_factor",
    "Stage wall time divided by audio duration (lower is faster).",
    ["stage"],
    buckets=(0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)
AUDIO_MINUTES = Counter("polymom_audio_minutes_processed_total", "Audio minutes fully processed.")
QUEUE_DEPTH = Gauge("polymom_queue_depth", "Jobs waiting per queue.", ["queue"])
STAGE_RETRIES = Counter("polymom_stage_retries_total", "Transient-failure retries.", ["stage"])
OOM_FALLBACKS = Counter("polymom_gpu_oom_fallbacks_total", "GPU OOMs recovered on CPU.", ["stage"])
LLM_TOKENS = Counter("polymom_llm_tokens_total", "LLM tokens.", ["kind"])
LLM_COST = Counter("polymom_llm_cost_usd_total", "Estimated LLM cost in USD.")
WEBHOOKS = Counter("polymom_webhooks_total", "Webhook deliveries.", ["result"])
