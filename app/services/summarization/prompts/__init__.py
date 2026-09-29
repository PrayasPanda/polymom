"""Versioned Jinja2 prompt templates.

Each template starts with ``{#- version: <name>@<n> -#}``. Bump ``n`` whenever the
wording changes; the versions are stored in ``model_info.prompt_version`` and
tagged on Langfuse spans, so every summary can be traced to its prompts.
"""

import re
from functools import cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

PROMPTS_DIR = Path(__file__).parent
_VERSION = re.compile(r"\{#-?\s*version:\s*(\S+?)\s*-?#\}")

# Plain-text prompts, not HTML: autoescaping would corrupt quotes in the transcript.
_env = Environment(  # noqa: S701
    loader=FileSystemLoader(PROMPTS_DIR),
    undefined=StrictUndefined,
    keep_trailing_newline=True,
)


@cache
def prompt_version(name: str) -> str:
    match = _VERSION.search((PROMPTS_DIR / name).read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"prompt {name} has no version header")
    return match.group(1)


def render(name: str, **context: Any) -> str:
    return _env.get_template(name).render(**context).strip()
