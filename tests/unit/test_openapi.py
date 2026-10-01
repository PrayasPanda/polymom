"""The committed OpenAPI spec (docs/openapi.json) must match the app. Regenerate: make openapi."""

import json
from pathlib import Path

from app.core.config import Settings
from app.main import create_app


def test_committed_openapi_is_current(tmp_path: Path) -> None:
    spec = create_app(Settings(_env_file=None, storage_dir=tmp_path)).openapi()
    committed = json.loads((Path(__file__).parents[2] / "docs" / "openapi.json").read_text("utf-8"))
    assert committed == spec, "docs/openapi.json is stale: run `make openapi`"
    assert not any(path.startswith("/ui") for path in spec["paths"])
