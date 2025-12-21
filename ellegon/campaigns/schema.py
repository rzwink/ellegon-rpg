from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from ellegon import config

_SCHEMA_CACHE: Dict[str, dict] = {}


def load_schema(path: Path | None = None) -> dict:
    schema_path = path or (config.PROJECT_ROOT / "ellegon-campaign.schema.json")
    cache_key = str(schema_path)
    if cache_key in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[cache_key]
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    _SCHEMA_CACHE[cache_key] = schema
    return schema
