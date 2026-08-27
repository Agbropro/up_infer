"""Load and validate application configuration."""

from pathlib import Path
from typing import Any

import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load YAML configuration from disk."""

    config_path = path or ROOT_DIR / "config" / "config.yaml"
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    models = config.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("config.yaml must contain at least one model")

    seen: set[str] = set()
    for item in models:
        if not isinstance(item, dict) or not item.get("id") or not item.get("path"):
            raise ValueError("Each model needs an id, name, and path")
        if item["id"] in seen:
            raise ValueError(f"Duplicate model id: {item['id']}")
        seen.add(item["id"])
        item.setdefault("name", item["id"])
    return config
