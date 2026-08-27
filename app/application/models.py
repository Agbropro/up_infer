"""Manage configured Ultralytics models."""

from pathlib import Path
from threading import Lock
from typing import Any

from app.application.config import ROOT_DIR
from app.domain.entities import ModelItem, ModelSummary


class ModelStore:
    """Load configured models once and expose their metadata."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Create a model store from validated configuration."""

        self.items = {item["id"]: item for item in config["models"]}
        self.loaded: dict[str, Any] = {}
        self.lock = Lock()

    def get_model(self, model_id: str) -> Any:
        """Return a cached model by id."""

        if model_id not in self.items:
            raise KeyError(f"Unknown model: {model_id}")
        with self.lock:
            if model_id not in self.loaded:
                from ultralytics import YOLO

                model_path = self.items[model_id]["path"]
                path = Path(model_path).expanduser()
                if not path.is_absolute() and path.suffix and path.parent != Path("."):
                    path = ROOT_DIR / path
                self.loaded[model_id] = YOLO(str(path))
        return self.loaded[model_id]

    def get_item(self, model_id: str) -> ModelItem:
        """Return metadata for one model."""

        model = self.get_model(model_id)
        item = self.items[model_id]
        names = model.names
        labels = (
            [names[key] for key in sorted(names)]
            if isinstance(names, dict)
            else list(names)
        )
        return {
            "id": model_id,
            "name": item["name"],
            "task": str(getattr(model, "task", "detect")),
            "labels": labels,
        }

    def list_items(self) -> list[ModelItem]:
        """Return metadata for every configured model."""

        return [self.get_item(model_id) for model_id in self.items]

    def list_summaries(self) -> list[ModelSummary]:
        """Return configured models without loading weights."""

        return [
            {"id": model_id, "name": item["name"]}
            for model_id, item in self.items.items()
        ]
