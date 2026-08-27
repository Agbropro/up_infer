"""Typed response entities shared by the application."""

from typing_extensions import TypedDict


class ModelItem(TypedDict):
    """Describe one configured model."""

    id: str
    name: str
    task: str
    labels: list[str]


class ScoreItem(TypedDict):
    """Describe one predicted object."""

    label: str
    confidence: float
    color: str


class ImageItem(TypedDict):
    """Describe one inferred image."""

    name: str
    image: str
    width: int
    height: int
    predictions: list[ScoreItem]


class InferReply(TypedDict):
    """Describe one inference response."""

    model: str
    task: str
    images: list[ImageItem]
