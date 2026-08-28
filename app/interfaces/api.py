"""Expose the UP Infer HTTP API."""

import json
import os
from typing import Annotated, Literal

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from app.application.config import load_config
from app.application.infer import run_infer
from app.application.models import ModelStore
from app.domain.entities import InferReply, ModelItem, ModelSummary


config = load_config()
store = ModelStore(config)
router = APIRouter(prefix="/api")
limits = config.get("upload", {})
max_files = int(limits.get("max_files", 20))
max_bytes = int(limits.get("max_mb", 15)) * 1024 * 1024


class FeedbackRequest(BaseModel):
    """Validate feedback submitted by the browser widget."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    ticket_type: Literal["misc", "bug", "feature"]
    title: str = Field(min_length=3, max_length=160)
    description: str = Field(min_length=3, max_length=10_000)
    page_url: str = Field(max_length=2_048)
    selected_model: str | None = Field(default=None, max_length=100)
    viewport: str | None = Field(default=None, max_length=50)


async def _send_ticket(
    url: str,
    api_key: str,
    payload: dict[str, object],
) -> dict[str, object]:
    """Send one ticket without blocking the FastAPI event loop."""

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                f"{url.rstrip('/')}/api/tickets",
                json=payload,
                headers={
                    "User-Agent": "up-infer/1.0",
                    "x-api-key": api_key,
                },
            )
    except httpx.RequestError as error:
        raise RuntimeError("Ticket service could not be reached") from error

    if response.is_error:
        try:
            message = response.json().get("error", {}).get("message")
        except (json.JSONDecodeError, AttributeError, TypeError):
            message = None
        raise RuntimeError(
            message or f"Ticket service returned HTTP {response.status_code}"
        )
    return response.json()


@router.get("/models", response_model=list[ModelSummary])
async def list_models() -> list[ModelSummary]:
    """List configured models without loading weights."""

    return store.list_summaries()


@router.post("/feedback", status_code=201)
async def submit_feedback(feedback: FeedbackRequest) -> dict[str, object]:
    """Forward browser feedback without exposing the central API key."""

    service_url = os.getenv("TICKET_SERVICE_URL", "https://ticket.agbropro.my.id")
    api_key = os.getenv("TICKET_SERVICE_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Feedback service is not configured",
        )

    type_mapping = {"misc": "general", "bug": "bug", "feature": "feedback"}
    metadata = {
        "feedback_type": feedback.ticket_type,
        "source": "feedback-widget",
        "page_url": feedback.page_url,
        "selected_model": feedback.selected_model,
        "viewport": feedback.viewport,
    }
    ticket_payload: dict[str, object] = {
        "project_name": "up-infer",
        "ticket_type": type_mapping[feedback.ticket_type],
        "content": f"{feedback.title}\n\n{feedback.description}",
        "metadata": {
            key: value for key, value in metadata.items() if value is not None
        },
    }

    try:
        result = await _send_ticket(service_url, api_key, ticket_payload)
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return result


@router.get("/models/{model_id}", response_model=ModelItem)
async def get_model(model_id: str) -> ModelItem:
    """Load one model and return its metadata."""

    try:
        return store.get_item(model_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=503, detail=f"Could not load model: {error}"
        ) from error


@router.post("/infer", response_model=InferReply)
async def infer_images(
    files: Annotated[list[UploadFile], File()],
    model: Annotated[str, Form()],
    labels: Annotated[str, Form()],
    confidence: Annotated[float, Form()] = 0.25,
    iou: Annotated[float, Form()] = 0.7,
) -> InferReply:
    """Infer one or more uploaded images."""

    if not files or len(files) > max_files:
        raise HTTPException(
            status_code=400, detail=f"Upload between 1 and {max_files} images"
        )
    if not 0 <= confidence <= 1 or not 0 <= iou <= 1:
        raise HTTPException(
            status_code=400, detail="Confidence and IoU must be between 0 and 1"
        )
    try:
        chosen = json.loads(labels)
    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=400, detail="Labels must be a JSON list"
        ) from error
    if not isinstance(chosen, list) or not all(
        isinstance(item, str) for item in chosen
    ):
        raise HTTPException(status_code=400, detail="Labels must be a list of names")
    if not chosen:
        raise HTTPException(status_code=400, detail="Select at least one label")

    uploads: list[tuple[str, bytes]] = []
    for upload in files:
        if upload.content_type and not upload.content_type.startswith("image/"):
            raise HTTPException(
                status_code=415, detail=f"{upload.filename} is not an image"
            )
        data = await upload.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(
                status_code=413, detail=f"{upload.filename} is larger than the limit"
            )
        uploads.append((upload.filename or "image", data))
    try:
        return run_infer(store, model, uploads, chosen, confidence, iou)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500, detail=f"Inference failed: {error}"
        ) from error
