"""Expose the UP Infer HTTP API."""

import json
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.application.config import load_config
from app.application.infer import run_infer
from app.application.models import ModelStore
from app.domain.entities import InferReply, ModelItem


config = load_config()
store = ModelStore(config)
router = APIRouter(prefix="/api")
limits = config.get("upload", {})
max_files = int(limits.get("max_files", 20))
max_bytes = int(limits.get("max_mb", 15)) * 1024 * 1024


@router.get("/models", response_model=list[ModelItem])
async def list_models() -> list[ModelItem]:
    """List available models, tasks, and labels."""

    try:
        return store.list_items()
    except Exception as error:
        raise HTTPException(
            status_code=503, detail=f"Could not load models: {error}"
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
