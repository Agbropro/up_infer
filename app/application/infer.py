"""Run inference and format browser-friendly results."""

import base64
from io import BytesIO

from PIL import Image, UnidentifiedImageError

from app.application.models import ModelStore
from app.domain.entities import ImageItem, InferReply, ScoreItem


def get_color(class_id: int) -> str:
    """Return the Ultralytics class color as CSS hex."""

    from ultralytics.utils.plotting import colors

    red, green, blue = colors(class_id, bgr=False)
    return f"#{red:02x}{green:02x}{blue:02x}"


def open_image(data: bytes) -> Image.Image:
    """Decode uploaded bytes into an RGB image."""

    try:
        image = Image.open(BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("The uploaded file is not a readable image") from error
    return image.convert("RGB")


def encode_image(array: object) -> str:
    """Encode a plotted BGR array as a data URL."""

    image = Image.fromarray(array[:, :, ::-1])  # type: ignore[index]
    output = BytesIO()
    image.save(output, format="JPEG", quality=90, optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def get_scores(result: object) -> list[ScoreItem]:
    """Extract labels and confidence values from a result."""

    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.cls is None or boxes.conf is None:
        return []
    names = result.names
    scores: list[ScoreItem] = []
    for class_value, confidence in zip(boxes.cls.tolist(), boxes.conf.tolist()):
        class_id = int(class_value)
        scores.append(
            {
                "label": str(names[class_id]),
                "confidence": round(float(confidence), 4),
                "color": get_color(class_id),
            }
        )
    return sorted(scores, key=lambda item: item["confidence"], reverse=True)


def run_infer(
    store: ModelStore,
    model_id: str,
    uploads: list[tuple[str, bytes]],
    labels: list[str],
    confidence: float,
    iou: float,
) -> InferReply:
    """Infer uploaded images with selected labels."""

    model = store.use_model(model_id)
    item = store.get_item(model_id)
    name_map = model.names
    name_items = name_map.items() if isinstance(name_map, dict) else enumerate(name_map)
    class_ids = [key for key, value in name_items if value in labels]
    options = store.options
    batch_size = max(1, int(options.get("batch_size", 1)))
    image_size = max(32, int(options.get("image_size", 640)))
    device = options.get("device", "auto")
    predict_args = {
        "conf": confidence,
        "iou": iou,
        "classes": class_ids,
        "imgsz": image_size,
        "half": bool(options.get("half", False)),
        "verbose": False,
    }
    if device != "auto":
        predict_args["device"] = device
    output: list[ImageItem] = []
    for start in range(0, len(uploads), batch_size):
        batch = uploads[start : start + batch_size]
        images = [open_image(data) for _, data in batch]
        results = model.predict(images, **predict_args)
        for (name, _), image, result in zip(batch, images, results):
            plotted = result.plot(conf=True, labels=True, boxes=True, masks=True)
            output.append(
                {
                    "name": name,
                    "image": encode_image(plotted),
                    "width": image.width,
                    "height": image.height,
                    "predictions": get_scores(result),
                }
            )
        del results
        store.clear_cache()
    return {"model": item["name"], "task": item["task"], "images": output}
