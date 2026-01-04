import base64
import io
import os
from typing import Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw, ImageFont

from .model import get_model

app = FastAPI(title="Action Recognition API", version="0.1.0")

# To keep local dev simple, allow all origins by default. You can tighten this
# via ALLOWED_ORIGINS (comma-separated) if desired.
env_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
allow_all = not env_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_all else env_origins,
    allow_origin_regex=None,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/labels")
def labels() -> Dict[str, list[str]]:
    model = get_model()
    return {"labels": model.class_names}


@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)) -> JSONResponse:
    """Accept a single image and predict the action (for image-trained models)."""
    if file.content_type is None or not file.content_type.startswith("image"):
        raise HTTPException(status_code=400, detail=f"Invalid image file type: {file.content_type}")

    try:
        file_bytes = await file.read()
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid image data: {exc}") from exc

    model = get_model()
    label, score, scores = model.predict(image)

    annotated = _annotate_image(image, label, score)
    encoded = _encode_image(annotated)

    note = (
        f"Model weights loaded from {model.weights_path}" if model.weights_loaded else "Model is randomly initialized; replace weights for real predictions."
    )

    payload = {
        "label": label,
        "score": round(score, 4),
        "scores": {k: round(v, 4) for k, v in scores.items()},
        "annotated_image_base64": encoded,
        "note": note,
    }
    return JSONResponse(payload)


@app.post("/predict")
async def predict(frames: list[UploadFile] = File(...)) -> JSONResponse:
    """Accept frames from a video; if a single image is provided, fallback to image prediction."""
    if len(frames) == 0:
        raise HTTPException(status_code=400, detail="No frames provided.")

    # Single image: delegate to image endpoint for image-only models
    if len(frames) == 1:
        return await predict_image(file=frames[0])

    if len(frames) < 8:
        raise HTTPException(status_code=400, detail=f"Expected at least 8 frames, got {len(frames)}.")

    try:
        frame_images = []
        for frame_file in frames:
            if frame_file.content_type is None or not frame_file.content_type.startswith("image"):
                raise ValueError(f"Invalid frame file type: {frame_file.content_type}")
            frame_bytes = await frame_file.read()
            frame_image = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
            frame_images.append(frame_image)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid frame data: {exc}") from exc

    model = get_model()
    label, score, scores = model.predict_sequence(frame_images)

    preview = frame_images[0].copy()
    annotated = _annotate_image(preview, label, score)
    encoded = _encode_image(annotated)

    note = (
        f"Model weights loaded from {model.weights_path}" if model.weights_loaded else "Model is randomly initialized; replace weights for real predictions."
    )

    payload = {
        "label": label,
        "score": round(score, 4),
        "scores": {k: round(v, 4) for k, v in scores.items()},
        "annotated_image_base64": encoded,
        "note": note,
    }
    return JSONResponse(payload)


def _annotate_image(image: Image.Image, label: str, score: float) -> Image.Image:
    copy = image.copy()
    draw = ImageDraw.Draw(copy)
    font = ImageFont.load_default()
    text = f"{label} ({score:.2f})"
    margin = 8
    # textbbox returns (left, top, right, bottom)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = right - left, bottom - top
    box = [margin, margin, margin + text_w + 8, margin + text_h + 8]
    draw.rectangle(box, fill="black")
    draw.text((margin + 4, margin + 4), text, fill="white", font=font)
    return copy


def _encode_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
