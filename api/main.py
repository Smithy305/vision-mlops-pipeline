"""FastAPI application for fine-grained image classification inference.

Endpoints:
  GET  /health   — liveness check
  POST /predict  — accepts an image upload, returns class + top-5 predictions
"""

import io
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from api.model import load_model, predict


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

class Prediction(BaseModel):
    """A single class prediction with its probability."""
    class_name: str
    confidence: float


class PredictResponse(BaseModel):
    """Response schema for the /predict endpoint."""
    predicted_class: str
    confidence: float
    top5_predictions: list[Prediction]
    inference_time_ms: float


class HealthResponse(BaseModel):
    """Response schema for the /health endpoint."""
    status: str
    model_loaded: bool
    num_classes: int | None


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at startup so the first request isn't slow."""
    try:
        load_model()
        print("Model loaded successfully at startup.")
    except FileNotFoundError as exc:
        print(f"WARNING: {exc}")
    yield


app = FastAPI(
    title="Vision MLOps Pipeline",
    description="Fine-grained image classification API",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """Return the liveness status and model info."""
    try:
        _, class_names, _ = load_model()
        return HealthResponse(
            status="ok",
            model_loaded=True,
            num_classes=len(class_names),
        )
    except FileNotFoundError:
        return HealthResponse(status="model_not_found", model_loaded=False, num_classes=None)


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["inference"],
    responses={
        400: {"description": "Invalid image"},
        422: {"description": "Unsupported file type"},
        503: {"description": "Model not loaded"},
    },
)
async def predict_endpoint(file: UploadFile = File(...)) -> PredictResponse:
    """Accept an image upload and return the top-5 class predictions.

    - **file**: JPEG or PNG image to classify.
    """
    # Validate content type
    allowed_types = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type '{file.content_type}'. "
                   f"Accepted: {sorted(allowed_types)}",
        )

    # Read and decode image
    raw_bytes = await file.read()
    if len(raw_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    try:
        image = Image.open(io.BytesIO(raw_bytes))
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode image. Ensure the file is a valid JPEG or PNG.",
        )

    # Run inference
    try:
        t0 = time.perf_counter()
        result = predict(image)
        elapsed_ms = (time.perf_counter() - t0) * 1000
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )

    return PredictResponse(
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        top5_predictions=[Prediction(**p) for p in result["top5_predictions"]],
        inference_time_ms=round(elapsed_ms, 2),
    )
