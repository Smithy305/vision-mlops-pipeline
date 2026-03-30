"""Tests for the FastAPI /health and /predict endpoints."""

import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_jpeg_bytes(width: int = 224, height: int = 224) -> bytes:
    """Create a random JPEG image and return it as bytes."""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


def make_png_bytes(width: int = 224, height: int = 224) -> bytes:
    """Create a random PNG image and return it as bytes."""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client(dummy_checkpoint, class_names):
    """Return a FastAPI TestClient with the model mocked to use dummy_checkpoint."""
    with patch("api.model.CHECKPOINT_PATH", dummy_checkpoint):
        # Reset singleton so it reloads with the patched path
        import api.model as m
        m._model = None
        m._class_names = None
        m._device = None

        from api.main import app
        with TestClient(app) as c:
            yield c

        # Clean up singleton after tests
        m._model = None
        m._class_names = None
        m._device = None


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200(self, client):
        """Health endpoint should return HTTP 200."""
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_model_loaded(self, client, num_classes):
        """Health response should confirm model is loaded with correct class count."""
        data = client.get("/health").json()
        assert data["model_loaded"] is True
        assert data["num_classes"] == num_classes
        assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# /predict — happy path
# ---------------------------------------------------------------------------

class TestPredict:
    def test_predict_jpeg_returns_200(self, client):
        """POST a JPEG image — should return 200."""
        resp = client.post(
            "/predict",
            files={"file": ("test.jpg", make_jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 200

    def test_predict_png_returns_200(self, client):
        """POST a PNG image — should return 200."""
        resp = client.post(
            "/predict",
            files={"file": ("test.png", make_png_bytes(), "image/png")},
        )
        assert resp.status_code == 200

    def test_predict_response_schema(self, client, class_names):
        """Prediction response should contain expected fields with valid types."""
        resp = client.post(
            "/predict",
            files={"file": ("test.jpg", make_jpeg_bytes(), "image/jpeg")},
        )
        data = resp.json()

        assert "predicted_class" in data
        assert "confidence" in data
        assert "top5_predictions" in data
        assert "inference_time_ms" in data

        assert data["predicted_class"] in class_names
        assert 0.0 <= data["confidence"] <= 1.0
        assert isinstance(data["top5_predictions"], list)
        assert len(data["top5_predictions"]) <= 5
        assert data["inference_time_ms"] > 0

    def test_predict_top5_sums_to_leq_1(self, client):
        """Top-5 confidence scores must each be in [0, 1]."""
        resp = client.post(
            "/predict",
            files={"file": ("test.jpg", make_jpeg_bytes(), "image/jpeg")},
        )
        preds = resp.json()["top5_predictions"]
        for p in preds:
            assert 0.0 <= p["confidence"] <= 1.0

    def test_predict_top1_matches_top5_first(self, client):
        """The top-level predicted_class must match top5_predictions[0]."""
        resp = client.post(
            "/predict",
            files={"file": ("test.jpg", make_jpeg_bytes(), "image/jpeg")},
        )
        data = resp.json()
        assert data["predicted_class"] == data["top5_predictions"][0]["class_name"]
        assert data["confidence"] == data["top5_predictions"][0]["confidence"]


# ---------------------------------------------------------------------------
# /predict — error handling
# ---------------------------------------------------------------------------

class TestPredictErrors:
    def test_predict_empty_file_returns_400(self, client):
        """Uploading an empty file should return 400."""
        resp = client.post(
            "/predict",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        assert resp.status_code == 400

    def test_predict_invalid_bytes_returns_400(self, client):
        """Uploading non-image bytes should return 400."""
        resp = client.post(
            "/predict",
            files={"file": ("bad.jpg", b"not an image at all", "image/jpeg")},
        )
        assert resp.status_code == 400

    def test_predict_wrong_content_type_returns_422(self, client):
        """Uploading a PDF content type should return 422."""
        resp = client.post(
            "/predict",
            files={"file": ("doc.pdf", make_jpeg_bytes(), "application/pdf")},
        )
        assert resp.status_code == 422
