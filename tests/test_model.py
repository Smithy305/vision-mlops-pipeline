"""Model sanity checks — output shape, valid probabilities, checkpoint loading."""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torchvision import models


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_resnet50(num_classes: int) -> nn.Module:
    """Build a ResNet-50 with a custom head (no pretrained weights)."""
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestModelOutputShape:
    def test_output_shape_matches_num_classes(self, num_classes):
        """Model output should have shape (batch, num_classes)."""
        model = build_resnet50(num_classes)
        model.eval()
        dummy = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (4, num_classes)

    def test_single_image_output_shape(self, num_classes):
        """Single-image forward pass should return shape (1, num_classes)."""
        model = build_resnet50(num_classes)
        model.eval()
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (1, num_classes)


class TestModelProbabilities:
    def test_softmax_sums_to_one(self, num_classes):
        """Softmax over logits must sum to 1.0 for every sample in the batch."""
        model = build_resnet50(num_classes)
        model.eval()
        dummy = torch.randn(8, 3, 224, 224)
        with torch.no_grad():
            logits = model(dummy)
        probs = torch.softmax(logits, dim=1)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(8), atol=1e-5)

    def test_probabilities_in_zero_one(self, num_classes):
        """All softmax probabilities must lie in [0, 1]."""
        model = build_resnet50(num_classes)
        model.eval()
        dummy = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            logits = model(dummy)
        probs = torch.softmax(logits, dim=1)
        assert probs.min().item() >= 0.0
        assert probs.max().item() <= 1.0


class TestCheckpointLoading:
    def test_load_model_from_dummy_checkpoint(self, dummy_checkpoint, num_classes, class_names):
        """load_model() should successfully load the dummy checkpoint."""
        with patch("api.model.CHECKPOINT_PATH", dummy_checkpoint):
            import api.model as m
            # Reset singleton
            m._model = None
            m._class_names = None
            m._device = None

            model, loaded_classes, device = m.load_model()

            assert model is not None
            assert loaded_classes == class_names
            assert isinstance(device, torch.device)

            # Clean up
            m._model = None
            m._class_names = None
            m._device = None

    def test_predict_returns_valid_result(self, dummy_checkpoint, class_names, sample_pil_image):
        """predict() should return a dict with expected keys and valid values."""
        with patch("api.model.CHECKPOINT_PATH", dummy_checkpoint):
            import api.model as m
            m._model = None
            m._class_names = None
            m._device = None

            result = m.predict(sample_pil_image)

            assert "predicted_class" in result
            assert "confidence" in result
            assert "top5_predictions" in result
            assert result["predicted_class"] in class_names
            assert 0.0 <= result["confidence"] <= 1.0
            assert 1 <= len(result["top5_predictions"]) <= 5

            m._model = None
            m._class_names = None
            m._device = None

    def test_missing_checkpoint_raises_file_not_found(self, tmp_path):
        """load_model() must raise FileNotFoundError for a missing checkpoint."""
        import api.model as m
        m._model = None
        m._class_names = None
        m._device = None

        missing = tmp_path / "does_not_exist.pt"
        with patch("api.model.CHECKPOINT_PATH", missing):
            with pytest.raises(FileNotFoundError):
                m.load_model()

        m._model = None
        m._class_names = None
        m._device = None
