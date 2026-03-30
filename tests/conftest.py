"""Shared pytest fixtures."""

import io
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models


NUM_CLASSES = 10
CLASS_NAMES = [f"class_{i:02d}" for i in range(NUM_CLASSES)]


@pytest.fixture(scope="session")
def num_classes() -> int:
    """Return the number of classes used in test fixtures."""
    return NUM_CLASSES


@pytest.fixture(scope="session")
def class_names() -> list[str]:
    """Return class name list used in test fixtures."""
    return CLASS_NAMES


@pytest.fixture(scope="session")
def dummy_checkpoint(tmp_path_factory) -> Path:
    """Create a minimal ResNet-50 checkpoint for testing."""
    tmp_dir = tmp_path_factory.mktemp("checkpoints")
    ckpt_path = tmp_dir / "model_best.pt"

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    torch.save(
        {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "val_acc": 0.99,
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "dataset": "fgvc-aircraft",
        },
        ckpt_path,
    )
    return ckpt_path


@pytest.fixture()
def sample_image_bytes() -> bytes:
    """Return a valid 224x224 JPEG image as bytes."""
    img = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture()
def sample_pil_image() -> Image.Image:
    """Return a valid 224x224 PIL image."""
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)
