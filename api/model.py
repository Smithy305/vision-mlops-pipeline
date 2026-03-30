"""Model loading singleton shared by the FastAPI app and Gradio UI."""

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torchvision import models

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
_model: Optional[nn.Module] = None
_class_names: Optional[list[str]] = None
_device: Optional[torch.device] = None

CHECKPOINT_PATH = Path(
    __file__
).parent.parent / "checkpoints" / "model_best.pt"


def get_transform() -> T.Compose:
    """Return the standard inference transform."""
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_model(checkpoint_path: Optional[Path] = None) -> tuple[nn.Module, list[str], torch.device]:
    """Load the ResNet-50 checkpoint.

    Uses a module-level singleton so the model is only loaded once per process.
    Returns (model, class_names, device).
    """
    global _model, _class_names, _device

    if _model is not None:
        return _model, _class_names, _device

    path = checkpoint_path or CHECKPOINT_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {path}. "
            "Run src/train.py first to generate checkpoints/model_best.pt"
        )

    if torch.cuda.is_available():
        _device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        _device = torch.device("mps")
    else:
        _device = torch.device("cpu")
    ckpt = torch.load(path, map_location=_device)

    _class_names = ckpt["class_names"]
    num_classes = len(_class_names)

    _model = models.resnet50(weights=None)
    _model.fc = nn.Linear(_model.fc.in_features, num_classes)
    _model.load_state_dict(ckpt["model_state_dict"])
    _model.to(_device)
    _model.eval()

    return _model, _class_names, _device


@torch.no_grad()
def predict(image: Image.Image, top_k: int = 5) -> dict:
    """Run inference on a PIL image.

    Returns a dict with keys: predicted_class, confidence, top5_predictions.
    Each entry in top5_predictions is {class_name, confidence}.
    """
    model, class_names, device = load_model()
    transform = get_transform()

    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0)

    k = min(top_k, len(class_names))
    top_probs, top_indices = probs.topk(k)

    top_predictions = [
        {
            "class_name": class_names[idx.item()],
            "confidence": round(prob.item(), 4),
        }
        for prob, idx in zip(top_probs, top_indices)
    ]

    return {
        "predicted_class": top_predictions[0]["class_name"],
        "confidence": top_predictions[0]["confidence"],
        "top5_predictions": top_predictions,
    }
