"""Evaluation script — loads a checkpoint and evaluates on the test set.

Outputs results/metrics.json with top-1 accuracy, top-5 accuracy,
and per-class breakdown. Used as the CI quality gate.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision import datasets, models


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DIR = Path("checkpoints")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_model(num_classes: int) -> nn.Module:
    """Return an uninitialised ResNet-50 with the correct output head."""
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_val_transform():
    """Standard ImageNet-style validation transform."""
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_test_dataset(dataset_name: str, data_dir: Path, transform):
    """Load the test split for the given dataset name."""
    if dataset_name == "fgvc-aircraft":
        ds = datasets.FGVCAircraft(
            root=str(data_dir), split="test", annotation_level="variant",
            transform=transform, download=True,
        )
        return ds, ds.classes
    elif dataset_name == "food-101":
        ds = datasets.Food101(
            root=str(data_dir), split="test", transform=transform, download=True,
        )
        return ds, ds.classes
    else:
        raise ValueError(f"Unknown dataset: {dataset_name!r}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_evaluation(
    model: nn.Module,
    loader: DataLoader,
    class_names: list[str],
    device: torch.device,
) -> dict:
    """Run full evaluation and return a metrics dictionary.

    Computes top-1 accuracy, top-5 accuracy, and per-class accuracy.
    """
    model.eval()
    num_classes = len(class_names)

    # Per-class counters
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    top1_correct = 0
    top5_correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        batch_size = images.size(0)

        # Top-1
        preds_top1 = logits.argmax(dim=1)
        top1_correct += (preds_top1 == labels).sum().item()

        # Top-5
        top5_preds = logits.topk(min(5, num_classes), dim=1).indices
        for i in range(batch_size):
            if labels[i].item() in top5_preds[i].tolist():
                top5_correct += 1

        # Per-class
        for pred, label in zip(preds_top1.cpu().tolist(), labels.cpu().tolist()):
            class_total[label] += 1
            if pred == label:
                class_correct[label] += 1

        total += batch_size

    top1_acc = top1_correct / total
    top5_acc = top5_correct / total

    per_class = {}
    for idx, name in enumerate(class_names):
        n = class_total[idx]
        c = class_correct[idx]
        per_class[name] = {
            "accuracy": round(c / n, 4) if n > 0 else None,
            "correct": c,
            "total": n,
        }

    return {
        "top1_accuracy": round(top1_acc, 4),
        "top5_accuracy": round(top5_acc, 4),
        "total_samples": total,
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate fine-grained classifier checkpoint")
    parser.add_argument(
        "--checkpoint", type=str,
        default="checkpoints/model_best.pt",
        help="Path to checkpoint file",
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--output", type=str, default="results/metrics.json",
        help="Path to write metrics JSON",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    class_names: list[str] = ckpt["class_names"]
    dataset_name: str = ckpt["dataset"]
    num_classes = len(class_names)
    print(f"Loaded checkpoint — dataset={dataset_name}, classes={num_classes}, "
          f"best_val_acc={ckpt.get('val_acc', 'N/A')}")

    # Build and load model
    model = build_model(num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # Dataset
    transform = get_val_transform()
    test_ds, _ = load_test_dataset(dataset_name, Path(args.data_dir), transform)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Evaluate
    print("Evaluating on test set …")
    metrics = run_evaluation(model, test_loader, class_names, device)

    print(f"\nTop-1 accuracy : {metrics['top1_accuracy']:.4f}")
    print(f"Top-5 accuracy : {metrics['top5_accuracy']:.4f}")
    print(f"Total samples  : {metrics['total_samples']}")

    # Write results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics written to: {output_path}")


if __name__ == "__main__":
    main()
