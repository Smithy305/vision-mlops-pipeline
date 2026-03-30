"""Drift monitoring using Evidently AI.

Compares a reference distribution (clean test images) against a degraded
distribution (Gaussian blur + JPEG compression + Gaussian noise) to simulate
real-world image quality degradation.

Usage:
    python monitoring/drift_report.py \
        --checkpoint checkpoints/model_best.pt \
        --data-dir data \
        --n-samples 500

Outputs:
    monitoring/reports/drift_report.html  — full Evidently HTML report
    monitoring/reports/drift_summary.json — machine-readable summary
"""

import argparse
import json
import os
import sys
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from api.model import load_model, predict

REPORTS_DIR = Path("monitoring/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Image degradation helpers
# ---------------------------------------------------------------------------

def apply_gaussian_blur(image: Image.Image, radius: float = 3.0) -> Image.Image:
    """Apply Gaussian blur to simulate out-of-focus or motion blur."""
    from PIL import ImageFilter
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_jpeg_compression(image: Image.Image, quality: int = 20) -> Image.Image:
    """Apply JPEG compression artefacts to simulate lossy transmission."""
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def apply_gaussian_noise(image: Image.Image, std: float = 30.0) -> Image.Image:
    """Add Gaussian noise to simulate sensor noise."""
    arr = np.array(image, dtype=np.float32)
    noise = np.random.normal(0, std, arr.shape).astype(np.float32)
    degraded = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(degraded)


def degrade_image(image: Image.Image) -> Image.Image:
    """Apply the full degradation pipeline: blur → JPEG → noise."""
    image = apply_gaussian_blur(image, radius=2.5)
    image = apply_jpeg_compression(image, quality=25)
    image = apply_gaussian_noise(image, std=20.0)
    return image


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def get_prediction_features(
    images: list[Image.Image],
    class_names: list[str],
    batch_size: int = 32,
) -> pd.DataFrame:
    """Run inference on a list of PIL images and return a feature DataFrame.

    Columns: predicted_class, confidence, top1_label_index,
             entropy (prediction uncertainty).
    """
    model, _class_names, device = load_model()
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    rows = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch_pil = images[i : i + batch_size]
            tensors = torch.stack([
                transform(img.convert("RGB")) for img in batch_pil
            ]).to(device)

            logits = model(tensors)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            for prob_row in probs:
                pred_idx = int(prob_row.argmax())
                confidence = float(prob_row.max())
                # Shannon entropy (uncertainty measure)
                entropy = float(-np.sum(prob_row * np.log(prob_row + 1e-9)))
                rows.append({
                    "predicted_class": class_names[pred_idx],
                    "confidence": confidence,
                    "top1_label_index": pred_idx,
                    "entropy": entropy,
                })

    return pd.DataFrame(rows)


def load_sample_images(
    dataset_name: str,
    data_dir: Path,
    n_samples: int,
) -> tuple[list[Image.Image], list[str]]:
    """Load n_samples raw PIL images from the test split."""
    # Load without any transform (we want raw PIL)
    if dataset_name == "fgvc-aircraft":
        ds = datasets.FGVCAircraft(
            root=str(data_dir), split="test", annotation_level="variant",
            transform=None, download=True,
        )
        class_names = ds.classes
    else:
        ds = datasets.Food101(
            root=str(data_dir), split="test", transform=None, download=True,
        )
        class_names = ds.classes

    n = min(n_samples, len(ds))
    indices = np.random.choice(len(ds), n, replace=False).tolist()

    images = []
    for idx in indices:
        img, _ = ds[idx]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        images.append(img.convert("RGB"))

    return images, class_names


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

def compute_distribution_shift(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
) -> dict:
    """Compute simple statistical drift metrics between reference and current.

    Returns a dict with:
      - confidence_mean_shift: absolute difference in mean confidence
      - entropy_mean_shift: absolute difference in mean entropy
      - top1_class_distribution_shift: L1 distance between class frequency vectors
    """
    conf_shift = abs(ref_df["confidence"].mean() - cur_df["confidence"].mean())
    entropy_shift = abs(ref_df["entropy"].mean() - cur_df["entropy"].mean())

    # Class distribution
    all_classes = sorted(
        set(ref_df["predicted_class"]) | set(cur_df["predicted_class"])
    )
    ref_counts = ref_df["predicted_class"].value_counts(normalize=True)
    cur_counts = cur_df["predicted_class"].value_counts(normalize=True)

    l1 = sum(
        abs(ref_counts.get(c, 0.0) - cur_counts.get(c, 0.0))
        for c in all_classes
    )

    return {
        "confidence_mean_shift": round(conf_shift, 4),
        "entropy_mean_shift": round(entropy_shift, 4),
        "class_distribution_l1": round(l1, 4),
        "ref_mean_confidence": round(ref_df["confidence"].mean(), 4),
        "cur_mean_confidence": round(cur_df["confidence"].mean(), 4),
        "ref_mean_entropy": round(ref_df["entropy"].mean(), 4),
        "cur_mean_entropy": round(cur_df["entropy"].mean(), 4),
    }


def flag_drift(
    drift_metrics: dict,
    confidence_threshold: float = 0.10,
    entropy_threshold: float = 0.50,
    distribution_threshold: float = 0.30,
) -> tuple[bool, list[str]]:
    """Return (drift_detected, reasons) based on configurable thresholds.

    Args:
        drift_metrics: output of compute_distribution_shift().
        confidence_threshold: max allowed absolute drop in mean confidence.
        entropy_threshold: max allowed absolute rise in mean entropy.
        distribution_threshold: max allowed L1 class distribution shift.

    Returns:
        (drift_detected: bool, reasons: list of human-readable strings)
    """
    reasons = []

    if drift_metrics["confidence_mean_shift"] > confidence_threshold:
        reasons.append(
            f"Mean confidence dropped by "
            f"{drift_metrics['confidence_mean_shift']:.3f} "
            f"(threshold={confidence_threshold})"
        )

    if drift_metrics["entropy_mean_shift"] > entropy_threshold:
        reasons.append(
            f"Mean prediction entropy increased by "
            f"{drift_metrics['entropy_mean_shift']:.3f} "
            f"(threshold={entropy_threshold})"
        )

    if drift_metrics["class_distribution_l1"] > distribution_threshold:
        reasons.append(
            f"Class distribution L1 shift = "
            f"{drift_metrics['class_distribution_l1']:.3f} "
            f"(threshold={distribution_threshold})"
        )

    return bool(reasons), reasons


# ---------------------------------------------------------------------------
# Evidently report
# ---------------------------------------------------------------------------

def generate_evidently_report(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    output_path: Path,
):
    """Generate an Evidently HTML data drift report for numerical features.

    Falls back to a plain-JSON summary if Evidently is not installed.
    """
    try:
        from evidently import ColumnMapping
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset

        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ])

        # Evidently expects a column mapping; treat predicted_class as target
        column_mapping = ColumnMapping(
            target="top1_label_index",
            numerical_features=["confidence", "entropy"],
            categorical_features=["predicted_class"],
        )

        report.run(
            reference_data=ref_df,
            current_data=cur_df,
            column_mapping=column_mapping,
        )
        report.save_html(str(output_path))
        print(f"Evidently HTML report saved to: {output_path}")

    except ImportError:
        # Evidently not installed — write a minimal HTML summary
        drift_summary = compute_distribution_shift(ref_df, cur_df)
        html = (
            "<html><body><h1>Drift Report (Evidently not installed)</h1>"
            f"<pre>{json.dumps(drift_summary, indent=2)}</pre>"
            "</body></html>"
        )
        output_path.write_text(html)
        print(f"Plain HTML summary saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Generate image drift report")
    p.add_argument("--checkpoint", default="checkpoints/model_best.pt")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--n-samples", type=int, default=500,
                   help="Number of test images to use for each split")
    p.add_argument("--confidence-threshold", type=float, default=0.10)
    p.add_argument("--entropy-threshold", type=float, default=0.50)
    p.add_argument("--distribution-threshold", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    from pathlib import Path as _Path
    from api import model as _model_module
    _model_module.CHECKPOINT_PATH = _Path(args.checkpoint)

    # Load checkpoint metadata to get dataset name
    device = torch.device("cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    dataset_name = ckpt["dataset"]
    class_names = ckpt["class_names"]
    print(f"Dataset: {dataset_name} | Classes: {len(class_names)}")

    # Load clean reference images
    print(f"Loading {args.n_samples} reference images …")
    ref_images, _ = load_sample_images(dataset_name, Path(args.data_dir), args.n_samples)

    # Apply degradation to create the current split
    print("Applying degradation pipeline (blur + JPEG + noise) …")
    cur_images = [degrade_image(img) for img in ref_images]

    # Extract prediction features
    print("Running inference on reference images …")
    ref_df = get_prediction_features(ref_images, class_names)

    print("Running inference on degraded images …")
    cur_df = get_prediction_features(cur_images, class_names)

    # Compute drift metrics
    drift_metrics = compute_distribution_shift(ref_df, cur_df)
    drift_detected, reasons = flag_drift(
        drift_metrics,
        confidence_threshold=args.confidence_threshold,
        entropy_threshold=args.entropy_threshold,
        distribution_threshold=args.distribution_threshold,
    )

    # Console summary
    print("\n── Drift Metrics ──────────────────────────────")
    for k, v in drift_metrics.items():
        print(f"  {k}: {v}")
    print()
    if drift_detected:
        print("DRIFT DETECTED:")
        for r in reasons:
            print(f"  • {r}")
    else:
        print("No significant drift detected.")

    # Save JSON summary
    summary = {
        "drift_detected": drift_detected,
        "reasons": reasons,
        "metrics": drift_metrics,
        "thresholds": {
            "confidence": args.confidence_threshold,
            "entropy": args.entropy_threshold,
            "distribution": args.distribution_threshold,
        },
        "n_samples": args.n_samples,
    }
    summary_path = REPORTS_DIR / "drift_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nDrift summary saved to: {summary_path}")

    # Generate full Evidently HTML report
    report_path = REPORTS_DIR / "drift_report.html"
    generate_evidently_report(ref_df, cur_df, report_path)


if __name__ == "__main__":
    main()
