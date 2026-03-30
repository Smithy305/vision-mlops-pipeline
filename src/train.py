"""Training script for fine-grained image classification on FGVC-Aircraft.

Falls back to Food-101 if FGVC-Aircraft is unavailable.
Logs training/validation metrics, hyperparameters, and sample predictions to W&B.
Saves the best checkpoint as model_best.pt.
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import wandb
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def get_aircraft_transforms():
    """Return train/val image transforms for FGVC-Aircraft."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tf = T.Compose([
        T.RandomResizedCrop(224, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    val_tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train_tf, val_tf


def load_aircraft(data_dir: Path, train_tf, val_tf):
    """Load FGVC-Aircraft splits (variant-level, 100 classes).

    Returns (train_dataset, val_dataset, test_dataset, class_names).
    """
    train_ds = datasets.FGVCAircraft(
        root=str(data_dir), split="train", annotation_level="variant",
        transform=train_tf, download=True,
    )
    val_ds = datasets.FGVCAircraft(
        root=str(data_dir), split="val", annotation_level="variant",
        transform=val_tf, download=True,
    )
    test_ds = datasets.FGVCAircraft(
        root=str(data_dir), split="test", annotation_level="variant",
        transform=val_tf, download=True,
    )
    return train_ds, val_ds, test_ds, train_ds.classes


def load_food101(data_dir: Path, train_tf, val_tf):
    """Fallback: load Food-101 (101 classes).

    Returns (train_dataset, val_dataset, test_dataset, class_names).
    """
    train_ds = datasets.Food101(
        root=str(data_dir), split="train", transform=train_tf, download=True,
    )
    test_ds = datasets.Food101(
        root=str(data_dir), split="test", transform=val_tf, download=True,
    )
    # Use 10 % of train as validation
    n_val = int(0.1 * len(train_ds))
    indices = list(range(len(train_ds)))
    random.shuffle(indices)
    val_ds = Subset(train_ds, indices[:n_val])
    # Re-apply val transforms to the subset's underlying dataset
    val_ds.dataset = datasets.Food101(
        root=str(data_dir), split="train", transform=val_tf, download=False,
    )
    train_ds = Subset(train_ds, indices[n_val:])
    return train_ds, val_ds, test_ds, test_ds.classes


def get_datasets(data_dir: Path, train_tf, val_tf):
    """Load FGVC-Aircraft; fall back to Food-101 on any error."""
    try:
        print("Attempting to load FGVC-Aircraft …")
        splits = load_aircraft(data_dir, train_tf, val_tf)
        print(f"Loaded FGVC-Aircraft — {len(splits[3])} classes")
        return splits, "fgvc-aircraft"
    except Exception as exc:
        print(f"FGVC-Aircraft unavailable ({exc}). Falling back to Food-101.")
        splits = load_food101(data_dir, train_tf, val_tf)
        print(f"Loaded Food-101 — {len(splits[3])} classes")
        return splits, "food-101"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(num_classes: int) -> nn.Module:
    """Return a pretrained ResNet-50 with the final layer replaced."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    """Run one training epoch.

    Returns (mean_loss, accuracy).
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        if batch_idx % 50 == 0:
            print(
                f"  Epoch {epoch} [{batch_idx}/{len(loader)}]  "
                f"loss={loss.item():.4f}"
            )

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model on a dataloader.

    Returns (mean_loss, accuracy).
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# W&B prediction logging
# ---------------------------------------------------------------------------

@torch.no_grad()
def log_sample_predictions(
    model: nn.Module,
    loader: DataLoader,
    class_names: list[str],
    device: torch.device,
    n_samples: int = 16,
):
    """Log a grid of sample predictions with confidence scores to W&B."""
    model.eval()
    images_batch, labels_batch = next(iter(loader))
    images_batch = images_batch[:n_samples].to(device)
    labels_batch = labels_batch[:n_samples]

    logits = model(images_batch)
    probs = torch.softmax(logits, dim=1).cpu()
    preds = probs.argmax(dim=1)
    confidences = probs.max(dim=1).values

    # Un-normalise for display
    inv_mean = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225])
    inv_std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225])

    wandb_images = []
    for i in range(len(images_batch)):
        img = images_batch[i].cpu().clone()
        for c in range(3):
            img[c] = img[c] / inv_std[c] - inv_mean[c]  # approximate un-norm
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()
        img_pil = Image.fromarray((img * 255).astype(np.uint8))

        true_label = class_names[labels_batch[i].item()]
        pred_label = class_names[preds[i].item()]
        conf = confidences[i].item()
        caption = f"GT: {true_label}\nPred: {pred_label} ({conf:.1%})"
        wandb_images.append(wandb.Image(img_pil, caption=caption))

    wandb.log({"sample_predictions": wandb_images})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train ResNet-50 fine-grained classifier")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None, metavar="CHECKPOINT",
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--wandb-project", type=str, default="vision-mlops-pipeline")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Initialise W&B — resume the original run if a checkpoint is provided
    use_wandb = not args.no_wandb and os.environ.get("WANDB_API_KEY") is not None
    if use_wandb:
        resume_run_id = None
        if args.resume:
            _ckpt_peek = torch.load(args.resume, map_location="cpu")
            resume_run_id = _ckpt_peek.get("wandb_run_id")

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            id=resume_run_id,
            resume="allow" if resume_run_id else None,
        )
    else:
        if not args.no_wandb:
            print("WANDB_API_KEY not set — running without W&B logging.")

    # Data
    train_tf, val_tf = get_aircraft_transforms()
    (train_ds, val_ds, test_ds, class_names), dataset_name = get_datasets(
        Path(args.data_dir), train_tf, val_tf
    )
    num_classes = len(class_names)
    print(f"Dataset: {dataset_name} | Classes: {num_classes}")

    # Save class names for inference
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_DIR / "class_names.json", "w") as f:
        json.dump(class_names, f)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory,
    )

    # Model, loss, optimiser
    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    if use_wandb:
        wandb.config.update({"num_classes": num_classes, "dataset": dataset_name})

    # Resume from checkpoint if requested
    start_epoch = 1
    best_val_acc = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt["val_acc"]
        # Advance scheduler to match resumed epoch
        for _ in range(ckpt["epoch"]):
            scheduler.step()
        print(f"Resumed from epoch {ckpt['epoch']} (best val_acc={best_val_acc:.4f})")

    # Training loop
    best_ckpt_path = CHECKPOINT_DIR / "model_best.pt"

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "lr": scheduler.get_last_lr()[0],
            })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "num_classes": num_classes,
                    "class_names": class_names,
                    "dataset": dataset_name,
                    "wandb_run_id": wandb.run.id if use_wandb else None,
                },
                best_ckpt_path,
            )
            print(f"  ✓ Saved best checkpoint (val_acc={val_acc:.4f})")

    # Log sample predictions from validation set
    if use_wandb:
        model.load_state_dict(
            torch.load(best_ckpt_path, map_location=device)["model_state_dict"]
        )
        log_sample_predictions(model, val_loader, class_names, device)
        wandb.summary["best_val_acc"] = best_val_acc
        wandb.finish()

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint saved to: {best_ckpt_path}")


if __name__ == "__main__":
    main()
