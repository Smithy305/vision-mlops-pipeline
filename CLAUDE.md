# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end MLOps pipeline for fine-grained image classification. ResNet-50 fine-tuned on FGVC-Aircraft (falls back to Food-101), served via FastAPI, with Gradio UI, Evidently AI drift monitoring, and a GitHub Actions CI quality gate.

## Commands

```bash
# Install
pip install -r requirements.txt

# Train (downloads dataset automatically; requires WANDB_API_KEY env var or --no-wandb)
python src/train.py --epochs 30 --batch-size 32 --data-dir data
python src/train.py --no-wandb   # without W&B

# Evaluate checkpoint → results/metrics.json
python src/evaluate.py --checkpoint checkpoints/model_best.pt

# Run FastAPI server
uvicorn api.main:app --reload --port 8000

# Run Gradio UI
python api/gradio_app.py

# Generate drift report
python monitoring/drift_report.py --checkpoint checkpoints/model_best.pt --data-dir data

# Tests
pytest tests/ -v
pytest tests/test_api.py -v                             # API tests only
pytest tests/test_drift.py::TestFlagDrift -v            # specific class

# Docker
docker build -t vision-mlops-pipeline .
docker run -p 8000:8000 vision-mlops-pipeline
```

## Architecture

```
src/train.py ──► checkpoints/model_best.pt
                        │
              ┌─────────┴──────────┐
              │                    │
         api/model.py          (also used by)
              │                monitoring/drift_report.py
    ┌─────────┴──────────┐
    │                    │
api/main.py         api/gradio_app.py
(FastAPI)           (Gradio UI)
    │
Dockerfile (port 8000)
```

**Key design decisions:**
- `api/model.py` — singleton pattern; model loaded once at startup, shared by FastAPI and Gradio
- `src/train.py` — saves `class_names` and `dataset` name inside the checkpoint so evaluate/serve/monitor are self-contained with no config files to keep in sync
- Drift monitoring uses prediction-level features (confidence, entropy, class distribution) extracted by running the model on clean vs. degraded images — not raw pixel statistics
- CI quality gate reads `results/metrics.json` produced by `evaluate.py`; skips gracefully if no checkpoint exists

## W&B

Set `WANDB_API_KEY` environment variable before training. The script skips W&B silently if the key is absent and `--no-wandb` is not passed.

## Dataset notes

FGVC-Aircraft requires accepting a license on first download — `torchvision` handles this automatically. If the download fails, the script falls back to Food-101. The dataset name is stored in the checkpoint and used by `evaluate.py`, `drift_report.py`, and the API automatically.
