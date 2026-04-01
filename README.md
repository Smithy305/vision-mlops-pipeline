---
title: Vision MLOps Pipeline
emoji: ✈️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.31.4
app_file: app.py
pinned: false
---

# Vision MLOps Pipeline

End-to-end MLOps pipeline for fine-grained image classification — ResNet-50 fine-tuned on FGVC-Aircraft, served via FastAPI, monitored with Evidently AI, and gated by GitHub Actions CI.

> **Live demo:** [Hugging Face Spaces](https://huggingface.co/spaces/Smithy305/vision-mlops-pipeline)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Training                                │
│  torchvision (FGVC-Aircraft / Food-101)                         │
│        │                                                        │
│        ▼                                                        │
│  src/train.py  ──── W&B (metrics, hyperparams, predictions) ── │
│        │                                                        │
│        ▼                                                        │
│  checkpoints/model_best.pt                                      │
└────────────────────┬────────────────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
┌─────────────────┐   ┌──────────────────────┐
│  Serving        │   │  Monitoring          │
│                 │   │                      │
│  api/main.py    │   │  monitoring/         │
│  (FastAPI)      │   │  drift_report.py     │
│       │         │   │  (Evidently AI)      │
│       ▼         │   │                      │
│  api/           │   │  Degradation:        │
│  gradio_app.py  │   │  blur + JPEG + noise │
│  (Gradio UI)    │   └──────────────────────┘
│       │         │
│       ▼         │
│  Dockerfile     │
│  (port 8000)    │
└─────────────────┘
          │
          ▼
┌─────────────────────────────────────────────┐
│  CI/CD  (.github/workflows/ci.yml)          │
│  • pytest test suite                        │
│  • evaluate.py → top-1 ≥ 70% gate          │
└─────────────────────────────────────────────┘
```

---

## Setup

```bash
# Clone
git clone https://github.com/Smithy305/vision-mlops-pipeline.git
cd vision-mlops-pipeline

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Set your W&B API key (required for experiment tracking):

```bash
export WANDB_API_KEY=your_key_here
```

---

## Training

```bash
python src/train.py \
  --epochs 30 \
  --batch-size 32 \
  --lr 1e-3 \
  --data-dir data \
  --wandb-project vision-mlops-pipeline
```

The script downloads FGVC-Aircraft automatically (falls back to Food-101 if unavailable).
The best checkpoint is saved to `checkpoints/model_best.pt`.

To train without W&B:

```bash
python src/train.py --no-wandb
```

---

## Evaluation

```bash
python src/evaluate.py \
  --checkpoint checkpoints/model_best.pt \
  --output results/metrics.json
```

Outputs `results/metrics.json` with top-1 accuracy, top-5 accuracy, and per-class breakdown.

---

## Running the API locally

```bash
uvicorn api.main:app --reload --port 8000
```

- **POST** `/predict` — upload an image, receive top-5 predictions
- **GET**  `/health`  — liveness check

Interactive docs at <http://localhost:8000/docs>.

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/image.jpg"
```

---

## Gradio UI

```bash
python api/gradio_app.py
```

Opens at <http://localhost:7860>.

---

## Docker

```bash
# Build (requires checkpoints/model_best.pt to exist)
docker build -t vision-mlops-pipeline .

# Run
docker run -p 8000:8000 vision-mlops-pipeline
```

---

## Drift monitoring

```bash
python monitoring/drift_report.py \
  --checkpoint checkpoints/model_best.pt \
  --data-dir data \
  --n-samples 500
```

Generates:
- `monitoring/reports/drift_report.html` — full Evidently HTML report
- `monitoring/reports/drift_summary.json` — machine-readable drift flags

The degradation pipeline applies Gaussian blur, JPEG compression, and Gaussian noise to simulate real-world image quality degradation.

---

## Tests

```bash
# All tests
pytest tests/ -v

# Single test file
pytest tests/test_api.py -v

# Single test
pytest tests/test_drift.py::TestFlagDrift::test_drift_detected_on_degraded_data -v
```

---

## CI/CD

GitHub Actions runs on every push to `main`:

1. Install dependencies
2. Run full pytest suite
3. If `checkpoints/model_best.pt` exists, run `evaluate.py` and gate on top-1 ≥ 70 %

See `.github/workflows/ci.yml`.

---

## Project structure

```
vision-mlops-pipeline/
├── src/
│   ├── train.py          # Training script (ResNet-50 + W&B logging)
│   └── evaluate.py       # Checkpoint evaluation → results/metrics.json
├── api/
│   ├── main.py           # FastAPI app (/predict, /health)
│   ├── model.py          # Model loading singleton
│   └── gradio_app.py     # Gradio UI (standalone + HF Spaces)
├── monitoring/
│   └── drift_report.py   # Evidently drift report with image degradation
├── tests/
│   ├── conftest.py        # Shared fixtures (dummy checkpoint, sample images)
│   ├── test_api.py        # FastAPI endpoint tests
│   ├── test_drift.py      # Drift detection tests
│   └── test_model.py      # Model sanity checks
├── .github/workflows/
│   └── ci.yml            # CI pipeline
├── checkpoints/          # Saved model weights (git-ignored)
├── results/              # Evaluation outputs (git-ignored)
├── data/                 # Downloaded datasets (git-ignored)
├── monitoring/reports/   # Generated drift reports
├── Dockerfile
└── requirements.txt
```
