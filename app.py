"""Hugging Face Spaces entrypoint.

Downloads the model checkpoint from the HF Hub model repo on cold start,
then launches the Gradio interface.
"""

import os
from pathlib import Path

# Download checkpoint from HF Hub if not already present
CHECKPOINT_PATH = Path("checkpoints/model_best.pt")
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "Smithy305/vision-mlops-aircraft")

if not CHECKPOINT_PATH.exists():
    print(f"Downloading checkpoint from {HF_MODEL_REPO} …")
    from huggingface_hub import hf_hub_download
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename="model_best.pt",
        local_dir=str(CHECKPOINT_PATH.parent),
    )
    print("Checkpoint downloaded.")

from api.gradio_app import build_interface

demo = build_interface()

demo.launch(server_name="0.0.0.0", server_port=7860)
