"""Gradio front-end for the fine-grained image classifier.

Can be run standalone:
    python api/gradio_app.py

Or launched as a Hugging Face Space by setting the entrypoint to this file.
Set the CHECKPOINT_PATH environment variable to override the default checkpoint location.
"""

import os
from pathlib import Path

import gradio as gr
from PIL import Image

# Allow override via env var (useful for HF Spaces)
_env_ckpt = os.environ.get("CHECKPOINT_PATH")
if _env_ckpt:
    from api import model as _model_module
    _model_module.CHECKPOINT_PATH = Path(_env_ckpt)

from api.model import predict


# ---------------------------------------------------------------------------
# Inference wrapper
# ---------------------------------------------------------------------------

def classify_image(image: Image.Image) -> tuple[dict, str]:
    """Run inference and return results in Gradio-compatible format.

    Returns (label_confidences_dict, markdown_table_string).
    """
    if image is None:
        return {}, "Please upload an image."

    result = predict(image)

    # Label dict for gr.Label component (class_name -> confidence)
    label_dict = {
        p["class_name"]: p["confidence"]
        for p in result["top5_predictions"]
    }

    # Markdown summary table
    top1 = result["predicted_class"]
    conf = result["confidence"]
    rows = "\n".join(
        f"| {i+1} | {p['class_name']} | {p['confidence']:.1%} |"
        for i, p in enumerate(result["top5_predictions"])
    )
    md = (
        f"### Prediction: **{top1}** ({conf:.1%} confidence)\n\n"
        f"| Rank | Class | Confidence |\n"
        f"|------|-------|------------|\n"
        f"{rows}"
    )

    return label_dict, md


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

def build_interface() -> gr.Blocks:
    """Construct and return the Gradio Blocks interface."""
    with gr.Blocks(
        title="Fine-Grained Image Classifier",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # Fine-Grained Image Classifier
            Upload an image to classify it. The model is a ResNet-50 fine-tuned on
            [FGVC-Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) (or Food-101 as fallback).
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Input Image",
                    height=320,
                )
                submit_btn = gr.Button("Classify", variant="primary")

            with gr.Column(scale=1):
                label_output = gr.Label(
                    num_top_classes=5,
                    label="Top-5 Predictions",
                )
                markdown_output = gr.Markdown(label="Details")

        submit_btn.click(
            fn=classify_image,
            inputs=image_input,
            outputs=[label_output, markdown_output],
        )
        image_input.change(
            fn=classify_image,
            inputs=image_input,
            outputs=[label_output, markdown_output],
        )

        gr.Markdown(
            """
            ---
            *Part of the [vision-mlops-pipeline](https://github.com/Smithy305/vision-mlops-pipeline) project.*
            """
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
