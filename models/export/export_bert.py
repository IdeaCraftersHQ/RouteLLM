"""Export BERT router model to ONNX format for Triton Inference Server.

Loads AutoModelForSequenceClassification from a HuggingFace checkpoint,
exports with dynamic batch and sequence length axes, and validates
the exported model against PyTorch outputs.

Usage:
    python models/export/export_bert.py
    python models/export/export_bert.py --checkpoint path/or/hf-id
    python models/export/export_bert.py --output models/triton/bert/1/model.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_model(checkpoint: str):
    """Load BERT model and tokenizer from checkpoint."""
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=3
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model.eval()
    return model, tokenizer


def export_onnx(
    model: torch.nn.Module,
    tokenizer,
    output_path: Path,
) -> None:
    """Export BERT model to ONNX with dynamic axes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Dummy input
    dummy_text = "What is the capital of France?"
    inputs = tokenizer(dummy_text, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"},
    }

    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        str(output_path),
        opset_version=17,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
    )

    # Validate ONNX graph
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model saved and validated: {output_path}")


def validate(
    model: torch.nn.Module,
    tokenizer,
    onnx_path: Path,
    *,
    atol: float = 1e-4,
) -> None:
    """Compare PyTorch and ONNX Runtime outputs."""
    test_text = "Explain quantum computing in simple terms."
    inputs = tokenizer(test_text, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # PyTorch
    with torch.no_grad():
        pt_out = model(input_ids, attention_mask=attention_mask).logits.numpy()

    # ONNX Runtime
    sess = ort.InferenceSession(str(onnx_path))
    ort_out = sess.run(
        None,
        {
            "input_ids": input_ids.numpy(),
            "attention_mask": attention_mask.numpy(),
        },
    )[0]

    diff = np.abs(pt_out - ort_out).max()
    print(f"Max absolute diff (PyTorch vs ONNX): {diff:.2e}")
    if diff > atol:
        raise ValueError(
            f"Validation FAILED: max diff {diff:.2e} > tolerance {atol:.2e}"
        )
    print("Validation PASSED")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export BERT router model to ONNX for Triton",
    )
    parser.add_argument(
        "--checkpoint",
        default="routellm/routellm_bert",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--output",
        default="models/triton/bert/1/model.onnx",
        help="Output ONNX path",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip PyTorch vs ONNX validation step",
    )
    args = parser.parse_args()

    output_path = Path(args.output)

    print(f"Loading BERT model from {args.checkpoint} ...")
    model, tokenizer = load_model(args.checkpoint)

    print("Exporting to ONNX ...")
    export_onnx(model, tokenizer, output_path)

    if not args.skip_validation:
        print("Validating ...")
        validate(model, tokenizer, output_path)

    print("Done.")


if __name__ == "__main__":
    main()
