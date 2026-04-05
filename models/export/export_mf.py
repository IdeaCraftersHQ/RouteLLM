"""Export MFModel to ONNX format for Triton Inference Server.

Loads the matrix factorization model from a HuggingFace checkpoint,
wraps its tensor operations into an ONNX-exportable forward pass,
and validates the exported model against PyTorch outputs.

Usage:
    python models/export/export_mf.py
    python models/export/export_mf.py --checkpoint path/or/hf-id
    python models/export/export_mf.py --output models/triton/mf/1/model.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

# ---------------------------------------------------------------------------
# ONNX-exportable wrapper
# ---------------------------------------------------------------------------
# The original MFModel.forward() calls the OpenAI API for embeddings and
# accepts raw Python lists for model_id. Neither is ONNX-compatible.
# This wrapper takes pre-computed tensors and performs the math only.
# ---------------------------------------------------------------------------


class MFModelONNX(nn.Module):
    """Thin wrapper around MFModel weights for ONNX export.

    Inputs (all tensors):
        strong_id      : int64 [1]   — model index for the strong model
        weak_id        : int64 [1]   — model index for the weak model
        embedding      : float32 [1536] — pre-computed text embedding

    Output:
        win_rate       : float32 [1] — sigmoid(logit_strong - logit_weak)
    """

    def __init__(self, mf_model: nn.Module):
        super().__init__()
        self.P = mf_model.P
        self.text_proj = mf_model.text_proj
        self.classifier = mf_model.classifier

    def forward(
        self,
        strong_id: torch.Tensor,
        weak_id: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        # Stack model ids: [strong, weak] -> [2]
        model_ids = torch.cat([strong_id, weak_id], dim=0)

        # Embedding lookup + L2 normalise
        model_embed = self.P(model_ids)  # [2, dim]
        model_embed = nn.functional.normalize(model_embed, p=2, dim=1)

        # Project text embedding -> [dim]
        prompt_embed = self.text_proj(embedding)  # [dim]

        # Broadcast multiply + classify
        logits = self.classifier(model_embed * prompt_embed).squeeze(-1)  # [2]

        # Win-rate = sigma(logit_strong - logit_weak)
        win_rate = torch.sigmoid(logits[0:1] - logits[1:2])  # [1]
        return win_rate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_mf_model(checkpoint: str) -> nn.Module:
    """Load MFModel from HuggingFace checkpoint."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from routellm.routers.matrix_factorization.model import MFModel

    model = MFModel.from_pretrained(checkpoint)
    model.eval()
    return model


def export_onnx(
    wrapper: nn.Module,
    output_path: Path,
) -> None:
    """Export the ONNX-wrapped model."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Dummy inputs
    strong_id = torch.tensor([0], dtype=torch.long)
    weak_id = torch.tensor([1], dtype=torch.long)
    embedding = torch.randn(1536, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        (strong_id, weak_id, embedding),
        str(output_path),
        opset_version=17,
        input_names=["strong_id", "weak_id", "embedding"],
        output_names=["win_rate"],
        dynamic_axes=None,  # fixed shapes
    )

    # Validate ONNX graph
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model saved and validated: {output_path}")


def validate(
    wrapper: nn.Module,
    onnx_path: Path,
    *,
    atol: float = 1e-5,
) -> None:
    """Compare PyTorch and ONNX Runtime outputs."""
    strong_id = torch.tensor([0], dtype=torch.long)
    weak_id = torch.tensor([1], dtype=torch.long)
    embedding = torch.randn(1536, dtype=torch.float32)

    # PyTorch
    with torch.no_grad():
        pt_out = wrapper(strong_id, weak_id, embedding).numpy()

    # ONNX Runtime
    sess = ort.InferenceSession(str(onnx_path))
    ort_out = sess.run(
        None,
        {
            "strong_id": strong_id.numpy(),
            "weak_id": weak_id.numpy(),
            "embedding": embedding.numpy(),
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
        description="Export MFModel to ONNX for Triton",
    )
    parser.add_argument(
        "--checkpoint",
        default="routellm/routellm_mf",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--output",
        default="models/triton/mf/1/model.onnx",
        help="Output ONNX path",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip PyTorch vs ONNX validation step",
    )
    args = parser.parse_args()

    output_path = Path(args.output)

    print(f"Loading MFModel from {args.checkpoint} ...")
    mf_model = load_mf_model(args.checkpoint)

    wrapper = MFModelONNX(mf_model)
    wrapper.eval()

    print("Exporting to ONNX ...")
    export_onnx(wrapper, output_path)

    if not args.skip_validation:
        print("Validating ...")
        validate(wrapper, output_path)

    print("Done.")


if __name__ == "__main__":
    main()
