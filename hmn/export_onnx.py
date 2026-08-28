#!/usr/bin/env python3
"""v9.7 Production — ONNX export for ReTop models.

BLOCKED: IdentityRegister.stats() uses data-dependent branching
(`if (ids == self.asi_id).any()`) which is incompatible with torch.export.
This is the same root cause as torch.compile graph breaks (v9.3).

To enable ONNX export, IR needs to be refactored to use tensor ops
instead of Python conditionals. This is a v10+ task.

Exports trained models to ONNX format for production deployment.
"""

import argparse
import os
import sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hmn import HMNConfig, create_model


def export_to_onnx(checkpoint, output_path, dim=96, n_layers=3,
                   variant="ssm", asi_id=129):
    """Export model to ONNX format.

    NOTE: This will FAIL due to data-dependent branching in IR.stats().
    See module docstring for details.
    """
    cfg = HMNConfig(
        vocab_size=3190, dim=dim, n_layers=n_layers, variant=variant,
        seam_addr=False, stem_addr=False, attn_ptr=False,
        use_moe=False, asi_id=asi_id,
    )
    model = create_model(cfg)

    state_dict = torch.load(checkpoint, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    dummy_input = torch.randint(0, 3190, (1, 64))

    torch.onnx.export(
        model, dummy_input, output_path,
        export_params=True, opset_version=14,
        do_constant_folding=True,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size", 1: "seq_len"},
        },
    )
    print(f"Exported to {output_path}")
    return output_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="model.onnx")
    p.add_argument("--dim", type=int, default=96)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--variant", default="ssm")
    args = p.parse_args()

    export_to_onnx(args.checkpoint, args.output, args.dim, args.layers,
                   args.variant)
