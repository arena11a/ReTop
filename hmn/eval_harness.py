"""v7 M11 — Evaluation harness for HMN models.

Provides automated evaluation across all task families:
  - slot-copy (seen/unseen)
  - chain (multi-step copy)
  - reorder (omega-seam)
  - permutation (rotation)

Outputs JSON reports and supports checkpoint comparison.
"""
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from tokenizers import Tokenizer

from hmn.v3 import HMN3
from hmn.v7 import HMN3AttentionWR
from hmn.recipe import (eval_slots, eval_slot_chains, eval_reorders,
                        make_reorder_batch, loss_v33, seed_guardrail,
                        ASSIST, EOS, CHAIN_SLOTS_A, CHAIN_SLOTS_B,
                        CHAIN_SLOTS_A_U, CHAIN_SLOTS_B_U)

ROOT = os.path.join(os.path.dirname(__file__), "..")


@dataclass
class EvalResult:
    task: str
    accuracy: float
    gate_avg: float
    n_samples: int
    elapsed_s: float
    details: Optional[dict] = None


@dataclass
class EvalReport:
    model_path: str
    checkpoint_step: Optional[int]
    results: list
    total_elapsed_s: float

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        d["results"] = [EvalResult(**r) for r in d["results"]]
        return cls(**d)


def eval_checkpoint(model_path, tokenizer_path=None, device="cpu",
                    n_slot=40, n_chain=20, n_reorder=20, seed=42):
    """Run full eval suite on a checkpoint. Returns EvalReport."""
    seed_guardrail(seed)
    tok = Tokenizer.from_file(tokenizer_path or os.path.join(ROOT, "retop_tokenizer.json"))
    asid = tok.token_to_id(ASSIST)

    # Load model
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if "model" in ckpt:
        state_dict = ckpt["model"]
        step = ckpt.get("step")
    else:
        state_dict = ckpt
        step = None

    # Detect model config from state dict
    dim = state_dict["embed.weight"].shape[1]
    vocab = state_dict["embed.weight"].shape[0]
    is_attention = "blocks.0.qkv.weight" in state_dict
    if is_attention:
        n_layers = len([k for k in state_dict if k.startswith("blocks.") and
                        k.endswith(".ln1.weight")])
        has_moe = any("moe" in k for k in state_dict)
        has_seam = any("seed_ptr" in k for k in state_dict)
        m = HMN3AttentionWR(vocab, dim=dim, n_layers=n_layers,
                            n_experts=16, top_k=2, use_moe=has_moe,
                            gate_bias=-1.0, asi_id=asid, seam_addr=has_seam)
    else:
        n_layers = len([k for k in state_dict if k.startswith("blocks.") and
                        k.endswith(".F1.ln.weight")])
        state_dim = state_dict["blocks.0.F1.A_log"].shape[1]
        has_moe = any("moe" in k for k in state_dict)
        has_seam = any("seed_ptr" in k for k in state_dict)
        m = HMN3(vocab, dim=dim, state_dim=state_dim, n_layers=n_layers,
                 n_experts=16, top_k=2, use_moe=has_moe, gate_bias=-1.0,
                 asi_id=asid, seam_addr=has_seam)
    m.load_state_dict(state_dict, strict=False)
    m.to(device)
    m.eval()

    results = []
    t0 = time.time()

    # Slot-copy (unseen)
    val_slots = [f"pkg{i:03d}" for i in range(60, 60 + n_slot)]
    acc, gate, ngen = eval_slots(m, tok, val_slots, mode="blend",
                                  boundary_eos=True, device=device)
    results.append(EvalResult("slot_copy_unseen", acc, gate, n_slot,
                               time.time() - t0))

    # Chain (unseen)
    if n_chain > 0:
        t1 = time.time()
        acc_c, gate_c, _ = eval_slot_chains(m, tok, CHAIN_SLOTS_A_U[:n_chain],
                                             CHAIN_SLOTS_B_U[:n_chain],
                                             mode="blend", boundary_eos=True,
                                             device=device)
        results.append(EvalResult("chain_unseen", acc_c, gate_c, n_chain,
                                   time.time() - t1))

    # Reorder
    if n_reorder > 0 and has_seam:
        from hmn.recipe import eval_reorders
        t2 = time.time()
        acc_r, gate_r = eval_reorders(m, tok,
                                       [f"pkg{i:03d}" for i in range(60, 60 + n_reorder)],
                                       [f"lib{i:03d}" for i in range(60, 60 + n_reorder)],
                                       mode="hard", pos_eos=True, device=device)
        results.append(EvalResult("reorder_unseen", acc_r, gate_r, n_reorder,
                                   time.time() - t2))

    total_time = time.time() - t0
    return EvalReport(model_path=model_path, checkpoint_step=step,
                      results=results, total_elapsed_s=total_time)


def compare_reports(report_a, report_b):
    """Compare two EvalReports. Returns diff dict."""
    diffs = {}
    for r_a in report_a.results:
        for r_b in report_b.results:
            if r_a.task == r_b.task:
                diffs[r_a.task] = {
                    "accuracy_a": r_a.accuracy,
                    "accuracy_b": r_b.accuracy,
                    "diff": r_b.accuracy - r_a.accuracy,
                    "gate_a": r_a.gate_avg,
                    "gate_b": r_b.gate_avg,
                }
    return diffs


def main():
    """CLI entry point for eval harness."""
    import argparse
    parser = argparse.ArgumentParser(description="HMN evaluation harness")
    parser.add_argument("--model", required=True, help="Path to checkpoint")
    parser.add_argument("--tokenizer", default=None, help="Path to tokenizer")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--compare", default=None, help="Compare with another report")
    parser.add_argument("--device", default="cpu", help="Compute device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    report = eval_checkpoint(args.model, args.tokenizer, args.device, seed=args.seed)

    if args.output:
        report.save(args.output)
        print(f"Report saved to {args.output}")

    if args.compare:
        report_b = EvalReport.load(args.compare)
        diffs = compare_reports(report, report_b)
        print("\nComparison:")
        for task, d in diffs.items():
            sign = "+" if d["diff"] > 0 else ""
            print(f"  {task}: {d['accuracy_a']:.3f} -> {d['accuracy_b']:.3f} "
                  f"({sign}{d['diff']:.3f})")

    # Print summary
    print(f"\nEval Report ({args.model}):")
    for r in report.results:
        print(f"  {r.task:20s}: {r.accuracy:.3f} (gate={r.gate_avg:.3f}, "
              f"n={r.n_samples}, {r.elapsed_s:.1f}s)")
    print(f"  Total: {report.total_elapsed_s:.1f}s")


if __name__ == "__main__":
    main()
