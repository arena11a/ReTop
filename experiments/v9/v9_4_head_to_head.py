#!/usr/bin/env python3
"""v9.4 Head-to-Head — attention vs SSM comparison.

Compares both variants on:
1. Slot-copy task (basic)
2. Reorder task (ptr3)
3. Training speed
4. Parameter count
"""

import argparse
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
from tokenizers import Tokenizer

from hmn import HMNConfig, create_model
from hmn.recipe import (
    ASSIST, make_reorder_batch, eval_reorders, eval_slots,
    make_slot_batch, loss_v33, seam_losses, resolve_device,
)

tok = Tokenizer.from_file("retop_tokenizer.json")
ASI = tok.token_to_id("<|assistant|>")
VOCAB = tok.get_vocab_size()
DEV = resolve_device(None)

SLOTS = [f"pkg{i:03d}" for i in range(80)]
LIBS = [f"lib{i:03d}" for i in range(80, 160)]


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train_and_eval(variant, dim, n_layers, task, steps=1000, bs=8, lr=3e-4):
    """Train model on task and return metrics."""
    cfg = HMNConfig(
        vocab_size=VOCAB, dim=dim, n_layers=n_layers,
        variant=variant, seam_addr=True, stem_addr=True,
        attn_ptr=(variant == "attention"), use_moe=False, asi_id=ASI,
    )
    model = create_model(cfg).to(DEV)
    n_params = count_params(model)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)

    t0 = time.time()
    for step in range(1, steps + 1):
        if task == "slot":
            Xb, Yb, YcB, Gb = make_slot_batch(tok, SLOTS[:40], bs=bs,
                                               seed=step, device=DEV)
            out = model(Xb)
            logits = out.get("logits", out.get("gen_logits"))
            loss = lossf(logits.reshape(-1, logits.shape[-1]), Yb.reshape(-1))
        elif task == "reorder":
            Xb, Yb, YcB, Gb, Ab, Sb, Rb = make_reorder_batch(
                tok, SLOTS[:40], LIBS[:40], bs=bs, seed=step, device=DEV)
            out = model(Xb, seam_anchor=Ab)
            loss, lb, lg, lc = loss_v33(out, Yb, YcB, Gb, lossf=lossf)
            lp, ll = seam_losses(out, Sb, Rb, Ab)
            loss = loss + lp + ll

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        opt.zero_grad()

    train_time = time.time() - t0

    # Eval
    model.eval()
    if task == "slot":
        acc, gate, _ = eval_slots(model, tok, SLOTS[:10], seed=7, device=DEV)
    elif task == "reorder":
        acc, gate = eval_reorders(model, tok, SLOTS[:10], LIBS[:10],
                                  seed=7, device=DEV)

    return {
        "variant": variant,
        "task": task,
        "dim": dim,
        "n_layers": n_layers,
        "n_params": n_params,
        "train_time": round(train_time, 1),
        "ms_per_step": round(train_time / steps * 1000, 1),
        "accuracy": round(acc, 3),
        "gate": round(gate, 3),
    }


def run_experiment(args):
    print("=" * 60)
    print("v9.4 Head-to-Head: Attention vs SSM")
    print("=" * 60)

    results = []

    # Compare at same parameter count
    configs = [
        (64, 2),   # ~100K params
        (128, 3),  # ~500K params
        (256, 3),  # ~2M params
    ]

    for dim, nl in configs:
        print(f"\n--- D={dim} L={nl} ---")

        for variant in ["ssm", "attention"]:
            for task in ["slot", "reorder"]:
                print(f"  {variant} on {task}...", end=" ", flush=True)
                r = train_and_eval(variant, dim, nl, task,
                                   steps=args.steps, bs=args.bs)
                results.append(r)
                print(f"acc={r['accuracy']:.3f} "
                      f"params={r['n_params']:,} "
                      f"speed={r['ms_per_step']:.1f}ms")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Group by variant and task
    for variant in ["ssm", "attention"]:
        print(f"\n{variant.upper()}:")
        for task in ["slot", "reorder"]:
            subset = [r for r in results if r["variant"] == variant
                      and r["task"] == task]
            if subset:
                avg_acc = sum(r["accuracy"] for r in subset) / len(subset)
                avg_speed = sum(r["ms_per_step"] for r in subset) / len(subset)
                print(f"  {task}: avg_acc={avg_acc:.3f} avg_speed={avg_speed:.1f}ms")

    # Winner by task
    print("\nWinners:")
    for task in ["slot", "reorder"]:
        ssm_results = [r for r in results if r["variant"] == "ssm"
                       and r["task"] == task]
        attn_results = [r for r in results if r["variant"] == "attention"
                        and r["task"] == task]
        if ssm_results and attn_results:
            ssm_acc = sum(r["accuracy"] for r in ssm_results) / len(ssm_results)
            attn_acc = sum(r["accuracy"] for r in attn_results) / len(attn_results)
            winner = "attention" if attn_acc > ssm_acc else "ssm"
            print(f"  {task}: {winner} ({attn_acc:.3f} vs {ssm_acc:.3f})")

    out_path = os.path.join(os.path.dirname(__file__), "v9_4_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--bs", type=int, default=8)
    args = p.parse_args()
    run_experiment(args)
