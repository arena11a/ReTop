#!/usr/bin/env python3
"""v9.3 Benchmark — torch.compile speedup on CPU.

Measures forward + backward pass time for eager vs compiled models.
"""

import argparse
import time
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
from tokenizers import Tokenizer

from hmn import HMNConfig, create_model
from hmn.recipe import make_reorder_batch, loss_v33, seam_losses, resolve_device

tok = Tokenizer.from_file("retop_tokenizer.json")
ASI = tok.token_to_id("<|assistant|>")
VOCAB = tok.get_vocab_size()
DEV = resolve_device(None)


def benchmark_step(model, batch, lossf, steps=50, warmup=5):
    """Benchmark training steps (forward + backward)."""
    Xb, Yb, YcB, Gb, Ab, Sb, Rb = batch

    # Warmup
    for _ in range(warmup):
        out = model(Xb, seam_anchor=Ab)
        loss, _, _, _ = loss_v33(out, Yb, YcB, Gb, lossf=lossf)
        lp, ll = seam_losses(out, Sb, Rb, Ab)
        (loss + lp + ll).backward()
        model.zero_grad()

    # Benchmark
    t0 = time.time()
    for _ in range(steps):
        out = model(Xb, seam_anchor=Ab)
        loss, _, _, _ = loss_v33(out, Yb, YcB, Gb, lossf=lossf)
        lp, ll = seam_losses(out, Sb, Rb, Ab)
        (loss + lp + ll).backward()
        model.zero_grad()
    elapsed = time.time() - t0
    return elapsed


def run_benchmark(args):
    print("=" * 60)
    print("v9.3 torch.compile Benchmark (CPU)")
    print("=" * 60)

    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    batch = make_reorder_batch(
        tok, [f"pkg{i:03d}" for i in range(40)],
        [f"lib{i:03d}" for i in range(80, 120)],
        bs=args.bs, seed=42, device=DEV)

    results = {}

    for dim, nl in [(64, 2), (128, 3), (256, 3)]:
        print(f"\n--- D={dim} L={nl} ---")

        cfg = HMNConfig(
            vocab_size=VOCAB, dim=dim, n_layers=nl,
            variant="attention", seam_addr=True, stem_addr=True,
            attn_ptr=True, use_moe=False, asi_id=ASI,
        )

        # Eager
        model = create_model(cfg).to(DEV)
        eager_time = benchmark_step(model, batch, lossf, steps=args.steps, warmup=args.warmup)
        eager_ms = eager_time / args.steps * 1000
        print(f"  Eager:    {eager_ms:.1f} ms/step ({eager_time:.2f}s)")
        del model

        # Compiled
        model = create_model(cfg).to(DEV)
        compiled_model = torch.compile(model, mode=args.mode)
        compiled_time = benchmark_step(compiled_model, batch, lossf, steps=args.steps, warmup=args.warmup)
        compiled_ms = compiled_time / args.steps * 1000
        print(f"  Compiled: {compiled_ms:.1f} ms/step ({compiled_time:.2f}s)")

        speedup = eager_time / compiled_time
        print(f"  Speedup:  {speedup:.2f}x")

        results[f"D{dim}"] = {
            "eager_ms": round(eager_ms, 1),
            "compiled_ms": round(compiled_ms, 1),
            "speedup": round(speedup, 2),
        }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    avg_speedup = sum(r["speedup"] for r in results.values()) / len(results)
    for k, v in results.items():
        print(f"  {k}: {v['eager_ms']:.1f}ms -> {v['compiled_ms']:.1f}ms ({v['speedup']:.2f}x)")
    print(f"\n  Average speedup: {avg_speedup:.2f}x")

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--mode", type=str, default="default",
                   choices=["default", "reduce-overhead", "max-autotune"])
    args = p.parse_args()
    run_benchmark(args)
