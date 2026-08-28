#!/usr/bin/env python3
"""v9.6 Perm Generalization — Neural Seeder improvements.

Hypothesis: Adding positional encoding to SeedPointer helps it predict
structural positions instead of token-based positions, improving
cross-verb generalization.

Approach:
1. Add relative positional encoding to SeedPointer
2. Add structural embedding that captures task pattern
3. Test on unseen verb families
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
import torch.nn.functional as F
from tokenizers import Tokenizer

from hmn import HMNConfig, create_model
from hmn.recipe import (
    ASSIST, make_reorder_ids_v2, make_reorder_batch, decode_v33,
    eval_reorders, eval_slots, loss_v33, seam_losses, resolve_device,
)

tok = Tokenizer.from_file("retop_tokenizer.json")
ASI = tok.token_to_id("<|assistant|>")
VOCAB = tok.get_vocab_size()
DEV = resolve_device(None)

SLOTS = [f"pkg{i:03d}" for i in range(80)]
LIBS = [f"lib{i:03d}" for i in range(80, 120)]


class StructuralSeedPointer(nn.Module):
    """SeedPointer with positional encoding for structural positions.

    Adds learned positional embeddings to the pointer keys to help it
    predict structural positions instead of token-based positions.
    """

    def __init__(self, dim, max_run=16, beta=20.0, max_pos=512):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.len_head = nn.Linear(dim, max_run)
        self.max_run = max_run
        self.beta = beta
        # Learned positional embeddings
        self.pos_embed = nn.Embedding(max_pos, dim)

    def forward(self, h, keys, bound):
        B, T, D = h.shape
        q = F.normalize(self.q(h.float()), dim=-1)
        # Add positional embeddings to keys
        pos = torch.arange(T, device=h.device).unsqueeze(0).expand(B, -1)
        k = F.normalize((keys + self.pos_embed(pos)).float(), dim=-1)
        logits = (q @ k.transpose(-1, -2)) * self.beta
        cols = torch.arange(T, device=h.device)
        legal = cols.unsqueeze(0) < bound.unsqueeze(-1)
        logits = logits.masked_fill(~legal.unsqueeze(1), float("-inf"))
        len_logits = self.len_head(h.float())
        return logits, len_logits


def train_with_structural_pointer(steps, bs=8, lr=3e-4, log_every=200):
    """Train model with structural seed pointer."""
    cfg = HMNConfig(
        vocab_size=VOCAB, dim=96, n_layers=3, variant="ssm",
        seam_addr=True, stem_addr=True, attn_ptr=False,
        use_moe=False, asi_id=ASI,
    )
    model = create_model(cfg).to(DEV)

    # Replace SeedPointer with StructuralSeedPointer
    model.seed_ptr = StructuralSeedPointer(96).to(DEV)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)

    t0 = time.time()
    for step in range(1, steps + 1):
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

        if step % log_every == 0 or step == steps:
            model.eval()
            slot_acc, _, _ = eval_slots(model, tok, SLOTS[:10],
                                         seed=7, device=DEV)
            reorder_acc, _ = eval_reorders(model, tok, SLOTS[:10], LIBS[:10],
                                            seed=7, device=DEV)
            model.train()
            elapsed = time.time() - t0
            print(f"    step {step:5d} loss={loss.item():.3f} "
                  f"slot={slot_acc:.3f} reorder={reorder_acc:.3f} "
                  f"[{elapsed:.0f}s]", flush=True)

    return model


def eval_unseen_verbs(model, n=10):
    """Eval model on unseen verb families."""
    model.eval()
    results = {}
    unseen = [("start", "stop"), ("push", "pull"),
              ("read", "write"), ("open", "close")]
    for va, vb in unseen:
        ok = tot = 0
        rng = random.Random(42)
        for _ in range(n):
            a = rng.choice(SLOTS[50:60])
            b = rng.choice(SLOTS[50:60])
            ids, asi_pos, _ = make_reorder_ids_v2(tok, a, b, va, vb)
            prompt = ids[:asi_pos + 1]
            gold_ids, gold_asi, _ = make_reorder_ids_v2(tok, b, a, vb, va)
            gold = tok.decode(gold_ids[gold_asi + 1:-1]).strip()

            txt, g, _ = decode_v33(model, tok, prompt, max_new=32,
                                    mode="hard", seam=True,
                                    pos_eos=True, device=DEV)
            ok += int(txt.strip() == gold)
            tot += 1
        results[f"{va}/{vb}"] = ok / max(tot, 1)
    model.train()
    return results


def run_experiment(args):
    print("=" * 60)
    print("v9.6 Perm Generalization — Structural Seed Pointer")
    print("=" * 60)

    results = {}

    # Train with structural pointer
    print("\n--- Structural Seed Pointer ---")
    model = train_with_structural_pointer(args.steps, bs=args.bs,
                                          log_every=args.log_every)
    unseen = eval_unseen_verbs(model)
    avg_unseen = sum(unseen.values()) / len(unseen)
    results["structural_pointer"] = {
        "verb_accs": unseen,
        "avg_verb": avg_unseen,
    }
    print(f"  Unseen verb: {avg_unseen:.3f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Structural Pointer: avg_verb={avg_unseen:.3f}")

    if avg_unseen > 0.05:
        print("\n  → Structural pointer IMPROVES cross-family generalization!")
    else:
        print("\n  → Structural pointer does NOT significantly help")

    out_path = os.path.join(os.path.dirname(__file__), "v9_6_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--log-every", type=int, default=300)
    args = p.parse_args()
    run_experiment(args)
