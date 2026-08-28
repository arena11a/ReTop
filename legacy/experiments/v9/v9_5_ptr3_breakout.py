#!/usr/bin/env python3
"""v9.5 ptr3 Breakout — multi-task + curriculum training.

Hypothesis: Training on slot-copy + reorder simultaneously with curriculum
learning helps the model learn structural patterns instead of memorizing tokens.

Approach:
1. Multi-task: alternate between slot-copy and reorder batches
2. Curriculum: start with 100% slot-copy, gradually increase reorder ratio
3. Use SSM variant (proven better in v9.4)
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
    ASSIST, make_reorder_batch, make_slot_batch, eval_reorders, eval_slots,
    loss_v33, seam_losses, resolve_device,
)

tok = Tokenizer.from_file("retop_tokenizer.json")
ASI = tok.token_to_id("<|assistant|>")
VOCAB = tok.get_vocab_size()
DEV = resolve_device(None)

SLOTS = [f"pkg{i:03d}" for i in range(80)]
LIBS = [f"lib{i:03d}" for i in range(80, 120)]


def train_multitask(steps, bs=8, lr=3e-4, reorder_ratio=0.5, log_every=200):
    """Multi-task training: slot-copy + reorder at fixed ratio."""
    cfg = HMNConfig(
        vocab_size=VOCAB, dim=96, n_layers=3, variant="ssm",
        seam_addr=True, stem_addr=True, attn_ptr=False,
        use_moe=False, asi_id=ASI,
    )
    model = create_model(cfg).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)

    t0 = time.time()
    for step in range(1, steps + 1):
        use_reorder = random.random() < reorder_ratio

        if use_reorder:
            Xb, Yb, YcB, Gb, Ab, Sb, Rb = make_reorder_batch(
                tok, SLOTS[:40], LIBS[:40], bs=bs, seed=step, device=DEV)
            out = model(Xb, seam_anchor=Ab)
            loss, lb, lg, lc = loss_v33(out, Yb, YcB, Gb, lossf=lossf)
            lp, ll = seam_losses(out, Sb, Rb, Ab)
            loss = loss + lp + ll
        else:
            Xb, Yb, YcB, Gb = make_slot_batch(
                tok, SLOTS[:40], bs=bs, seed=step, device=DEV)
            out = model(Xb)
            logits = out.get("logits", out.get("gen_logits"))
            loss = lossf(logits.reshape(-1, logits.shape[-1]), Yb.reshape(-1))

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


def train_curriculum(steps, bs=8, lr=3e-4, log_every=200):
    """Curriculum training: start with slot-copy, gradually add reorder.

    Phase 1 (0-33%): 100% slot-copy
    Phase 2 (33-66%): 50% slot-copy + 50% reorder
    Phase 3 (66-100%): 100% reorder
    """
    cfg = HMNConfig(
        vocab_size=VOCAB, dim=96, n_layers=3, variant="ssm",
        seam_addr=True, stem_addr=True, attn_ptr=False,
        use_moe=False, asi_id=ASI,
    )
    model = create_model(cfg).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)

    t0 = time.time()
    for step in range(1, steps + 1):
        # Curriculum schedule
        progress = step / steps
        if progress < 0.33:
            reorder_ratio = 0.0
            phase = "phase1"
        elif progress < 0.66:
            reorder_ratio = 0.5
            phase = "phase2"
        else:
            reorder_ratio = 1.0
            phase = "phase3"

        use_reorder = random.random() < reorder_ratio

        if use_reorder:
            Xb, Yb, YcB, Gb, Ab, Sb, Rb = make_reorder_batch(
                tok, SLOTS[:40], LIBS[:40], bs=bs, seed=step, device=DEV)
            out = model(Xb, seam_anchor=Ab)
            loss, lb, lg, lc = loss_v33(out, Yb, YcB, Gb, lossf=lossf)
            lp, ll = seam_losses(out, Sb, Rb, Ab)
            loss = loss + lp + ll
        else:
            Xb, Yb, YcB, Gb = make_slot_batch(
                tok, SLOTS[:40], bs=bs, seed=step, device=DEV)
            out = model(Xb)
            logits = out.get("logits", out.get("gen_logits"))
            loss = lossf(logits.reshape(-1, logits.shape[-1]), Yb.reshape(-1))

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
            print(f"    step {step:5d} [{phase}] loss={loss.item():.3f} "
                  f"slot={slot_acc:.3f} reorder={reorder_acc:.3f} "
                  f"[{elapsed:.0f}s]", flush=True)

    return model


def train_baseline(steps, bs=8, lr=3e-4, log_every=200):
    """Baseline: reorder-only training (v9.2 setup)."""
    cfg = HMNConfig(
        vocab_size=VOCAB, dim=96, n_layers=3, variant="ssm",
        seam_addr=True, stem_addr=True, attn_ptr=False,
        use_moe=False, asi_id=ASI,
    )
    model = create_model(cfg).to(DEV)
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
            # Build prompt
            from hmn.recipe import make_reorder_ids_v2, _find_word
            ids, asi_pos, _ = make_reorder_ids_v2(tok, a, b, va, vb)
            prompt = ids[:asi_pos + 1]
            gold_ids, gold_asi, _ = make_reorder_ids_v2(tok, b, a, vb, va)
            gold = tok.decode(gold_ids[gold_asi + 1:-1]).strip()

            from hmn.recipe import decode_v33
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
    print("v9.5 ptr3 Breakout — Multi-task + Curriculum")
    print("=" * 60)

    results = {}

    # Baseline: reorder-only
    print("\n--- Baseline (reorder-only) ---")
    m_base = train_baseline(args.steps, bs=args.bs, log_every=args.log_every)
    v_base = eval_unseen_verbs(m_base)
    avg_base = sum(v_base.values()) / len(v_base)
    results["baseline"] = {"verb_accs": v_base, "avg_verb": avg_base}
    print(f"  Unseen verb: {avg_base:.3f}")

    # Multi-task: 50/50
    print("\n--- Multi-task (50/50) ---")
    m_mt = train_multitask(args.steps, bs=args.bs, reorder_ratio=0.5,
                           log_every=args.log_every)
    v_mt = eval_unseen_verbs(m_mt)
    avg_mt = sum(v_mt.values()) / len(v_mt)
    results["multitask_50"] = {"verb_accs": v_mt, "avg_verb": avg_mt}
    print(f"  Unseen verb: {avg_mt:.3f}")

    # Curriculum
    print("\n--- Curriculum ---")
    m_curr = train_curriculum(args.steps, bs=args.bs, log_every=args.log_every)
    v_curr = eval_unseen_verbs(m_curr)
    avg_curr = sum(v_curr.values()) / len(v_curr)
    results["curriculum"] = {"verb_accs": v_curr, "avg_verb": avg_curr}
    print(f"  Unseen verb: {avg_curr:.3f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Baseline:     avg_verb={avg_base:.3f}")
    print(f"  Multi-task:   avg_verb={avg_mt:.3f}")
    print(f"  Curriculum:   avg_verb={avg_curr:.3f}")

    if avg_mt > avg_base + 0.05 or avg_curr > avg_base + 0.05:
        print("\n  → Multi-task/curriculum IMPROVES cross-family generalization!")
    else:
        print("\n  → Multi-task/curriculum does NOT significantly help")

    out_path = os.path.join(os.path.dirname(__file__), "v9_5_results.json")
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
