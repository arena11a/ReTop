#!/usr/bin/env python3
"""v9.3 Multi-Family Training — force structural learning by training on
multiple verb families simultaneously.

Hypothesis: Training on multiple verb families prevents token memorization
and forces the model to learn the abstract "swap two segments" operation.
"""

import argparse, json, os, random, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
from tokenizers import Tokenizer

from hmn import HMNConfig, create_model
from hmn.recipe import (
    ASSIST, BOS, USER, EOS, REORDER_AND,
    decode_v33, loss_v33, seam_losses,
    make_reorder_batch, make_reorder_batch_v2,
    eval_reorders, resolve_device, _find_word,
)

tok = Tokenizer.from_file("retop_tokenizer.json")
ASI = tok.token_to_id(ASSIST)
VOCAB = tok.get_vocab_size()
DEV = resolve_device(None)

# Slot pools
SLOTS_A = [f"pkg{i:03d}" for i in range(80)]
SLOTS_B = [f"lib{i:03d}" for i in range(80, 160)]

# Verb families
FAMILIES = [
    ("fetch", "deploy"),
    ("load", "unload"),
    ("check", "clean"),
    ("search", "find"),
    ("start", "stop"),
    ("push", "pull"),
    ("read", "write"),
    ("open", "close"),
]

# Evaluation splits
TRAIN_FAMILIES = FAMILIES[:4]       # fetch/deploy, load/unload, check/clean, search/find
UNSEEN_FAMILIES = FAMILIES[4:]      # start/stop, push/pull, read/write, open/close


def train_multi_family(n_families, steps, bs=8, lr=3e-4, log_every=500):
    """Train on multiple verb families."""
    families = TRAIN_FAMILIES[:n_families]
    print(f"  Training on {n_families} families: {families}")

    cfg = HMNConfig(
        vocab_size=VOCAB, dim=96, n_layers=3,
        variant="attention", seam_addr=True, stem_addr=True,
        attn_ptr=True, use_moe=False, asi_id=ASI,
    )
    model = create_model(cfg).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)

    t0 = time.time()
    for step in range(1, steps + 1):
        # Randomly choose a family for this batch
        va, vb = families[step % len(families)]
        Xb, Yb, YcB, Gb, Ab, Sb, Rb = make_reorder_batch_v2(
            tok, SLOTS_A[:40], SLOTS_B[:40], bs=bs, seed=step,
            verb_a=va, verb_b=vb, device=DEV)
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
            # Eval on training families
            sg = 0
            for va, vb in families:
                a, _ = eval_reorders(model, tok,
                                     [f"pkg{i:03d}" for i in range(10)],
                                     [f"lib{i:03d}" for i in range(80, 90)],
                                     seed=7, device=DEV)
                sg += a
            sg /= len(families)

            # Eval on unseen families
            ug = 0
            for va, vb in UNSEEN_FAMILIES:
                a, _ = eval_reorders(model, tok,
                                     [f"pkg{i:03d}" for i in range(50, 60)],
                                     [f"lib{i:03d}" for i in range(130, 140)],
                                     seed=9, device=DEV)
                ug += a
            ug /= len(UNSEEN_FAMILIES)

            model.train()
            elapsed = time.time() - t0
            print(f"    step {step:5d} loss={loss.item():.3f} "
                  f"seen_families={sg:.3f} unseen_families={ug:.3f} "
                  f"[{elapsed:.0f}s]", flush=True)

    return model


def eval_verb_generalization(model, n=10):
    """Eval model on unseen verb families."""
    model.eval()
    results = {}
    for va, vb in UNSEEN_FAMILIES:
        ok = tot = 0
        rng = random.Random(42)
        for _ in range(n):
            a = rng.choice(SLOTS_A[50:60])
            b = rng.choice(SLOTS_B[50:60])
            # Build prompt
            bos = tok.token_to_id(BOS)
            uid = tok.token_to_id(USER)
            asid = tok.token_to_id(ASSIST)
            eos = tok.token_to_id(EOS)
            text = f"{va} {a} and {vb} {b}"
            U = list(tok.encode(text).ids)
            i_u = _find_word(tok, U, REORDER_AND)
            Gt = U[i_u + 1:] + [U[i_u]] + U[:i_u]
            ids = [bos, uid] + U + [asid] + Gt + [eos]
            prompt = ids[:2 + len(U) + 1]
            gold = tok.decode(Gt).strip()

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
    print("v9.3 Multi-Family Training Experiment")
    print("=" * 60)

    results = {}

    # Test 1 family (baseline from v9.2)
    print("\n--- 1 Family (baseline) ---")
    m1 = train_multi_family(1, args.steps, bs=args.bs, log_every=args.log_every)
    v1 = eval_verb_generalization(m1)
    avg1 = sum(v1.values()) / len(v1)
    results["1_family"] = {"verb_accs": v1, "avg_verb": avg1}
    print(f"  Avg unseen verb accuracy: {avg1:.3f}")

    # Test 2 families
    print("\n--- 2 Families ---")
    m2 = train_multi_family(2, args.steps, bs=args.bs, log_every=args.log_every)
    v2 = eval_verb_generalization(m2)
    avg2 = sum(v2.values()) / len(v2)
    results["2_families"] = {"verb_accs": v2, "avg_verb": avg2}
    print(f"  Avg unseen verb accuracy: {avg2:.3f}")

    # Test 4 families
    print("\n--- 4 Families ---")
    m4 = train_multi_family(4, args.steps, bs=args.bs, log_every=args.log_every)
    v4 = eval_verb_generalization(m4)
    avg4 = sum(v4.values()) / len(v4)
    results["4_families"] = {"verb_accs": v4, "avg_verb": avg4}
    print(f"  Avg unseen verb accuracy: {avg4:.3f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  1 family:  avg_verb={avg1:.3f}")
    print(f"  2 families: avg_verb={avg2:.3f}")
    print(f"  4 families: avg_verb={avg4:.3f}")

    if avg4 > avg1 + 0.1:
        print("\n  → Multi-family training IMPROVES cross-family generalization!")
    elif avg4 > avg1:
        print("\n  → Multi-family training helps slightly")
    else:
        print("\n  → Multi-family training does NOT help cross-family generalization")
        print("  → The model still memorizes tokens despite seeing multiple families")

    out_path = os.path.join(os.path.dirname(__file__), "v9_3_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--log-every", type=int, default=500)
    args = p.parse_args()
    run_experiment(args)
