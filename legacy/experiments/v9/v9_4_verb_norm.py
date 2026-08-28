#!/usr/bin/env python3
"""v9.4 Verb Normalization — replace specific verbs with generic tokens
to force structural learning.

Hypothesis: If we train with generic verb tokens, the model learns the
abstract "swap two segments" operation rather than memorizing specific verbs.
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
    make_reorder_batch, make_reorder_batch_v2, make_reorder_ids_v2,
    eval_reorders, resolve_device, _find_word, reorder_anchors,
)

tok = Tokenizer.from_file("retop_tokenizer.json")
ASI = tok.token_to_id(ASSIST)
VOCAB = tok.get_vocab_size()
DEV = resolve_device(None)

SA = [f"pkg{i:03d}" for i in range(80)]
SB = [f"lib{i:03d}" for i in range(80, 160)]

TRAIN_FAMILIES = [("fetch", "deploy"), ("load", "unload"),
                  ("check", "clean"), ("search", "find")]
UNSEEN_FAMILIES = [("start", "stop"), ("push", "pull"),
                   ("read", "write"), ("open", "close")]


def make_reorder_ids_normalized(tok, a_s, b_s):
    """Build reorder IDs with normalized verbs (replaced by generic tokens).

    Instead of "fetch pkg001 and deploy lib080", we build:
    "A pkg001 and B lib080" where A/B are the first tokens of each verb.
    This forces the model to learn the structural pattern.
    """
    bos = tok.token_to_id(BOS)
    uid = tok.token_to_id(USER)
    asid = tok.token_to_id(ASSIST)
    eos = tok.token_to_id(EOS)

    # Use generic markers that appear in training
    # "A" and "B" are common single-token words
    U = list(tok.encode(f"A {a_s} and B {b_s}").ids)
    i_u = _find_word(tok, U, REORDER_AND)
    if i_u <= 0:
        raise AssertionError("make_reorder_ids_normalized: 'and' not found")
    Gt = U[i_u + 1:] + [U[i_u]] + U[:i_u]
    ids = [bos, uid] + U + [asid] + Gt + [eos]
    return ids, 2 + len(U), i_u


def make_reorder_batch_normalized(tok, a_slots, b_slots, bs, seed, device=None):
    """Build reorder batch with normalized verbs."""
    rng = random.Random(seed)
    eos = tok.token_to_id(EOS)
    asid = tok.token_to_id(ASSIST)
    X, Y, YC, G, AN, S, R = [], [], [], [], [], [], []
    for _ in range(bs):
        a_s = rng.choice(a_slots)
        b_s = rng.choice(b_slots)
        ids, asi_pos, _ = make_reorder_ids_normalized(tok, a_s, b_s)
        anchors, seams, runs = reorder_anchors(ids, asid, tok)
        targets = ids[1:] + [eos]
        Tn = len(ids)
        y, yc, gt = [-100] * Tn, [-100] * Tn, [-1.0] * Tn
        for t in range(asi_pos, Tn):
            tgt = targets[t]
            y[t] = tgt
            if tgt == eos:
                yc[t] = -100
                gt[t] = 0.0
            else:
                yc[t] = tgt
                gt[t] = 1.0
        X.append(ids); Y.append(y); YC.append(yc); G.append(gt)
        AN.append(anchors); S.append(seams); R.append(runs)
    dev = resolve_device(device)
    T = max(len(x) for x in X)
    Xb = torch.full((bs, T), eos, dtype=torch.long, device=dev)
    Yb = torch.full((bs, T), -100, dtype=torch.long, device=dev)
    YcB = torch.full((bs, T), -100, dtype=torch.long, device=dev)
    Gb = torch.full((bs, T), -1.0, dtype=torch.float, device=dev)
    Ab = torch.full((bs, T), -100, dtype=torch.long, device=dev)
    Sb = torch.zeros((bs, T), dtype=torch.bool, device=dev)
    Rb = torch.full((bs, T), -100, dtype=torch.long, device=dev)
    for j in range(bs):
        L = len(X[j])
        Xb[j, :L] = torch.tensor(X[j], device=dev)
        Yb[j, :L] = torch.tensor(Y[j], device=dev)
        YcB[j, :L] = torch.tensor(YC[j], device=dev)
        Gb[j, :L] = torch.tensor(G[j], device=dev)
        Ab[j, :L] = torch.tensor(AN[j], device=dev)
        Sb[j, :L] = torch.tensor(S[j], device=dev)
        Rb[j, :L] = torch.tensor(R[j], device=dev)
    return Xb, Yb, YcB, Gb, Ab, Sb, Rb


def eval_with_real_verbs(model, n=10):
    """Eval model trained with normalized verbs on real verb families."""
    model.eval()
    results = {}
    for va, vb in UNSEEN_FAMILIES:
        ok = tot = 0
        rng = random.Random(42)
        for _ in range(n):
            a = rng.choice(SA[50:60])
            b = rng.choice(SB[50:60])
            prompt, gold = _build_prompt_real(va, a, vb, b)
            txt, g, _ = decode_v33(model, tok, prompt, max_new=32,
                                    mode="hard", seam=True,
                                    pos_eos=True, device=DEV)
            ok += int(txt.strip() == gold)
            tot += 1
        results[f"{va}/{vb}"] = ok / max(tot, 1)
    model.train()
    return results


def _build_prompt_real(va, a, vb, b):
    """Build prompt with real verbs for eval."""
    bos = tok.token_to_id(BOS)
    uid = tok.token_to_id(USER)
    asid = tok.token_to_id(ASSIST)
    eos = tok.token_to_id(EOS)
    U = list(tok.encode(f"{va} {a} and {vb} {b}").ids)
    i_u = _find_word(tok, U, REORDER_AND)
    Gt = U[i_u + 1:] + [U[i_u]] + U[:i_u]
    ids = [bos, uid] + U + [asid] + Gt + [eos]
    prompt = ids[:2 + len(U) + 1]
    gold = tok.decode(Gt).strip()
    return prompt, gold


def run_experiment(args):
    print("=" * 60)
    print("v9.4 Verb Normalization Experiment")
    print("=" * 60)

    results = {}

    # Test 1: Normalized training (A/B markers)
    print("\n--- Normalized Training (A/B markers) ---")
    cfg = HMNConfig(
        vocab_size=VOCAB, dim=128, n_layers=3,
        variant="attention", seam_addr=True, stem_addr=True,
        attn_ptr=True, use_moe=False, asi_id=ASI,
    )
    model = create_model(cfg).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        Xb, Yb, YcB, Gb, Ab, Sb, Rb = make_reorder_batch_normalized(
            tok, SA[:40], SB[:40], bs=args.bs, seed=step, device=DEV)
        out = model(Xb, seam_anchor=Ab)
        loss, lb, lg, lc = loss_v33(out, Yb, YcB, Gb, lossf=lossf)
        lp, ll = seam_losses(out, Sb, Rb, Ab)
        loss = loss + lp + ll
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        opt.zero_grad()

        if step % args.log_every == 0 or step == args.steps:
            model.eval()
            sg, _ = eval_reorders(model, tok, SA[:10], SB[:10],
                                  seed=7, device=DEV)
            ug, _ = eval_reorders(model, tok, SA[50:60], SB[50:60],
                                  seed=9, device=DEV)
            model.train()
            elapsed = time.time() - t0
            print(f"    step {step:5d} loss={loss.item():.3f} "
                  f"seen={sg:.3f} unseen_slot={ug:.3f} "
                  f"[{elapsed:.0f}s]", flush=True)

    # Eval on unseen verb families
    vgen = eval_with_real_verbs(model)
    avg_v = sum(vgen.values()) / len(vgen)
    print(f"\n  Unseen verb generalization: {avg_v:.3f}")
    for vn, va in vgen.items():
        print(f"    {vn}: {va:.3f}")

    results["normalized"] = {
        "verb_accs": vgen,
        "avg_verb": avg_v,
    }

    # Test 2: Baseline (real verbs, fetch/deploy only)
    print("\n--- Baseline (fetch/deploy only) ---")
    model2 = create_model(cfg).to(DEV)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=3e-4)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        Xb, Yb, YcB, Gb, Ab, Sb, Rb = make_reorder_batch(
            tok, SA[:40], SB[:40], bs=args.bs, seed=step, device=DEV)
        out = model2(Xb, seam_anchor=Ab)
        loss, lb, lg, lc = loss_v33(out, Yb, YcB, Gb, lossf=lossf)
        lp, ll = seam_losses(out, Sb, Rb, Ab)
        loss = loss + lp + ll
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model2.parameters(), 5.0)
        opt2.step()
        opt2.zero_grad()

    vgen2 = eval_with_real_verbs(model2)
    avg_v2 = sum(vgen2.values()) / len(vgen2)
    print(f"  Unseen verb generalization: {avg_v2:.3f}")

    results["baseline"] = {
        "verb_accs": vgen2,
        "avg_verb": avg_v2,
    }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Normalized: avg_verb={avg_v:.3f}")
    print(f"  Baseline:   avg_verb={avg_v2:.3f}")

    if avg_v > avg_v2 + 0.05:
        print("\n  → Verb normalization IMPROVES cross-family generalization!")
    elif avg_v > avg_v2:
        print("\n  → Verb normalization helps slightly")
    else:
        print("\n  → Verb normalization does NOT help")

    out_path = os.path.join(os.path.dirname(__file__), "v9_4_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--log-every", type=int, default=500)
    args = p.parse_args()
    run_experiment(args)
