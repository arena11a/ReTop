#!/usr/bin/env python3
"""v9.2 All-in-one probe: ptr3 plateau root cause analysis.

Trains on reorder tasks (fetch/deploy) then evaluates:
  A) Capacity: D=64/128/256, unseen slot generalization
  B) Signal: cross-verb generalization (load/unload, check/clean, ...)
  C) Gap: seen vs unseen accuracy
  D) Data: 10/20/40 training pairs, unseen slot + unseen verb eval
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
    make_reorder_batch, make_reorder_ids,
    eval_reorders, resolve_device, _find_word, _find_all_word,
)

tok = Tokenizer.from_file("retop_tokenizer.json")
ASI = tok.token_to_id(ASSIST)
VOCAB = tok.get_vocab_size()
DEV = resolve_device(None)

PKGS = [f"pkg{i:03d}" for i in range(80)]
LIBS = [f"lib{i:03d}" for i in range(80, 160)]

TRAIN_A, TRAIN_B = PKGS[:40], LIBS[:40]
UNSEEN_A, UNSEEN_B = PKGS[50:60], LIBS[50:60]

VERB_PAIRS = [("load", "unload"), ("check", "clean"),
              ("search", "find"), ("start", "stop")]


def build_reorder_prompt(tok, a_s, b_s, verb_a="fetch", verb_b="deploy"):
    """Build reorder prompt+gold for arbitrary verb pair."""
    bos = tok.token_to_id(BOS)
    uid = tok.token_to_id(USER)
    asid = tok.token_to_id(ASSIST)
    eos = tok.token_to_id(EOS)
    text = f"{verb_a} {a_s} and {verb_b} {b_s}"
    U = list(tok.encode(text).ids)
    i_u = _find_word(tok, U, REORDER_AND)
    Gt = U[i_u + 1:] + [U[i_u]] + U[:i_u]
    ids = [bos, uid] + U + [asid] + Gt + [eos]
    prompt = ids[:2 + len(U) + 1]
    gold = tok.decode(Gt).strip()
    return prompt, gold


def train_model(dim, n_layers, steps, bs=8, lr=3e-4,
                train_a=TRAIN_A, train_b=TRAIN_B, log_every=500):
    """Train model on reorder tasks. Returns (model, log_dict)."""
    cfg = HMNConfig(
        vocab_size=VOCAB, dim=dim, n_layers=n_layers,
        variant="attention", seam_addr=True, stem_addr=True,
        attn_ptr=True, use_moe=False, asi_id=ASI,
    )
    model = create_model(cfg).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)

    log = {"step": [], "loss": [], "ptr": [], "seen": [], "unseen": []}
    t0 = time.time()

    for step in range(1, steps + 1):
        Xb, Yb, YcB, Gb, Ab, Sb, Rb = make_reorder_batch(
            tok, train_a, train_b, bs=bs, seed=step, device=DEV)
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
            sg, _ = eval_reorders(model, tok, TRAIN_A[:10], TRAIN_B[:10],
                                  seed=7, device=DEV)
            ug, _ = eval_reorders(model, tok, UNSEEN_A, UNSEEN_B,
                                  seed=9, device=DEV)
            model.train()
            log["step"].append(step)
            log["loss"].append(loss.item())
            log["ptr"].append(lp.item())
            log["seen"].append(sg)
            log["unseen"].append(ug)
            elapsed = time.time() - t0
            print(f"    step {step:5d} loss={loss.item():.3f} "
                  f"ptr={lp.item():.3f} seen={sg:.3f} unseen={ug:.3f} "
                  f"[{elapsed:.0f}s]", flush=True)

    elapsed = time.time() - t0
    return model, log, elapsed


def eval_verb_generalization(model, n=10):
    """Eval model on unseen verb families."""
    model.eval()
    results = {}
    for va, vb in VERB_PAIRS:
        ok = tot = 0
        rng = random.Random(42)
        for _ in range(n):
            a = rng.choice(UNSEEN_A[:n])
            b = rng.choice(UNSEEN_B[:n])
            prompt, gold = build_reorder_prompt(tok, a, b, va, vb)
            txt, g, _ = decode_v33(model, tok, prompt, max_new=32,
                                    mode="hard", seam=True,
                                    pos_eos=True, device=DEV)
            ok += int(txt.strip() == gold)
            tot += 1
        results[f"{va}/{vb}"] = ok / max(tot, 1)
    model.train()
    return results


def run_all(args):
    all_results = {}

    # === PROBE A: Capacity ===
    print("\n" + "=" * 60)
    print("PROBE A: Capacity — does bigger model break ptr3 plateau?")
    print("=" * 60)
    results_a = {}
    for dim, nl in [(64, 2), (128, 3), (256, 3)]:
        print(f"\n  --- D={dim} L={nl} ---", flush=True)
        m, log, t = train_model(dim, nl, args.steps, bs=args.bs,
                                log_every=args.log_every)
        m.eval()
        sg, _ = eval_reorders(m, tok, TRAIN_A[:10], TRAIN_B[:10],
                              seed=7, device=DEV)
        ug, _ = eval_reorders(m, tok, UNSEEN_A, UNSEEN_B,
                              seed=9, device=DEV)
        vgen = eval_verb_generalization(m, n=10)
        avg_v = sum(vgen.values()) / len(vgen)
        results_a[f"D{dim}"] = {
            "seen": sg, "unseen_slot": ug, "avg_verb": avg_v,
            "verb_accs": vgen, "time": t,
        }
        print(f"  FINAL D={dim}: seen={sg:.3f} unseen_slot={ug:.3f} "
              f"verb={avg_v:.3f}")
    all_results["A"] = results_a

    # === PROBE B: Signal — per-family unseen eval ===
    print("\n" + "=" * 60)
    print("PROBE B: Signal — per-family unseen generalization")
    print("=" * 60)
    m, log, t = train_model(96, 3, args.steps, bs=args.bs,
                            log_every=args.log_every)
    m.eval()
    sg, _ = eval_reorders(m, tok, TRAIN_A[:10], TRAIN_B[:10],
                          seed=7, device=DEV)
    ug, _ = eval_reorders(m, tok, UNSEEN_A, UNSEEN_B,
                          seed=9, device=DEV)
    vgen = eval_verb_generalization(m, n=10)
    avg_v = sum(vgen.values()) / len(vgen)
    print(f"  seen={sg:.3f} unseen_slot={ug:.3f} avg_verb={avg_v:.3f}")
    for vn, va in vgen.items():
        print(f"    {vn}: {va:.3f}")
    all_results["B"] = {
        "seen": sg, "unseen_slot": ug,
        "verb_accs": vgen, "avg_verb": avg_v, "time": t,
    }

    # === PROBE C: Seen-Unseen Gap ===
    print("\n" + "=" * 60)
    print("PROBE C: Seen-Unseen Gap — how large is the generalization gap?")
    print("=" * 60)
    gap = sg - ug
    print(f"  seen={sg:.3f} unseen={ug:.3f} gap={gap:.3f}")
    if gap > 0.1:
        print("  → Significant generalization gap")
    elif gap < 0.05:
        print("  → Small gap, generalization is good")
    all_results["C"] = {"seen": sg, "unseen": ug, "gap": gap}

    # === PROBE D: Data Scaling ===
    print("\n" + "=" * 60)
    print("PROBE D: Data — training pairs vs unseen generalization")
    print("=" * 60)
    results_d = {}
    for n_pairs in [10, 20, 40]:
        ta, tb = PKGS[:n_pairs], LIBS[:n_pairs]
        print(f"\n  --- {n_pairs} pairs ---", flush=True)
        m, log, t = train_model(96, 3, args.steps, bs=args.bs,
                                train_a=ta, train_b=tb,
                                log_every=args.log_every)
        m.eval()
        sg, _ = eval_reorders(m, tok, TRAIN_A[:10], TRAIN_B[:10],
                              seed=7, device=DEV)
        ug, _ = eval_reorders(m, tok, UNSEEN_A, UNSEEN_B,
                              seed=9, device=DEV)
        vgen = eval_verb_generalization(m, n=5)
        avg_v = sum(vgen.values()) / len(vgen)
        results_d[str(n_pairs)] = {
            "seen": sg, "unseen_slot": ug, "avg_verb": avg_v, "time": t,
        }
        print(f"  FINAL {n_pairs}: seen={sg:.3f} unseen_slot={ug:.3f} "
              f"verb={avg_v:.3f}")
    all_results["D"] = results_d

    # === Summary ===
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nProbe A (Capacity):")
    for k, v in all_results["A"].items():
        print(f"  {k}: seen={v['seen']:.3f} unseen_slot={v['unseen_slot']:.3f} "
              f"verb={v['avg_verb']:.3f}")
    print(f"\nProbe B (Signal): avg_verb={all_results['B']['avg_verb']:.3f}")
    print(f"\nProbe C (Gap): gap={all_results['C']['gap']:.3f}")
    print("\nProbe D (Data):")
    for k, v in all_results["D"].items():
        print(f"  {k}_pairs: seen={v['seen']:.3f} unseen_slot={v['unseen_slot']:.3f} "
              f"verb={v['avg_verb']:.3f}")

    out_path = os.path.join(os.path.dirname(__file__), "probe_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return all_results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--log-every", type=int, default=500)
    args = p.parse_args()
    run_all(args)
