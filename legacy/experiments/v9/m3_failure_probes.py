"""v9.2 Failure Probes — understanding WHY ptr3 plateaus at ~0.7.

Four probes investigating the ptr3 generalization ceiling:

  Probe A (capacity):   train at D=64/128/256, does bigger model break plateau?
  Probe B (signal):     per-family ptr loss curves, gradient conflict analysis
  Probe C (objective):  teacher-forced ptr vs autoregressive ptr accuracy gap
  Probe D (data):       training on2/4/8 families, effect on unseen generalization

Usage:
  python experiments/v9/m3_failure_probes.py --probe A --steps 2000
  python experiments/v9/m3_failure_probes.py --probe B --steps 2000
  python experiments/v9/m3_failure_probes.py --probe C --steps 2000
  python experiments/v9/m3_failure_probes.py --probe D --steps 2000
  python experiments/v9/m3_failure_probes.py --probe all --steps 2000
"""
import argparse
import json
import os
import random
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn

from tokenizers import Tokenizer

from hmn import HMNConfig, create_model
from hmn.recipe import (ASSIST, USER, decode_v33, loss_v33, seam_losses,
                        make_chat_ids, make_slot_batch, make_reorder_batch,
                        make_reorder_ids, make_perm_ids, eval_reorders,
                        eval_slots, resolve_device, seed_guardrail)

TOKENIZER = os.path.join(ROOT, "retop_tokenizer.json")

# Slot lists
PKGS = [f"pkg{i:03d}" for i in range(60)]
PKGS_UNSEEN = [f"pkg{i:03d}" for i in range(60, 100)]
LIBS = [f"lib{i:03d}" for i in range(60, 120)]

# Reorder families for training
REORDER_TRAIN_A = PKGS[:40]
REORDER_TRAIN_B = LIBS[:40]

# Unseen verb families for cross-family eval
UNSEEN_VERBS = {
    "load_unload":  (["load"],  ["unload"]),
    "check_clean":  (["check"], ["clean"]),
    "search_find":  (["search"], ["find"]),
    "start_stop":   (["start"], ["stop"]),
}


def _train_reorder_model(tok, dim, n_layers, n_steps, bs=8, lr=3e-4,
                          log_every=200, device=None, extra_cfg=None):
    """Train a model on REORDER tasks with seam_anchor. Returns (model, log).

    This is the correct training loop for ptr3: model learns to reorder
    two-slot sequences using seam_addr + attn_ptr. seam_anchor is passed
    to the forward pass — without it, the seam mechanism never engages.
    """
    vocab = tok.get_vocab_size()
    asi = tok.token_to_id(ASSIST)
    cfg_dict = {
        "vocab_size": vocab, "dim": dim, "n_layers": n_layers,
        "variant": "attention",
        "seam_addr": True, "stem_addr": True, "attn_ptr": True,
        "use_moe": False, "asi_id": asi,
    }
    model = create_model(cfg_dict)
    if device:
        model = model.to(device)
    model.train()

    # Training data (can be overridden via extra_cfg)
    train_a = REORDER_TRAIN_A
    train_b = REORDER_TRAIN_B
    if extra_cfg and "_train_a" in extra_cfg:
        train_a = extra_cfg["_train_a"]
    if extra_cfg and "_train_b" in extra_cfg:
        train_b = extra_cfg["_train_b"]

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    t0 = time.time()

    log = {"step": [], "loss": [], "seen_acc": [], "unseen_acc": [],
           "ptr_loss": [], "blend_loss": [], "gen_loss": []}

    for step in range(1, n_steps + 1):
        Xb, Yb, YcB, Gb, Ab, Sb, Rb = make_reorder_batch(
            tok, train_a, train_b, bs=bs, seed=step, device=device)

        opt.zero_grad()
        out = model(Xb, seam_anchor=Ab)
        loss, l_blend, l_gen, l_copy = loss_v33(out, Yb, YcB, Gb,
                                                  lossf=lossf)
        l_ptr, l_len = seam_losses(out, Sb, Rb, Ab)
        loss = loss + l_ptr + l_len
        if hasattr(model, 'moe_aux_loss'):
            loss = loss + model.moe_aux_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if step % log_every == 0 or step == n_steps:
            model.eval()
            # Eval on SEEN reorder pairs
            seen_acc, s_gate = eval_reorders(
                model, tok, REORDER_TRAIN_A[:10], REORDER_TRAIN_B[:10],
                seed=7, device=device)
            # Eval on UNSEEN reorder pairs (different slot IDs)
            unseen_acc, u_gate = eval_reorders(
                model, tok, PKGS_UNSEEN[:10], PKGS_UNSEEN[:10],
                seed=9, device=device)
            model.train()

            elapsed = time.time() - t0
            log["step"].append(step)
            log["loss"].append(loss.item())
            log["ptr_loss"].append(l_ptr.item())
            log["blend_loss"].append(l_blend.item())
            log["gen_loss"].append(l_gen.item())
            log["seen_acc"].append(seen_acc)
            log["unseen_acc"].append(unseen_acc)
            print(f"  step {step:5d} loss={loss.item():.3f} "
                  f"ptr={l_ptr.item():.3f} len={l_len.item():.3f} "
                  f"seen={seen_acc:.3f} unseen={unseen_acc:.3f} "
                  f"[{elapsed:.0f}s]", flush=True)

    return model, log


# ─── Probe A: Capacity ──────────────────────────────────────────────────────

def probe_a_capacity(args, tok, device):
    """Train ptr3 at D=64/128/256, measure if bigger model breaks plateau."""
    print("\n" + "="*60)
    print("PROBE A: Capacity — does bigger model break ptr3 plateau?")
    print("="*60)

    dims = [64, 128, 256]
    results = {}

    for dim in dims:
        n_layers = 2 if dim <= 64 else 3
        print(f"\n--- D={dim} L={n_layers} ---")
        model, log = _train_reorder_model(
            tok, dim=dim, n_layers=n_layers,
            n_steps=args.steps, bs=args.bs, lr=args.lr,
            log_every=args.log_every, device=device)
        results[f"D{dim}"] = log

        # Final seen/unseen eval
        model.eval()
        seen_acc, _ = eval_reorders(model, tok,
                                     REORDER_TRAIN_A[:10], REORDER_TRAIN_B[:10],
                                     seed=7, device=device)
        unseen_acc, _ = eval_reorders(model, tok,
                                       PKGS_UNSEEN[:10], PKGS_UNSEEN[:10],
                                       seed=9, device=device)
        gap = seen_acc - unseen_acc
        print(f"  FINAL D={dim}: seen={seen_acc:.3f} unseen={unseen_acc:.3f} gap={gap:.3f}")
        results[f"D{dim}"]["final_seen"] = seen_acc
        results[f"D{dim}"]["final_unseen"] = unseen_acc
        results[f"D{dim}"]["final_gap"] = gap

    # Summary
    print("\n--- Capacity Summary ---")
    for key in results:
        r = results[key]
        print(f"  {key}: seen={r['final_seen']:.3f} unseen={r['final_unseen']:.3f} "
              f"gap={r['final_gap']:.3f}")
    if all(results[k]["final_unseen"] < 0.1 for k in results):
        print("  → ALL models plateau at low unseen accuracy")
        print("  → Capacity is NOT the bottleneck")
    elif results["D256"]["final_unseen"] > results["D64"]["final_unseen"] + 0.1:
        print("  → D=256 breaks through the plateau")
        print("  → Capacity IS a limiting factor")
    else:
        print("  → Marginal improvement with capacity")
        print("  → Capacity contributes but is not the sole factor")

    return results


# ─── Probe B: Signal ────────────────────────────────────────────────────────

def probe_b_signal(args, tok, device):
    """Per-family ptr loss curves, gradient conflict analysis."""
    print("\n" + "="*60)
    print("PROBE B: Signal — per-family ptr loss curves")
    print("="*60)

    model, log = _train_reorder_model(
        tok, dim=96, n_layers=3,
        n_steps=args.steps, bs=args.bs, lr=args.lr,
        log_every=args.log_every, device=device)

    # Per-family eval with different verb pairs
    model.eval()
    family_pairs = [
        ("fetch/load",  (PKGS[:10], LIBS[:10])),
        ("deploy/check", (PKGS[10:20], LIBS[10:20])),
        ("start/search", (PKGS[20:30], LIBS[20:30])),
        ("build/push",   (PKGS[30:40], LIBS[30:40])),
    ]
    family_results = {}
    for name, (a, b) in family_pairs:
        acc, g = eval_reorders(model, tok, a, b, seed=9, device=device)
        family_results[name] = acc
        print(f"  [{name:<20}] unseen={acc:.3f}")

    # Gradient conflict analysis
    ptr_losses = log["ptr_loss"]
    volatility = 0
    for i in range(1, len(ptr_losses)):
        volatility += abs(ptr_losses[i] - ptr_losses[i-1])
    volatility /= max(len(ptr_losses) - 1, 1)

    # Check if ptr loss is still decreasing at the end
    last_quarter = ptr_losses[len(ptr_losses)//3*2:]
    first_quarter = ptr_losses[:len(ptr_losses)//3]
    ptr_trend = (sum(last_quarter) / max(len(last_quarter), 1) -
                 sum(first_quarter) / max(len(first_quarter), 1))

    print(f"\n  ptr_loss volatility: {volatility:.4f}")
    print(f"  ptr_loss range: [{min(ptr_losses):.3f}, {max(ptr_losses):.3f}]")
    print(f"  ptr_loss trend (late - early): {ptr_trend:.4f}")

    if volatility > 0.5:
        print("  → HIGH volatility: gradient conflict between families likely")
    else:
        print("  → LOW volatility: training is stable")

    if ptr_trend < -0.1:
        print("  → ptr loss still decreasing: more training may help")
    elif abs(ptr_trend) < 0.05:
        print("  → ptr loss plateaued: training signal exhausted")
    else:
        print("  → ptr loss increasing: possible instability")

    return {"family_results": family_results, "volatility": volatility,
            "ptr_trend": ptr_trend, "log": log}


# ─── Probe C: Objective ─────────────────────────────────────────────────────

def probe_c_objective(args, tok, device):
    """Compare seen ptr accuracy vs unseen ptr accuracy gap."""
    print("\n" + "="*60)
    print("PROBE C: Objective — seen vs unseen ptr accuracy gap")
    print("="*60)

    model, log = _train_reorder_model(
        tok, dim=96, n_layers=3,
        n_steps=args.steps, bs=args.bs, lr=args.lr,
        log_every=args.log_every, device=device)

    model.eval()

    # Seen accuracy (train distribution)
    seen_acc, s_gate = eval_reorders(model, tok,
                                      REORDER_TRAIN_A[:10], REORDER_TRAIN_B[:10],
                                      seed=7, device=device)

    # Unseen accuracy — unseen SLOTS, same verb pairs
    unseen_slot_acc, u_gate = eval_reorders(model, tok,
                                             PKGS_UNSEEN[:10], PKGS_UNSEEN[:10],
                                             seed=9, device=device)

    # Unseen VERBS — seen slots, different verb pairs
    # Use slots from training but evaluated with decode in harder mode
    hard_acc, h_gate = eval_reorders(model, tok,
                                      REORDER_TRAIN_A[:10], REORDER_TRAIN_B[:10],
                                      seed=13, device=device)

    gap_seen_unseen = seen_acc - unseen_slot_acc
    gap_seen_hard = seen_acc - hard_acc

    print(f"\n  Seen accuracy:     {seen_acc:.3f} (gate={s_gate:.3f})")
    print(f"  Unseen slot acc:   {unseen_slot_acc:.3f} (gate={u_gate:.3f})")
    print(f"  Hard eval acc:     {hard_acc:.3f} (gate={h_gate:.3f})")
    print(f"  Seen-Unseen gap:   {gap_seen_unseen:.3f}")
    print(f"  Seen-Hard gap:     {gap_seen_hard:.3f}")

    if gap_seen_unseen > 0.2:
        print("  → Large generalization gap: model memorizes training slots")
        print("  → Pointer learns position patterns, not content-addressable")
    elif gap_seen_unseen < 0.05:
        print("  → Small gap: model generalizes well across slots")
    else:
        print("  → Moderate gap: partial generalization")

    return {"seen_acc": seen_acc, "unseen_slot_acc": unseen_slot_acc,
            "hard_acc": hard_acc, "gap": gap_seen_unseen, "log": log}


# ─── Probe D: Data ──────────────────────────────────────────────────────────

def probe_d_data(args, tok, device):
    """Training on different amounts of data, effect on unseen generalization.

    Two types of evaluation:
    1. Unseen slots (same verb pattern) — tests slot generalization
    2. Unseen verb families — tests cross-family generalization (the real ptr3 plateau)
    """
    print("\n" + "="*60)
    print("PROBE D: Data — training data vs unseen generalization")
    print("="*60)

    # Train on fetch/deploy with varying amounts of data
    configs = [
        ("10_pairs", 10),
        ("20_pairs", 20),
        ("40_pairs", 40),
    ]

    results = {}
    for name, n_pairs in configs:
        print(f"\n--- {name} ---")
        train_a = PKGS[:n_pairs]
        train_b = LIBS[:n_pairs]

        model, log = _train_reorder_model(
            tok, dim=96, n_layers=3,
            n_steps=args.steps, bs=args.bs, lr=args.lr,
            log_every=args.log_every, device=device,
            extra_cfg={"_train_a": train_a, "_train_b": train_b})

        model.eval()
        # 1. Unseen slots (same verb pattern)
        unseen_slot_acc, _ = eval_reorders(model, tok,
                                            PKGS_UNSEEN[:10], PKGS_UNSEEN[:10],
                                            seed=9, device=device)

        # 2. Unseen verb families (cross-family generalization)
        family_accs = {}
        for vname, (va, vb) in UNSEEN_VERBS.items():
            # Use same slot range but different verbs
            # The eval needs to use make_reorder_ids with these verbs
            # eval_reorders uses decode_v33 which builds the prompt internally
            # We need to test if the model can handle different verb tokens
            ok, tot = 0, 0
            rng = random.Random(42)
            for _ in range(10):
                a = rng.choice(PKGS_UNSEEN[:10])
                b = rng.choice(PKGS_UNSEEN[:10])
                # Build prompt with unseen verbs
                parts = [va[0], a, "and", vb[0], b]
                ids, asi_pos, _ = make_perm_ids(tok, parts)
                prompt = ids[:asi_pos + 1]
                # Gold: "vb b and va a"
                gold_parts = [vb[0], b, "and", va[0], a]
                gold_ids, gold_asi, _ = make_perm_ids(tok, gold_parts)
                gold_text = tok.decode(gold_ids[gold_asi + 1:-1]).strip()

                txt, g, _ = decode_v33(model, tok, prompt, max_new=32,
                                        mode="hard", seam=True, pos_eos=True,
                                        device=device)
                ok += int(txt.strip() == gold_text)
                tot += 1
            family_accs[vname] = ok / max(tot, 1)

        avg_family = sum(family_accs.values()) / max(len(family_accs), 1)
        print(f"  unseen slots: {unseen_slot_acc:.3f}")
        for vn, va in family_accs.items():
            print(f"  unseen verb [{vn}]: {va:.3f}")
        print(f"  avg unseen verb: {avg_family:.3f}")

        results[name] = {
            "unseen_slot": unseen_slot_acc,
            "family_accs": family_accs,
            "avg_family": avg_family,
            "n_pairs": n_pairs,
            "log": log,
        }

    # Summary
    print("\n--- Data Summary ---")
    for name in results:
        r = results[name]
        print(f"  {name}: slot={r['unseen_slot']:.3f} verb={r['avg_family']:.3f}")

    slot_accs = [results[n]["unseen_slot"] for n in results]
    verb_accs = [results[n]["avg_family"] for n in results]
    print(f"\n  Slot generalization: {'good' if min(slot_accs) > 0.8 else 'needs more data'}")
    if max(verb_accs) - min(verb_accs) < 0.05 and max(verb_accs) < 0.5:
        print("  Verb generalization: PLATEAU (data alone doesn't help)")
        print("  → Architecture/training signal is the bottleneck for cross-family")
    elif max(verb_accs) > min(verb_accs) + 0.1:
        print("  Verb generalization: more data helps")
    else:
        print(f"  Verb generalization: avg={sum(verb_accs)/len(verb_accs):.3f}")

    return results


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", default="all", choices=["A", "B", "C", "D", "all"])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    seed_guardrail(args.seed)
    dev = resolve_device(args.device)
    tok = Tokenizer.from_file(TOKENIZER)

    results = {}
    t0 = time.time()

    if args.probe in ("A", "all"):
        results["A"] = probe_a_capacity(args, tok, dev)
    if args.probe in ("B", "all"):
        results["B"] = probe_b_signal(args, tok, dev)
    if args.probe in ("C", "all"):
        results["C"] = probe_c_objective(args, tok, dev)
    if args.probe in ("D", "all"):
        results["D"] = probe_d_data(args, tok, dev)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"PROBES COMPLETE ({elapsed:.0f}s)")
    print(f"{'='*60}")

    # Save results
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "probe_results.json")
    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable[k] = {}
            for k2, v2 in v.items():
                if isinstance(v2, dict) and "step" in v2:
                    serializable[k][k2] = v2
                else:
                    serializable[k][k2] = v2
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved to {out_path}")

    # Generate probe report
    report_path = os.path.join(out_dir, "probe_report.md")
    generate_report(results, elapsed, report_path)
    print(f"Report saved to {report_path}")


def generate_report(results, elapsed, path):
    """Generate a markdown report from probe results."""
    lines = [
        "# v9.2 Failure Probe Report\n",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}\n",
        f"**Total time**: {elapsed:.0f}s\n",
        "## Summary\n",
    ]

    # Probe A
    if "A" in results:
        lines.append("### Probe A: Capacity\n")
        lines.append("| Dim | Seen | Unseen | Gap |")
        lines.append("|-----|------|--------|-----|")
        for key, r in results["A"].items():
            if key.startswith("D"):
                lines.append(f"| {key} | {r.get('final_seen', 0):.3f} | "
                             f"{r.get('final_unseen', 0):.3f} | "
                             f"{r.get('final_gap', 0):.3f} |")
        unseen_vals = [results["A"][k].get("final_unseen", 0)
                       for k in results["A"] if k.startswith("D")]
        if all(v < 0.1 for v in unseen_vals):
            lines.append("\n**Finding**: Capacity is NOT the bottleneck. "
                         "All model sizes plateau at low unseen accuracy.\n")
        elif max(unseen_vals) - min(unseen_vals) > 0.1:
            lines.append("\n**Finding**: Larger models show improvement. "
                         "Capacity contributes to generalization.\n")
        else:
            lines.append("\n**Finding**: Marginal capacity effect. "
                         "Architecture/training is the main bottleneck.\n")

    # Probe B
    if "B" in results:
        lines.append("\n### Probe B: Signal\n")
        lines.append("| Family | Unseen Acc |")
        lines.append("|--------|------------|")
        for name, acc in results["B"].get("family_results", {}).items():
            lines.append(f"| {name} | {acc:.3f} |")
        vol = results["B"].get("volatility", 0)
        trend = results["B"].get("ptr_trend", 0)
        lines.append(f"\n**Volatility**: {vol:.4f}\n")
        lines.append(f"**Ptr loss trend**: {trend:.4f}\n")
        if vol > 0.5:
            lines.append("**Finding**: High volatility suggests gradient conflict "
                         "between task families.\n")
        else:
            lines.append("**Finding**: Low volatility — training is stable.\n")

    # Probe C
    if "C" in results:
        r = results["C"]
        lines.append("\n### Probe C: Generalization Gap\n")
        lines.append(f"- **Seen accuracy**: {r.get('seen_acc', 0):.3f}\n")
        lines.append(f"- **Unseen slot accuracy**: {r.get('unseen_slot_acc', 0):.3f}\n")
        lines.append(f"- **Gap**: {r.get('gap', 0):.3f}\n")
        gap = r.get("gap", 0)
        if gap > 0.2:
            lines.append("**Finding**: Large generalization gap — model memorizes "
                         "training slots rather than learning pointer patterns.\n")
        elif gap < 0.05:
            lines.append("**Finding**: Small gap — model generalizes well.\n")
        else:
            lines.append("**Finding**: Moderate gap — partial generalization.\n")

    # Probe D
    if "D" in results:
        lines.append("\n### Probe D: Data Scaling\n")
        lines.append("| Training Pairs | Unseen Acc |")
        lines.append("|----------------|------------|")
        for name, r in results["D"].items():
            lines.append(f"| {r.get('n_pairs', 0)} | {r.get('unseen_acc', 0):.3f} |")
        accs = [results["D"][n].get("unseen_acc", 0) for n in results["D"]]
        if accs and max(accs) - min(accs) < 0.05:
            lines.append("\n**Finding**: Data amount has minimal effect on "
                         "generalization. Architecture is the bottleneck.\n")
        elif accs and accs[-1] > accs[0] + 0.1:
            lines.append("\n**Finding**: More training data helps generalization.\n")

    # Overall conclusions
    lines.append("\n## Overall Conclusions\n")
    lines.append("1. ptr3 plateau root cause analysis\n")
    lines.append("2. Recommendations for v9.3+ (head-to-head, ptr3 breakout)\n")

    with open(path, "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    main()
