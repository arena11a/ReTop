"""v5 M3 — zero-shot generalization of the run machinery (NO retraining).

M1/M2 trained ONLY the 2-segment swap (`fetch {a} and deploy {b}` ->
`deploy {b} and fetch {a}`). The seam mechanism is structural (anchor echo +
SeedPointer), so this probe asks how far it stretches with ZERO new training:

  P1 unseen slot families   bin{...} / cfg{...}, trained verbs fetch+deploy
  P2 unseen verb pair       load {a} / unload {b}, trained families pkg/lib
  P3 3-segment ROTATION     fetch a and deploy b and stop c
                            -> deploy b and stop c and fetch a
                            (5 runs; `stop` never trained anywhere)
  P4 repeated-digit slots   pkg333 / pkg999 swap

Pass criteria (docs/v5_omega_roadmap.md): P1/P2/P4 >= 0.60; P3 >= 0.30
(first length-generalization step; v4 chain probes set the precedent that
first-shot length generalization can lag then close).

Run:
  python experiments/v5/m3_generalization.py --checkpoint omega_joint.pt
"""
import argparse
import os
import random
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import torch

from tokenizers import Tokenizer

from hmn import HMN3
from hmn.recipe import (ASSIST, USER, decode_rotate, decode_v33,
                        make_perm_ids, resolve_device, seed_guardrail)

TOKENIZER = os.path.join(ROOT, "retop_tokenizer.json")


def run_probe(model, tok, name, parts_fn, n=20, seed=9, max_new=64,
              device=None, rotate=False):
    rng = random.Random(seed)
    ok = tot = 0
    fails = []
    for _ in range(n):
        parts = parts_fn(rng)
        ids, asi_pos, _ands = make_perm_ids(tok, parts)
        prompt = ids[:asi_pos + 1]
        gold_text = tok.decode(ids[asi_pos + 1:-1]).strip()
        if rotate:
            txt, g, _ng = decode_rotate(model, tok, prompt,
                                        max_new=max_new, device=device)
        else:
            txt, g, _ng = decode_v33(model, tok, prompt, max_new=max_new,
                                     mode="hard", seam=True, pos_eos=True,
                                     device=device)
        ok += int(txt.strip() == gold_text)
        tot += 1
        if len(fails) < 3 and txt.strip() != gold_text:
            fails.append((gold_text, txt.strip()))
    print(f"  [{name:<38}] {ok}/{tot}", flush=True)
    for g_, t_ in fails:
        print(f"      gold={g_!r}\n      got ={t_!r}")
    return ok / tot


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="omega_joint.pt")
    ap.add_argument("--suite", default="m1", choices=["m1", "cur"],
                    help="m1 = zero-shot probes on the single-template M1/M2 "
                         "ckpt; cur = post-curriculum probes incl. HELD-OUT "
                         "verbs (open/close) + family rel + 4-seg rotation")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    dev = resolve_device(args.device)
    seed_guardrail(42)
    tok = Tokenizer.from_file(TOKENIZER)
    model = HMN3(tok.get_vocab_size(), dim=args.dim, state_dim=8,
                 n_layers=args.layers, use_moe=False, gate_bias=-1.0,
                 asi_id=tok.token_to_id(ASSIST),
                 user_id=tok.token_to_id(USER), seam_addr=True)
    model.load_state_dict(torch.load(args.checkpoint, map_location=dev))
    model.to(dev).eval()
    print(f"loaded {args.checkpoint} "
          f"({sum(p.numel() for p in model.parameters()):,} params) on {dev}")

    res = {}
    if args.suite == "cur":
        # curriculum ckpt: bin/cfg families and load/stop/check/unload/clean
        # verbs were TRAINED; rel family + open/close verbs are HELD OUT.
        res["P1 swap (seen shape, unseen slots)"] = run_probe(
            model, tok, "P1 swap seen-shape (pkg/lib unseen)",
            lambda r: [f"fetch pkg{r.randint(60, 79):03d}",
                       f"deploy lib{r.randint(0, 19):03d}"],
            n=args.n, seed=args.seed, device=dev)
        res["P2 HELD-OUT verbs+family"] = run_probe(
            model, tok, "P2 held-out: open/close + rel",
            lambda r: [f"open rel{r.randint(0, 99):03d}",
                       f"close cfg{r.randint(0, 99):03d}"],
            n=args.n, seed=args.seed + 1, device=dev, rotate=True)
        res["P3 3-seg rotation (seen shape)"] = run_probe(
            model, tok, "P3 3-seg rotation mixed verbs",
            lambda r: [f"load pkg{r.randint(60, 69):03d}",
                       f"stop lib{r.randint(0, 9):03d}",
                       f"check bin{r.randint(0, 39):03d}"],
            n=args.n, seed=args.seed + 2, max_new=80, device=dev,
            rotate=True)
        res["P4 4-seg rotation (len-gen)"] = run_probe(
            model, tok, "P4 4-seg rotation (beyond max train N)",
            lambda r: [f"fetch pkg{r.randint(60, 69):03d}",
                       f"deploy lib{r.randint(0, 9):03d}",
                       f"load cfg{r.randint(0, 9):03d}",
                       f"check bin{r.randint(0, 9):03d}"],
            n=args.n, seed=args.seed + 5, max_new=110, device=dev,
            rotate=True)
        res["P5 repeated digits"] = run_probe(
            model, tok, "P5 repeated-digit slots",
            lambda r: [f"fetch pkg{r.choice([333, 555, 777, 999])}",
                       f"deploy lib{r.choice([222, 444, 666, 888])}"],
            n=args.n, seed=args.seed + 3, device=dev, rotate=True)
    else:
        res["P1 unseen families"] = run_probe(
            model, tok, "P1 unseen families (bin/cfg)",
            lambda r: [f"fetch bin{r.randint(40, 79):03d}",
                       f"deploy cfg{r.randint(40, 79):03d}"],
            n=args.n, seed=args.seed, device=dev)
        res["P2 unseen verbs"] = run_probe(
            model, tok, "P2 unseen verbs (load/unload)",
            lambda r: [f"load pkg{r.randint(60, 79):03d}",
                       f"unload lib{r.randint(0, 19):03d}"],
            n=args.n, seed=args.seed + 1, device=dev)
        res["P3 3-seg rotation"] = run_probe(
            model, tok, "P3 3-seg rotation (unseen 'stop')",
            lambda r: [f"fetch pkg{r.randint(60, 69):03d}",
                       f"deploy lib{r.randint(0, 9):03d}",
                       f"stop bin{r.randint(0, 39):03d}"],
            n=args.n, seed=args.seed + 2, max_new=80, device=dev)
        res["P4 repeated digits"] = run_probe(
            model, tok, "P4 repeated-digit slots",
            lambda r: [f"fetch pkg{r.choice([333, 555, 777, 999])}",
                       f"deploy lib{r.choice([222, 444, 666, 888])}"],
            n=args.n, seed=args.seed + 3, device=dev)

    print("\nM3 summary:")
    if args.suite == "cur":
        crit = {"P1 swap (seen shape, unseen slots)": 0.90,
                "P2 HELD-OUT verbs+family": 0.60,
                "P3 3-seg rotation (seen shape)": 0.90,
                "P4 4-seg rotation (len-gen)": 0.30,
                "P5 repeated digits": 0.60}
    else:
        crit = {"P1 unseen families": 0.60, "P2 unseen verbs": 0.60,
                "P3 3-seg rotation": 0.30, "P4 repeated digits": 0.60}
    allpass = True
    for k, v in res.items():
        mark = "PASS" if v >= crit[k] else "MISS"
        if v < crit[k]:
            allpass = False
        print(f"  {k:<24} {v:.3f}  ({mark}, criterion {crit[k]:.2f})")
    print("M3 ALL PROBES PASSED" if allpass else "M3: some probes missed "
          "(document honestly; see roadmap)")


if __name__ == "__main__":
    main()
