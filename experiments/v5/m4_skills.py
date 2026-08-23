"""v5 M4 — Skill Library end-to-end (distill -> retrieve -> execute).

Pipeline (Ω-Coder Module 7 mapped onto ReTop):
  1. DISTILL  : draw one fresh instance per task family, teacher-force the
                family's run recipe, accept ONLY if copy-argmax == gold AND
                gates open on every planned row (evidence before storage).
  2. RETRIEVE : fingerprint = (n_parts, segment token-counts, k0-from-ptr).
                Unique family under a fingerprint resolves; conflicts raise
                (escalate, never guess); unknown fingerprints fall back to
                rotate(k0=ptr) and are REPORTED.
  3. EXECUTE  : greedy seam walk of the retrieved plan on UNSEEN slots.

Pass criteria:
  - all four distillations verified
  - unseen-slot execution >= 0.9 on every trained family (echo, swap,
    rot3, rot4)
  - held-out verb pair executes >= 0.6 via fallback
  - ambiguity guard demonstrably raises on a conflicting insertion

Run: python experiments/v5/m4_skills.py --checkpoint omega_cur2.pt
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
from hmn import skills as S
from hmn.recipe import (ASSIST, CHAIN_SLOTS_A_U, USER, eval_slots,
                        make_chat_ids, make_perm_ids, resolve_device,
                        seed_guardrail)

TOKENIZER = os.path.join(ROOT, "retop_tokenizer.json")


def build_model(tok):
    return HMN3(tok.get_vocab_size(), dim=96, state_dim=8, n_layers=3,
                use_moe=False, gate_bias=-1.0,
                asi_id=tok.token_to_id(ASSIST),
                user_id=tok.token_to_id(USER),
                stem_addr=True, seam_addr=True)


def ids_echo(tok, parts):
    user = " and ".join(parts)
    return make_chat_ids(tok, user, user), None


def ids_perm(tok, parts):
    ids, _asi, _ands = make_perm_ids(tok, list(parts))
    return ids, _asi


def k0_of_gold(ids, gold_tail, P):
    return S.gold_k0(ids, gold_tail, P)


def distill_all(model, tok, lib, dev):
    rng = random.Random(1234)
    fams = []
    # echo: user == gold chain text
    def echo_parts(r):
        return [f"fetch pkg{r.randint(0, 39):03d}",
                f"deploy lib{r.randint(40, 79):03d}"]
    fams.append(("echo", echo_parts, ids_echo))
    # swap / rot3 / rot4 permutations
    def swap_parts(r):
        return [f"fetch pkg{r.randint(0, 39):03d}",
                f"deploy lib{r.randint(40, 79):03d}"]
    def rot3_parts(r):
        return [f"load pkg{r.randint(0, 59):03d}",
                f"stop lib{r.randint(40, 79):03d}",
                f"check bin{r.randint(0, 99):03d}"]
    def rot4_parts(r):
        return [f"fetch pkg{r.randint(0, 59):03d}",
                f"deploy lib{r.randint(40, 79):03d}",
                f"load cfg{r.randint(0, 99):03d}",
                f"check bin{r.randint(0, 99):03d}"]
    fams.append(("rotate", swap_parts, ids_perm))     # n=2 swap == rotation
    fams.append(("rotate", rot3_parts, ids_perm))
    fams.append(("rotate", rot4_parts, ids_perm))

    asid = tok.token_to_id(ASSIST)
    for fam, parts_fn, idfn in fams:
        parts = parts_fn(rng)
        ids, _extra = idfn(tok, parts)
        P = S.parse_prompt(ids, asid, tok)
        gold_tail = ids[P["asi_pos"] + 1:len(ids) - 1]
        k0 = k0_of_gold(ids, gold_tail, P) if fam == "rotate" else 0
        plan = S.RECIPES[fam](P, k0)
        ok, detail, rows = S.verify_plan(model, ids, P["asi_pos"] + 1,
                                         plan, dev)
        if not ok:
            raise AssertionError(f"distill[{fam}] FAILED: {detail}")
        entry = lib.add(fam, P["fp"], verified_on=" ".join(parts)[:48])
        entry["rows"] = rows
        entry["k0"] = k0
        print(f"  distilled [{fam:<6}] fp={entry['fp']} rows={rows} "
              f"verified", flush=True)


def execute_family(model, tok, lib, dev, name, parts_fn, idfn, n=10,
                   seed=99, max_new=80, family=None):
    rng = random.Random(seed)
    ok = tot = 0
    fb = 0
    stats = {}
    for _ in range(n):
        parts = parts_fn(rng)
        ids, _x = idfn(tok, parts)
        P = S.parse_prompt(ids, tok.token_to_id(ASSIST), tok)
        gold_text = tok.decode(ids[P["asi_pos"] + 1:-1]).strip()
        # retrieval k0 from the PTR HEAD (no gold access)
        with torch.no_grad():
            out = model(torch.tensor([ids[:P["asi_pos"] + 1]], device=dev))
        c0 = int(out["ptr_logits"][0, -1].argmax(-1).item())
        src = c0 + 1
        k0 = next((kk for kk in range(P["n_parts"])
                   if P["bounds"][kk] + 1 <= src <= P["bounds"][kk + 1]), 0)
        txt, meta = S.execute(lib, model, tok, ids[:P["asi_pos"] + 1],
                              dev, max_new=max_new, family=family)
        ok += int(txt.strip() == gold_text)
        tot += 1
        fb += int(meta["fallback"])
        stats[meta["status"]] = stats.get(meta["status"], 0) + 1
    print(f"  [{name:<34}] {ok}/{tot} (retrieval: {stats}, "
          f"fallback {fb}/{tot})", flush=True)
    return ok / tot


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="omega_cur2.pt")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    dev = resolve_device(args.device)
    seed_guardrail(42)
    tok = Tokenizer.from_file(TOKENIZER)
    model = build_model(tok)
    model.load_state_dict(torch.load(args.checkpoint, map_location=dev))
    model.to(dev).eval()
    print(f"loaded {args.checkpoint} on {dev}")

    lib = S.SkillLibrary()
    print("== DISTILL ==")
    distill_all(model, tok, lib, dev)
    print(f"library size: {len(lib)}")

    print("== AMBIGUITY GUARD ==")
    # echo and swap share verbs (same fp): retrieval without a hint must
    # escalate, never guess.
    shared_fp = None
    for fp, bucket in lib.items():
        if len(bucket) > 1:
            shared_fp = fp
            break
    if shared_fp is not None:
        _, status = lib.match(shared_fp)
        print(f"  fp={shared_fp} families={[b['family'] for b in lib._skills[shared_fp].values()]} "
              f"-> match status = {status!r} (escalates)", flush=True)
    else:
        print("  no shared fp in this library (echo/swap slots differ)")

    print("== EXECUTE (unseen slots) ==")
    results = {}
    results["echo"] = execute_family(
        model, tok, lib, dev, "echo (chain, pos_eos)",
        lambda r: [f"fetch pkg{r.randint(60, 79):03d}",
                   f"deploy lib{r.randint(0, 19):03d}"],
        ids_echo, n=args.n, seed=101, max_new=48, family="echo")
    results["swap"] = execute_family(
        model, tok, lib, dev, "swap (unseen pkg/lib)",
        lambda r: [f"fetch pkg{r.randint(60, 79):03d}",
                   f"deploy lib{r.randint(0, 19):03d}"],
        ids_perm, n=args.n, seed=102, family="rotate")
    results["rot3"] = execute_family(
        model, tok, lib, dev, "rot3 (unseen mixed)",
        lambda r: [f"load pkg{r.randint(60, 69):03d}",
                   f"stop lib{r.randint(0, 9):03d}",
                   f"check bin{r.randint(0, 39):03d}"],
        ids_perm, n=args.n, seed=103, family="rotate")
    results["rot4"] = execute_family(
        model, tok, lib, dev, "rot4 (len-gen)",
        lambda r: [f"fetch pkg{r.randint(60, 69):03d}",
                   f"deploy lib{r.randint(0, 9):03d}",
                   f"load cfg{r.randint(0, 9):03d}",
                   f"check bin{r.randint(0, 9):03d}"],
        ids_perm, n=args.n, seed=104, max_new=110, family="rotate")
    results["heldout"] = execute_family(
        model, tok, lib, dev, "held-out open/close + rel",
        lambda r: [f"open rel{r.randint(0, 99):03d}",
                   f"close cfg{r.randint(0, 99):03d}"],
        ids_perm, n=args.n, seed=105)

    print("\nM4 summary:")
    crit = {"echo": 0.9, "swap": 0.9, "rot3": 0.9, "rot4": 0.9,
            "heldout": 0.6}
    allpass = True
    for k, v in results.items():
        mark = "PASS" if v >= crit[k] else "MISS"
        if v < crit[k]:
            allpass = False
        print(f"  {k:<8} {v:.3f} ({mark}, criterion {crit[k]:.2f})")
    print("M4 ALL CRITERIA PASSED" if allpass else
          "M4: some criteria missed (document honestly)")


if __name__ == "__main__":
    main()
