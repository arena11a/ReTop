"""v5 omega-seam (M1) — can seam re-seeding close the REORDER wall?

The M12 wall (m12_reorder_probe.py / _long / m12b / m12c / m12d): the reorder
task `fetch {a} and deploy {b}` -> `deploy {b} and fetch {a}` stays 0.000 in
every config because (a) fragment seams collapse the gate and (b) nothing can
START a copy run mid-answer. omega-seam adds exactly that mechanism:

  - within a run: deterministic positional echo of the anchor column
    (generalized stem-addr; forced via seam_anchor, gate opens by mass_same)
  - at seams: SeedPointer predicts the next run's start column + run length
    (ptr CE + len CE); decode consumes them greedily
  - gold is assembled FROM prompt token variants so every answer row has an
    exact identity twin (ByteLevel space-prefix safe)

Milestones (docs/v5_omega_roadmap.md):
  M0  baseline reproduce   seam-OFF control on the same data -> expect ~0.000
  M1b train fit            seen reorder high at <= --steps steps (CPU)
  M1c HYPOTHESIS TEST      unseen reorder >= 0.60 (was 0.000 everywhere)
  M1d robustness           seeds sweep + echo-chain regression stays solved

Run:
  python experiments/v5/omega_seam.py                     # full pipeline
  python experiments/v5/omega_seam.py --steps 600         # shorter horizon
  python experiments/v5/omega_seam.py --seeds 42 43 44    # robustness sweep
"""
import argparse
import os
import random
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn

from tokenizers import Tokenizer

from hmn import HMN3
from hmn.recipe import (ASSIST, USER, CHAIN_SLOTS_A, CHAIN_SLOTS_A_U,
                        CHAIN_SLOTS_B, CHAIN_SLOTS_B_U, decode_v33,
                        eval_reorders, eval_slot_chains, loss_v33,
                        make_perm_batch, make_reorder_batch,
                        make_reorder_ids, make_slot_chain_batch,
                        resolve_device, seam_losses, seed_guardrail)

TOKENIZER = os.path.join(ROOT, "retop_tokenizer.json")
A_UNSEEN = [f"pkg{i:03d}" for i in range(60, 80)]     # disjoint from A_TRAIN
B_UNSEEN = [f"lib{i:03d}" for i in range(0, 20)]      # disjoint from B_TRAIN

# v5 M3b curriculum pools (HOLD OUT: verbs open/close, family `rel` -> probes)
FAM_TRAIN = {"pkg": lambda r: f"pkg{r.randint(0, 59):03d}",
             "lib": lambda r: f"lib{r.randint(40, 79):03d}",
             "bin": lambda r: f"bin{r.randint(0, 99):03d}",
             "cfg": lambda r: f"cfg{r.randint(0, 99):03d}"}
PARTS_2 = [lambda r, v=("fetch", "deploy"): (
    f"{v[0]} {FAM_TRAIN['pkg'](r)}", f"{v[1]} {FAM_TRAIN['lib'](r)}")]
PARTS_3 = [
    lambda r: (f"load {FAM_TRAIN['bin'](r)}",
               f"stop {FAM_TRAIN['cfg'](r)}",
               f"check {FAM_TRAIN['pkg'](r)}"),
    lambda r: (f"fetch {FAM_TRAIN['lib'](r)}",
               f"unload {FAM_TRAIN['cfg'](r)}",
               f"clean {FAM_TRAIN['bin'](r)}"),
]


def _pick(pool, rng):
    return pool[rng.randrange(len(pool))](rng)


def train_curriculum(model, tok, args, dev):
    """v5 M3b: anti-lexicon curriculum — echo chain + 2-seg swaps + 3-seg
    rotations over MIXED verb/family pools, so SeedPointer learns run
    STRUCTURE instead of binding to one template (the v4-M3 lesson)."""
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        kind = step % 3
        if kind == 0:
            X, Y, Yc, G = make_slot_chain_batch(tok, args.bs, step,
                                                stem_row0=True, device=dev)
            out = model(X)
            _, lb, lg, lc = loss_v33(out, Y, Yc, G, lossf=lossf)
            loss = lb + lg + lc
            tag = "ec"
        else:
            prng = random.Random(step * 31 + kind)
            if kind == 1:
                plist = list(_pick(PARTS_2, prng))
            else:
                plist = list(_pick(PARTS_3, prng))
            X, Y, Yc, G, A, S, R = make_perm_batch(
                tok, lambda _r, pl=plist: pl, args.bs, step, device=dev)
            out = model(X, seam_anchor=A)
            _, lb, lg, lc = loss_v33(out, Y, Yc, G, lossf=lossf)
            lp, ll = seam_losses(out, S, R, A)
            loss = lb + lg + lc + args.w_ptr * lp + args.w_len * ll
            tag = f"p{len(plist)}"
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            ch_acc, ch_g, _ = eval_slot_chains(
                model, tok, A_UNSEEN, B_UNSEEN, seed=9, boundary_eos=True,
                cycle_break=True, pos_eos=True, device=dev)
            ro_s, _g1 = eval_reorders(model, tok, CHAIN_SLOTS_A,
                                      CHAIN_SLOTS_B, seed=7, device=dev)
            ro_u, _g2 = eval_reorders(model, tok, A_UNSEEN, B_UNSEEN,
                                      seed=9, device=dev)
            # SeedPointer convergence readout on the 3-seg shape
            model.eval()
            with torch.no_grad():
                Xp, _Yp, _Ycp, _Gp, Ap, Sp, Rp = make_perm_batch(
                    tok, lambda r: list(_pick(PARTS_3, r)), 16, step + 7,
                    device=dev)
                op = model(Xp, seam_anchor=Ap)
                bi, ti = Sp.nonzero(as_tuple=True)
                ph = (op["ptr_logits"][bi, ti].argmax(-1) == Ap[bi, ti]
                      ).float().mean().item()
                lh = (op["len_logits"][bi, ti].argmax(-1) == Rp[bi, ti]
                      ).float().mean().item()
            model.train()
            print(f"step {step:4d} [{tag}] loss={loss.item():.3f} "
                  f"chain_u={ch_acc:.3f} reorder s/u={ro_s:.3f}/{ro_u:.3f} "
                  f"ptr3={ph:.2f} len3={lh:.2f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)


def build_tok():
    return Tokenizer.from_file(TOKENIZER)


def build_model(tok, args, seam_addr=None):
    return HMN3(tok.get_vocab_size(), dim=args.dim, state_dim=8,
                n_layers=args.layers, use_moe=False, gate_bias=-1.0,
                asi_id=tok.token_to_id(ASSIST),
                user_id=tok.token_to_id(USER),
                stem_addr=(args.task in ("joint", "joint3")),
                seam_addr=args.seam_addr if seam_addr is None else seam_addr)


def eval_reorder_plain(model, tok, a_slots, b_slots, seed=0, max_new=32,
                       device=None):
    """Seam-OFF reorder accuracy: free blend decode (the m12 protocol)."""
    model.eval()
    rng = random.Random(seed)
    ok = tot = 0
    for _ in a_slots:
        a = rng.choice(a_slots)
        b = rng.choice(b_slots)
        ids, asi_pos, _ = make_reorder_ids(tok, a, b)
        prompt = ids[:asi_pos + 1]
        gold_text = tok.decode(ids[asi_pos + 1:-1]).strip()
        txt, _, _ = decode_v33(model, tok, prompt, max_new=max_new,
                               mode="blend", boundary_eos=True, device=device)
        tot += 1
        ok += int(txt.strip() == gold_text)
    model.train()
    return ok / tot


def train_seam(model, tok, args, dev):
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    t0 = time.time()
    hist = []
    for step in range(1, args.steps + 1):
        X, Y, Yc, G, A, S, R = make_reorder_batch(
            tok, CHAIN_SLOTS_A, CHAIN_SLOTS_B, args.bs, step, device=dev)
        opt.zero_grad()
        out = model(X, seam_anchor=A)
        _, lb, lg, lc = loss_v33(out, Y, Yc, G, lossf=lossf)
        lp, ll = seam_losses(out, S, R, A)
        loss = lb + lg + lc + args.w_ptr * lp + args.w_len * ll
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            s_acc, s_g = eval_reorders(model, tok, CHAIN_SLOTS_A,
                                       CHAIN_SLOTS_B, seed=7, device=dev)
            u_acc, u_g = eval_reorders(model, tok, A_UNSEEN, B_UNSEEN,
                                       seed=9, device=dev)
            hist.append((step, s_acc, u_acc))
            print(f"step {step:4d} loss={loss.item():.3f} "
                  f"(b{lb.item():.2f} c{lc.item():.2f} "
                  f"p{lp.item():.2f} l{ll.item():.2f}) "
                  f"seen={s_acc:.3f}(g{s_g:.2f}) unseen={u_acc:.3f}(g{u_g:.2f}) "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    return hist


def train_joint(model, tok, args, dev):
    """v5 M2: ONE model, TWO tasks — echo chain (stem-addr path, no anchors)
    and reorder (seam-anchor path). Alternates batches per step; each task
    keeps its own decode machinery (boundary_eos+pos_eos vs run machine)."""
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        if step % 2 == 0:
            X, Y, Yc, G, A, S, R = make_reorder_batch(
                tok, CHAIN_SLOTS_A, CHAIN_SLOTS_B, args.bs, step, device=dev)
            out = model(X, seam_anchor=A)
            _, lb, lg, lc = loss_v33(out, Y, Yc, G, lossf=lossf)
            lp, ll = seam_losses(out, S, R, A)
            loss = lb + lg + lc + args.w_ptr * lp + args.w_len * ll
            tag = "ro"
        else:
            X, Y, Yc, G = make_slot_chain_batch(tok, args.bs, step,
                                                stem_row0=True, device=dev)
            out = model(X)
            _, lb, lg, lc = loss_v33(out, Y, Yc, G, lossf=lossf)
            loss = lb + lg + lc
            tag = "ec"
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            ch_acc, ch_g, _ = eval_slot_chains(
                model, tok, A_UNSEEN, B_UNSEEN, seed=9, boundary_eos=True,
                cycle_break=True, pos_eos=True, device=dev)
            ro_s, _g1 = eval_reorders(model, tok, CHAIN_SLOTS_A,
                                      CHAIN_SLOTS_B, seed=7, device=dev)
            ro_u, _g2 = eval_reorders(model, tok, A_UNSEEN, B_UNSEEN,
                                      seed=9, device=dev)
            print(f"step {step:4d} [{tag}] loss={loss.item():.3f} "
                  f"chain_unseen={ch_acc:.3f}(g{ch_g:.2f}) "
                  f"reorder seen={ro_s:.3f} unseen={ro_u:.3f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)


def run_m0(tok, args, dev):
    """Seam-OFF control: same data/slots, blend-CE-only, free decode."""
    torch.manual_seed(args.seeds[0])
    m = build_model(tok, args, seam_addr=False).to(dev)
    print(f"== M0: seam OFF baseline, params={sum(p.numel() for p in m.parameters()):,} ==")
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr)
    steps = min(240, args.steps)
    t0 = time.time()
    for step in range(1, steps + 1):
        X, Y, *_rest = make_reorder_batch(
            tok, CHAIN_SLOTS_A, CHAIN_SLOTS_B, args.bs, step, device=dev)
        opt.zero_grad()
        out = m(X)
        vocab = out["logits"].shape[-1]
        loss = lossf(out["logits"].reshape(-1, vocab), Y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        opt.step()
        if step % 120 == 0 or step == steps:
            s_acc = eval_reorder_plain(m, tok, CHAIN_SLOTS_A, CHAIN_SLOTS_B,
                                       seed=7, device=dev)
            u_acc = eval_reorder_plain(m, tok, A_UNSEEN, B_UNSEEN,
                                       seed=9, device=dev)
            print(f"step {step:4d} loss={loss.item():.3f} seen={s_acc:.3f} "
                  f"unseen={u_acc:.3f} [{time.time()-t0:.0f}s]", flush=True)


def run_seed(seed, tok, args, dev):
    seed_guardrail(seed)
    torch.manual_seed(seed)
    model = build_model(tok, args).to(dev)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=dev))
        print(f"resumed from {args.init}", flush=True)
    print(f"== M1 seam ON seed={seed}, "
          f"params={sum(p.numel() for p in model.parameters()):,} ==")
    if args.task == "joint3":
        train_curriculum(model, tok, args, dev)
        ch_acc, ch_g, _ = eval_slot_chains(model, tok, A_UNSEEN, B_UNSEEN,
                                           seed=9, boundary_eos=True,
                                           cycle_break=True, pos_eos=True,
                                           device=dev)
        s_acc, s_g = eval_reorders(model, tok, CHAIN_SLOTS_A, CHAIN_SLOTS_B,
                                   seed=7, device=dev)
        u_acc, u_g = eval_reorders(model, tok, A_UNSEEN, B_UNSEEN, seed=9,
                                   device=dev)
        print(f"FINAL(curriculum) seed={seed}: chain_unseen={ch_acc:.3f} "
              f"reorder seen={s_acc:.3f} unseen={u_acc:.3f}", flush=True)
    elif args.task == "joint":
        train_joint(model, tok, args, dev)
        ch_acc, ch_g, _ = eval_slot_chains(model, tok, A_UNSEEN, B_UNSEEN,
                                           seed=9, boundary_eos=True,
                                           cycle_break=True, pos_eos=True,
                                           device=dev)
        s_acc, s_g = eval_reorders(model, tok, CHAIN_SLOTS_A, CHAIN_SLOTS_B,
                                   seed=7, device=dev)
        u_acc, u_g = eval_reorders(model, tok, A_UNSEEN, B_UNSEEN, seed=9,
                                   device=dev)
        print(f"FINAL(joint) seed={seed}: chain_unseen={ch_acc:.3f} "
              f"reorder seen={s_acc:.3f} unseen={u_acc:.3f}", flush=True)
    else:
        train_seam(model, tok, args, dev)
        s_acc, s_g = eval_reorders(model, tok, CHAIN_SLOTS_A, CHAIN_SLOTS_B,
                                   seed=7, device=dev)
        u_acc, u_g = eval_reorders(model, tok, A_UNSEEN, B_UNSEEN, seed=9,
                                   device=dev)
        print(f"FINAL seed={seed}: reorder seen={s_acc:.3f} unseen={u_acc:.3f}",
              flush=True)
    if args.save and seed == args.seeds[0]:
        torch.save(model.state_dict(), args.save)
        print(f"saved -> {args.save}", flush=True)
    return s_acc, u_acc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--w-ptr", type=float, default=1.0)
    ap.add_argument("--w-len", type=float, default=0.3)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--task", default="reorder",
                    choices=["reorder", "joint", "joint3"],
                    help="reorder = M1 (seam only); joint = v5 M2 (echo+swap); "
                         "joint3 = v5 M3b anti-lexicon curriculum (echo + 2-seg "
                         "swaps + 3-seg rotations, mixed verb/family pools)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--no-seam-addr", dest="seam_addr", action="store_false")
    ap.add_argument("--save", default="omega_seam.pt")
    ap.add_argument("--init", default=None,
                    help="load this checkpoint before training (resume)")
    ap.add_argument("--skip-m0", action="store_true",
                    help="skip the seam-OFF baseline control")
    ap.add_argument("--skip-regression", action="store_true",
                    help="skip the echo-chain regression check")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    dev = resolve_device(args.device)
    print(f"device: {dev}", flush=True)
    tok = build_tok()

    if not args.skip_m0:
        run_m0(tok, args, dev)

    results = {}
    for sd in args.seeds:
        results[sd] = run_seed(sd, tok, args, dev)

    if (not args.skip_regression and len(args.seeds) == 1
            and args.task == "reorder"):
        # NOTE: an M1 model is trained ONLY on reorder, so it is NOT expected
        # to solve the echo chain zero-shot. The no-regression guarantee for
        # existing tasks lives in the flag-OFF path: test_hmn.py +
        # slot_v33_seed42.py stay green (bit-identical). joint/joint3 runs
        # carry their own chain eval in the training loop instead.
        model = build_model(tok, args).to(dev)
        model.load_state_dict(torch.load(args.save, map_location=dev))
        ch_acc, ch_gate, _ = eval_slot_chains(
            model, tok, CHAIN_SLOTS_A_U[:10], CHAIN_SLOTS_B_U[:10], seed=9,
            boundary_eos=True, cycle_break=True, pos_eos=True, device=dev)
        print(f"  chain(unseen subset) on the reorder-only ckpt = {ch_acc:.3f} "
              f"(gate {ch_gate:.3f}) — expected ~0: M1 is task-specialized",
              flush=True)

    accs = [u for _, u in results.values()]
    print(f"DONE. unseen reorder over seeds: "
          f"{[f'{u:.3f}' for u in accs]} | mean {sum(accs)/len(accs):.3f} "
          f"| M1c pass criterion >= 0.600")


if __name__ == "__main__":
    main()
