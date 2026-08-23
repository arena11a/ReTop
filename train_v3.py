"""v3 training harness — slot-copy task (pointer-register v3.1).

Goal (docs/hmn_v3_design.md §3): hard slot-copy of an UNSEEN token from the
prompt.
  train: 60 seen packages   val: 40 UNSEEN packages, zero overlap.
  v2 ceiling was 0/40 exact on val (softmax head cannot re-emit a prompt token).
  v3 must >= 85% via the Identity Register copy lane.

Architecture (v3.1 — "pointer register with coordinated gate"):
  IR returns position attention a (pointer) + vocab mass (copy dist) + ctx.
  Three supervisions shape three responsibilities:
    blend CE   : final (1-g)*gen + g*copy is correct
    copy CE    : copy_dist peaks at the CORRECT TOKEN (aux_copy)
    gen CE     : gen owns ONLY the generate rows (first answer token 'pip',
                 EOS, absent targets) — copy rows are masked OUT
    gate BCE   : gate is ON iff the target token exists in the prompt
  Eval reports three readouts: blend, hard-copy (gate>0.5 -> copy argmax),
  and copy-only (channel quality), on seen AND unseen slots.

The loss math + decode rule + eval are centralized in hmn/recipe.py (v3.3) so
every entry point shares one implementation.

Usage:
  python train_v3.py --steps 3000 --dim 96 --layers 3
  python train_v3.py --arch baseline ...      # blend CE only (no aux)
  python train_v3.py --arch noreg ...          # no copy channel (softmax only)
  python train_v3.py --templates "pip install {slot}|import {slot}" ...
      # v4 M3: multi-template curriculum (per-record random template). When
      # set, eval reports each template + templates NEVER trained (measured
      # generalization: does the copy gate stop being lexicon-bound?)
"""
import argparse
import os
import time

import torch
import torch.nn as nn

from tokenizers import Tokenizer

from hmn import HMN3, HMN3_NoReg
from hmn.recipe import (CHAIN_SLOTS_A, CHAIN_SLOTS_A_U, CHAIN_SLOTS_B,
                        CHAIN_SLOTS_B_U, DEFAULT_TEMPLATE, eval_slot_chains,
                        eval_slots, loss_v33, make_slot_batch,
                        make_slot_chain_batch, resolve_device, seed_guardrail)

ROOT = os.path.dirname(os.path.abspath(__file__))
EOS = "</s>"
ASI = "<|assistant|>"
USER = "<|user|>"

PKGS_SEEN = [f"pkg{i:03d}" for i in range(60)]
PKGS_UNSEEN = [f"pkg{i:03d}" for i in range(60, 100)]


def build_tok(path=os.path.join(ROOT, "retop_tokenizer.json")):
    return Tokenizer.from_file(path)


def asi_id(tok):
    return tok.token_to_id(ASI)


def user_id(tok):
    return tok.token_to_id(USER)


def build_model(arch, args, tok):
    vocab = tok.get_vocab_size()
    if arch == "noreg":
        return HMN3_NoReg(vocab, dim=args.dim, state_dim=8, n_layers=args.layers)
    # v6 M1-B: the v31 recipe consumes the IRStats API; other archs keep the
    # legacy dense blended logits for their plain-CE branch.
    return HMN3(vocab, dim=args.dim, state_dim=8, n_layers=args.layers,
                use_moe=args.moe, gate_bias=args.gate_bias, asi_id=asi_id(tok), aux_copy=(arch == "v31"),
                sparse_marginal=args.sparse_marginal, gate_mode=args.gate_mode,
                use_think=args.use_think, k_max=args.k_max, user_id=user_id(tok),
                stem_addr=args.stem_addr,
                exact_blend=args.exact_blend or arch != "v31")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="v31", choices=["v31", "baseline", "noreg"],
                    help="v31 = copy+pointer+gate losses (default), baseline = copy only, noreg = no register")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gate-bias", type=float, default=-1.0)
    ap.add_argument("--w-copy", type=float, default=1.0, help="weight on copy+gen aux CE")
    ap.add_argument("--w-ptr", type=float, default=1.0, help="(vestigial) pointer CE weight")
    ap.add_argument("--w-gate", type=float, default=0.0,
                    help="v4 M2: BCE weight supervising a LEARNED gate against the copy mask "
                         "(default 0 = deterministic gate, no extra loss)")
    ap.add_argument("--gate-mode", default="deterministic", choices=["deterministic", "relative"],
                    help="v4 M2: deterministic same-token-id gate (default, v3.3-final) or the "
                         "learned RelativeGate ([h, gate_mass, gen_margin, behind, n_legal])")
    ap.add_argument("--use-think", action="store_true",
                    help="v4 M4: run LatentThinkingBuffer before the head (k_max re-runs of the "
                         "WR block stack refining h in latent space)")
    ap.add_argument("--k-max", type=int, default=4,
                    help="v4 M4: max latent re-runs for use_think")
    ap.add_argument("--task", default="slot", choices=["slot", "chain"],
                    help="v4 M4: slot = single-template copy (default); chain = two-slot "
                         "multi-step task (fetch {a} and deploy {b})")
    ap.add_argument("--stem-addr", action="store_true",
                    help="v4 M2-dev: stem-addressing — anchor the ASI boundary row's copy "
                         "attention onto the USER column so the template's first token is "
                         "copyable (row-0 no longer forced through gen). Default off = v3.3")
    ap.add_argument("--exact-blend", action="store_true",
                    help="v6 M1-B: force the legacy dense blended-logits oracle path "
                         "(default: IRStats index path for the v31 recipe)")
    ap.add_argument("--moe", action="store_true")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="hmn_v31.pt")
    ap.add_argument("--templates", default=None,
                    help="v4 M3: pipe-separated training template list, e.g. "
                         "'pip install {slot}|import {slot}|run {slot}|apt install {slot}' "
                         "(default: single DEFAULT_TEMPLATE)")
    ap.add_argument("--sparse-marginal", action="store_true",
                    help="v4 M1: use the sparse copy-marginal path (no (B,T,V) copy tensor)")
    ap.add_argument("--device", default=None,
                    help="compute device: auto (default; cuda->mps->cpu), a device str "
                         "(cuda:0, mps, cpu), or the RETOP_DEVICE env var. A real GPU "
                         "is used automatically when present.")
    args = ap.parse_args()

    dev = resolve_device(args.device)
    print(f"device: {dev}", flush=True)

    tpls = [DEFAULT_TEMPLATE] if not args.templates else args.templates.split("|")
    ALL_TEMPLATES = [  # the full pool; probes are chosen from here MINUS trained
        "pip install {slot}", "import {slot}", "run {slot}", "apt install {slot}",
        "remove {slot}", "delete {slot}", "get {slot}", "cache {slot}",
        "fetch {slot}", "pip uninstall {slot}", "mount {slot}", "uninstall {slot}",
        "clean {slot}", "check {slot}", "search {slot}",
    ]
    PROBE_TEMPLATES = [t for t in ALL_TEMPLATES if t not in tpls][:4]

    seed_guardrail(args.seed)
    tok = build_tok()
    model = build_model(args.arch, args, tok).to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters()):,} | arch={args.arch} "
          f"sparse={args.sparse_marginal}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    best_unseen = 0.0
    t0 = time.time()

    for step in range(1, args.steps + 1):
        if args.task == "chain":
            X, Y, Yc, G = make_slot_chain_batch(tok, args.bs, step,
                                                stem_row0=args.stem_addr,
                                                device=dev)
        else:
            X, Y, Yc, G = make_slot_batch(tok, PKGS_SEEN, args.bs, step,
                                          templates=tpls, stem_row0=args.stem_addr,
                                          device=dev)
        opt.zero_grad()
        out = model(X)
        if args.arch == "v31":
            loss, l_blend, l_gen, l_copy = loss_v33(out, Y, Yc, G, lossf=lossf,
                                                    w_copy=args.w_copy,
                                                    w_gate=args.w_gate)
        else:
            vocab = out["logits"].shape[-1]
            loss = lossf(out["logits"].reshape(-1, vocab), Y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if step % args.eval_every == 0 or step == args.steps:
            final = (step == args.steps)
            if args.task == "chain":
                # v4 M4: two-slot chain on UNSEEN slot pairs — the multi-step
                # eval. seen uses PKGS_SEEN pairings, unseen uses PKGS_UNSEEN.
                s_b, s_g, _ = eval_slot_chains(model, tok, CHAIN_SLOTS_A, CHAIN_SLOTS_B,
                                       seed=7, boundary_eos=True, cycle_break=True,
                                       device=dev)
                u_b, u_g, u_ng = eval_slot_chains(model, tok, CHAIN_SLOTS_A_U,
                                                  CHAIN_SLOTS_B_U, seed=9,
                                                  boundary_eos=True, cycle_break=True,
                                                  device=dev)
            else:
                s_b, s_g, _ = eval_slots(model, tok, PKGS_SEEN, mode="blend",
                                         seed=7, boundary_eos=True, template=tpls[0],
                                         device=dev)
                u_b, u_g, u_ng = eval_slots(model, tok, PKGS_UNSEEN, mode="blend",
                                            seed=9, boundary_eos=True, template=tpls[0],
                                            device=dev)
            el = time.time() - t0
            line = (f"step {step:5d} loss={loss.item():.3f} "
                    f"seen={s_b:.3f}(g{s_g:.2f}) unseen_blend={u_b:.3f}(g{u_g:.2f},gen{u_ng:.1f})")
            if final:
                if args.task == "chain":
                    u_h, _, _ = eval_slot_chains(model, tok, CHAIN_SLOTS_A_U,
                                                 CHAIN_SLOTS_B_U,
                                                 seed=9, mode="hard",
                                                 boundary_eos=True, cycle_break=True,
                                                 device=dev)
                else:
                    u_h, _, _ = eval_slots(model, tok, PKGS_UNSEEN, mode="hard",
                                           seed=9, boundary_eos=True, template=tpls[0],
                                           device=dev)
                    u_c, _, _ = eval_slots(model, tok, PKGS_UNSEEN, mode="copy", seed=9,
                                           template=tpls[0], device=dev)
                    line += f" hard={u_h:.3f} copy={u_c:.3f}"

                # v4 M3: per-template table — all TRAINED templates + NEVER-seen
                # probe templates. This is the whole point: does the copy gate
                # generalize once it has seen several templates at train time?
                if args.task != "chain":
                    print("  v4 M3 per-template (unseen slots, blend+boundary):")
                    for name, tpl in ([(f"trained:{t}", t) for t in tpls]
                                      + [(f"probe:{p}", p) for p in PROBE_TEMPLATES]):
                        a, g, _ = eval_slots(model, tok, PKGS_UNSEEN, mode="blend",
                                             seed=9, boundary_eos=True, template=tpl,
                                             device=dev)
                        print(f"    {name:<16} {a:.3f}  gate={g:.3f}")
                else:
                    line += f" hard={u_h:.3f}"

            print(line + f" [{el:.0f}s]", flush=True)
            if u_b > best_unseen:
                best_unseen = u_b
                torch.save(model.state_dict(), args.save)
                print(f"  * new best unseen {u_b:.3f} -> saved {args.save}", flush=True)

    print(f"DONE. best unseen blend: {best_unseen:.3f}")


if __name__ == "__main__":
    main()
