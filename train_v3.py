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
"""
import argparse
import os
import time

import torch
import torch.nn as nn

from tokenizers import Tokenizer

from hmn import HMN3, HMN3_NoReg
from hmn.recipe import DEFAULT_TEMPLATE, eval_slots, loss_v33, make_slot_batch, seed_guardrail

ROOT = os.path.dirname(os.path.abspath(__file__))
EOS = "</s>"
ASI = "<|assistant|>"

PKGS_SEEN = [f"pkg{i:03d}" for i in range(60)]
PKGS_UNSEEN = [f"pkg{i:03d}" for i in range(60, 100)]


def build_tok(path=os.path.join(ROOT, "retop_tokenizer.json")):
    return Tokenizer.from_file(path)


def asi_id(tok):
    return tok.token_to_id(ASI)


def build_model(arch, args, tok):
    vocab = tok.get_vocab_size()
    if arch == "noreg":
        return HMN3_NoReg(vocab, dim=args.dim, state_dim=8, n_layers=args.layers)
    return HMN3(vocab, dim=args.dim, state_dim=8, n_layers=args.layers,
                use_moe=args.moe, gate_bias=args.gate_bias, asi_id=asi_id(tok),
                keys_proj=args.keys_proj, aux_copy=(arch == "v31"))


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
    ap.add_argument("--w-gate", type=float, default=0.5, help="(vestigial) gate BCE weight")
    ap.add_argument("--moe", action="store_true")
    ap.add_argument("--keys-proj", action="store_true")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="hmn_v31.pt")
    args = ap.parse_args()

    seed_guardrail(args.seed)
    tok = build_tok()
    model = build_model(args.arch, args, tok)
    print(f"params: {sum(p.numel() for p in model.parameters()):,} | arch={args.arch}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    best_unseen = 0.0
    t0 = time.time()

    for step in range(1, args.steps + 1):
        X, Y, Yc, G = make_slot_batch(tok, PKGS_SEEN, args.bs, step,
                                      template=DEFAULT_TEMPLATE)
        opt.zero_grad()
        out = model(X)
        if args.arch == "v31":
            loss, l_blend, l_gen, l_copy = loss_v33(out, Y, Yc, G, lossf=lossf,
                                                    w_copy=args.w_copy)
        else:
            vocab = out["logits"].shape[-1]
            loss = lossf(out["logits"].reshape(-1, vocab), Y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if step % args.eval_every == 0 or step == args.steps:
            final = (step == args.steps)
            s_b, s_g, _ = eval_slots(model, tok, PKGS_SEEN, mode="blend",
                                     seed=7, boundary_eos=True)
            u_b, u_g, u_ng = eval_slots(model, tok, PKGS_UNSEEN, mode="blend",
                                        seed=9, boundary_eos=True)
            el = time.time() - t0
            line = (f"step {step:5d} loss={loss.item():.3f} "
                    f"seen={s_b:.3f}(g{s_g:.2f}) unseen_blend={u_b:.3f}(g{u_g:.2f},gen{u_ng:.1f})")
            if final:
                u_h, _, _ = eval_slots(model, tok, PKGS_UNSEEN, mode="hard",
                                       seed=9, boundary_eos=True)
                u_c, _, _ = eval_slots(model, tok, PKGS_UNSEEN, mode="copy", seed=9)
                line += f" hard={u_h:.3f} copy={u_c:.3f}"
            print(line + f" [{el:.0f}s]", flush=True)
            if u_b > best_unseen:
                best_unseen = u_b
                torch.save(model.state_dict(), args.save)
                print(f"  * new best unseen {u_b:.3f} -> saved {args.save}", flush=True)

    print(f"DONE. best unseen blend: {best_unseen:.3f}")


if __name__ == "__main__":
    main()
