"""v4 chain verification (2026-08-14) — stem-addr + pos_eos, reproducible.

Locks the chain-task claims that were previously documented-but-unguarded:
the 2-slot chain benchmark reached 1.000 only after pos_eos (M6), and it
generalizes in LENGTH to unseen 3-slot templates with zero retraining.

  python experiments/verified/chain_v4.py                # ~2 min CPU
  python experiments/verified/chain_v4.py 120            # CI short-horizon

ASSERTED (fails loudly, feeds v4_guardrail):
  - 2-slot chain, unseen pools, seeds {9,11,13,17,21}: 1.000 hard AND blend
    (boundary_eos + pos_eos)
  - 3-slot chain (unseen template count, zero extra training): 1.000 hard
  - the 3-slot WITHOUT pos_eos must NOT reach 1.000 (documents WHY the fix is
    needed: post-answer termination recursion) — printed, not asserted

PRINTED (informational): per-seed table + params + final loss.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
from tokenizers import Tokenizer

from hmn import HMN3
from hmn.recipe import (decode_v33, eval_slot_chains, make_chat_ids,
                        make_slot_chain_batch, resolve_device, seed_guardrail)

TOKENIZER = os.path.join(ROOT, "retop_tokenizer.json")
SEEN_A = [f"pkg{i:03d}" for i in range(0, 30)]
SEEN_B = [f"lib{i:03d}" for i in range(30, 60)]
UNSEEN_A = [f"pkg{i:03d}" for i in range(60, 80)]
UNSEEN_B = [f"lib{i:03d}" for i in range(0, 20)]
CHAIN_SEEDS = (9, 11, 13, 17, 21)
ID_POOL = ([f"pkg{i:03d}" for i in range(40, 60)] +
           [f"lib{i:03d}" for i in range(0, 40)] +
           [f"src{i:03d}" for i in range(20, 40)])
VERBS = ["fetch", "deploy", "stop"]


def build_model(tok, steps, device=None):
    torch.manual_seed(42)
    seed_guardrail(42)
    dev = resolve_device(device)
    m = HMN3(tok.get_vocab_size(), dim=96, state_dim=8, n_layers=3,
             use_moe=False, gate_bias=0.0, asi_id=tok.token_to_id("<|assistant|>"), aux_copy=True, sparse_marginal=True,
             gate_mode="deterministic", use_think=False, k_max=4,
             user_id=tok.token_to_id("<|user|>"), stem_addr=True).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    for step in range(1, steps + 1):
        X, Y, _, _ = make_slot_chain_batch(tok, 16, step, stem_row0=True,
                                           device=dev)
        opt.zero_grad()
        # v6 M1-B: legacy plain blend-CE training runs through the dense
        # ORACLE to stay bit-identical; decode exercises the stats API.
        logits = m(X, exact_blend=True)["logits"]
        loss = lossf(logits.reshape(-1, logits.shape[-1]), Y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        opt.step()
    return m, dev


def eval_three_slot(model, tok, trials=20, pos_eos=True, mode="hard", device=None):
    import random
    rng = random.Random(7)
    ok = tot = 0
    for _ in range(trials):
        u, w, v = rng.sample(ID_POOL, 3)
        user = f"{VERBS[0]} {u} and {VERBS[1]} {w} and {VERBS[2]} {v}"
        out, _, _ = decode_v33(model, tok, make_chat_ids(tok, user),
                               mode=mode, max_new=80, boundary_eos=True,
                               pos_eos=pos_eos, device=device)
        tot += 1
        ok += int(out.strip() == user)
    return ok / tot


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    device = None
    for i, a in enumerate(sys.argv[1:]):
        if a == "--device" and i + 1 < len(sys.argv[1:]):
            device = sys.argv[i + 2]
    tok = Tokenizer.from_file(TOKENIZER)
    m, dev = build_model(tok, steps, device=device)
    p = sum(pp.numel() for pp in m.parameters())
    print(f"trained {steps} steps | {p:,} params | device {dev}")
    m.eval()

    ok = True
    print("2-slot chain, unseen pools, pos_eos=True:")
    for seed in CHAIN_SEEDS:
        row = []
        for mode in ("hard", "blend"):
            a, _, _ = eval_slot_chains(m, tok, UNSEEN_A, UNSEEN_B, seed=seed,
                                       mode=mode, max_new=40, boundary_eos=True,
                                       pos_eos=True, device=dev)
            ok &= a >= 1.0
            row.append(f"{mode}={a:.3f}")
        print(f"  seed {seed:<3d} " + "  ".join(row))

    a3 = eval_three_slot(m, tok, trials=20, pos_eos=True, mode="hard", device=dev)
    ok &= a3 >= 1.0
    print(f"3-slot chain (unseen length, zero retrain) hard+pos_eos: {a3:.3f}")

    a3_raw = eval_three_slot(m, tok, trials=20, pos_eos=False, mode="hard",
                             device=dev)
    print(f"3-slot chain WITHOUT pos_eos (informational, must be < 1.0): "
          f"{a3_raw:.3f}")

    print("\n" + ("ALL V4 CHAIN GUARDS PASSED" if ok else "V4 CHAIN GUARD FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
