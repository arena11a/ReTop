"""M12 probe (2026-08-14) — can HMN copy+compose a REORDER transform?

Echo tasks (user == gold) are solved by stem-addr + pos_eos, but that is a
designed boundary: the register copies what the prompt SAYS, in order. This
probe asks whether the gen head can handle a NON-echo reorder:

    user = "fetch {a} and deploy {b}"
    gold = "deploy {b} and fetch {a}"     <- the two halves SWAPPED

Notice WHY this might gate-open: every gold token ("deploy _ and fetch _")
exists verbatim in the prompt in a different order, so the identity lane has
an exact twin for each answer token. The hard part is only ROW 0: the seed is
ASI and the first gold token ("deploy") is a MID-prompt word with no identity
anchor — so row-0 must be GEN-emitted. After "deploy" the pointer chains
naturally: deploy->lib055->and->fetch->pkg028.

Test matrix (flag-guarded, ~2 min CPU):
  stem-addr OFF  -> identity + gen compose (row-0 gen, rest pointer)
  stem-addr ON   -> row-0 anchored to USER (first user word "fetch") which is
                    WRONG for the swap -> should FAIL, and that is informative:
                    the anchor is echo-specific.

RESULT (2026-08-14, 240 steps x2, bs16, CPU):
  stem-addr OFF : loss 4.29->4.19, unseen_acc 0.000. Deterministic, non-noisy:
                  never learns either seen or unseen swaps.
  stem-addr ON  : loss 16.3 (flat), unseen_acc 0.000. The anchor forces the
                  USER's first token ("fetch") as answer row 0, which is wrong
                  for the swapped gold -> actively harmful.
  => CONCLUSION: reorder/transform is beyond the register+pointer lane even
     when every gold token exists verbatim in the prompt. Row-0 gen is not
     enough; the answer is a COMPOSITION (order changed), and the sparse
     identity attention has no mechanism to flip two attention sinks. This is
     the hard edge of M6's "echo-only" assumption and stays OPEN for M12.

Run: python experiments/v4/m12_reorder_probe.py [--steps 240]
"""
import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
from tokenizers import Tokenizer

from hmn import HMN3
from hmn.recipe import (EOS, decode_v33, eval_slot_chains, make_chat_ids,
                        make_slot_chain_batch, seed_guardrail)

TOKENIZER = os.path.join(ROOT, "retop_tokenizer.json")
A_TRAIN = [f"pkg{i:03d}" for i in range(0, 40)]
B_TRAIN = [f"lib{i:03d}" for i in range(40, 80)]
A_UNSEEN = [f"pkg{i:03d}" for i in range(60, 80)]
B_UNSEEN = [f"lib{i:03d}" for i in range(0, 20)]


def make_reorder_batch(tok, a_slots, b_slots, bs, seed):
    import random
    rng = random.Random(seed)
    eos = tok.token_to_id(EOS)
    asid = tok.token_to_id("<|assistant|>")
    X, Y = [], []
    for _ in range(bs):
        a = rng.choice(a_slots)
        b = rng.choice(b_slots)
        user = f"fetch {a} and deploy {b}"
        gold = f"deploy {b} and fetch {a}"
        ids = make_chat_ids(tok, user, gold)
        y = [-100] * len(ids)
        a_ = ids.index(asid)
        for t in range(a_, len(ids)):
            y[t] = (ids[1:] + [eos])[t]
        X.append(ids); Y.append(y)
    T = max(map(len, X))
    Xb = torch.full((bs, T), eos, dtype=torch.long)
    Yb = torch.full((bs, T), -100, dtype=torch.long)
    for j in range(len(X)):
        Xb[j, :len(X[j])] = torch.tensor(X[j])
        Yb[j, :len(Y[j])] = torch.tensor(Y[j])
    return Xb, Yb


def eval_reorder(model, tok, a_slots, b_slots, seed=0):
    import random
    rng = random.Random(seed)
    ok = tot = 0
    for _ in a_slots:
        a = rng.choice(a_slots)
        b = rng.choice(b_slots)
        base = f"deploy {b} and fetch {a}"
        user = f"fetch {a} and deploy {b}"
        gold = base
        out, _, _ = decode_v33(model, tok, make_chat_ids(tok, user),
                               mode="blend", max_new=24, boundary_eos=True)
        tot += 1
        ok += int(out.strip() == gold)
    return ok / tot


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()
    seed_guardrail(42)
    torch.manual_seed(42)
    tok = Tokenizer.from_file(TOKENIZER)
    vocab = tok.get_vocab_size()

    for stem in (False, True):
        m = HMN3(vocab, dim=96, state_dim=8, n_layers=3,
                 use_moe=False, gate_bias=0.0, asi_id=tok.token_to_id("<|assistant|>"),
                 keys_proj=False, aux_copy=True, sparse_marginal=True,
                 gate_mode="deterministic", use_think=False, k_max=4,
                 user_id=tok.token_to_id("<|user|>"), stem_addr=stem)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        lossf = nn.CrossEntropyLoss(ignore_index=-100)
        print(f"\n--- reorder probe, stem_addr={stem} ---")
        for step in range(1, args.steps + 1):
            X, Y = make_reorder_batch(tok, A_TRAIN, B_TRAIN, args.bs, step)
            opt.zero_grad()
            logits = m(X)["logits"]
            loss = lossf(logits.reshape(-1, vocab), Y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
            if step % 120 == 0 or step == args.steps:
                m.eval()
                acc = eval_reorder(m, tok, A_UNSEEN, B_UNSEEN, seed=step)
                print(f"  step {step:4d} loss={loss.item():.3f} unseen_acc={acc:.3f}")
                m.train()
    print("\nprobe done: see per-config accs above")


if __name__ == "__main__":
    main()