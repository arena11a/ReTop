"""M12c probe (2026-08-14) — does a structural 'swap:' marker make reorder learn?

M12b decomposed the reorder wall into (a) row-0 content-initiation (gen cannot
emit a non-anchored head token; gate won't open a copy for it) and (b)
fragment-seam backward jumps. This probe attacks (a) the cheapest possible way:
give the template a structural MARKER ("swap:") so the model has an explicit
signal that the answer starts with a specific reordered fragment.

  user = "swap: fetch {a} and deploy {b}"      gold = "deploy {b} and fetch {a}"

RESULT (2026-08-14, 400 steps, seed-42, bs16): loss 3.99->3.80, unseen_acc
  0.000 at every checkpoint — statistically identical to M12 WITHOUT the
  marker (plateau ~3.97-4.0, 0.000 at 1200 steps). The 'swap:' marker does
  NOT change anything. CONCLUSION: the row-0 content-initiation wall is in
  the GEN LANE CAPACITY, not the conditioning signal. Even given an explicit
  structural marker, the model cannot emit a non-anchored content token at
  row 0 → the gate/anchor mechanism (copy-lane bias) is what's missing, and
  it is not reachable by adding template diversity to the SAME architecture.
  This closes the 'signal conditioning' branch of M12: addressing requires a
  mechanism change (a seeded pointer the gate can open), not more/louder
  input features.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
from tokenizers import Tokenizer

from hmn import HMN3
from hmn.recipe import decode_v33, make_chat_ids, seed_guardrail
import train_v3 as tv

TOKENIZER = os.path.join(ROOT, "retop_tokenizer.json")
A_TRAIN = [f"pkg{i:03d}" for i in range(0, 40)]
B_TRAIN = [f"lib{i:03d}" for i in range(40, 80)]
A_UNSEEN = [f"pkg{i:03d}" for i in range(60, 80)]
B_UNSEEN = [f"lib{i:03d}" for i in range(0, 20)]


def make_marker_batch(tok, a_slots, b_slots, bs, seed):
    import random
    rng = random.Random(seed)
    eos = tok.token_to_id("</s>")
    asid = tok.token_to_id("<|assistant|>")
    X, Y = [], []
    for _ in range(bs):
        a = rng.choice(a_slots)
        b = rng.choice(b_slots)
        user = f"swap: fetch {a} and deploy {b}"
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


def eval_marker(model, tok, a_slots, b_slots, seed, steps):
    import random
    rng = random.Random(seed)
    ok = tot = 0
    for _ in a_slots:
        a = rng.choice(a_slots)
        b = rng.choice(b_slots)
        user = f"swap: fetch {a} and deploy {b}"
        gold = f"deploy {b} and fetch {a}"
        out, _, _ = decode_v33(model, tok, make_chat_ids(tok, user),
                               mode="blend", max_new=24, boundary_eos=True)
        tot += 1
        ok += int(out.strip() == gold)
    return ok / tot


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    tok = Tokenizer.from_file(TOKENIZER)
    seed_guardrail(42)
    torch.manual_seed(42)
    m = HMN3(tok.get_vocab_size(), dim=96, state_dim=8, n_layers=3,
             use_moe=False, gate_bias=0.0, asi_id=tv.asi_id(tok),
             keys_proj=False, aux_copy=True, sparse_marginal=True,
             gate_mode="deterministic", use_think=False, k_max=4,
             user_id=tok.token_to_id("<|user|>"), stem_addr=False)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    print("swap-with-marker, stem_addr=False, seed-42")
    for step in range(1, steps + 1):
        X, Y = make_marker_batch(tok, A_TRAIN, B_TRAIN, 16, step)
        opt.zero_grad()
        out = m(X)["logits"]
        loss = lossf(out.reshape(-1, out.shape[-1]), Y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        opt.step()
        if step % 100 == 0 or step == steps:
            m.eval()
            acc = eval_marker(m, tok, A_UNSEEN, B_UNSEEN, seed=step, steps=steps)
            print(f"  step {step:4d} loss={loss.item():.3f} unseen_acc={acc:.3f}",
                  flush=True)
            m.train()
    print("done")


if __name__ == "__main__":
    main()