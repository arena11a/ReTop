"""v4 slot verification (2026-08-13) — stem-addr + pos_eos, reproducible.

The v3.3 guardrail (slot_v33_seed42.py) locks the shipped hmn_v33.pt at
40/40 but its generalization matrix has known 0.00 rows (unseen templates,
repeated digits) that v4 closes. THIS script reproduces those numbers from a
fresh 600-step training instead of relying on /tmp artifacts:

  python experiments/verified/slot_v4.py                # ~2 min CPU

ASSERTED (fails loudly, feeds v4_guardrail):
  - 10 trained templates, unseen slots (pkg060..099): 1.000 (blend + hard,
    boundary_eos + pos_eos)
  - 4 never-trained probe templates (mount/uninstall/clean/check): 1.000
  - v3.3 matrix rows that were 0.00 / 0.33: 4-digit, 5-digit, alnum, repeated
    all 1.000

PRINTED (informational): per-template table + params + final loss.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import torch
from tokenizers import Tokenizer

from hmn import HMN3
from hmn.recipe import (eval_slots, make_slot_batch, resolve_device,
                        seed_guardrail)

TOKENIZER = os.path.join(ROOT, "retop_tokenizer.json")
SEEN = [f"pkg{i:03d}" for i in range(60)]
UNSEEN = [f"pkg{i:03d}" for i in range(60, 100)]
TPL_STR = ("pip install {slot}|import {slot}|run {slot}|apt install {slot}|"
           "remove {slot}|delete {slot}|get {slot}|cache {slot}|"
           "fetch {slot}|pip uninstall {slot}")
PROBES = ["mount {slot}", "uninstall {slot}", "clean {slot}", "check {slot}"]
MATRIX = {
    "4-digit": [f"pkg{i:04d}" for i in range(9000, 9030)],
    "5-digit": [f"pkg{i:05d}" for i in range(90000, 90030)],
    "alnum": [f"pkgA{i}" for i in range(10)] + [f"xy{i}" for i in range(15)],
    "repeated": [f"pkg{i:03d}" for i in (111, 222, 333, 444, 555, 666, 777, 888, 999)],
}


def build_model(tok, steps, device=None):
    torch.manual_seed(42)
    seed_guardrail(42)
    dev = resolve_device(device)
    tpls = [t.strip() for t in TPL_STR.split("|")]
    m = HMN3(tok.get_vocab_size(), dim=96, state_dim=8, n_layers=3,
             use_moe=False, gate_bias=0.0, asi_id=tok.token_to_id("<|assistant|>"), aux_copy=True, sparse_marginal=True,
             gate_mode="deterministic", use_think=False, k_max=4,
             user_id=tok.token_to_id("<|user|>"), stem_addr=True).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for step in range(1, steps + 1):
        X, Y, Yc, G = make_slot_batch(tok, SEEN, 16, step, templates=tpls,
                                      stem_row0=True, device=dev)
        opt.zero_grad()
        # v6 M1-B: this legacy guardrail trains with plain blend-CE (pre-v3.3
        # recipe); run it through the dense ORACLE to keep the historical
        # training math bit-identical — decode below exercises the stats API.
        logits = m(X, exact_blend=True)["logits"]
        loss = torch.nn.CrossEntropyLoss(ignore_index=-100)(
            logits.reshape(-1, logits.shape[-1]), Y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        opt.step()
    return m, dev


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--device")]
    steps = int(args[0]) if args else 600
    device = None
    for i, a in enumerate(sys.argv[1:]):
        if a == "--device" and i + 1 < len(sys.argv[1:]):
            device = sys.argv[i + 2]
    tok = Tokenizer.from_file(TOKENIZER)
    m, dev = build_model(tok, steps, device=device)
    p = sum(pp.numel() for pp in m.parameters())
    print(f"trained {steps} steps | {p:,} params | device {dev}")
    m.eval()
    tpls = [t.strip() for t in TPL_STR.split("|")]

    results = {}
    for t in tpls:
        a, _, _ = eval_slots(m, tok, UNSEEN, template=t, mode="hard", seed=11,
                             boundary_eos=True, pos_eos=True, device=dev)
        results[t] = a
    for t in PROBES:
        a, _, _ = eval_slots(m, tok, UNSEEN, template=t, mode="hard", seed=13,
                             boundary_eos=True, pos_eos=True, device=dev)
        results[t] = a
    for name, slots in MATRIX.items():
        a, _, _ = eval_slots(m, tok, slots, template="pip install {slot}",
                             mode="hard", seed=9, boundary_eos=True, pos_eos=True,
                             device=dev)
        results[name] = a

    ok = True
    for t in tpls:
        ok &= results[t] >= 1.0
        print(f"  trained {t.split()[0]:<10} {results[t]:.3f}")
    for t in PROBES:
        ok &= results[t] >= 1.0
        print(f"  probe   {t.split()[0]:<10} {results[t]:.3f}")
    for name in MATRIX:
        ok &= results[name] >= 1.0
        print(f"  matrix  {name:<10} {results[name]:.3f}")
    print("\n" + ("ALL V4 SLOT GUARDS PASSED" if ok else "V4 SLOT GUARD FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())