"""Verified eval: HMN v3.3 slot-copy on hmn_v33.pt (seed-42 guardrail).

One command reproduces the documented claims (docs/hmn_v3_design.md):

  python experiments/verified/slot_v33_seed42.py            # default checkpoint
  python experiments/verified/slot_v33_seed42.py --checkpoint PATH

ASSERTED (must pass, this is the guardrail):
  - 40/40 exact on UNSEEN slots (pkg060..pkg099) for blend AND hard decode
    with the boundary_eos rule
  - 40/40 exact on a structural variant the model never saw at train time
    ("pip install -r {slot}") — extends in length, same command word

PRINTED as a generalization matrix (known limitations, not asserted):
  - 4/5-digit slots (~0.92): extra unique digits mostly copy but leak sometimes
  - repeated-digit slots (pkg333..pkg999, ~0.33): the copy lane loops on the
    repeated token (gate stays ~0.93) and boundary_eos cannot fire — root
    cause #3, documented
  - different command words (import/run/apt install): 0.0 — the copy gate is
    lexicon-bound to the trained template, NOT a general "copy the prompt tail"
    operation. Honest boundary of the mechanism.

Reproducibility: greedy decode, CPU, torch.manual_seed(42); the slot lists and
templates fully determine the output (decode is seed-free).
"""
import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import torch

from tokenizers import Tokenizer

from hmn import HMN3
from hmn.recipe import eval_slots, resolve_device

CFG = dict(dim=96, layers=3, gate_bias=-1.0)
CHECKPOINT = os.path.join(ROOT, "hmn_v33.pt")
TOKENIZER = os.path.join(ROOT, "retop_tokenizer.json")

UNSEEN = [f"pkg{i:03d}" for i in range(60, 100)]
TEMPLATE = "pip install {slot}"


def load_model(checkpoint, device=None):
    tok = Tokenizer.from_file(TOKENIZER)
    dev = resolve_device(device)
    m = HMN3(tok.get_vocab_size(), dim=CFG["dim"], state_dim=8,
             n_layers=CFG["layers"], use_moe=False, gate_bias=CFG["gate_bias"],
             asi_id=tok.token_to_id("<|assistant|>"), keys_proj=False, aux_copy=True)
    m.load_state_dict(torch.load(checkpoint, map_location=dev))
    m.to(dev).eval()
    return m, tok, dev


def run_guards(checkpoint, device=None):
    """Run all verified guards + generalization probes on a slot-copy checkpoint.

    Returns a structured dict consumed by BOTH the CLI (this file) and the GUI
    (retop_gui.py "VERIFY" tab) so the verified numbers never drift:
      {n_params, blend: (acc, gate), hard: acc, structural: (acc, gate, template),
       matrix: [{name, ok_n, total, acc, gate}, ...], ok: bool}
    """
    torch.manual_seed(42)
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(checkpoint)
    m, tok, dev = load_model(checkpoint, device=device)
    n_params = sum(p.numel() for p in m.parameters())

    acc, gate, _ = eval_slots(m, tok, UNSEEN, template=TEMPLATE, mode="blend",
                              seed=42, boundary_eos=True, device=dev)
    acc_h, _, _ = eval_slots(m, tok, UNSEEN, template=TEMPLATE, mode="hard",
                             seed=42, boundary_eos=True, device=dev)
    struct = TEMPLATE.replace("{slot}", "-r {slot}")
    acc_s, gate_s, _ = eval_slots(m, tok, UNSEEN, template=struct, seed=42,
                                  boundary_eos=True, device=dev)

    matrix = []
    for name, slots, tpl in [
        ("4-digit slots", [f"pkg{i:04d}" for i in range(9000, 9040)], TEMPLATE),
        ("5-digit slots", [f"pkg{i:05d}" for i in range(90000, 90040)], TEMPLATE),
        ("repeated digit slots", [f"pkg{i:03d}" for i in [111, 222, 333, 444, 555,
                                                          666, 777, 888, 999]], TEMPLATE),
        ("template 'import'", UNSEEN, "import {slot}"),
        ("template 'run'", UNSEEN, "run {slot}"),
        ("template 'apt install'", UNSEEN, "apt install {slot}"),
    ]:
        a, g, _ = eval_slots(m, tok, slots, template=tpl, seed=42, boundary_eos=True,
                             device=dev)
        matrix.append({"name": name, "ok_n": int(round(a * len(slots))),
                       "total": len(slots), "acc": a, "gate": g})

    ok = acc >= 1.0 and acc_h >= 1.0 and acc_s >= 1.0
    return {"n_params": n_params, "blend": (acc, gate), "hard": acc_h,
            "structural": (acc_s, gate_s, struct), "matrix": matrix, "ok": ok}


def main(checkpoint, device=None):
    r = run_guards(checkpoint, device=device)
    print(f"checkpoint: {checkpoint} ({r['n_params']:,} params, cfg={CFG})")

    def guard(name, cond, detail):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")

    acc, gate = r["blend"]
    guard(f"unseen blend {TEMPLATE!r}", acc >= 1.0,
          f"{acc:.3f} ({len(UNSEEN)*acc:.0f}/{len(UNSEEN)}, gate {gate:.3f})")
    guard("unseen hard (gate>0.5 -> copy argmax)", r["hard"] >= 1.0, f"{r['hard']:.3f}")
    acc_s, gate_s, struct = r["structural"]
    guard(f"structural variant {struct!r}", acc_s >= 1.0,
          f"{acc_s:.3f} (gate {gate_s:.3f})")

    print("\ngeneralization matrix (known limitations, informational):")
    for row in r["matrix"]:
        print(f"  {row['name']:<22} {row['ok_n']:3d}/{row['total']:3d}"
              f"  acc={row['acc']:.3f} gate={row['gate']:.3f}")

    print(f"\n{'ALL GUARDS PASSED' if r['ok'] else 'GUARDRAIL FAILED'}")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--device", default=None,
                    help="compute device (auto-detect default; see resolve_device)")
    args = ap.parse_args()
    raise SystemExit(main(args.checkpoint, args.device))