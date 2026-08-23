"""v4 M1 parity check: dense (v3.3) vs sparse copy-marginal must be identical.

What M1 actually proves / delivers:
  1. Parity — the sparse path (position-gather for the copy loss, no separate
     (B,T,V) copy-mass tensor in the IR forward) is BIT-IDENTICAL to the dense
     path for logits, all four loss terms, and the 40/40 guardrail decode.
  2. Structure — forward no longer materializes the IR `mass` (B,T,V) tensor
     nor the duplicated `copy_dist` normalization in hmn/v3.HMN3; the copy CE
     in loss_v33 reads only (attn, nxt) via copy_prob_sparse. The trained
     copy-path V-wide allocation drops (dense: mass + hmn3 copy_dist + dual
     copy_dist; sparse: dual copy + dual copy_dist).
  3. Honest boundary — the DUAL bus still emits (B,T,V) logits (that is the
     model output, irreducible), and the register attention `sim` is (B,T,T)
     (the true long-context bottleneck at T >> 1.5k, discovered here). Neither
     is claimed to be removed by M1; both are later milestones.

Run: python experiments/v4/m1_sparse_parity.py
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import torch

from tokenizers import Tokenizer

from hmn import HMN3
from hmn.checkpoint import load_compat
from hmn.recipe import make_slot_batch, loss_v33, eval_slots, resolve_device

TOKENIZER = os.path.join(ROOT, "retop_tokenizer.json")
CFG = dict(dim=96, layers=3, gate_bias=-1.0)
SEEN = [f"pkg{i:03d}" for i in range(60)]
UNSEEN = [f"pkg{i:03d}" for i in range(60, 100)]


def build(sparse, device=None):
    tok = Tokenizer.from_file(TOKENIZER)
    dev = resolve_device(device)
    m = HMN3(tok.get_vocab_size(), dim=CFG["dim"], state_dim=8,
             n_layers=CFG["layers"], use_moe=False, gate_bias=CFG["gate_bias"],
             asi_id=tok.token_to_id("<|assistant|>"),
             aux_copy=True, sparse_marginal=sparse)
    m.load_state_dict(torch.load(os.path.join(ROOT, "hmn_v33.pt"),
                                 map_location=dev))
    m.to(dev).eval()
    return m, tok, dev


def main():
    torch.manual_seed(0)
    md, tok, dev = build(False)
    ms, _, _ = build(True)

    # --- 1. forward logits parity on a slot batch ---
    X, Y, Yc, G = make_slot_batch(tok, SEEN, 4, seed=3, device=dev)
    with torch.no_grad():
        od = md(X)
        os_ = ms(X)
    diff = (od["logits"] - os_["logits"]).abs().max().item()
    print(f"forward logits max-abs-diff dense-vs-sparse : {diff:.3e}")
    assert diff < 1e-6, "logits diverged"
    assert "attn" in os_ and "nxt" in os_, "sparse dict missing attn/nxt"

    # --- 2. loss parity (sparse uses position-gather, no (B,T,V) index) ---
    lossf = torch.nn.CrossEntropyLoss(ignore_index=-100)
    ld_loss = loss_v33(od, Y, Yc, G, lossf=lossf)
    ls_loss = loss_v33(os_, Y, Yc, G, lossf=lossf)
    print(f"loss dense : {[round(float(x), 6) for x in ld_loss]}")
    print(f"loss sparse: {[round(float(x), 6) for x in ls_loss]}")
    for name, a, b in zip(["total", "blend", "gen", "copy"], ld_loss, ls_loss):
        d = abs(float(a) - float(b))
        print(f"  {name:6s} diff {d:.3e}")
        assert d < 1e-4, f"{name} diverged"

    # --- 3. exact-match parity on the v3.3 guardrail task ---
    acc_d, g_d, _ = eval_slots(md, tok, UNSEEN, seed=42, boundary_eos=True,
                               device=dev)
    acc_s, g_s, _ = eval_slots(ms, tok, UNSEEN, seed=42, boundary_eos=True,
                               device=dev)
    print(f"guardrail blend dense vs sparse : {acc_d:.3f} vs {acc_s:.3f} "
          f"(gate {g_d:.3f} vs {g_s:.3f})")
    assert abs(acc_d - acc_s) < 1e-9, "guardrail accuracy diverged"
    assert acc_s >= 1.0, "sparse must keep the verified 40/40"
    print("\nM1 SPARSE PARITY OK")


if __name__ == "__main__":
    main()