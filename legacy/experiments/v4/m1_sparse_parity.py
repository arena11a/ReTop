"""v4-M1/v6-M1 parity check: dense oracle vs index (IRStats) path must agree.

What this proves / delivers:

  v4 M1 (2026-08-13): the sparse copy-marginal (position-gather for the copy
  loss, no separate (B,T,V) copy-mass tensor in the IR forward) was proven
  BIT-IDENTICAL to the dense path and kept the 40/40 guardrail. Both variants
  live on unchanged behind HMN3(exact_blend=True).

  v6 M1-B (2026-08-23): the default forward no longer materializes ANY
  register tensor — consumers read the IRStats inverted-index API. The copy
  lane snaps to own-group payload histograms (cross-id epsilon-mass dropped;
  the DECLARED semantic change of M1). Parity therefore becomes behavioral:
    1. gate inputs (mass_same/n_legal) and ctx match the dense oracle to FP
       noise (group-space collapse is LOSSLESS),
    2. blend/gen CE terms match tightly; copy CE matches to within the snap,
    3. the trained 40/40 slot guardrail decodes IDENTICALLY on both paths.

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


def build(exact, device=None):
    tok = Tokenizer.from_file(TOKENIZER)
    dev = resolve_device(device)
    m = HMN3(tok.get_vocab_size(), dim=CFG["dim"], state_dim=8,
             n_layers=CFG["layers"], use_moe=False, gate_bias=CFG["gate_bias"],
             asi_id=tok.token_to_id("<|assistant|>"),
             aux_copy=True, exact_blend=exact)
    missing, unexpected = load_compat(m, os.path.join(ROOT, "hmn_v33.pt"),
                                      device=dev)
    assert not missing and not unexpected, \
        f"checkpoint mismatch: missing={missing} unexpected={unexpected}"
    m.to(dev).eval()
    return m, tok, dev


def main():
    torch.manual_seed(0)
    md, tok, dev = build(True)      # dense oracle (v3.3 behavior)
    ms, _, _ = build(False)         # v6 M1-B index/stats path

    # --- 1. forward parity: gate inputs + ctx are a lossless collapse -------
    X, Y, Yc, G = make_slot_batch(tok, SEEN, 4, seed=3, device=dev)
    with torch.no_grad():
        od = md(X)
        os_ = ms(X)
        st = os_["stats"]
    gm_diff = (od["g"] - os_["g"]).abs().max().item()
    print(f"gate value max-abs-diff oracle-vs-stats     : {gm_diff:.3e}")
    assert gm_diff < 1e-4, "gate diverged"
    gl_diff = (od["gen_logits"] - os_["gen_logits"]).abs().max().item()
    print(f"gen logits max-abs-diff oracle-vs-stats     : {gl_diff:.3e}")
    assert gl_diff < 1e-4, "gen head diverged"
    assert st.mass_same.shape == X.shape, "stats shape"
    assert not hasattr(st, "attn"), "stats must not carry dense attention"

    # --- 2. loss parity (blend/gen tight; copy within the declared snap) ----
    lossf = torch.nn.CrossEntropyLoss(ignore_index=-100)
    ld_loss = loss_v33(od, Y, Yc, G, lossf=lossf)
    ls_loss = loss_v33(os_, Y, Yc, G, lossf=lossf)
    print(f"loss oracle : {[round(float(x), 6) for x in ld_loss]}")
    print(f"loss stats  : {[round(float(x), 6) for x in ls_loss]}")
    for name, a, b, tol in zip(["total", "blend", "gen", "copy"],
                               ld_loss, ls_loss,
                               [1e-3, 1e-4, 1e-6, 5e-3]):
        d = abs(float(a) - float(b))
        print(f"  {name:6s} diff {d:.3e} (tol {tol:.0e})")
        assert d < tol, f"{name} diverged"

    # --- 3. exact-match parity on the v3.3 guardrail task -------------------
    acc_d, g_d, ng_d = eval_slots(md, tok, UNSEEN, seed=42, boundary_eos=True,
                                  device=dev)
    acc_s, g_s, ng_s = eval_slots(ms, tok, UNSEEN, seed=42, boundary_eos=True,
                                  device=dev)
    print(f"guardrail blend oracle vs stats : {acc_d:.3f}/{acc_s:.3f} "
          f"(gate {g_d:.3f}/{g_s:.3f}, gen-tokens {ng_d:.2f}/{ng_s:.2f})")
    assert acc_d == acc_s == 1.0, "guardrail accuracy diverged or broke"
    assert abs(g_d - g_s) < 1e-3, "decode gate trace diverged"
    print("\nM1 PARITY OK (dense oracle == IRStats index path)")


if __name__ == "__main__":
    main()