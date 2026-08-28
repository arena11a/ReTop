"""v6 M5 — sequence packing: doc-masked padding, per-document positions,
loss parity, ReversibleFunction autograd compat.

Pass criteria (this file; roadmap M5 data-side half):
  * pack/unpack round-trip: unpack(pack(docs)) == original docs (exact)
  * position_ids reset at every doc boundary (exact)
  * loss parity: packed forward loss == sum of individual per-document
    losses, AND packed parameter gradients == their summed gradients
    (exact up to FP summation order)
  * ReversibleFunction.backward works at packed sequence shapes (matches
    plain autograd; gradient reaches every packed document)

Honest boundaries (stated, measured — never hidden):
  * Exact parity is proven on architecturally ISOLATED documents: the test
    swaps the recurrent SelectiveSSM coupling for a pointwise reversible
    block, keeps everything else production (embed, IRStats index path,
    deterministic gate, dual head, loss_v33), uses EOS-bounded docs with
    disjoint token ids (no cross-document token-twin groups) and asi-free
    rows (the register would otherwise bind the FIRST <|assistant|> column
    of the row). This is precisely the contract the model-level half of M5
    (block-diagonal masking / SSM state resets under FSDP2/DeepSpeed) has
    to enforce; the packing utilities themselves carry documents, masks
    and positions exactly.
  * With the PRODUCTION recurrent WR, earlier documents remain causally
    untouched (asserted exact for doc 1), while later documents inherit
    SSM state from earlier ones — the residual cross-doc effect is
    REPORTED numerically below and is what the state-reset work closes.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn
from tokenizers import Tokenizer

import hmn.v3 as v3mod
from hmn.packing import (doc_position_ids, pack_batch,
                         pack_sequences, unpack_outputs)
from hmn.v3 import HMN3, HelixCouplingBlock, ReversibleFunction
from hmn.recipe import EOS, loss_v33, make_chat_ids, seed_guardrail

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")

# isolated-construction constants: toy vocab, EOS-terminated docs, disjoint
# id ranges per document (bases 10/20/30) so no token-twin group spans docs
EOS_T = 1
V_ISO = 48


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


# ---------------------------------------------------------------------------
# shared builders for the parity constructions
# ---------------------------------------------------------------------------
def make_iso_doc(base):
    """Echo-style doc with a PERMUTED answer: [E, x,y,z, x,z,y, E].

    Answer rows carry real copy mass (target present as a prompt payload at
    fraction 1/2), an EOS-closing gen row, and a masked tail row — i.e. the
    Y/Yc/G triple a training step consumes. The leading/trailing EOS align
    the register's payload legality between the packed row and standalone
    runs (an inter-doc boundary column is 'bad payload' in BOTH).
    """
    u = [base + 1, base + 2, base + 3]
    return [EOS_T] + u + [u[0], u[2], u[1]] + [EOS_T]


def make_iso_targets(ids):
    """Hand-built recipe triple for make_iso_doc docs (answer = rows 4..7)."""
    T = len(ids)
    Y, YC, G = [-100] * T, [-100] * T, [-1.0] * T
    for t, (yc, g) in zip(range(4, T),
                          [(ids[5], 1.0), (ids[6], 1.0),
                           (-100, 0.0), (-100, -1.0)]):
        Y[t] = ids[t + 1] if t + 1 < T else -100
        YC[t] = yc
        G[t] = g
    Y[T - 1] = -100                      # nothing may target the NEXT doc
    return Y, YC, G


def place_triples(triples, lens, L):
    """Per-doc (Y,Yc,G) lists -> packed (1,L) tensors with -100/-1 outside
    every document's own rows (roadmap 2.4: masks carried through)."""
    Y = torch.full((1, L), -100, dtype=torch.long)
    YC = torch.full((1, L), -100, dtype=torch.long)
    G = torch.full((1, L), -1.0, dtype=torch.float)
    off = 0
    for (y, yc, g), ln in zip(triples, lens):
        Y[0, off:off + ln] = torch.tensor(y, dtype=torch.long)
        YC[0, off:off + ln] = torch.tensor(yc, dtype=torch.long)
        G[0, off:off + ln] = torch.tensor(g, dtype=torch.float)
        off += ln
    return Y, YC, G


def mask_only(triple, lens, L, k):
    """Full-length targets carrying ONLY doc k's rows (for per-doc losses
    evaluated against the packed forward's outputs)."""
    Y, YC, G = place_triples([triple], [lens[k]], L)
    Yfull = torch.full((1, L), -100, dtype=torch.long)
    YCfull = torch.full((1, L), -100, dtype=torch.long)
    Gfull = torch.full((1, L), -1.0, dtype=torch.float)
    off = sum(lens[:k])
    Yfull[0, off:off + lens[k]] = Y[0, :lens[k]]
    YCfull[0, off:off + lens[k]] = YC[0, :lens[k]]
    Gfull[0, off:off + lens[k]] = G[0, :lens[k]]
    return Yfull, YCfull, Gfull


def loss_counts(Y, Yc, G):
    """Row counts behind loss_v33's three mean reductions."""
    return (int((Y != -100).sum()),
            int(((Y != -100) & (G == 0.0)).sum()),
            int((Yc != -100).sum()))


class PointwiseCoupling(nn.Module):
    """Test-scoped reversible coupling with NO temporal mixing.

    Signature-compatible drop-in for HelixCouplingBlock (patched into
    hmn.v3 only for the parity construction): F1/F2 act per position, so
    document k's hidden states cannot read document <k content — the one
    leak channel left once ids are disjoint and rows are asi-free. The
    inverse is the exact algebraic reverse, so ReversibleFunction's
    reconstruct-and-recompute backward operates identically.
    """

    def __init__(self, dim, state_dim=None):
        super().__init__()
        half = dim // 2

        def f():
            return nn.Sequential(nn.Linear(half, 2 * half), nn.GELU(),
                                 nn.Linear(2 * half, half))
        self.F1, self.F2 = f(), f()

    def forward(self, h):
        a, b = h.chunk(2, dim=-1)
        a = a + self.F1(b)
        b = b + self.F2(a)
        return torch.cat([a, b], dim=-1)

    def inverse(self, h):
        a, b = h.chunk(2, dim=-1)
        b = b - self.F2(a)
        a = a - self.F1(b)
        return torch.cat([a, b], dim=-1)


def isolated_hmn3():
    """Production HMN3 minus the recurrent WR channel (see module docstring)."""
    orig = v3mod.HelixCouplingBlock
    v3mod.HelixCouplingBlock = PointwiseCoupling
    try:
        m = HMN3(V_ISO, dim=16, state_dim=4, n_layers=2, use_moe=False,
                 asi_id=None)
    finally:
        v3mod.HelixCouplingBlock = orig
    return m


# ---------------------------------------------------------------------------
# M5-1: pack/unpack round-trip
# ---------------------------------------------------------------------------
def test_pack_round_trip():
    print("[M5-1: pack/unpack round-trip]")
    tok = Tokenizer.from_file(os.path.join(ROOT, "retop_tokenizer.json"))
    eos = tok.token_to_id(EOS)
    docs = [make_chat_ids(tok, f"pip install {s}", f"pip install {s}")
            for s in ("pkg007", "lib042", "pkg123")]
    packed, lens = pack_sequences(docs)
    check(sum(lens) == len(packed), f"boundaries cover the row "
          f"({len(docs)} docs, lens {lens}, packed {len(packed)})")

    L = len(packed) + 5                       # deliberate trailing padding
    b = pack_batch(docs, max_len=L, pad_id=eos)
    unp = unpack_outputs(b["input_ids"], lens)
    back = [u.squeeze(0).tolist() for u in unp]
    check(back == docs, "unpack(pack(docs)) == original docs (exact)")

    am = b["attention_mask"]
    check(int(am.sum()) == len(packed) and bool(am[0, :len(packed)].all())
          and int(am[0, len(packed):].sum()) == 0,
          "attention mask marks exactly the real document tokens")
    pad_cols = (am == 0).nonzero(as_tuple=True)[1]
    check(bool((b["input_ids"][0, pad_cols] == eos).all()),
          f"padding filled with EOS ({int(pad_cols.numel())} pad columns)")
    did = b["doc_id"][0]
    want = []
    for k, ln in enumerate(lens):
        want += [k] * ln
    check(did[:len(packed)].tolist() == want and bool((did[len(packed):] == -1)
                                                      .all()),
          "doc_id maps every column to its owning document")


# ---------------------------------------------------------------------------
# M5-2: position_ids reset at doc boundaries
# ---------------------------------------------------------------------------
def test_position_ids_reset():
    print("[M5-2: position_ids reset at doc boundaries]")
    tok = Tokenizer.from_file(os.path.join(ROOT, "retop_tokenizer.json"))
    eos = tok.token_to_id(EOS)
    docs = [make_chat_ids(tok, f"pip install {s}", f"pip install {s}")
            for s in ("pkg007", "lib042", "pkg123")]
    _, lens = pack_sequences(docs)
    L = sum(lens) + 4
    pos = doc_position_ids(lens, L)
    off = 0
    for k, ln in enumerate(lens):
        seg = pos[0, off:off + ln].tolist()
        check(seg == list(range(ln)),
              f"doc {k}: positions restart 0..{ln - 1} (got {seg})")
        off += ln
    check(bool((pos[0, off:] == 0).all()), "padding columns carry position 0")
    resets = [int(pos[0, sum(lens[:k])]) for k in range(len(lens))]
    check(resets == [0] * len(lens),
          f"every doc boundary resets the clock (starts {resets}) — the "
          f"IR is content-addressed so it imposes no positional scheme; "
          f"this is the convention RoPE/attention consumers expect")


# ---------------------------------------------------------------------------
# M5-3: loss parity — packed forward == sum of per-document losses
# ---------------------------------------------------------------------------
def test_loss_parity_isolated():
    print("[M5-3: loss parity (isolated documents, production heads/index)]")
    seed_guardrail(42)
    docs = [make_iso_doc(b) for b in (10, 20, 30)]
    triples = [make_iso_targets(d) for d in docs]
    packed, lens = pack_sequences(docs)
    L = len(packed) + 3
    batch = pack_batch(docs, max_len=L, pad_id=EOS_T)
    Y, YC, G = place_triples(triples, lens, L)
    nb_t, ng_t, nc_t = loss_counts(Y, YC, G)
    m = isolated_hmn3()
    m.train()

    # ---- packed forward: one row, all documents -------------------------
    out_p = m(batch["input_ids"])
    lp, lb_p, lg_p, lc_p = loss_v33(out_p, Y, YC, G)
    packed_sums = (float(lb_p.detach()) * nb_t, float(lg_p.detach()) * ng_t,
                   float(lc_p.detach()) * nc_t)

    # ---- (a) decomposition: per-doc losses FROM the packed forward ------
    decomp = torch.zeros(3)
    for k, tr in enumerate(triples):
        Yk, YCk, Gk = mask_only(tr, lens, L, k)
        _, lbk, lgk, lck = loss_v33(out_p, Yk, YCk, Gk)
        nb, ng, nc = loss_counts(Yk, YCk, Gk)
        decomp += torch.tensor([float(lbk.detach()) * nb,
                                float(lgk.detach()) * ng,
                                float(lck.detach()) * nc])
    derr = float((decomp - torch.tensor(packed_sums)).abs().max())
    check(derr < 1e-4,
          f"packed blend/gen/copy sums decompose into per-doc parts "
          f"(max err {derr:.2e})")

    # ---- (b) TRUE parity: independent per-document forwards -------------
    iso_sums = torch.zeros(3)
    for d, tr in zip(docs, triples):
        Yd = torch.tensor([tr[0]], dtype=torch.long)
        YCd = torch.tensor([tr[1]], dtype=torch.long)
        Gd = torch.tensor([tr[2]], dtype=torch.float)
        out_d = m(torch.tensor([d], dtype=torch.long))
        _, lb, lg, lc = loss_v33(out_d, Yd, YCd, Gd)
        nb, ng, nc = loss_counts(Yd, YCd, Gd)
        iso_sums += torch.tensor([float(lb.detach()) * nb,
                                  float(lg.detach()) * ng,
                                  float(lc.detach()) * nc])
    perr = float((iso_sums - torch.tensor(packed_sums)).abs().max())
    total_p = sum(packed_sums)
    check(perr < 1e-4,
          f"PACKED LOSS == SUM OF INDIVIDUAL DOC LOSSES "
          f"(blend/gen/copy max err {perr:.2e}; sum-form total "
          f"{total_p:.6f} vs {float(iso_sums.sum()):.6f})")

    # ---- (c) per-document outputs: packed slices == standalone ----------
    outs_d = []
    for d in docs:
        with torch.no_grad():
            outs_d.append(m(torch.tensor([d], dtype=torch.long)))
    max_gl = max_g = 0.0
    for k in range(len(docs)):
        sl = unpack_outputs(out_p, lens)[k]
        max_gl = max(max_gl, float((sl["gen_logits"]
                                    - outs_d[k]["gen_logits"]).abs()
                                   .max().detach()))
        max_g = max(max_g, float((sl["g"] - outs_d[k]["g"]).abs()
                                 .max().detach()))
    check(max_gl < 1e-5 and max_g < 1e-5,
          f"unpacked gen_logits/g match standalone per doc "
          f"(gen_logits {max_gl:.2e}, g {max_g:.2e})")

    # ---- (d) gradients: packed backward == summed per-doc backwards -----
    m.zero_grad(set_to_none=True)
    lp.backward()
    packed_grads = {n: (None if p.grad is None else p.grad.detach().clone())
                    for n, p in m.named_parameters()}
    m.zero_grad(set_to_none=True)
    for d, tr in zip(docs, triples):
        Yd = torch.tensor([tr[0]], dtype=torch.long)
        YCd = torch.tensor([tr[1]], dtype=torch.long)
        Gd = torch.tensor([tr[2]], dtype=torch.float)
        out_d = m(torch.tensor([d], dtype=torch.long))
        _, lb, lg, lc = loss_v33(out_d, Yd, YCd, Gd)
        nb, ng, nc = loss_counts(Yd, YCd, Gd)
        # scale by GLOBAL row fractions so the accumulated scalar equals the
        # packed loss exactly (means over matching row sets)
        scaled = (lb * (nb / nb_t) + lg * (ng / ng_t) + lc * (nc / nc_t))
        scaled.backward()                    # accumulates across documents
    worst, worst_n, n_cmp = 0.0, "", 0
    for n, p in m.named_parameters():
        a, b = packed_grads[n], p.grad
        if a is None and b is None:
            continue
        if a is None or b is None:
            raise AssertionError(f"grad presence mismatch for {n}")
        e = float((a - b).abs().max())
        n_cmp += 1
        if e > worst:
            worst, worst_n = e, n
    check(worst < 1e-5,
          f"packed parameter gradients == sum of per-document backwards "
          f"({n_cmp} params, max err {worst:.2e} on {worst_n})")
    print("  boundary: exactness here uses isolated documents (pointwise WR, "
          "disjoint ids, asi-free rows); the recurrent-WR residual is "
          "quantified in M5-4")


# ---------------------------------------------------------------------------
# M5-4: production recurrent WR — causality + honest cross-doc measurement
# ---------------------------------------------------------------------------
def test_real_model_causality_and_backward():
    print("[M5-4: production SSM-WR — doc-1 causality, cross-doc report, "
          "packed backward]")
    seed_guardrail(7)
    docs = [make_iso_doc(b) for b in (10, 20, 30)]
    triples = [make_iso_targets(d) for d in docs]
    packed, lens = pack_sequences(docs)
    L = len(packed) + 3
    batch = pack_batch(docs, max_len=L, pad_id=EOS_T)
    Y, YC, G = place_triples(triples, lens, L)
    m = HMN3(V_ISO, dim=32, state_dim=8, n_layers=2, use_moe=False,
             asi_id=None)                 # REAL SelectiveSSM WR — unpatched
    m.eval()

    with torch.no_grad():
        out_p = m(batch["input_ids"])
        lp, *_ = loss_v33(out_p, Y, YC, G)
        deltas = []
        for k, tr in enumerate(triples):
            Yk, YCk, Gk = mask_only(tr, lens, L, k)
            lk, *_ = loss_v33(out_p, Yk, YCk, Gk)
            Yd = torch.tensor([tr[0]], dtype=torch.long)
            YCd = torch.tensor([tr[1]], dtype=torch.long)
            Gd = torch.tensor([tr[2]], dtype=torch.float)
            ld, *_ = loss_v33(m(torch.tensor([docs[k]], dtype=torch.long)),
                              Yd, YCd, Gd)
            deltas.append(float(lk) - float(ld))
    check(abs(deltas[0]) < 1e-4,
          f"doc 1 loss untouched by packing (delta {deltas[0]:.2e}) — "
          f"causality: nothing after doc 1 may change it")
    for k, dv in enumerate(deltas[1:], start=1):
        check(dv == dv and abs(dv) < 1e3,
              f"doc {k + 1} packed-vs-standalone delta {dv:+.4f} (finite; "
              f"documented SSM state carry + first-boundary binding — closed "
              f"by model-level state resets in the M5 distributed wrap)")
    print(f"  packed total loss {float(lp):.6f}; per-doc deltas "
          f"{['%+.4f' % d for d in deltas]}")

    # backward through the packed loss: custom Function + full recipe head
    m.train()
    m.zero_grad(set_to_none=True)
    out_p = m(batch["input_ids"])
    lp, _, _, _ = loss_v33(out_p, Y, YC, G)
    lp.backward()
    n_grad = sum(1 for p in m.parameters() if p.grad is not None)
    all_fin = all(bool(torch.isfinite(p.grad).all())
                  for p in m.parameters() if p.grad is not None)
    check(n_grad > 0 and all_fin,
          f"ReversibleFunction backward on the packed row reaches "
          f"{n_grad} params, all finite")
    emb = m.embed.weight.grad
    reach, sums = [], []
    for k, d in enumerate(docs):
        content = [t for t in d if t != EOS_T]
        s = float(emb[content].abs().sum())
        sums.append(s)
        reach.append(s > 0)
    check(all(reach),
          f"embedding gradient lands in EVERY packed document "
          f"(per-doc content |grad| sums "
          f"{['%.2e' % s for s in sums]})")


# ---------------------------------------------------------------------------
# M5-5: ReversibleFunction backward correctness AT PACKED SHAPES
# ---------------------------------------------------------------------------
def test_reversible_backward_packed_shape():
    print("[M5-5: ReversibleFunction vs naive autograd at packed length]")
    torch.manual_seed(0)
    docs = [make_iso_doc(b) for b in (10, 20, 30)]
    packed, lens = pack_sequences(docs)
    Lp, D = len(packed), 32
    blk = HelixCouplingBlock(D, 4)

    # reconstruction identity at the packed length first
    x0 = torch.randn(1, Lp, D)
    rec = (x0 - blk.inverse(blk.forward(x0))).abs().max().item()
    check(rec < 1e-4, f"forward->inverse reconstructs at T={Lp} "
                      f"(max err {rec:.2e})")

    x = torch.randn(1, Lp, D).requires_grad_(True)
    y1 = ReversibleFunction.apply(x, [blk])
    y1.square().mean().backward()
    gx1 = x.grad.clone()
    gw1 = {n: p.grad.clone() for n, p in blk.named_parameters()
           if p.grad is not None}

    x2 = x.detach().clone().requires_grad_(True)
    for p in blk.parameters():
        p.grad = None
    y2 = blk.forward(x2)
    y2.square().mean().backward()
    check(torch.allclose(gx1, x2.grad, atol=1e-5),
          "packed input gradient matches naive autograd")
    for n, p in blk.named_parameters():
        check(p.grad is not None and torch.allclose(gw1[n], p.grad, atol=1e-5),
              f"packed param grad matches naive ({n})")


if __name__ == "__main__":
    test_pack_round_trip()
    test_position_ids_reset()
    test_loss_parity_isolated()
    test_real_model_causality_and_backward()
    test_reversible_backward_packed_shape()
    print("\nM5 ALL PASSED")
