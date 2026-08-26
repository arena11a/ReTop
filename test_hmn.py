"""Model-level tests for the HMN architecture (v2 + v3).

Covers the pieces the data-generator tests don't:
  - forward/backward on every model class
  - reversibility of the coupling backbone (reconstruction identity)
  - gradient correctness of ReversibleFunction vs a naive forward
  - MoE aux loss / v3 copy path / latent thinking buffer

Run: python test_hmn.py        (CPU, a few seconds)
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizers import Tokenizer

from hmn import (
    HMN3,
    HMN3_NoReg,
    HelixCouplingBlock,
    ReversibleFunction,
    SeedPointer,
    SparseConditionalCompute,
)
from hmn.v7 import (
    HMN3AttentionWR,
    SparseConditionalComputeV2,
    AttentionBlock,
    RMSNorm,
    RotaryPositionEmbedding,
    SwiGLUFFN,
)
from hmn.config import HMNConfig, create_model
from hmn.recipe import (ASSIST, USER, decode_v33, eval_reorders, loss_v33,
                        make_chat_ids, make_perm_ids, make_reorder_batch,
                        make_reorder_ids, reorder_anchors, seam_losses)

ROOT = os.path.dirname(os.path.abspath(__file__))
VOCAB = 3190
DIM = 64


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def rand_ids(bs=2, t=16):
    return torch.randint(0, VOCAB, (bs, t))


def test_coupling_reversible():
    print("[coupling reversibility]")
    blk = HelixCouplingBlock(DIM, 8)
    x = torch.randn(2, 16, DIM)
    y = blk.forward(x)
    xr = blk.inverse(y)
    err = (x - xr).abs().max().item()
    check(err < 1e-4, f"forward->inverse reconstructs input (max err {err:.2e})")


def test_reversible_autograd_vs_naive():
    print("[ReversibleFunction gradient vs naive]")
    # Compare the custom autograd backward against plain autograd (same forward
    # math): both should give identical grads for the input and the block params.
    blk = HelixCouplingBlock(32, 4)
    x = torch.randn(1, 8, 32).requires_grad_(True)

    y1 = ReversibleFunction.apply(x, [blk])
    loss1 = y1.square().mean()
    loss1.backward()
    g_x1 = x.grad.clone()
    g_w1 = {n: p.grad.clone() for n, p in blk.named_parameters() if p.grad is not None}

    # plain-autograd reference on identical params
    x2 = x.detach().clone().requires_grad_(True)
    for p in blk.parameters():
        p.grad = None
    y2 = blk.forward(x2)
    loss2 = y2.square().mean()
    loss2.backward()
    check(torch.allclose(g_x1, x2.grad, atol=1e-5), "input gradient matches naive")
    for n, p in blk.named_parameters():
        check(p.grad is not None and torch.allclose(g_w1[n], p.grad, atol=1e-5),
              f"param grad matches naive ({n})")


def test_v3_forward_and_copy():
    print("[HMN3 forward + copy path]")
    m = HMN3(VOCAB, dim=96, state_dim=8, n_layers=3, use_moe=False,
             gate_bias=-2.0, asi_id=5)
    out = m(rand_ids())
    # v6 M1-B: the default forward returns the IRStats API — NO dense blended
    # "logits" / "copy_dist" (those are oracle-only behind exact_blend=True).
    check("stats" in out and "gen_logits" in out and "g" in out,
          "returns stats/gen_logits/g")
    check(not hasattr(out["stats"], "attn"), "stats carries no dense attention")
    check(out["gen_logits"].shape == (2, 16, VOCAB), "gen_logits shape")
    check(torch.isfinite(out["gen_logits"]).all(), "gen_logits finite")
    # backward incl. aux copy loss (via the v3.3 recipe on the stats API)
    Y = rand_ids()
    Yc = Y.clone()
    Yc[:, 0] = -100
    G = torch.ones_like(Y, dtype=torch.float)
    G[:, 0] = 0.0
    loss, l_blend, l_gen, l_copy = loss_v33(out, Y, Yc, G)
    loss.backward()
    n_grad = sum(1 for p in m.parameters() if p.grad is not None)
    check(n_grad > 0, f"gradients reached {n_grad} params")
    check(torch.isfinite(loss), "loss_v33 finite")
    # oracle regime (--exact-blend): legacy dense contract preserved
    mo = HMN3(VOCAB, dim=96, state_dim=8, n_layers=3, use_moe=False,
              gate_bias=-2.0, asi_id=5, exact_blend=True)
    oo = mo(rand_ids())
    check(oo["logits"].shape == (2, 16, VOCAB), "oracle logits shape")
    check(oo["copy_dist"].shape == (2, 16, VOCAB), "oracle copy_dist shape")
    check(torch.allclose(oo["copy_dist"].sum(-1), torch.ones(2, 16), atol=1e-3),
          "oracle copy_dist sums to 1 per position")


def test_v3_noreg_and_think():
    print("[HMN3_NoReg + HMN3 with thinking buffer]")
    nr = HMN3_NoReg(VOCAB, dim=96, state_dim=8, n_layers=3)
    logits = nr(rand_ids())
    check(logits.shape == (2, 16, VOCAB), "NoReg logits shape")
    t = HMN3(VOCAB, dim=96, state_dim=8, n_layers=3, use_think=True, k_max=4,
             use_moe=False, gate_bias=-2.0, asi_id=5)
    out = t(rand_ids())
    check(torch.isfinite(out["gen_logits"]).all(), "thinking-buffer gen_logits finite")


def test_moe_aux_loss():
    print("[MoE aux loss]")
    moe = SparseConditionalCompute(32, n_experts=16, top_k=2)
    moe.train()
    moe(torch.randn(2, 10, 32))
    l = moe.last_aux_loss
    check(torch.isfinite(l) and 0.0 <= l.item() <= 1.0, f"aux loss in [0,1] ({l.item():.3f})")
    mm = HMN3(VOCAB, DIM, 8, 2, use_moe=True)
    a = mm.moe_aux_loss()
    check(a.numel() == 1, "HMN.moe_aux_loss() scalar")


def test_recipe_copy_ce_is_manual_logp():
    print("[recipe copy CE is manual -log p (NOT CE-on-probs)]")
    vocab = 32
    bs, t = 2, 8
    torch.manual_seed(0)
    cp_logits = torch.randn(bs, t, vocab)
    copy_dist = cp_logits.softmax(-1)  # already a probability distribution
    Yc = torch.randint(0, vocab, (bs, t)).long()
    Yc[0, 0] = -100
    Yc[1, 1] = -100
    Y = Yc.clone()
    G = torch.ones(bs, t)
    G[0, 0] = G[1, 1] = 0.0
    out = {"logits": cp_logits,
           "gen_logits": cp_logits.log_softmax(-1),
           "copy_dist": copy_dist}
    loss, lb, lg, lc = loss_v33(out, Y, Yc, G)
    mask = Yc != -100
    manual = -(copy_dist[mask].gather(1, Yc[mask].unsqueeze(1)).squeeze(-1)).log().mean()
    check(torch.allclose(lc, manual, atol=1e-6),
          f"copy CE == -log p_target ({lc.item():.4f} vs manual {manual.item():.4f})")
    buggy = F.cross_entropy(copy_dist.reshape(-1, vocab), Yc.reshape(-1), ignore_index=-100)
    check(not torch.allclose(lc, buggy, atol=0.5),
          f"copy CE is NOT CE-on-probs ({lc.item():.4f} vs buggy {buggy.item():.4f})")


def test_recipe_chat_ids_no_special_split():
    print("[recipe make_chat_ids: chat specials stay single ids]")
    tok = Tokenizer.from_file(os.path.join(ROOT, "retop_tokenizer.json"))
    asid = tok.token_to_id("<|assistant|>")
    eos = tok.token_to_id("</s>")
    ids = make_chat_ids(tok, "pip install pkg042")
    check(ids[0] == tok.token_to_id("<s>"), "<s> single id")
    check(ids[1] == tok.token_to_id("<|user|>"), "<|user|> single id")
    check(ids.count(asid) == 1, "<|assistant|> present exactly once")
    check(eos not in ids, "no </s> without a gold answer")
    full = make_chat_ids(tok, "pip install pkg042", "pip install pkg042")
    check(full.count(eos) == 1, "</s> present exactly once with gold")
    check(ids.index(asid) == 2 + len(tok.encode("pip install pkg042").ids),
          "<|assistant|> lands exactly after the encoded user text")
    a = full.index(asid)
    check(tok.decode(full[a + 1:]) == "pip install pkg042", "answer region decodes to gold")


def test_recipe_decode_boundary_rule():
    print("[recipe decode_v33 boundary_eos forces EOS on low gate]")
    tok = Tokenizer.from_file(os.path.join(ROOT, "retop_tokenizer.json"))
    vocab = tok.get_vocab_size()
    bad = 42  # a non-EOS id the dummy model keeps argmaxing

    class Dummy(nn.Module):
        def forward(self, x):
            T = x.shape[1]
            logits = torch.zeros(1, T, vocab)
            logits[0, -1, bad] = 10.0
            return {"logits": logits,
                    "gen_logits": logits.log_softmax(-1),
                    "copy_dist": logits.softmax(-1),
                    "g": torch.zeros(1, T).fill_(0.1)}

    prompt = make_chat_ids(tok, "pip install pkg042")
    m = Dummy()
    ob, _, _ = decode_v33(m, tok, prompt, max_new=8, boundary_eos=True)
    onb, _, _ = decode_v33(m, tok, prompt, max_new=8, boundary_eos=False)
    check(ob != "", "boundary decode still emits the first answer token")
    check(len(ob) < len(onb),
          f"boundary rule stops early ({ob!r} -> {len(ob)} chars vs no-boundary {len(onb)} chars)")


def test_recipe_cycle_break_self_pair():
    print("[recipe decode_v33 cycle_break ignores (x,x) self-pairs]")
    tok = Tokenizer.from_file(os.path.join(ROOT, "retop_tokenizer.json"))
    vocab = tok.get_vocab_size()
    digit = tok.token_to_id("9")  # repeated-identical output token (pkg99999 run)

    prompt = make_chat_ids(tok, "pip install pkg09990")

    # repeat-dummy: always next = '9' (legit repeated identical emission).
    # max_new=6 -> '999999' emitted regardless of cycle_break what token.
    class RepeatDummy(nn.Module):
        def forward(self, x):
            T = x.shape[1]
            logits = torch.zeros(1, T, vocab)
            logits[0, -1, digit] = 10.0
            return {"logits": logits,
                    "gen_logits": logits.log_softmax(-1),
                    "copy_dist": logits.softmax(-1),
                    "g": torch.ones(1, T).fill_(0.997)}

    out_cb, _, _ = decode_v33(RepeatDummy(), tok, prompt, max_new=6,
                              cycle_break=True, mode="copy", pos_eos=False)
    check(out_cb == "999999",
          f"repeated identical tokens survive cycle_break ({out_cb!r})")

    # true replay (non-self transitions within a recopied segment) still stops.
    # A deterministic 9,8,9,8... cycle produces pairs (9,8) and (8,9); the
    # second (9,8) is a segment replay and must fire EOS.
    alt = tok.token_to_id("8")
    ctr = {"n": 0}

    class ReplayDummy(nn.Module):
        def forward(self, x):
            T_ = x.shape[1]
            logits = torch.zeros(1, T_, vocab)
            pick = digit if ctr["n"] % 2 == 0 else alt
            ctr["n"] += 1
            logits[0, -1, pick] = 10.0
            return {"logits": logits,
                    "gen_logits": logits.log_softmax(-1),
                    "copy_dist": logits.softmax(-1),
                    "g": torch.ones(1, T_).fill_(0.997)}

    ctr["n"] = 0
    out_rb, _, _ = decode_v33(ReplayDummy(), tok, prompt, max_new=12,
                              cycle_break=True, mode="copy", pos_eos=False)
    # emits 9,8,9,8 then the (8,9) replay pair fires on the 5th candidate ->
    # text is exactly "9898" (stops BEFORE an unbounded replay)
    check(out_rb == "9898",
          f"true replay triggers cycle EOS, got {out_rb!r} (expected '9898')")


def test_seam_anchor_none_parity():
    print("[v5 seam: seam_anchor=None is bit-identical to no-anchor]")
    torch.manual_seed(0)
    m = HMN3(VOCAB, dim=64, state_dim=8, n_layers=2, use_moe=False,
             gate_bias=-1.0, asi_id=5, user_id=1, seam_addr=True)
    x = rand_ids()
    with torch.no_grad():
        o1 = m(x)
        o2 = m(x, seam_anchor=torch.full_like(x, -100))
    check(torch.equal(o1["gen_logits"], o2["gen_logits"]), "gen_logits identical")
    check(torch.equal(o1["g"], o2["g"]), "gate identical")
    check(torch.equal(o1["stats"].mass_same, o2["stats"].mass_same),
          "stats mass_same identical")


def test_v6_m1a_index_stats():
    print("[v6 M1-A: index-derived n_legal/mass_same == brute-force dense]")
    from hmn.v3 import IdentityRegister
    ir = IdentityRegister(dim=32, asi_id=5, eos_id=1)
    torch.manual_seed(3)
    # keys MUST come from an id->vector table (the model invariant
    # keys = embed(ids)): same id => identical embedding => exactly equal
    # sims, which is what the twins-uniform mass_same identity rests on.
    table = torch.randn(128, 32)
    for trial, (asi_on, dup) in enumerate([(True, True), (False, True),
                                           (True, False), (False, False)]):
        B, T = 3, 14
        ids = torch.randint(10, 80, (B, T))
        if dup:
            ids[:, 4] = ids[:, 9]            # twins straddling a would-be ASI bound
            ids[:, 0] = ids[:, 2]            # twins at the front edge
        if asi_on:
            ids[:, 7] = 5                    # ASI boundary mid-sequence
            ids[:, 10] = 1                   # EOS payload inside the prompt region
        keys = table[ids]
        a, nxt, n_legal, ctx, behind, mass_same, mask = ir._attn(keys, ids)
        # brute-force v3.3 reference math (pre-M1-A)
        beta = ir.beta.abs() + 1.0
        qk = F.normalize(keys, dim=-1)
        sim = (qk @ qk.transpose(-1, -2)) * beta
        m = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        if (ids == 5).any():
            bnd = torch.where((ids == 5).any(-1),
                              (ids == 5).float().argmax(-1),
                              torch.full((B,), T)).long()
            m = m.unsqueeze(0) | (torch.arange(T).unsqueeze(0)
                                  >= bnd.unsqueeze(-1)).unsqueeze(1)
        nxt_col = torch.cat([ids[:, 1:], torch.full_like(ids[:, :1], -1)], 1)
        bad = (nxt_col == 5) | (nxt_col == 1) | (nxt_col == -1)
        m = m | bad.unsqueeze(1)
        sim = sim.masked_fill(m, float("-inf"))
        a_ref = sim.softmax(-1)
        ref_n_legal = (~m).sum(-1)
        ref_mass = (a_ref * (ids.unsqueeze(1) == ids.unsqueeze(2)).float()).sum(-1)
        check(torch.equal(n_legal, ref_n_legal),
              f"trial{trial}: n_legal exact match")
        err = (mass_same - ref_mass).abs().max().item()
        # 1e-4 = FP summation-order noise only; a structural bug (wrong twin
        # group, boundary off-by-one, bad count) errs at weight magnitude >=1e-2
        check(err < 1e-4,
              f"trial{trial}: mass_same matches brute force (max err {err:.2e})")
    # seam forcing still opens the gate through the overridden stats path
    anchor = torch.full((1, 14), -100, dtype=torch.long)
    anchor[0, 11] = 3
    *_, ms_seam, _ = ir._attn(table[ids[:1]], ids[:1], seam_anchor=anchor)
    check(ms_seam[0, 11].item() == 1.0, "seam-forced row keeps mass_same=1.0")


def test_v6_m1b_stats_vs_oracle():
    print("[v6 M1-B: IRStats index path == dense oracle (losses + decode)]")
    torch.manual_seed(7)
    mo = HMN3(VOCAB, dim=48, state_dim=8, n_layers=2, use_moe=False,
              gate_bias=-1.0, asi_id=5)
    ms = HMN3(VOCAB, dim=48, state_dim=8, n_layers=2, use_moe=False,
              gate_bias=-1.0, asi_id=5)
    ms.load_state_dict(mo.state_dict())          # identical weights
    # realistic slot-style batch: the answer repeats prompt tokens so the copy
    # lane carries REAL twin mass (not just cross-id epsilon), which is what
    # trained workloads look like
    ids = torch.tensor([[100, 201, 202, 203, 204, 205, 206, 5,
                         202, 203, 204, 205, 206, 1]])
    X = ids
    T = ids.shape[1]
    Y = ids.roll(-1, dims=1).clone()
    Y[:, :7] = -100                              # targets live in answer rows
    Yc = Y.clone()
    Yc[0, 7] = -100                              # first answer row: seed only
    Yc[0, T - 1] = -100                          # EOS not copyable
    G = torch.zeros(1, T)
    G[0, 8:T - 1] = 1.0
    with torch.no_grad():
        od = mo(X, exact_blend=True)
        os_ = ms(X)
        st = os_["stats"]
        # gate inputs / ctx are a LOSSLESS group-space collapse of the oracle
        x_emb = mo.embed(X)
        _, _, n_legal_o, ctx_o, behind_o, gm_o, _ = mo.ir._attn(x_emb, X)
        check(torch.equal(st.n_legal, n_legal_o), "n_legal == oracle")
        check(torch.equal(st.behind, behind_o), "behind == oracle")
        gm_err = (st.mass_same - gm_o).abs().max().item()
        check(gm_err < 1e-4, f"mass_same == oracle ({gm_err:.2e})")
        # ctx: EXACT at answer rows (the only rows loss/decode read); prompt
        # rows use the documented full-group segment mean
        bound_t = 7
        arow = torch.arange(T).unsqueeze(0) >= bound_t
        ctx_err = (st.ctx - ctx_o)[arow].abs().max().item()
        check(ctx_err < 1e-3,
              f"ctx == attention blend at answer rows ({ctx_err:.2e})")
        ld = loss_v33(od, Y, Yc, G)
        ls = loss_v33(os_, Y, Yc, G)
    names = ["total", "blend", "gen", "copy"]
    tols = [2e-3, 2e-3, 1e-5, 1e-3]
    for nme, a, b, tol in zip(names[2:], ld[2:], ls[2:], tols[2:]):
        d = abs(float(a) - float(b))
        check(d < tol, f"{nme} CE parity ({d:.2e} < {tol:.0e})")
    # gen/copy CE match tightly on ANY batch; the BLEND term additionally
    # matches row-by-row wherever the blend probability sits above the
    # oracle's own eps floor (p.clamp(min=1e-8) inside DualHeadDecoder) — an
    # untrained head pushes garbage rows below that floor. End-to-end blend
    # parity on a TRAINED checkpoint lives in experiments/v4/m1_sparse_parity.
    with torch.no_grad():
        lo_rows = od["logits"].squeeze(0)          # oracle blended log-probs
        g_s = os_["g"].squeeze(-1)[0]
        pc_r = st.prob_at(Yc)[0]
        lg_y = os_["gen_logits"][0].gather(1, Y.squeeze(0).clamp(min=0).unsqueeze(-1)).squeeze(-1)
        lp_c = torch.full_like(pc_r, float("-inf"))
        nzr = pc_r > 0
        lp_c[nzr] = pc_r[nzr].log()
        import math as _m
        ok_rows = 0
        for t in range(Y.shape[1]):
            y = int(Y[0, t])
            if y == -100 or float(lo_rows[t, y]) <= math.log(1e-8) + 1e-4:
                continue                           # floored by the oracle itself
            gc = float(g_s[t])
            a_ = math.log(max(1 - gc, 1e-12)) + float(lg_y[t])
            b_ = math.log(max(gc, 1e-12)) + float(lp_c[t])
            l_st = -(max(a_, b_) + _m.log(_m.exp(min(a_, b_) - max(a_, b_)) + 1))
            check(abs(l_st - float(lo_rows[t, y])) < 2e-2,
                  f"blend row {t} parity ({l_st:.4f} vs {float(lo_rows[t, y]):.4f})")
            ok_rows += 1
    check(ok_rows >= 4, f"enough above-floor blend rows compared ({ok_rows})")
    p = st.prob_at(Yc)
    check(bool(((p >= 0) & (p <= 1 + 1e-5)).all()), "prob_at in [0,1]")
    check(float(p[0, 9]) > 0.4, f"copy target probability sharp ({float(p[0, 9]):.3f})")


def test_v6_m1c_blend_argmax_bound():
    print("[v6 M1-C: {gen-argmax} ∪ payload scheme == full-vocab brute force]")
    from hmn.recipe import blend_argmax
    torch.manual_seed(11)
    V = 500
    bad = 0
    for trial in range(300):
        gen = torch.log_softmax(torch.randn(V), -1)
        g = float(torch.randint(0, 4, (1,)).float() / 3)   # 0, 1/3, 2/3, 1
        npay = int(torch.randint(0, 9, (1,)))
        if npay:
            pay = torch.randperm(V)[:npay]
            raw = torch.rand(npay) + 0.01
            fr = raw / raw.sum()
        else:
            pay = torch.zeros(0, dtype=torch.long)
            fr = torch.zeros(0)
        got = blend_argmax(gen, g, pay, fr)
        p = (1 - g) * gen.exp()
        if npay:
            p.index_add_(0, pay, g * fr)
        want = int(p.argmax())
        # optimality within FP noise: id flips are only allowed between
        # numerically-equal candidates (knife-edge ties from exp/log rounding)
        if float(p[got]) < float(p[want]) - 1e-9:
            bad += 1
    check(bad == 0, f"300 randomized trials match brute force ({bad} misses)")


def test_reorder_anchors_and_batch():
    print("[v5 seam: reorder anchors force the exact gold payload]")
    tok = Tokenizer.from_file(os.path.join(ROOT, "retop_tokenizer.json"))
    asid = tok.token_to_id(ASSIST)
    ids, asi_pos, i_u = make_reorder_ids(tok, "pkg028", "lib055")
    an, s, r = reorder_anchors(ids, asid, tok)
    # every anchored answer row's forced payload == the next gold token
    ok_rows = 0
    for t in range(asi_pos, len(ids) - 1):
        if an[t] >= 0:
            check(ids[an[t] + 1] == ids[t + 1],
                  f"row {t}: payload(ids[{an[t]}+1]) == gold ids[{t}+1]")
            ok_rows += 1
    check(ok_rows == len(ids) - asi_pos - 1 - 1,
          f"all {ok_rows} non-EOS gold rows are copy-anchored (EOS excluded)")
    check(sum(s) == 3, f"3 runs -> 3 seam rows (got {sum(s)})")
    gold = tok.decode(ids[asi_pos + 1:]).strip()
    check(gold.startswith("deploy lib055") and gold.endswith("pkg028"),
          f"swapped gold decodes correctly ({gold!r})")
    # batch tensors + losses + backward
    X, Y, Yc, G, A, S, Rn = make_reorder_batch(
        tok, [f"pkg{i:03d}" for i in range(40)], [f"lib{i:03d}" for i in range(40, 80)],
        4, seed=1)
    m = HMN3(tok.get_vocab_size(), dim=64, state_dim=8, n_layers=2,
             gate_bias=-1.0, asi_id=tok.token_to_id(ASSIST),
             user_id=tok.token_to_id(USER), seam_addr=True)
    out = m(X, seam_anchor=A)
    _, lb, lg, lc = loss_v33(out, Y, Yc, G)
    lp, ll = seam_losses(out, S, Rn, A)
    (lb + lg + lc + lp + ll).backward()
    n_grad = sum(1 for p in m.parameters() if p.grad is not None)
    check(n_grad > 0, f"seam training step reaches {n_grad} param grads")


def test_decode_seam_mechanics():
    print("[v5 seam: decode_v33 seam loop runs and terminates]")
    tok = Tokenizer.from_file(os.path.join(ROOT, "retop_tokenizer.json"))
    torch.manual_seed(0)
    m = HMN3(tok.get_vocab_size(), dim=48, state_dim=8, n_layers=2,
             gate_bias=-1.0, asi_id=tok.token_to_id(ASSIST),
             user_id=tok.token_to_id(USER), seam_addr=True)
    ids, asi_pos, _ = make_reorder_ids(tok, "pkg007", "lib039")
    prompt = ids[:asi_pos + 1]
    # direct decode call (mechanics); eval_reorders exercised below
    out_txt, gate_avg, _ = decode_v33(m, tok, prompt, max_new=32,
                                      mode="hard", seam=True, pos_eos=True)
    check(isinstance(out_txt, str), f"seam decode returns text ({out_txt!r})")
    acc, gavg = eval_reorders(m, tok, ["pkg007", "pkg008"],
                              ["lib039", "lib040"], seed=0)
    check(0.0 <= acc <= 1.0, f"eval_reorders mechanics OK (acc={acc:.2f})")


def test_skills_recipes_and_coverage():
    print("[v5 M4: skill recipes cover gold exactly (echo off-by-one guard)]")
    from hmn import skills as S
    tok = Tokenizer.from_file(os.path.join(ROOT, "retop_tokenizer.json"))
    asid = tok.token_to_id(ASSIST)
    # rotate recipe: total planned length == gold tail length, anchors valid
    ids, asi_pos, _ands = make_perm_ids(tok, ["fetch pkg028", "deploy lib055"])
    P = S.parse_prompt(ids, asid, tok)
    plan = S.rotate_recipe(P, 1)
    gold_len = len(ids) - (asi_pos + 1) - 1        # strip EOS
    check(sum(L for _, L in plan) == gold_len,
          f"rotate plan covers gold ({sum(L for _, L in plan)} == {gold_len})")
    ok_rows = all(0 <= c < asi_pos and c + 1 < len(ids) for c, _L in plan)
    check(ok_rows, "rotate anchor columns are legal prompt columns")
    # the historical bug: echo must INCLUDE separators (sum(seg_lens) didn't)
    Pe = S.parse_prompt(make_chat_ids(tok, "fetch pkg028 and deploy lib055",
                                       None) if False else
                        make_chat_ids(tok, "fetch pkg028 and deploy lib055"),
                        asid, tok)
    ep = S.echo_recipe(Pe)
    check(sum(L for _, L in ep) == len(Pe["U"]),
          f"echo plan includes separators ({sum(L for _, L in ep)} == "
          f"{len(Pe['U'])})")
    # library ambiguity: same fp, two families -> escalate on hintless match
    lib = S.SkillLibrary()
    lib.add("echo", Pe["fp"], "t1")
    lib.add("rotate", Pe["fp"], "t2")
    entry, status = lib.match(Pe["fp"])
    check(entry is None and status == "ambiguous",
          f"shared-verb fp escalates (status={status!r})")
    entry, status = lib.match(Pe["fp"], family="echo")
    check(status == "hit" and entry["family"] == "echo",
          "hint resolves the ambiguity")


# ---------------------------------------------------------------------------
# v7 tests
# ---------------------------------------------------------------------------

def test_v7_attention_wr_forward():
    print("[v7: HMN3AttentionWR forward + backward]")
    m = HMN3AttentionWR(VOCAB, dim=64, n_layers=2, use_moe=False,
                        gate_bias=-1.0, asi_id=5)
    out = m(rand_ids())
    check("stats" in out and "gen_logits" in out and "g" in out,
          "returns stats/gen_logits/g")
    check(out["gen_logits"].shape == (2, 16, VOCAB), "gen_logits shape")
    check(torch.isfinite(out["gen_logits"]).all(), "gen_logits finite")
    Y = rand_ids()
    Yc = Y.clone()
    Yc[:, 0] = -100
    G = torch.ones_like(Y, dtype=torch.float)
    G[:, 0] = 0.0
    loss, l_blend, l_gen, l_copy = loss_v33(out, Y, Yc, G)
    loss.backward()
    n_grad = sum(1 for p in m.parameters() if p.grad is not None)
    check(n_grad > 0, f"gradients reached {n_grad} params")
    check(torch.isfinite(loss), "loss_v33 finite")


def test_v7_attention_wr_moe():
    print("[v7: HMN3AttentionWR with MoE (SparseConditionalComputeV2)]")
    m = HMN3AttentionWR(VOCAB, dim=64, n_layers=2, use_moe=True,
                        n_experts=16, top_k=2, gate_bias=-1.0, asi_id=5)
    out = m(rand_ids())
    check("stats" in out, "forward with MoE returns stats")
    aux = m.moe_aux_loss()
    check(torch.isfinite(aux) and aux.numel() == 1, f"MoE aux loss scalar ({aux.item():.4f})")
    loss, _, _, _ = loss_v33(out, rand_ids(), rand_ids(), torch.ones(2, 16))
    loss.backward()
    n_grad = sum(1 for p in m.parameters() if p.grad is not None)
    check(n_grad > 0, f"MoE backward reaches {n_grad} params")


def test_v7_sparse_v2():
    print("[v7: SparseConditionalComputeV2 — noisy gates, z-loss]")
    moe = SparseConditionalComputeV2(32, n_experts=16, top_k=2)
    moe.train()
    moe(torch.randn(2, 10, 32))
    check(torch.isfinite(moe.last_aux_loss), "aux loss finite")
    check(torch.isfinite(moe.last_z_loss), "z-loss finite")
    check(moe.last_z_loss.item() >= 0, "z-loss non-negative")


def test_v7_attention_block():
    print("[v7: AttentionBlock forward]")
    blk = AttentionBlock(64, n_heads=2, max_seq_len=256)
    x = torch.randn(2, 16, 64)
    y = blk(x)
    check(y.shape == (2, 16, 64), "output shape matches input")
    check(torch.isfinite(y).all(), "output finite")


def test_v7_rms_norm():
    print("[v7: RMSNorm]")
    norm = RMSNorm(64)
    x = torch.randn(2, 16, 64)
    y = norm(x)
    check(y.shape == x.shape, "shape preserved")
    # RMSNorm normalizes to unit RMS
    rms = y.pow(2).mean(-1).sqrt()
    check(torch.allclose(rms, torch.ones_like(rms), atol=1e-5),
          "output has unit RMS")


def test_v7_config_factory():
    print("[v7: HMNConfig + create_model factory]")
    cfg = HMNConfig.from_preset("attn-small")
    check(cfg.variant == "attention", "attn-small preset variant")
    check(cfg.dim == 64, "attn-small dim")
    m = create_model(cfg, asi_id=5)
    check(isinstance(m, HMN3AttentionWR), "create_model returns HMN3AttentionWR")
    out = m(rand_ids(bs=1, t=8))
    check("stats" in out, "factory model forward works")
    # SSM preset
    cfg2 = HMNConfig.from_preset("cpu-small")
    m2 = create_model(cfg2, asi_id=5)
    check(not isinstance(m2, HMN3AttentionWR), "cpu-small returns HMN3 (SSM)")
    # string shortcut
    m3 = create_model("attn-medium", asi_id=5)
    check(isinstance(m3, HMN3AttentionWR), "string preset works")


def test_v7_config_param_estimate():
    print("[v7: HMNConfig.param_count_estimate branches on variant]")
    cfg_ssm = HMNConfig(dim=96, n_layers=3, variant="ssm")
    cfg_attn = HMNConfig(dim=96, n_layers=3, variant="attention")
    # attention should have more params than SSM for same dim/layers
    check(cfg_attn.param_count_estimate() > cfg_ssm.param_count_estimate(),
          f"attention estimate ({cfg_attn.param_count_estimate()}) > "
          f"SSM estimate ({cfg_ssm.param_count_estimate()})")


def test_v7_decode_with_attention():
    print("[v7: decode_v33 on attention-WR model]")
    tok = Tokenizer.from_file(os.path.join(ROOT, "retop_tokenizer.json"))
    m = HMN3AttentionWR(tok.get_vocab_size(), dim=64, n_layers=2,
                        gate_bias=-1.0, asi_id=tok.token_to_id(ASSIST))
    prompt = make_chat_ids(tok, "pip install pkg042")
    out_txt, gate_avg, n_gen = decode_v33(m, tok, prompt, max_new=16,
                                          boundary_eos=True)
    check(isinstance(out_txt, str), f"decode returns text ({out_txt!r})")
    check(0.0 <= gate_avg <= 1.0, f"gate_avg in [0,1] ({gate_avg:.3f})")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_coupling_reversible()
    test_reversible_autograd_vs_naive()
    test_v3_forward_and_copy()
    test_v3_noreg_and_think()
    test_moe_aux_loss()
    test_recipe_copy_ce_is_manual_logp()
    test_recipe_chat_ids_no_special_split()
    test_recipe_decode_boundary_rule()
    test_recipe_cycle_break_self_pair()
    test_seam_anchor_none_parity()
    test_v6_m1a_index_stats()
    test_v6_m1b_stats_vs_oracle()
    test_v6_m1c_blend_argmax_bound()
    test_reorder_anchors_and_batch()
    test_decode_seam_mechanics()
    test_skills_recipes_and_coverage()
    # v7 tests
    test_v7_attention_wr_forward()
    test_v7_attention_wr_moe()
    test_v7_sparse_v2()
    test_v7_attention_block()
    test_v7_rms_norm()
    test_v7_config_factory()
    test_v7_config_param_estimate()
    test_v7_decode_with_attention()
    print("\nALL TESTS PASSED")
