"""Model-level tests for the HMN architecture (v2 + v3).

Covers the pieces the data-generator tests don't:
  - forward/backward on every model class
  - reversibility of the coupling backbone (reconstruction identity)
  - gradient correctness of ReversibleFunction vs a naive forward
  - MoE aux loss / v3 copy path / latent thinking buffer

Run: python test_hmn.py        (CPU, a few seconds)
"""
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
    check("logits" in out and "g" in out and "copy_dist" in out, "returns logits/g/copy_dist")
    check(out["logits"].shape == (2, 16, VOCAB), "logits shape")
    check(out["copy_dist"].shape == (2, 16, VOCAB), "copy_dist shape")
    check(torch.isfinite(out["logits"]).all(), "logits finite")
    # copy_dist is a probability distribution over vocab
    check(torch.allclose(out["copy_dist"].sum(-1), torch.ones(2, 16), atol=1e-3),
          "copy_dist sums to 1 per position")
    # backward incl. aux copy loss (via the v3.3 recipe, NOT CE-on-probs)
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


def test_v3_noreg_and_think():
    print("[HMN3_NoReg + HMN3 with thinking buffer]")
    nr = HMN3_NoReg(VOCAB, dim=96, state_dim=8, n_layers=3)
    logits = nr(rand_ids())
    check(logits.shape == (2, 16, VOCAB), "NoReg logits shape")
    t = HMN3(VOCAB, dim=96, state_dim=8, n_layers=3, use_think=True, k_max=4,
             use_moe=False, gate_bias=-2.0, asi_id=5)
    out = t(rand_ids())
    check(torch.isfinite(out["logits"]).all(), "thinking-buffer logits finite")


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
    check(torch.equal(o1["logits"], o2["logits"]), "logits identical")
    check(torch.equal(o1["copy_dist"], o2["copy_dist"]), "copy_dist identical")
    check(torch.equal(o1["g"], o2["g"]), "gate identical")


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
    test_reorder_anchors_and_batch()
    test_decode_seam_mechanics()
    test_skills_recipes_and_coverage()
    print("\nALL TESTS PASSED")
