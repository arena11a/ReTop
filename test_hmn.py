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

from hmn import (
    HMN,
    HMN_Option1,
    HMN3,
    HMN3_NoReg,
    HelixCouplingBlock,
    ReversibleFunction,
    SparseConditionalCompute,
)

VOCAB = 3190
DIM = 64


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def rand_ids(bs=2, t=16):
    return torch.randint(0, VOCAB, (bs, t))


def test_v2_forward_backward():
    print("[HMN v2 forward/backward]")
    m = HMN(VOCAB, DIM, 8, 2, n_experts=16, top_k=2, n_mem_cells=8, mem_top_k=4)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = rand_ids()
    y = rand_ids()
    lossf = nn.CrossEntropyLoss()
    logits = m(x)
    check(logits.shape == (2, 16, VOCAB), f"logits shape {tuple(logits.shape)}")
    check(torch.isfinite(logits).all(), "logits finite")
    loss = lossf(logits.reshape(-1, VOCAB), y.reshape(-1)) + m.moe_aux_loss()
    loss.backward()
    n_grad = sum(1 for p in m.parameters() if p.grad is not None)
    check(n_grad > 0, f"gradients reached {n_grad}/{len(list(m.parameters()))} params")
    g0 = loss.item()
    for _ in range(3):
        opt.zero_grad()
        l = lossf(m(x).reshape(-1, VOCAB), y.reshape(-1)) + m.moe_aux_loss()
        l.backward()
        opt.step()
    check(l.item() < g0, f"loss decreased over 3 steps ({g0:.3f} -> {l.item():.3f})")


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


def test_option1_forward():
    print("[HMN_Option1 forward]")
    m = HMN_Option1(VOCAB, dim=64, state_dim=8, n_layers=2, n_mem_cells=64,
                    mem_top_k=4, beta_init=30.0, usage_decay=True,
                    combined=True, exempt_combined=True, n_pairs=4)
    logits = m(rand_ids())
    check(logits.shape == (2, 16, VOCAB), f"single-head logits {tuple(logits.shape)}")
    mh = HMN_Option1(VOCAB, dim=64, state_dim=8, n_layers=2, n_mem_cells=64,
                     mem_top_k=4, use_multi_head=True, n_digits=2)
    heads = mh(rand_ids())
    check(isinstance(heads, list) and len(heads) == 2, "multi-head returns 2 digit heads")
    check(heads[0].shape == (2, 16, 10), f"digit head shape {tuple(heads[0].shape)}")


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
    # backward incl. aux copy loss
    lossf = nn.CrossEntropyLoss()
    loss = lossf(out["logits"].reshape(-1, VOCAB), rand_ids().reshape(-1)) \
        + lossf(out["copy_dist"].reshape(-1, VOCAB), rand_ids().reshape(-1))
    loss.backward()
    n_grad = sum(1 for p in m.parameters() if p.grad is not None)
    check(n_grad > 0, f"gradients reached {n_grad} params")


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
    m = HMN(VOCAB, DIM, 8, 2, n_experts=16, top_k=2, n_mem_cells=8, mem_top_k=4)
    a = m.moe_aux_loss()
    check(a.numel() == 1, "HMN.moe_aux_loss() scalar")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_v2_forward_backward()
    test_coupling_reversible()
    test_reversible_autograd_vs_naive()
    test_option1_forward()
    test_v3_forward_and_copy()
    test_v3_noreg_and_think()
    test_moe_aux_loss()
    print("\nALL TESTS PASSED")
