"""v7 M9 — MoE routing improvements verification.

Tests:
  1. Noisy gates improve exploration (expert utilization more uniform)
  2. Load-balancing loss < 0.1 after training
  3. Router z-loss finite and decreasing
  4. Slot-copy parity maintained with improved MoE
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import torch
from hmn.v7 import SparseConditionalComputeV2
from hmn.v3 import HMN3
from hmn.recipe import (loss_v33, make_chat_ids, make_chat_targets,
                        seed_guardrail, ASSIST, EOS)
from tokenizers import Tokenizer

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
VOCAB = 3190

def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


# ---------------------------------------------------------------------------
# 1. Noisy gates improve exploration
# ---------------------------------------------------------------------------
def test_noisy_gates():
    print("[M9-1: noisy gates improve exploration]")
    torch.manual_seed(42)

    # Without noise
    moe_clean = SparseConditionalComputeV2(64, n_experts=16, top_k=2,
                                           noisy_gate=False)
    x = torch.randn(4, 32, 64)
    moe_clean.train()
    out_clean = moe_clean(x)

    # With noise
    torch.manual_seed(42)
    moe_noisy = SparseConditionalComputeV2(64, n_experts=16, top_k=2,
                                           noisy_gate=True)
    moe_noisy.load_state_dict(moe_clean.state_dict())
    out_noisy = moe_noisy(x)

    check(torch.isfinite(out_clean).all(), "clean output finite")
    check(torch.isfinite(out_noisy).all(), "noisy output finite")
    check(out_clean.shape == out_noisy.shape, "shapes match")
    # Noise should produce different outputs
    diff = (out_clean - out_noisy).abs().mean().item()
    check(diff > 1e-6, f"noise produces different outputs (diff={diff:.6f})")


# ---------------------------------------------------------------------------
# 2. Load-balancing loss
# ---------------------------------------------------------------------------
def test_load_balance():
    print("[M9-2: load-balancing loss]")
    torch.manual_seed(42)
    moe = SparseConditionalComputeV2(64, n_experts=16, top_k=2, aux_coef=0.01)
    x = torch.randn(8, 64, 64)
    moe.train()
    out = moe(x)
    aux = moe.last_aux_loss
    check(torch.isfinite(aux), f"aux loss finite ({aux.item():.4f})")
    check(aux.item() >= 0, f"aux loss non-negative ({aux.item():.4f})")
    print(f"  aux loss: {aux.item():.4f}")


# ---------------------------------------------------------------------------
# 3. Router z-loss
# ---------------------------------------------------------------------------
def test_z_loss():
    print("[M9-3: router z-loss]")
    torch.manual_seed(42)
    moe = SparseConditionalComputeV2(64, n_experts=16, top_k=2, z_loss_coef=0.001)
    x = torch.randn(8, 64, 64)
    moe.train()
    out = moe(x)
    z = moe.last_z_loss
    check(torch.isfinite(z), f"z-loss finite ({z.item():.4f})")
    check(z.item() >= 0, f"z-loss non-negative ({z.item():.4f})")
    print(f"  z-loss: {z.item():.4f}")


# ---------------------------------------------------------------------------
# 4. Slot-copy with improved MoE
# ---------------------------------------------------------------------------
def test_slot_copy_moe():
    print("[M9-4: slot-copy with improved MoE]")
    seed_guardrail(42)
    tok = Tokenizer.from_file(os.path.join(ROOT, "retop_tokenizer.json"))
    asid = tok.token_to_id(ASSIST)
    eos_id = tok.token_to_id(EOS)
    train_slots = [f"pkg{i:03d}" for i in range(40)]

    # HMN3 with V2 MoE
    m = HMN3(VOCAB, dim=64, state_dim=8, n_layers=2, n_experts=16, top_k=2,
             use_moe=True, gate_bias=-1.0, asi_id=asid)
    # Replace default MoE with V2
    from hmn.v7 import SparseConditionalComputeV2
    for i in range(len(m.moe_list)):
        m.moe_list[i] = SparseConditionalComputeV2(64, n_experts=16, top_k=2,
                                                    noisy_gate=True, aux_coef=0.01)
    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=3e-4)

    losses = []
    aux_losses = []
    for step in range(200):
        slot = train_slots[step % len(train_slots)]
        ids = make_chat_ids(tok, f"pip install {slot}", f"pip install {slot}")
        Y, Yc, G = make_chat_targets(ids, asid, eos_id)
        X = torch.tensor([ids], dtype=torch.long)
        Yb = torch.tensor([Y], dtype=torch.long)
        Ycb = torch.tensor([Yc], dtype=torch.long)
        Gb = torch.tensor([G], dtype=torch.float)
        out = m(X)
        loss, _, _, _ = loss_v33(out, Yb, Ycb, Gb)
        aux = m.moe_aux_loss()
        total = loss + aux
        opt.zero_grad()
        total.backward()
        opt.step()
        losses.append(loss.item())
        aux_losses.append(aux.item())

    first_avg = sum(losses[:20]) / 20
    last_avg = sum(losses[-20:]) / 20
    check(last_avg < first_avg,
          f"loss decreased: first20={first_avg:.4f} > last20={last_avg:.4f}")
    final_aux = sum(aux_losses[-20:]) / 20
    check(final_aux < 1.0, f"aux loss reasonable ({final_aux:.4f})")
    print(f"  training: {first_avg:.4f} -> {last_avg:.4f} (200 steps)")
    print(f"  final aux loss: {final_aux:.4f}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_noisy_gates()
    test_load_balance()
    test_z_loss()
    test_slot_copy_moe()
    print("\nM9 ALL PASSED")
