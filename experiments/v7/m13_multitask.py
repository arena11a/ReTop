"""v7 M13 — Multi-task training verification.

Tests:
  1. Joint training on two distinct copy tasks (install/deploy)
  2. No single task drops > 10% vs single-task baseline
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import torch
from hmn.v3 import HMN3
from hmn.recipe import (loss_v33, make_chat_ids, make_chat_targets,
                        eval_slots, seed_guardrail, ASSIST, EOS)
from tokenizers import Tokenizer

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
VOCAB = 3190


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def train_copy(model, tok, asid, eos_id, template, slots, n_steps, lr=3e-4):
    """Train copy on a specific template."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(n_steps):
        slot = slots[step % len(slots)]
        ids = make_chat_ids(tok, template.format(slot), template.format(slot))
        Y, Yc, G = make_chat_targets(ids, asid, eos_id)
        X = torch.tensor([ids], dtype=torch.long)
        Yb = torch.tensor([Y], dtype=torch.long)
        Ycb = torch.tensor([Yc], dtype=torch.long)
        Gb = torch.tensor([G], dtype=torch.float)
        out = model(X)
        loss, _, _, _ = loss_v33(out, Yb, Ycb, Gb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()


def test_multi_task():
    print("[M13: multi-task two copy templates]")
    seed_guardrail(42)
    tok = Tokenizer.from_file(os.path.join(ROOT, "retop_tokenizer.json"))
    asid = tok.token_to_id(ASSIST)
    eos_id = tok.token_to_id(EOS)
    slots_a = [f"pkg{i:03d}" for i in range(20)]
    slots_b = [f"lib{i:03d}" for i in range(20)]
    val_a = [f"pkg{i:03d}" for i in range(20)]
    val_b = [f"lib{i:03d}" for i in range(20)]

    # Single-task baselines
    seed_guardrail(42)
    model_a = HMN3(VOCAB, dim=48, state_dim=8, n_layers=2, use_moe=False,
                   gate_bias=-1.0, asi_id=asid)
    train_copy(model_a, tok, asid, eos_id, "pip install {}", slots_a, 300)
    acc_a, _, _ = eval_slots(model_a, tok, val_a, mode="blend", boundary_eos=True)

    seed_guardrail(42)
    model_b = HMN3(VOCAB, dim=48, state_dim=8, n_layers=2, use_moe=False,
                   gate_bias=-1.0, asi_id=asid)
    train_copy(model_b, tok, asid, eos_id, "deploy {}", slots_b, 300)
    acc_b, _, _ = eval_slots(model_b, tok, val_b, mode="blend", boundary_eos=True)

    print(f"  single-task A: {acc_a:.3f}")
    print(f"  single-task B: {acc_b:.3f}")

    # Multi-task (alternating)
    seed_guardrail(42)
    model_m = HMN3(VOCAB, dim=48, state_dim=8, n_layers=2, use_moe=False,
                   gate_bias=-1.0, asi_id=asid)
    model_m.train()
    opt = torch.optim.Adam(model_m.parameters(), lr=3e-4)
    for step in range(300):
        if step % 2 == 0:
            slot = slots_a[step % len(slots_a)]
            ids = make_chat_ids(tok, f"pip install {slot}", f"pip install {slot}")
        else:
            slot = slots_b[step % len(slots_b)]
            ids = make_chat_ids(tok, f"deploy {slot}", f"deploy {slot}")
        Y, Yc, G = make_chat_targets(ids, asid, eos_id)
        X = torch.tensor([ids], dtype=torch.long)
        Yb = torch.tensor([Y], dtype=torch.long)
        Ycb = torch.tensor([Yc], dtype=torch.long)
        Gb = torch.tensor([G], dtype=torch.float)
        out = model_m(X)
        loss, _, _, _ = loss_v33(out, Yb, Ycb, Gb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model_m.eval()

    ma_acc, _, _ = eval_slots(model_m, tok, val_a, mode="blend", boundary_eos=True)
    mb_acc, _, _ = eval_slots(model_m, tok, val_b, mode="blend", boundary_eos=True)
    print(f"  multi-task A: {ma_acc:.3f}")
    print(f"  multi-task B: {mb_acc:.3f}")

    # Negative transfer check
    if acc_a > 0:
        drop_a = (acc_a - ma_acc) / acc_a
        print(f"  A drop: {drop_a:.1%}")
        check(drop_a < 0.10, f"A drop < 10% ({drop_a:.1%})")
    if acc_b > 0:
        drop_b = (acc_b - mb_acc) / acc_b
        print(f"  B drop: {drop_b:.1%}")
        check(drop_b < 0.10, f"B drop < 10% ({drop_b:.1%})")


if __name__ == "__main__":
    test_multi_task()
    print("\nM13 ALL PASSED")
