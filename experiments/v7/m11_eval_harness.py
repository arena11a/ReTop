"""v7 M11 — Evaluation harness verification.

Tests:
  1. EvalReport save/load round-trip
  2. eval_checkpoint runs on a trained model
  3. compare_reports shows diffs
"""
import os, sys, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import torch
from hmn.eval_harness import EvalReport, EvalResult, eval_checkpoint, compare_reports
from hmn.v3 import HMN3
from hmn.recipe import (loss_v33, make_chat_ids, make_chat_targets,
                        seed_guardrail, ASSIST, EOS)
from tokenizers import Tokenizer

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")

def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def test_report_roundtrip():
    print("[M11-1: EvalReport save/load round-trip]")
    results = [
        EvalResult("slot_copy_unseen", 0.75, 0.8, 40, 5.2),
        EvalResult("chain_unseen", 0.5, 0.7, 20, 3.1),
    ]
    report = EvalReport(model_path="/tmp/test.ckpt", checkpoint_step=100,
                        results=results, total_elapsed_s=8.3)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name
    report.save(path)
    loaded = EvalReport.load(path)
    check(loaded.model_path == report.model_path, "model_path match")
    check(len(loaded.results) == 2, "2 results")
    check(loaded.results[0].accuracy == 0.75, "accuracy preserved")
    os.unlink(path)


def test_eval_checkpoint():
    print("[M11-2: eval_checkpoint on trained model]")
    seed_guardrail(42)
    tok = Tokenizer.from_file(os.path.join(ROOT, "retop_tokenizer.json"))
    asid = tok.token_to_id(ASSIST)
    eos_id = tok.token_to_id(EOS)

    # Train a small model
    m = HMN3(3190, dim=48, state_dim=8, n_layers=2, use_moe=False,
             gate_bias=-1.0, asi_id=asid)
    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=3e-4)
    for step in range(100):
        ids = make_chat_ids(tok, f"pip install pkg{step:03d}", f"pip install pkg{step:03d}")
        Y, Yc, G = make_chat_targets(ids, asid, eos_id)
        X = torch.tensor([ids], dtype=torch.long)
        Yb = torch.tensor([Y], dtype=torch.long)
        Ycb = torch.tensor([Yc], dtype=torch.long)
        Gb = torch.tensor([G], dtype=torch.float)
        out = m(X)
        loss, _, _, _ = loss_v33(out, Yb, Ycb, Gb)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Save checkpoint
    path = "/tmp/opencode/m11_test_ckpt.pt"
    torch.save({"model": m.state_dict(), "step": 100}, path)

    # Run eval harness
    report = eval_checkpoint(path, n_slot=5, n_chain=0, n_reorder=0)
    check(len(report.results) >= 1, f"got {len(report.results)} results")
    r = report.results[0]
    check(r.task == "slot_copy_unseen", f"task is {r.task}")
    check(0.0 <= r.accuracy <= 1.0, f"accuracy in [0,1] ({r.accuracy})")
    print(f"  slot_copy: {r.accuracy:.3f} (gate={r.gate_avg:.3f})")
    os.unlink(path)


def test_compare():
    print("[M11-3: compare_reports]")
    a = EvalReport("a.ckpt", 100, [
        EvalResult("slot_copy", 0.5, 0.7, 40, 1.0),
    ], 1.0)
    b = EvalReport("b.ckpt", 200, [
        EvalResult("slot_copy", 0.8, 0.9, 40, 1.0),
    ], 1.0)
    diffs = compare_reports(a, b)
    check("slot_copy" in diffs, "slot_copy in diffs")
    check(abs(diffs["slot_copy"]["diff"] - 0.3) < 1e-9, f"diff is 0.3 ({diffs['slot_copy']['diff']})")


if __name__ == "__main__":
    test_report_roundtrip()
    test_eval_checkpoint()
    test_compare()
    print("\nM11 ALL PASSED")
