"""v6 M5 — Packed sequence verification."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import torch
from hmn.v3 import HMN3
from hmn.v2 import HelixCouplingBlock, ReversibleFunction
from hmn.packing import (pack_sequences, unpack_outputs, doc_masked_padding,
                          doc_position_ids)
from hmn.recipe import (loss_v33, make_chat_ids, make_chat_targets,
                        seed_guardrail, ASSIST, EOS)
from tokenizers import Tokenizer

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
VOCAB = 3190

def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")

def test_pack_unpack_roundtrip():
    print("[M5-1: pack/unpack round-trip]")
    docs = [[10, 20, 30, 40], [50, 60], [70, 80, 90]]
    bins = pack_sequences(docs, max_len=100)
    check(len(bins) == 1, f"1 bin (got {len(bins)})")
    b = bins[0]
    check(b.ids.numel() == 9, "total 9 tokens")
    check(b.lens.tolist() == [4, 2, 3], "doc lens correct")
    unpacked = unpack_outputs(b.ids, b.lens)
    check(len(unpacked) == 3, "3 docs unpacked")
    check(unpacked[0].tolist() == [10, 20, 30, 40], "doc 0")
    check(unpacked[1].tolist() == [50, 60], "doc 1")
    check(unpacked[2].tolist() == [70, 80, 90], "doc 2")

def test_position_ids_reset():
    print("[M5-2: position_ids reset at doc boundaries]")
    docs = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    b = pack_sequences(docs, max_len=100)[0]
    pos = b.position_ids
    check(pos[:3].tolist() == [0, 1, 2], "doc 0 pos")
    check(pos[3:5].tolist() == [0, 1], "doc 1 pos reset")
    check(pos[5:9].tolist() == [0, 1, 2, 3], "doc 2 pos reset")
    check(b.doc_index[:3].tolist() == [0, 0, 0], "doc 0 idx")
    check(b.doc_index[3:5].tolist() == [1, 1], "doc 1 idx")
    check(b.doc_index[5:9].tolist() == [2, 2, 2, 2], "doc 2 idx")

def test_doc_masked_padding():
    print("[M5-3: doc_masked_padding]")
    docs = [[10, 20, 30], [40, 50]]
    pad = doc_masked_padding(docs, max_len=5, pad_id=1)
    check(pad.ids.shape == (2, 5), "ids shape")
    check(pad.attn_mask.shape == (2, 5), "mask shape")
    check(pad.lens.tolist() == [3, 2], "lens")
    check(pad.ids[0, :3].tolist() == [10, 20, 30], "doc 0 content")
    check(pad.ids[0, 3:].tolist() == [1, 1], "doc 0 padded")
    check(pad.attn_mask[0, :3].tolist() == [1, 1, 1], "doc 0 mask")
    check(pad.attn_mask[0, 3:].tolist() == [0, 0], "doc 0 pad mask")

def test_packed_forward():
    print("[M5-4: packed forward output]")
    seed_guardrail(42)
    tok = Tokenizer.from_file(os.path.join(ROOT, "retop_tokenizer.json"))
    asid = tok.token_to_id(ASSIST)
    m = HMN3(VOCAB, dim=48, state_dim=8, n_layers=2, use_moe=False,
             gate_bias=-1.0, asi_id=asid)
    m.eval()
    docs = [make_chat_ids(tok, f"pip install pkg{i:03d}", f"pip install pkg{i:03d}")
            for i in range(3)]
    b = pack_sequences(docs, max_len=200)[0]
    X = b.ids.unsqueeze(0)
    with torch.no_grad():
        out = m(X)
    T_total = sum(len(d) for d in docs)
    check(out["gen_logits"].shape[1] == T_total, f"output {T_total}")
    check(torch.isfinite(out["gen_logits"]).all(), "finite")

def test_reversible_backward():
    print("[M5-5: ReversibleFunction backward]")
    torch.manual_seed(42)
    blk = HelixCouplingBlock(32, 4)
    x = torch.randn(1, 12, 32).requires_grad_(True)
    y = ReversibleFunction.apply(x, [blk])
    y.square().mean().backward()
    check(torch.isfinite(x.grad).all(), "input grad finite")

if __name__ == "__main__":
    test_pack_unpack_roundtrip()
    test_position_ids_reset()
    test_doc_masked_padding()
    test_packed_forward()
    test_reversible_backward()
    print("\nM5 ALL PASSED")
