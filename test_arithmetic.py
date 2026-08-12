"""Unit tests for gen_arithmetic.py — verify format BEFORE training (fail fast).

Run: python test_arithmetic.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gen_arithmetic import (
    collect_keys,
    compute_stats,
    enumerate_stage,
    expr_from_meta,
    gen_stage_file,
    key_from_meta,
    sampled_stage_d,
    split_keys,
)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def test_stage_enumeration():
    print("[enumeration]")
    for stage, size in [("A", 55), ("C", 55)]:
        metas = enumerate_stage(stage)
        check(len(metas) == size, f"stage {stage} space size {size} (got {len(metas)})")
        # verify answers
        for m in metas:
            ans = {"+": lambda a, b: a + b, "*": lambda a, b: a * b}[m["op"]](m["a"], m["b"])
            check(m["ans"] == ans, f"{m['a']}{m['op']}{m['b']} answer correct")
    b = enumerate_stage("B")
    check(len(b) == 8190, f"stage B space size 8190 (got {len(b)})")
    for m in b:
        if m["op"] == "+":
            check(m["a"] <= m["b"], "B add normalized a<=b")
            check(m["ans"] == m["a"] + m["b"], "B add answer")
        else:
            check(m["a"] >= m["b"] and m["ans"] == m["a"] - m["b"], "B sub answer")


def test_split_disjoint():
    print("[split disjoint]")
    for seed in [0, 1, 42]:
        metas = enumerate_stage("B")
        tr, va = split_keys(metas, 1000, seed)
        check(len(tr) + len(va) == 8190, f"split covers full space (seed {seed})")
        tr_keys = {key_from_meta(m) for m in tr}
        va_keys = {key_from_meta(m) for m in va}
        check(len(tr_keys & va_keys) == 0, f"train/val disjoint (seed {seed})")
        check(len(va) == 1000, f"val has exactly val_count (seed {seed})")


def test_commutative_key():
    print("[commutative key]")
    k1 = key_from_meta({"op": "+", "a": 5, "b": 7, "ans": 12})
    k2 = key_from_meta({"op": "+", "a": 7, "b": 5, "ans": 12})
    check(k1 == k2, "5+7 and 7+5 share dedup key")
    k3 = key_from_meta({"op": "-", "a": 7, "b": 5, "ans": 2})
    k4 = key_from_meta({"op": "-", "a": 5, "b": 7, "ans": -2})
    check(k3 != k4, "subtraction is order-sensitive")


def test_full_generation_roundtrip():
    print("[full gen roundtrip]")
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "stageB")
        tr_s, va_s, tr_keys, va_keys = gen_stage_file("B", out, 20000, 1000, seed=7)
        check(tr_s["n"] == 20000, f"train count 20000 (got {tr_s['n']})")
        check(va_s["n"] == 1000, f"val count 1000 (got {va_s['n']})")
        check(len(tr_keys & va_keys) == 0, "generated files disjoint")
        check(set(tr_s["op_dist"]) == {"+", "-"}, "train has both + and -")
        check(abs(sum(tr_s["op_dist"].values()) - 1) < 0.01, "op_dist sums to ~1")
        # format check on a real file
        with open(os.path.join(out, "train", "chunk_00000.jsonl"), encoding="utf-8") as f:
            line = f.readline().strip()
        import json
        rec = json.loads(line)
        check(set(rec.keys()) == {"messages", "meta"}, "record keys messages+meta")
        msgs = rec["messages"]
        check(msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant", "roles correct")
        m = rec["meta"]
        check(msgs[0]["content"] == expr_from_meta(m), "user content == expr")
        check(msgs[1]["content"] == str(m["ans"]), "assistant content == answer")


def test_chunking():
    print("[chunking]")
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "stageA")
        gen_stage_file("A", out, 25000, 10, seed=7)
        files = sorted(os.listdir(os.path.join(out, "train")))
        check(len(files) == 3, f"25k train -> 3 chunk files (got {len(files)})")
        sizes = []
        for fn in files:
            with open(os.path.join(out, "train", fn), encoding="utf-8") as f:
                sizes.append(sum(1 for _ in f))
        check(sizes == [10000, 10000, 5000], f"chunk sizes {sizes}")
        stats = compute_stats(os.path.join(out, "val"))
        check(stats["n"] == 10, f"val size 10 (got {stats['n']})")


def test_val_too_large_raises():
    print("[val bound]")
    try:
        metas = enumerate_stage("A")
        split_keys(metas, 56, 0)
        check(False, "should raise when val_count > space")
    except ValueError:
        check(True, "ValueError raised for val_count > space size")


def test_stage_d():
    print("[stage D sampler]")
    import random
    rng = random.Random(0)
    for _ in range(1000):
        m = sampled_stage_d(rng)
        check(10 <= m["a"] <= 999 and 10 <= m["b"] <= 999, "D operands 2-3 digit")
        ans = {"+": lambda a, b: a + b, "-": lambda a, b: a - b,
               "*": lambda a, b: a * b}[m["op"]](m["a"], m["b"])
        check(m["ans"] == ans, "D answer correct")
        if m["op"] == "-":
            check(m["a"] >= m["b"], "D sub non-negative")


if __name__ == "__main__":
    test_stage_enumeration()
    test_split_disjoint()
    test_commutative_key()
    test_full_generation_roundtrip()
    test_chunking()
    test_val_too_large_raises()
    test_stage_d()
    print("\nALL TESTS PASSED")
