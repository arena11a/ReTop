"""Unit tests for gen_recall.py (Task 2) — verify format BEFORE training (fail fast).

Run: python test_recall.py
"""

import json
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gen_recall import (
    compute_stats,
    iter_records,
    make_record,
    sample_recall_sample,
    stream_recall,
    write_chunked,
)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def parse_user_text(user_text):
    """v2 format: 'START k1 v1 k2 v2 ... ? kQ' -> pairs, query_key."""
    parts = [p for p in user_text.split() if p != "START"]
    pairs = []
    query = None
    i = 0
    while i < len(parts):
        if parts[i] == "?":
            query = int(parts[i + 1])
            i += 2
        else:
            pairs.append((int(parts[i]), int(parts[i + 1])))
            i += 2
    return pairs, query


def test_sample_shape():
    print("[sample shape]")
    rng = random.Random(0)
    for _ in range(5000):
        pairs, qk, pool = sample_recall_sample(rng)
        check(3 <= len(pairs) <= 15, "n_pairs in [3,15]")
        check(20 <= pool <= 200, "pool_size in [20,200]")
        keys = [k for k, _ in pairs]
        check(len(set(keys)) == len(keys), "keys distinct within sequence")
        check(qk in keys, "query key present")
        check(qk != keys[-1], "query is NEVER last-seen key (recency 0%)")


def test_make_record_format():
    print("[record format]")
    pairs = [(3, 42), (7, 15), (9, 88)]
    rec = make_record(pairs, 7, pool_size=50)
    check(set(rec.keys()) == {"messages", "meta"}, "record keys")
    msgs = rec["messages"]
    check(msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant", "roles")
    pairs_parsed, qk = parse_user_text(msgs[0]["content"])
    check(pairs_parsed == pairs, "user text roundtrips to pairs")
    check(qk == 7, "query key parsed")
    check(msgs[1]["content"] == "15", "answer == value of queried key")
    m = rec["meta"]
    check(m["n_pairs"] == 3 and m["pool_size"] == 50, "meta n_pairs/pool_size")
    check(m["query_pos"] == 1 and m["query_frac"] == round(1 / 3, 4), "meta query pos")


def test_stream_and_chunking():
    print("[stream + chunking]")
    with tempfile.TemporaryDirectory() as tmp:
        files, n = write_chunked(
            os.path.join(tmp, "train"),
            stream_recall(25000, random.Random(1)), chunk_size=10000)
        check(len(files) == 3, "25k -> 3 chunk files")
        sizes = [sum(1 for _ in open(os.path.join(tmp, "train", f))) for f in sorted(files)]
        check(sizes == [10000, 10000, 5000], f"chunk sizes {sizes}")
        check(n == 25000, "written count")


def test_stats_and_recency_zero():
    print("[stats + recency 0%]")
    with tempfile.TemporaryDirectory() as tmp:
        write_chunked(os.path.join(tmp, "train"),
                      stream_recall(3000, random.Random(2)), chunk_size=1000)
        s = compute_stats(os.path.join(tmp, "train"))
        check(s["n"] == 3000, "n counted")
        check(s["recency_violations"] == 0, "ZERO recency violations")
        check(s["n_pairs_min"] >= 3 and s["n_pairs_max"] <= 15, "n_pairs within range")
        check(s["pool_min"] >= 20 and s["pool_max"] <= 200, "pool within range")
        check(0.99 <= sum(s["query_frac_hist"].values()) <= 1.01, "query_frac_hist sums to 1")
        # query_frac bins should span more than just the last positions
        check(len(s["query_frac_hist"]) >= 3, f"query positions spread (bins={len(s['query_frac_hist'])})")


def test_deterministic():
    print("[deterministic]")
    a = list(stream_recall(500, random.Random(7)))
    b = list(stream_recall(500, random.Random(7)))
    check(a == b, "same seed -> identical records")


def test_every_answer_correct():
    print("[answer correctness on full stream]")
    rng = random.Random(3)
    bad = 0
    for rec in stream_recall(2000, rng):
        pairs, qk = parse_user_text(rec["messages"][0]["content"])
        expected = dict(pairs)[qk]
        if rec["messages"][1]["content"] != str(expected):
            bad += 1
    check(bad == 0, "all answers match their queried value")


if __name__ == "__main__":
    test_sample_shape()
    test_make_record_format()
    test_stream_and_chunking()
    test_stats_and_recency_zero()
    test_deterministic()
    test_every_answer_correct()
    print("\nALL TESTS PASSED")
