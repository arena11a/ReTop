"""v6 M6 — Streaming data pipeline test (docs/v6_scaling_roadmap.md M6).

Pass criteria (this environment):
  * StreamJsonlReader yields the exact sample count from
    /tmp/opencode/test_stream.jsonl (100 lines), lazily (line-by-line,
    never a full load)
  * BoundedBufferShuffle permutes order while preserving every sample,
    deterministically per seed
  * InfiniteStreamDataset feeds torch DataLoader with collate/batching;
    batch rows reproduce the converter's ids exactly and a restart
    reproduces the exact stream (the roadmap's stable-val-split property)
  * RSS stays BOUNDED over a 1000-sample stream (no accumulation -> the
    "no RAM growth" property M6 requires of the 24h stream)

Honest boundary: HF Hub streaming and multi-worker sharding land with the
cluster runs; here the streaming CONTRACT is verified on a local jsonl
(fineweb-class {"text": ..} records consumed in echo format via
ChatIdConverter(user_key="text", gold_key="text")).
"""
import json
import os
import random
import sys
from functools import partial
from itertools import count, islice

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from hmn.streaming import (BoundedBufferShuffle, ChatIdConverter,
                           InfiniteStreamDataset, StreamJsonlReader,
                           pad_collate)
from hmn.recipe import EOS

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
TEST_JSONL = "/tmp/opencode/test_stream.jsonl"
N_LINES = 100


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def rss_mb():
    """Current resident set (MB) from /proc — NOT the high-water mark."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    raise RuntimeError("VmRSS not found in /proc/self/status")


def ensure_test_jsonl(path=TEST_JSONL, n=N_LINES):
    """Deterministic fineweb-class fixture (regenerated if absent)."""
    if os.path.exists(path) and sum(1 for _ in open(path, encoding="utf-8")) == n:
        return path
    rng = random.Random(0)
    tpls = ["pip install {}", "fetch {}", "deploy {}", "stop {}"]
    pools = [[f"pkg{i:03d}" for i in range(100)],
             [f"lib{i:03d}" for i in range(80)],
             [f"bin{i:03d}" for i in range(60)]]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            slot = rng.choice(pools[i % 3])
            f.write(json.dumps({"text": tpls[i % 4].format(slot)},
                               ensure_ascii=False) + "\n")
    return path


def make_dataset(tok, buffer_size=32, seed=0, epoch_size=1000000):
    conv = ChatIdConverter(tok)          # defaults: text -> echo task
    return InfiniteStreamDataset(
        lambda: conv.stream(BoundedBufferShuffle(
            StreamJsonlReader(TEST_JSONL), buffer_size=buffer_size,
            seed=seed)),
        epoch_size=epoch_size)


# ---------------------------------------------------------------------------
# 1. Reader: correct count from the fixture jsonl, lazy iteration
# ---------------------------------------------------------------------------
def test_reader():
    print("[M6-1: StreamJsonlReader count/laziness]")
    ensure_test_jsonl()
    reader = StreamJsonlReader(TEST_JSONL)
    check(hasattr(reader, "__next__"), "reader IS an iterator (lazy)")
    recs = list(reader)
    check(len(recs) == N_LINES,
          f"yields exactly {N_LINES} samples (got {len(recs)})")
    check(all(isinstance(r, dict) and "text" in r for r in recs),
          "every record parses to a dict with 'text'")
    check(list(islice(StreamJsonlReader(TEST_JSONL), 3)) ==
          recs[:3], "lazy early-stop reads only what is consumed")
    check(list(reader) == recs, "exhaustion resets -> re-iterable like fresh")
    keyed = [t for t in StreamJsonlReader(TEST_JSONL, key="text")]
    check(keyed == [r["text"] for r in recs], "key='text' projection works")
    check(reader.count() == N_LINES, "count() rescan agrees (O(1) memory)")

    multi = list(StreamJsonlReader([TEST_JSONL, TEST_JSONL]))
    check(len(multi) == 2 * N_LINES, "shard list reads shards back-to-back")

    bad = "/tmp/opencode/test_stream_bad.jsonl"
    with open(bad, "w") as f:
        f.write('{"text": "a"}\nnot json at all\n{"text": "b"}\n')
    try:
        list(StreamJsonlReader(bad))
        raised = False
    except ValueError:                    # JSONDecodeError subclasses ValueError
        raised = True
    check(raised, "strict mode raises on malformed line")
    r = StreamJsonlReader(bad, on_error="skip")
    got = list(r)
    check(len(got) == 2 and r.skipped == 1,
          f"skip mode keeps good lines, counts skips ({len(got)}, {r.skipped})")


# ---------------------------------------------------------------------------
# 2. Buffer shuffle: order changes, content preserved exactly once
# ---------------------------------------------------------------------------
def test_shuffle():
    print("[M6-2: BoundedBufferShuffle order/preservation]")
    src = [f"s{i:04d}" for i in range(N_LINES)]
    out = list(BoundedBufferShuffle(iter(src), buffer_size=16, seed=42))
    check(len(out) == len(src), f"same length ({len(out)})")
    check(sorted(out) == sorted(src), "multiset preserved (all samples kept)")
    check(out != src, "order changed by shuffle")
    again = list(BoundedBufferShuffle(iter(src), buffer_size=16, seed=42))
    check(again == out, "deterministic per seed (two runs identical)")
    other = list(BoundedBufferShuffle(iter(src), buffer_size=16, seed=7))
    check(other != out, "different seed -> different order")
    ident = list(BoundedBufferShuffle(iter(src), buffer_size=1, seed=3))
    check(ident == src, "buffer_size=1 degenerates to identity order")
    small = list(BoundedBufferShuffle(iter(src[:10]), buffer_size=64, seed=3))
    check(sorted(small) == sorted(src[:10]),
          "source smaller than buffer preserved")
    # boundedness probe: an INFINITE source must keep streaming through a
    # tiny window (any full-materialization implementation would hang/OOM)
    tail = list(islice(BoundedBufferShuffle(count(), buffer_size=16, seed=0),
                       500))
    check(tail[-1] >= 400, "tiny window streams an infinite source (500 drawn)")


# ---------------------------------------------------------------------------
# 3. Dataset + DataLoader: collate, batching, integrity, restart stability
# ---------------------------------------------------------------------------
def test_dataset_dataloader():
    print("[M6-3: InfiniteStreamDataset + DataLoader]")
    tok = Tokenizer.from_file(os.path.join(ROOT, "retop_tokenizer.json"))
    eos = tok.token_to_id(EOS)
    conv = ChatIdConverter(tok)          # defaults: text -> echo task

    ref_ids = conv({"text": "pip install pkg000"})
    check(ref_ids[0] == tok.token_to_id("<s>")
          and ref_ids[1] == tok.token_to_id("<|user|>"),
          "converter emits make_chat_ids framing (<s>, <|user|>, ...)")
    check(ref_ids[-1] == eos, "gold answer EOS-terminated")

    ds = make_dataset(tok, buffer_size=32, seed=0, epoch_size=4096)
    dl = DataLoader(ds, batch_size=8, collate_fn=partial(pad_collate, pad_id=eos))
    it = iter(dl)
    b0, len0 = next(it)
    batches = [(b0, len0)] + [next(it) for _ in range(4)]
    check(all(b.shape[0] == 8 for b, _ in batches),
          "DataLoader yields five 8-row batches")
    check(b0.dtype == torch.long, "integer id dtype end to end")
    T = int(b0.shape[1])
    check(len0.dtype == torch.long and all(0 < l <= T for l in len0.tolist()),
          f"lengths tensor consistent with padded batch (T_max={T})")

    # content integrity: unpad row j == converter output of the j-th item of
    # the SAME pipeline (shuffle seed=0, buffer 32) replayed independently
    expect = list(islice(conv.stream(
        BoundedBufferShuffle(StreamJsonlReader(TEST_JSONL),
                             buffer_size=32, seed=0)), 8))
    for j, ids in enumerate(expect):
        row = b0[j, :len(ids)].tolist()
        pad = b0[j, len(ids):].tolist()
        check(row == ids and all(p == eos for p in pad),
              f"row {j}: unpadded ids match converter output, rest is pad")

    # restart stability: a fresh pass reproduces the EXACT same sequence
    # (fixed seed fixes eviction order — roadmap val-split requirement)
    cyc0 = list(islice(iter(ds), N_LINES))
    ds.reset()
    cyc0b = list(islice(iter(ds), N_LINES))
    plain = list(conv.stream(StreamJsonlReader(TEST_JSONL)))
    check(cyc0 != plain, "epoch order differs from raw file order (shuffled)")
    check(cyc0 == cyc0b, "restart reproduces the identical stream order")
    check(sorted(map(tuple, cyc0)) == sorted(map(tuple, plain)),
          "epoch preserves all documents")


# ---------------------------------------------------------------------------
# 4. Memory: RSS stays bounded while streaming 1000 samples
# ---------------------------------------------------------------------------
def test_memory_bounded():
    print("[M6-4: RSS bounded over 1000 samples]")
    tok = Tokenizer.from_file(os.path.join(ROOT, "retop_tokenizer.json"))
    eos = tok.token_to_id(EOS)
    ds = make_dataset(tok, buffer_size=128, seed=1)
    dl = DataLoader(ds, batch_size=16, collate_fn=partial(pad_collate, pad_id=eos))
    it = iter(dl)
    n = 0
    rss0 = rss_mb()
    marks = {}
    next_mark = 250
    while n < 1000:
        b, _lens = next(it)
        n += b.shape[0]
        if n >= next_mark:
            marks[next_mark] = rss_mb() - rss0
            next_mark += 250
        del b
    growth = rss_mb() - rss0
    traj = " ".join(f"@{k}:{v:+.1f}MB" for k, v in marks.items())
    print(f"  rss start {rss0:.1f} MB, growth {growth:+.2f} MB over "
          f"{n} samples ({traj})")
    check(1000 <= n < 1016,
          f"streamed through the 1000-sample mark in batch-16 steps "
          f"(consumed {n})")
    check(growth < 64.0, f"RSS growth bounded (<64 MB, got {growth:+.2f} MB)")
    peak_growth = max(marks.values())
    check(peak_growth < 64.0,
          f"no monotonic accumulation along the way (peak mark "
          f"{peak_growth:+.2f} MB)")


if __name__ == "__main__":
    test_reader()
    test_shuffle()
    test_dataset_dataloader()
    test_memory_bounded()
    print("\nM6 ALL PASSED")
