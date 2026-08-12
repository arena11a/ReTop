"""Task 1: Arithmetic Curriculum Dataset Generator.

Stages:
  A: single-digit addition       (0-9) + (0-9)        -> 55 unique pairs
  B: two-digit add/sub           10-99 +/- 10-99       -> ~8K unique
  C: single-digit multiplication (0-9) * (0-9)          -> 55 unique pairs
  D: mixed 2-3 digit add/sub/mul (10-999)               -> large, sampled

Core design:
  - Each stage's full key space is enumerated, then split deterministically
    (seeded shuffle) into train/val KEY SETS. This GUARANTEES train/val disjoint
    by construction, with no cross-contamination even for commutative ops.
  - train: sample with replacement from the train key set (curriculum repetition OK)
  - val:   each held-out key exactly once -> clean held-out generalization metric
  - streaming: only chunk_size records in RAM at a time; ~10K samples/chunk file

Hardware constraints honored:
  - never loads whole dataset into RAM (chunked .jsonl, streaming writer)
  - deterministic seed -> reproducible splits

Record format matches retop_dataset_math100k.jsonl exactly:
  {"messages": [{"role":"user","content":"5+7"},{"role":"assistant","content":"12"}]}
plus a "meta" field for stats / later breakdown.
"""

import argparse
import json
import os
import random
import time


# ---------------- per-stage enumeration of the full unique key space ----------------

def enumerate_stage(stage):
    """Return list of meta dicts covering the full unique key space of a stage."""
    metas = []
    if stage == "A":                       # (0-9)+(0-9), commutative -> a<=b
        for a in range(10):
            for b in range(a, 10):
                metas.append({"op": "+", "a": a, "b": b, "ans": a + b})
    elif stage == "B":                     # two-digit add (commutative) + sub (ordered)
        for a in range(10, 100):
            for b in range(a, 100):
                metas.append({"op": "+", "a": a, "b": b, "ans": a + b})
        for a in range(10, 100):
            for b in range(10, 100):
                if a >= b:
                    metas.append({"op": "-", "a": a, "b": b, "ans": a - b})
    elif stage == "C":                     # (0-9)*(0-9), commutative
        for a in range(10):
            for b in range(a, 10):
                metas.append({"op": "*", "a": a, "b": b, "ans": a * b})
    elif stage == "D":                     # 2-3 digit, too large to enumerate fully
        raise NotImplementedError("stage D uses sampled generation; call sampled_stage_d_stream()")
    else:
        raise KeyError(f"unknown stage {stage}")
    return metas


def sampled_stage_d(rng):
    """Sampled stage-D item (2-3 digit add/sub/mul with non-negative sub)."""
    a = rng.randint(10, 999)
    b = rng.randint(10, 999)
    op = rng.choice(["+", "-", "*"])
    if op == "-":
        a, b = max(a, b), min(a, b)
        return {"op": "-", "a": a, "b": b, "ans": a - b}
    if op == "+":
        return {"op": "+", "a": a, "b": b, "ans": a + b}
    return {"op": "*", "a": a, "b": b, "ans": a * b}


STAGE_SPACE_SIZES = {"A": 55, "B": 8190, "C": 55, "D": None}


def key_from_meta(m):
    """Canonical dedup key: commutative ops normalized (a<=b), sub keeps order."""
    if m["op"] in ("+", "*"):
        lo, hi = sorted((m["a"], m["b"]))
        return (m["op"], lo, hi)
    return (m["op"], m["a"], m["b"])


def expr_from_meta(m):
    return f"{m['a']}{m['op']}{m['b']}"


def make_record(meta):
    return {
        "messages": [
            {"role": "user", "content": expr_from_meta(meta)},
            {"role": "assistant", "content": str(meta["ans"])},
        ],
        "meta": meta,
    }


# ---------------- split helpers ----------------

def split_keys(metas, val_count, seed):
    """Deterministic split: shuffled by seed, last `val_count` go to val. Returns
    (train_metas, val_metas). Raises if val_count exceeds available distinct keys."""
    if val_count > len(metas):
        raise ValueError(
            f"val_count={val_count} exceeds distinct keys={len(metas)} — "
            f"pick val <= space size, or use a larger stage")
    rng = random.Random(seed)
    shuffled = list(metas)
    rng.shuffle(shuffled)
    return shuffled[: len(shuffled) - val_count], shuffled[len(shuffled) - val_count:]


# ---------------- streaming writers ----------------

def write_chunked(out_dir, records, chunk_size=10000):
    """Write an iterator of records into chunked .jsonl files. Returns (files, written)."""
    os.makedirs(out_dir, exist_ok=True)
    files = []
    written = 0
    chunk = []
    chunk_idx = 0
    for rec in records:
        chunk.append(rec)
        if len(chunk) >= chunk_size:
            files.append(_dump_chunk(chunk, out_dir, chunk_idx))
            chunk = []
            chunk_idx += 1
        written += 1
    if chunk:
        files.append(_dump_chunk(chunk, out_dir, chunk_idx))
    return files, written


def _dump_chunk(chunk, out_dir, idx):
    fn = os.path.join(out_dir, f"chunk_{idx:05d}.jsonl")
    with open(fn, "w", encoding="utf-8") as f:
        for rec in chunk:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return os.path.basename(fn)


def iter_records(out_dir):
    """Lazily iterate records from chunked files in a directory."""
    for fn in sorted(os.listdir(out_dir)):
        if not fn.endswith(".jsonl"):
            continue
        with open(os.path.join(out_dir, fn), encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


def collect_keys(out_dir):
    return {key_from_meta(r["meta"]) for r in iter_records(out_dir)}


def compute_stats(out_dir):
    ops = {}
    token_lens = []
    for r in iter_records(out_dir):
        m = r["meta"]
        ops[m["op"]] = ops.get(m["op"], 0) + 1
        token_lens.append(2 + len(m["op"]) + len(str(m["a"])) + len(str(m["b"])) + len(str(m["ans"])))
    n = sum(ops.values())
    return {
        "n": n,
        "op_dist": {k: round(v / n, 4) for k, v in sorted(ops.items())},
        "mean_tokens": round(sum(token_lens) / max(1, len(token_lens)), 2),
        "max_tokens": max(token_lens),
        "min_tokens": min(token_lens),
    }


# ---------------- generation drivers ----------------

def gen_stage_file(stage, out_dir, train_count, val_count, seed):
    """Generate one stage: deterministic split, streaming chunked files.
    Returns (train_stats, val_stats, train_keys, val_keys)."""
    if stage == "D":
        return gen_stage_d_file(out_dir, train_count, val_count, seed)
    metas = enumerate_stage(stage)
    train_metas, val_metas = split_keys(metas, val_count, seed)

    rng = random.Random(seed)
    def _train_recs():
        for _ in range(train_count):
            yield make_record(rng.choice(train_metas))
    tr_files, tr_n = write_chunked(os.path.join(out_dir, "train"), _train_recs())
    va_files, va_n = write_chunked(os.path.join(out_dir, "val"), (make_record(m) for m in val_metas))
    return compute_stats(os.path.join(out_dir, "train")), \
           compute_stats(os.path.join(out_dir, "val")), \
           collect_keys(os.path.join(out_dir, "train")), \
           collect_keys(os.path.join(out_dir, "val"))


def gen_stage_d_file(out_dir, train_count, val_count, seed):
    """Stage D (large space): stream-sampled train, val sampled AFTER reserving train keys."""
    train_rng = random.Random(seed)
    val_rng = random.Random(seed + 1)
    train_recs = (make_record(sampled_stage_d(train_rng)) for _ in range(train_count))
    tr_files, tr_n = write_chunked(os.path.join(out_dir, "train"), train_recs)
    train_keys = collect_keys(os.path.join(out_dir, "train"))
    val_recs = []
    tries = 0
    while len(val_recs) < val_count and tries < val_count * 100:
        m = sampled_stage_d(val_rng)
        if key_from_meta(m) not in train_keys:
            val_recs.append(make_record(m))
        tries += 1
    va_files, va_n = write_chunked(os.path.join(out_dir, "val"), iter(val_recs))
    return compute_stats(os.path.join(out_dir, "train")), \
           compute_stats(os.path.join(out_dir, "val")), \
           train_keys, \
           collect_keys(os.path.join(out_dir, "val"))


# ---------------- entry point ----------------

def main():
    ap = argparse.ArgumentParser(description="Generate arithmetic curriculum dataset (Task 1)")
    ap.add_argument("--stages", default="A,B", help="comma-separated stages, default A,B (small first)")
    ap.add_argument("--train", default=10000, type=int, help="train samples per stage")
    ap.add_argument("--val", default=1000, type=int, help="distinct val keys held out per stage")
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--outdir", default="/home/yonoob/projects/ReTop/hmn_data/arithmetic")
    ap.add_argument("--chunk", default=10000, type=int)
    args = ap.parse_args()

    for stage in [s.strip().upper() for s in args.stages.split(",")]:
        space = STAGE_SPACE_SIZES.get(stage)
        val_count = args.val
        if space is not None and val_count > space:
            print(f"[stage {stage}] warning: val={args.val} > space size {space}, "
                  f"clamping val to {space}")
            val_count = space
        # keep >= half the distinct keys in train so train is meaningful
        if space is not None and val_count > space // 2:
            print(f"[stage {stage}] warning: val={val_count} too large for train "
                  f"(space {space}), clamping to {space // 2}")
            val_count = space // 2
        t0 = time.time()
        tr, va, tr_keys, va_keys = gen_stage_file(
            stage, os.path.join(args.outdir, f"stage{stage}"), args.train, val_count, args.seed)
        overlap = tr_keys & va_keys
        print(f"[stage {stage}] train n={tr['n']} {tr['op_dist']} "
              f"tokens~{tr['mean_tokens']} | val n={va['n']} {va['op_dist']} "
              f"| disjoint={len(overlap)==0} ({len(tr_keys)}/{len(va_keys)} keys) "
              f"| {round(time.time()-t0,2)}s")
        with open(os.path.join(args.outdir, f"stage{stage}", "stats.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"train": tr, "val": va, "seed": args.seed}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
