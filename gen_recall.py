"""Task 2: Extended Key-Value Recall Dataset Generator.

Extends the Stage-2 recall test (5 pairs / 50 keys) to a wider, harder regime:
  - n_pairs:    3-15 pairs per sequence
  - pool_size:  20-200 distinct keys
  - recency bias = 0% ALWAYS: the query key is never the last-seen pair's key,
    so the model MUST use memory, not the recency shortcut.
  - metadata per sample (n_pairs, query position vs write order, pool_size)
    stored for later eval breakdown.

Constraints (hardware): streaming generator, chunked .jsonl (~10K/file), never
loads whole dataset in RAM, deterministic seed -> reproducible.

Record format (memory-compatible, v2):
  messages.user   = "START k1 v1 k2 v2 ... kN vN ? kQ"
                    Adjacent key->value pairs so the memory write
                    key_proj(h_{t-1}) at the value token sees the key token
                    as its immediate predecessor (k=v format broke this by
                    inserting '=' between key and value). 'START' prefix +
                    ' ? k' query keep every key tokenized identically by the
                    BPE tokenizer (space-prefix consistency).
  messages.assistant = "valQ"
  meta            = {n_pairs, pool_size, query_pos, query_frac, keys_present}
"""

import argparse
import json
import os
import random
import time


def key_from_pair(pair):
    return pair[0]


def make_record(pairs, query_key, pool_size):
    """pairs: list of (key, value); query_key must be one of the keys.

    v2 format: 'START k1 v1 k2 v2 ... ? kQ'. Adjacent key->value pairs so
    the memory's write key (key_proj of the token before the value) is the
    actual key. 'START' + '? k' normalize tokenization across positions.
    """
    n_pairs = len(pairs)
    parts = ["START"]
    for k, v in pairs:
        parts.append(str(k)); parts.append(str(v))
    parts.append("?")
    parts.append(str(query_key))
    user_text = " ".join(parts)
    answer = dict(pairs)[query_key]
    query_pos = [k for k, _ in pairs].index(query_key)  # 0-based write order position
    return {
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": str(answer)},
        ],
        "meta": {
            "n_pairs": n_pairs,
            "pool_size": pool_size,
            "query_pos": query_pos,
            "query_frac": round(query_pos / n_pairs, 4),
            "keys_present": [k for k, _ in pairs],
        },
    }


def sample_recall_sample(rng, n_pairs_range=(3, 15), pool_size_range=(20, 200),
                         value_range=(0, 9999)):
    """Sample one (pairs, query_key, pool_size). Guarantees query is NOT the
    last-seen key (recency bias 0%). Keys drawn from a shared pool so different
    samples share keys (tests generalization over the key space, not memorization)."""
    pool_size = rng.randint(*pool_size_range)
    n_pairs = rng.randint(*n_pairs_range)
    keys = rng.sample(range(pool_size), n_pairs)          # distinct keys
    pairs = [(k, rng.randint(*value_range)) for k in keys]
    # query position: uniformly over [0, n_pairs-2] -> never last pair (recency 0%)
    query_pos = rng.randint(0, n_pairs - 2)
    query_key = keys[query_pos]
    return pairs, query_key, pool_size


def stream_recall(n, rng, n_pairs_range=(3, 15), pool_size_range=(20, 200),
                  value_range=(0, 9999)):
    """Yield n records."""
    for _ in range(n):
        pairs, qk, pool = sample_recall_sample(
            rng, n_pairs_range, pool_size_range, value_range)
        yield make_record(pairs, qk, pool)


def write_chunked(out_dir, records, chunk_size=10000):
    os.makedirs(out_dir, exist_ok=True)
    files = []
    written = 0
    chunk = []
    idx = 0
    for rec in records:
        chunk.append(rec)
        if len(chunk) >= chunk_size:
            files.append(_dump(chunk, out_dir, idx))
            chunk = []
            idx += 1
        written += 1
    if chunk:
        files.append(_dump(chunk, out_dir, idx))
    return files, written


def _dump(chunk, out_dir, idx):
    fn = os.path.join(out_dir, f"chunk_{idx:05d}.jsonl")
    with open(fn, "w", encoding="utf-8") as f:
        for rec in chunk:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return os.path.basename(fn)


def iter_records(out_dir):
    for fn in sorted(os.listdir(out_dir)):
        if not fn.endswith(".jsonl"):
            continue
        with open(os.path.join(out_dir, fn), encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


def compute_stats(out_dir):
    stats = {"n": 0, "n_pairs_min": 99, "n_pairs_max": 0, "pool_min": 9999,
             "pool_max": 0, "query_frac_hist": {}, "recency_violations": 0}
    for r in iter_records(out_dir):
        m = r["meta"]
        stats["n"] += 1
        stats["n_pairs_min"] = min(stats["n_pairs_min"], m["n_pairs"])
        stats["n_pairs_max"] = max(stats["n_pairs_max"], m["n_pairs"])
        stats["pool_min"] = min(stats["pool_min"], m["pool_size"])
        stats["pool_max"] = max(stats["pool_max"], m["pool_size"])
        qf = m["query_frac"]
        bin_ = f"{int(qf * 10)}"
        stats["query_frac_hist"][bin_] = stats["query_frac_hist"].get(bin_, 0) + 1
        if m["query_pos"] == m["n_pairs"] - 1:
            stats["recency_violations"] += 1
    stats["query_frac_hist"] = {k: round(v / stats["n"], 4)
                                for k, v in sorted(stats["query_frac_hist"].items())}
    return stats


def main():
    ap = argparse.ArgumentParser(description="Generate extended key-value recall dataset (Task 2)")
    ap.add_argument("--train", default=20000, type=int)
    ap.add_argument("--val", default=3000, type=int)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--outdir",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "hmn_data", "recall"))
    ap.add_argument("--chunk", default=10000, type=int)
    ap.add_argument("--n-pairs-min", default=3, type=int)
    ap.add_argument("--n-pairs-max", default=15, type=int)
    ap.add_argument("--pool-min", default=20, type=int)
    ap.add_argument("--pool-max", default=200, type=int)
    args = ap.parse_args()

    npr = (args.n_pairs_min, args.n_pairs_max)
    psr = (args.pool_min, args.pool_max)

    t0 = time.time()
    tr_files, tr_n = write_chunked(
        os.path.join(args.outdir, "train"),
        stream_recall(args.train, random.Random(args.seed), npr, psr), args.chunk)
    va_files, va_n = write_chunked(
        os.path.join(args.outdir, "val"),
        stream_recall(args.val, random.Random(args.seed + 1), npr, psr), args.chunk)
    tr_s = compute_stats(os.path.join(args.outdir, "train"))
    va_s = compute_stats(os.path.join(args.outdir, "val"))
    print(f"[train] n={tr_s['n']} pairs[{tr_s['n_pairs_min']}-{tr_s['n_pairs_max']}] "
          f"pool[{tr_s['pool_min']}-{tr_s['pool_max']}] "
          f"recency_viol={tr_s['recency_violations']} files={len(tr_files)} "
          f"({round(time.time()-t0,2)}s)")
    print(f"[val]   n={va_s['n']} pairs[{va_s['n_pairs_min']}-{va_s['n_pairs_max']}] "
          f"pool[{va_s['pool_min']}-{va_s['pool_max']}] "
          f"recency_viol={va_s['recency_violations']} files={len(va_files)}")
    with open(os.path.join(args.outdir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump({"train": tr_s, "val": va_s, "seed": args.seed,
                   "n_pairs_range": list(npr), "pool_range": list(psr)},
                  f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
