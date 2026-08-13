"""gen_slots — slot-copy dataset generator for the v3.3 recipe.

Produces chat-pairs .jsonl (the format retop.py / train_v3.py train on) split
into a SEEN set and an UNSEEN set with ZERO overlap — the whole point of the
Identity Register eval (docs/hmn_v3_design.md §2/§3).

  python gen_slots.py --out data/slots.jsonl --n-seen 600 --n-unseen 400 \
      --kind pkg3 --template "pip install {slot}" --seed 0

Every slot value is drawn deterministically from `seed`, so re-runs are
bit-identical. The first `--n-seen` values go to train, the next `--n-unseen`
go to val; because both come from ONE deterministic stream, a value can never
appear in both sets.

Kinds:
  pkg3   pkg000..pkg999          (canonical, matches hmn_v33.pt training)
  pkg4   pkg0000..pkg9999        (longer slot)
  pkg5   pkg00000..pkg99999
  alnum  pkg{L}{dd} on 10 sampled letters   (mixed letters+digits)
  repeat pkg111,pkg222,..pkg999  (repeated-token slots — known limitation #3)

Output: <out> for train, <out without .jsonl>-val.jsonl for unseen, plus a
sidecar <out>.meta.json recording kind/template/seeds/counts.
"""
import argparse
import json
import os
import random

NAME = "gen_slots"


def slot_values(kind, rng):
    """Deterministic stream of distinct slot values for `kind` (whole universe)."""
    if kind == "pkg3":
        return [f"pkg{i:03d}" for i in range(1000)]
    if kind == "pkg4":
        return [f"pkg{i:04d}" for i in range(10000)]
    if kind == "pkg5":
        return [f"pkg{i:05d}" for i in range(100000)]
    if kind == "alnum":
        letters = rng.sample("ABCDEFGHIJ", 10)  # 10 letters x 100 digits = 1000 distinct
        return [f"pkg{l}{d:02d}" for d in range(100) for l in letters]
    if kind == "repeat":
        base = [111, 222, 333, 444, 555, 666, 777, 888, 999]
        return [f"pkg{v:03d}" for v in base]
    raise ValueError(f"unknown --kind {kind!r}")


def record(slot, template):
    text = template.format(slot=slot)
    return {"messages": [
        {"role": "user", "content": text},
        {"role": "assistant", "content": text},
    ]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, fromfile_prefix_chars="@")
    ap.add_argument("--out", required=True, help="train .jsonl path (val is <out>-val.jsonl)")
    ap.add_argument("--n-seen", type=int, default=600)
    ap.add_argument("--n-unseen", type=int, default=400)
    ap.add_argument("--kind", default="pkg3", choices=["pkg3", "pkg4", "pkg5", "alnum", "repeat"])
    ap.add_argument("--template", default="pip install {slot}")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    meta = vars(args).copy()
    meta["generator"] = NAME
    val_path = args.out[:-6] + "-val.jsonl" if args.out.endswith(".jsonl") else args.out + "-val.jsonl"

    universe = slot_values(args.kind, rng)
    rng.shuffle(universe)                      # deterministic (same seed)
    if len(universe) < args.n_seen + args.n_unseen:
        raise SystemExit(
            f"--kind {args.kind} has only {len(universe)} distinct slots; "
            f"cannot fill seen={args.n_seen} + unseen={args.n_unseen} without "
            f"overlap. Lower counts or use a different --kind.")
    seen = universe[:args.n_seen]
    unseen = universe[args.n_seen:args.n_seen + args.n_unseen]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for path, slots in [(args.out, seen), (val_path, unseen)]:
        with open(path, "w", encoding="utf-8") as f:
            for s in slots:
                f.write(json.dumps(record(s, args.template), ensure_ascii=False) + "\n")
        print(f"{path}: {len(slots)} records", flush=True)

    with open(args.out + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"meta: {args.out}.meta.json")


if __name__ == "__main__":
    main()