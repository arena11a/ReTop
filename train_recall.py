"""Train HMN v2 on the extended key-value recall task (Task 2).

Streams chunked .jsonl from gen_recall.py. Reports answer accuracy broken down
by n_pairs bucket and by query_frac (early/middle/late) — this is the metadata
breakdown Task 2 was designed for.

Usage:
  python train_recall.py --steps 1500 --dim 64 --layers 4 --mem-cells 16
"""

import argparse
import os
import random
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_recall import iter_records
from hmn_v2 import HMN
from train_arithmetic import build_tokenizer

TOKENIZER_PATH = "/home/yonoob/projects/ReTop/retop_tokenizer.json"
DATA_ROOT = "/home/yonoob/projects/ReTop/hmn_data/recall"
VOCAB_SIZE = 3190
EOS_ID = 1
PAD_ID = 3


def parse_user_text(user_text):
    """v2 format: 'START k1 v1 k2 v2 ... ? kQ' -> pairs, query_key."""
    parts = user_text.split()
    parts = [p for p in parts if p != "START"]
    pairs = []
    query = None
    i = 0
    while i < len(parts):
        if parts[i] == "?":
            query = int(parts[i + 1])
            i += 2
        else:
            k = int(parts[i]); v = int(parts[i + 1])
            pairs.append((k, v))
            i += 2
    return pairs, query


def encode_recall(tok, user_text, answer):
    """Template: <s><|user|>{user_text}<|assistant|>{answer}</s>"""
    text = f"<s><|user|>{user_text}<|assistant|>{answer}</s>"
    ids = tok.encode(text).ids
    targets = ids[1:] + [EOS_ID]
    ans_start = ids.index(5) + 1  # 5 = <|assistant|>
    return ids, targets, ans_start


class StreamingRecallDataset:
    def __init__(self, tok, batch_size, seed=0):
        self.tok = tok
        self.batch_size = batch_size
        self.gen = iter_records(os.path.join(DATA_ROOT, "train"))
        self.rng = random.Random(seed)

    def __iter__(self):
        return self

    def __next__(self):
        batch_ids, batch_tgt, batch_ans = [], [], []
        while len(batch_ids) < self.batch_size:
            try:
                rec = next(self.gen)
            except StopIteration:
                if not batch_ids:
                    raise
                break
            u = rec["messages"][0]["content"]
            a = rec["messages"][1]["content"]
            ids, tgt, ast = encode_recall(self.tok, u, a)
            batch_ids.append(ids); batch_tgt.append(tgt); batch_ans.append(ast)
        if not batch_ids:
            raise StopIteration
        T = max(len(x) for x in batch_ids)
        X = torch.full((len(batch_ids), T), PAD_ID, dtype=torch.long)
        Y = torch.full((len(batch_ids), T), PAD_ID, dtype=torch.long)
        for i, (ids, tgt) in enumerate(zip(batch_ids, batch_tgt)):
            X[i, :len(ids)] = torch.tensor(ids)
            Y[i, :len(tgt)] = torch.tensor(tgt)
        return X, Y, torch.tensor(batch_ans)


def eval_recall(model, tok, n=300, seed=1):
    """Answer accuracy + breakdown by n_pairs and query_frac (metadata-driven)."""
    model.eval()
    recs = list(iter_records(os.path.join(DATA_ROOT, "val")))
    random.Random(seed).shuffle(recs)
    recs = recs[:n]
    ok = 0
    buckets = {}   # n_pairs bucket -> [correct, total]
    qfrac = {}     # query_frac bucket -> [correct, total]
    with torch.no_grad():
        for rec in recs:
            u = rec["messages"][0]["content"]
            ans = rec["messages"][1]["content"]
            pairs, qk = parse_user_text(u)
            # answer is multi-token if >9; decode greedily over answer token space
            text = f"<s><|user|>{u}<|assistant|>"
            inp = torch.tensor([tok.encode(text).ids])
            pred_ids = []
            for _ in range(8):
                out = model(inp)
                idx = out[0, -1].argmax(-1).item()
                if idx == EOS_ID:
                    break
                pred_ids.append(idx)
                inp = torch.cat([inp, torch.tensor([[idx]])], dim=-1)
            pred = tok.decode(pred_ids)
            correct = (pred == ans)
            ok += correct
            np_ = len(pairs)
            b = f"{np_//3*3}-{np_//3*3+2}" if np_ <= 15 else "15+"
            e = buckets.setdefault(b, [0, 0]); e[0] += correct; e[1] += 1
            qf = rec["meta"]["query_frac"]
            qb = "early(<0.3)" if qf < 0.3 else ("mid(0.3-0.7)" if qf < 0.7 else "late(>=0.7)")
            e2 = qfrac.setdefault(qb, [0, 0]); e2[0] += correct; e2[1] += 1
    model.train()
    out = {"overall": ok / max(1, len(recs))}
    out["by_n_pairs"] = {k: round(v[0] / v[1], 3) for k, v in sorted(buckets.items())}
    out["by_query_frac"] = {k: round(v[0] / v[1], 3) for k, v in sorted(qfrac.items())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default=1500, type=int)
    ap.add_argument("--dim", default=64, type=int)
    ap.add_argument("--state", default=8, type=int)
    ap.add_argument("--layers", default=4, type=int)
    ap.add_argument("--mem-cells", default=16, type=int)
    ap.add_argument("--bs", default=8, type=int)
    ap.add_argument("--lr", default=3e-4, type=float)
    ap.add_argument("--eval-every", default=300, type=int)
    ap.add_argument("--seed", default=0, type=int)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tok = build_tokenizer()
    model = HMN(VOCAB_SIZE, args.dim, args.state, args.layers, n_experts=16, top_k=2,
                n_mem_cells=args.mem_cells, mem_top_k=4, memory_interval=2)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    train = StreamingRecallDataset(tok, args.bs, seed=args.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    lossf = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    step = 0
    t0 = time.time()
    while step < args.steps:
        try:
            X, Y, AS = next(train)
        except StopIteration:
            train = StreamingRecallDataset(tok, args.bs, seed=args.seed)
            continue
        opt.zero_grad()
        logits = model(X)
        loss = lossf(logits.reshape(-1, VOCAB_SIZE), Y.reshape(-1))
        (loss + model.moe_aux_loss()).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step()
        step += 1
        if step == 1 or step % 100 == 0:
            print(f"step {step}/{args.steps} loss {loss.item():.4f} "
                  f"{ (time.time()-t0)/step:.2f}s/step", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            r = eval_recall(model, tok)
            print(f"  >> val recall {r['overall']:.3f}  n_pairs={r['by_n_pairs']} "
                  f" qfrac={r['by_query_frac']}", flush=True)

    print("DONE. final:")
    print(eval_recall(model, tok))


if __name__ == "__main__":
    main()
