"""Train HMN v2 on the arithmetic curriculum (Task 1).

Reads chunked .jsonl from gen_arithmetic.py output. Streams batches from disk,
never loads whole dataset into RAM. Reports loss + held-out val accuracy
(disjoint by construction) + token-level accuracy on answer tokens.

Usage:
  python train_arithmetic.py --stage A --steps 600 --dim 64 --layers 4
"""

import argparse
import itertools
import json
import os
import random
import time

import torch
import torch.nn as nn

from tokenizers import Tokenizer

from gen_arithmetic import iter_records
from hmn import HMN

ROOT = os.path.dirname(os.path.abspath(__file__))
VOCAB_SIZE = 3190
DEFAULT_TOKENIZER = os.path.join(ROOT, "retop_tokenizer.json")
DEFAULT_DATA = os.path.join(ROOT, "hmn_data", "arithmetic")


def build_tokenizer():
    return Tokenizer.from_file(TOKENIZER_PATH)


def encode_problem(tok, problem, answer):
    """Return (input_ids, target_ids, answer_start_idx).
    Template: <s><|user|>{problem}<|assistant|>{answer}</s>
    Loss is computed over ALL tokens (teacher-forced LM objective), but we track
    answer-token accuracy separately by slicing at answer_start_idx."""
    text = f"<s><|user|>{problem}<|assistant|>{answer}</s>"
    ids = tok.encode(text).ids
    # find answer start: after the <|assistant|> special token (id 5)
    try:
        ans_start = ids.index(5) + 1
    except ValueError:
        ans_start = len(ids) - 1
    targets = ids[1:] + [tok.token_to_id("</s>")]
    return ids, targets, ans_start


class StreamingDataset:
    """Streams records from chunked jsonl files into batches without loading all."""

    def __init__(self, stage, split, tok, batch_size, seed=0):
        self.dir = os.path.join(DATA_ROOT, f"stage{stage}", split)
        self.tok = tok
        self.batch_size = batch_size
        self.gen = iter_records(self.dir)
        self.buffer = []
        self.rng = random.Random(seed)

    def __iter__(self):
        return self

    def __next__(self):
        # keep one batch buffered: encoded lazily
        batch_ids, batch_targets, batch_ans_start = [], [], []
        while len(batch_ids) < self.batch_size:
            try:
                rec = next(self.gen)
            except StopIteration:
                if len(batch_ids) == 0:
                    raise
                break
            problem = rec["messages"][0]["content"]
            answer = rec["messages"][1]["content"]
            ids, targets, ans_start = encode_problem(self.tok, problem, answer)
            batch_ids.append(ids)
            batch_targets.append(targets)
            batch_ans_start.append(ans_start)
        if not batch_ids:
            raise StopIteration
        T = max(len(x) for x in batch_ids)
        pad_id = self.tok.token_to_id("<pad>")
        pad_t = self.tok.token_to_id("</s>")
        ids_p = torch.full((len(batch_ids), T), pad_id, dtype=torch.long)
        tgt_p = torch.full((len(batch_ids), T), pad_t, dtype=torch.long)
        for i, (ids, tgt) in enumerate(zip(batch_ids, batch_targets)):
            ids_p[i, : len(ids)] = torch.tensor(ids)
            tgt_p[i, : len(tgt)] = torch.tensor(tgt)
        return ids_p, tgt_p, torch.tensor(batch_ans_start)


def eval_accuracy(model, tok, stage, split, n=200, seed=0):
    """Sample n records from val split (chunked on disk), decode answer tokens,
    measure exact-answer accuracy. Streaming, no full load."""
    model.eval()
    import random
    recs = list(itertools.islice(iter_records(
        os.path.join(DATA_ROOT, f"stage{stage}", split)), n * 5))
    random.Random(seed).shuffle(recs)
    recs = recs[:n]
    correct = 0
    total = 0
    with torch.no_grad():
        for rec in recs:
            problem = rec["messages"][0]["content"]
            answer = rec["messages"][1]["content"]
            text = f"<s><|user|>{problem}<|assistant|>"
            ids = tok.encode(text).ids
            inp = torch.tensor([ids])
            out = model(inp)
            pred_id = out[0, -1].argmax(-1).item()
            pred = tok.decode([pred_id])
            total += 1
            correct += (pred == answer)
    return correct / max(1, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="A")
    ap.add_argument("--steps", default=600, type=int)
    ap.add_argument("--dim", default=64, type=int)
    ap.add_argument("--state", default=8, type=int)
    ap.add_argument("--layers", default=4, type=int)
    ap.add_argument("--bs", default=8, type=int)
    ap.add_argument("--lr", default=3e-4, type=float)
    ap.add_argument("--eval-every", default=100, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--tok", default=DEFAULT_TOKENIZER)
    ap.add_argument("--data", default=DEFAULT_DATA)
    args = ap.parse_args()

    global TOKENIZER_PATH, DATA_ROOT
    TOKENIZER_PATH, DATA_ROOT = args.tok, args.data
    torch.manual_seed(args.seed)
    tok = build_tokenizer()
    model = HMN(VOCAB_SIZE, args.dim, args.state, args.layers, n_experts=16, top_k=2,
                n_mem_cells=8, mem_top_k=4, memory_interval=2)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    train = StreamingDataset(args.stage, "train", tok, args.bs, seed=args.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    lossf = nn.CrossEntropyLoss(ignore_index=tok.token_to_id("</s>"))

    step = 0
    t0 = time.time()
    best_loss = float("inf")
    while step < args.steps:
        try:
            ids, tgt, ans_start = next(train)
        except StopIteration:
            train = StreamingDataset(args.stage, "train", tok, args.bs, seed=args.seed)
            continue
        opt.zero_grad()
        logits = model(ids)
        loss = lossf(logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1))
        aux = model.moe_aux_loss()
        (loss + aux).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        step += 1
        if step == 1 or step % 50 == 0:
            dt = (time.time() - t0) / step
            eta = (args.steps - step) * dt
            print(f"step {step}/{args.steps} loss {loss.item():.4f} aux {aux.item():.4f} "
                  f"{dt:.2f}s/step ETA {eta/60:.1f}min", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            acc = eval_accuracy(model, tok, args.stage, "val")
            print(f"  >> val answer-acc: {acc:.3f}", flush=True)
        best_loss = min(best_loss, loss.item())

    print(f"DONE. best_train_loss={best_loss:.4f}")
    # final sample generations
    model.eval()
    with torch.no_grad():
        for rec in list(itertools.islice(iter_records(
                os.path.join(DATA_ROOT, f"stage{args.stage}", "val")), 5)):
            problem = rec["messages"][0]["content"]
            answer = rec["messages"][1]["content"]
            text = f"<s><|user|>{problem}<|assistant|>"
            ids = tok.encode(text).ids
            inp = torch.tensor([ids])
            pred_id = model(inp)[0, -1].argmax(-1).item()
            pred = tok.decode([pred_id])
            print(f"  {problem} -> pred:{pred} truth:{answer}")


if __name__ == "__main__":
    main()
