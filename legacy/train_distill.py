"""Distillation training for HMN on the Python+build domain (Step 4).

Teacher = assistant-authored (prompt, response) pairs in hmn_data/distill_python.
The distillation objective here is teacher-forced supervised fine-tuning on the
teacher's response tokens (the "distilled distribution"). Optional --standard
mode trains on the same data with a plain LM objective for A/B comparison.

Eval (Step 5) runs the eval protocol from distill_design.md:
  syntax:  ast.parse(code response) succeeds
  api:     required API tokens (meta.api) all present in the response
  run:     executable sample's stdout matches meta.expected_stdout

Baseline teacher val score (metadata): syntax 9/9, api 17/17, run 6/6 (100%).
"""
import argparse, ast, json, os, random, subprocess, sys, time
from paths import DISTILL_TEMPLATES
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from hmn import HMN

ROOT = os.path.dirname(os.path.abspath(__file__))
VOCAB_SIZE = 3190
TOKENIZER_PATH = os.path.join(ROOT, "retop_tokenizer.json")
DATA_ROOT = DISTILL_TEMPLATES


def build_tokenizer():
    return Tokenizer.from_file(TOKENIZER_PATH)


def encode_pair(tok, user, assistant):
    text = f"<s><|user|>{user}<|assistant|>{assistant}</s>"
    ids = tok.encode(text).ids
    targets = ids[1:] + [tok.token_to_id("</s>")]
    return ids, targets


def encode_pair_masked(tok, user, assistant):
    """encode_pair + mask the loss to the ASSISTANT region only.

    During distillation the student should only be graded on reproducing the
    teacher's response; the user-prompt tokens are context, not targets. Logging
    loss over the whole sequence lets the (longer) prompt dominate the gradient,
    which starves the response head (probe showed 14.7 nats answer loss — worse
    than random — while total loss read 1.9).

    Alignment: targets[i] = ids[i+1]. ids[a] = <|assistant|>, so the first gold
    response token ids[a+1] is predicted at target index a. Mask indices < a.
    """
    ids, targets = encode_pair(tok, user, assistant)
    asi = tok.encode("a<|assistant|>").ids[-1]
    a = ids.index(asi)
    for i in range(a):
        targets[i] = -100
    return ids, targets


def iter_records(out_dir):
    for fn in sorted(os.listdir(out_dir)):
        if not fn.endswith(".jsonl"):
            continue
        with open(os.path.join(out_dir, fn), encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


class StreamingPairs:
    def __init__(self, split, tok, batch_size, seed=0):
        self.recs = list(iter_records(os.path.join(DATA_ROOT, split)))
        self.tok = tok
        self.bs = batch_size
        self.rng = random.Random(seed)

    def batches(self):
        while True:
            self.rng.shuffle(self.recs)
            for i in range(0, len(self.recs), self.bs):
                batch = self.recs[i:i + self.bs]
                T = max(len(encode_pair(self.tok, r["messages"][0]["content"],
                                        r["messages"][1]["content"])[0]) for r in batch)
                X = torch.full((len(batch), T), self.tok.token_to_id("</s>"), dtype=torch.long)
                Y = torch.full((len(batch), T), -100, dtype=torch.long)
                for j, r in enumerate(batch):
                    ids, tgt = encode_pair_masked(self.tok, r["messages"][0]["content"],
                                                  r["messages"][1]["content"])
                    X[j, :len(ids)] = torch.tensor(ids)
                    Y[j, :len(tgt)] = torch.tensor(tgt)
                yield X, Y


def eval_protocol(model, tok, split="val", n=15, seed=0, max_new=128):
    """Step 5 metric: syntax / API / run scores on the distillation val set."""
    model.eval()
    recs = list(iter_records(os.path.join(DATA_ROOT, split)))
    random.Random(seed).shuffle(recs)
    recs = recs[:n]
    n_syn = n_api = n_run = 0
    ok_syn = ok_api = ok_run = 0
    with torch.no_grad():
        for rec in recs:
            user = rec["messages"][0]["content"]
            gold = rec["messages"][1]["content"]
            meta = rec["meta"]
            text = f"<s><|user|>{user}<|assistant|>"
            ids = tok.encode(text).ids
            pred = decode_response(model, tok, ids, max_new=max_new)
            if meta["kind"] == "code":
                n_syn += 1
                try:
                    ast.parse(pred)
                    ok_syn += 1
                except SyntaxError:
                    pass
            if meta["api"]:
                n_api += 1
                if all(a in pred for a in meta["api"]):
                    ok_api += 1
            if meta.get("run"):
                n_run += 1
                if pred == gold:
                    ok_run += 1  # exact match on runnable gold (strongest signal)
    model.train()
    return {
        "syntax": ok_syn / max(1, n_syn),
        "api": ok_api / max(1, n_api),
        "run_exact": ok_run / max(1, n_run),
    }


def decode_response(model, tok, prompt_ids, max_new=32):
    ids = list(prompt_ids)
    with torch.no_grad():
        for _ in range(max_new):
            inp = torch.tensor([ids])
            logits = model(inp)
            nxt = logits[0, -1].argmax(-1).item()
            if nxt == tok.token_to_id("</s>"):
                break
            ids.append(nxt)
    return tok.decode(ids[len(prompt_ids):])


def answer_loss(model, tok, split="train", n=15, seed=0):
    """CE on the assistant-response region only — recomputed with the SAME masked
    encoding used in training (encode_pair_masked), so alignment is guaranteed."""
    model.eval()
    recs = list(iter_records(os.path.join(DATA_ROOT, split)))
    random.Random(seed).shuffle(recs)
    lossf = nn.CrossEntropyLoss()
    tot = num = 0.0
    with torch.no_grad():
        for rec in recs[:n]:
            user, gold = rec["messages"][0]["content"], rec["messages"][1]["content"]
            ids_ids, tgt = encode_pair_masked(tok, user, gold)
            X = torch.tensor([ids_ids])
            Y = torch.tensor([tgt])
            logits = model(X)
            m = Y != -100
            if m.sum() == 0:
                continue
            tot += lossf(logits[m].reshape(-1, VOCAB_SIZE),
                         torch.where(m, Y, torch.zeros_like(Y))[m].reshape(-1)).item()
            num += 1
    model.train()
    return tot / max(1, num)


def main():
    global TOKENIZER_PATH, DATA_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default=1200, type=int)
    ap.add_argument("--dim", default=64, type=int)
    ap.add_argument("--state", default=8, type=int)
    ap.add_argument("--layers", default=4, type=int)
    ap.add_argument("--bs", default=6, type=int)
    ap.add_argument("--lr", default=3e-4, type=float)
    ap.add_argument("--eval-every", default=200, type=int)
    ap.add_argument("--sched", default="constant", choices=["cosine", "constant"],
                    help="LR schedule: constant (default) avoids premature LR collapse "
                         "that starved memorization in the 1200-step cosine run")
    ap.add_argument("--save", default=None,
                    help="checkpoint path (best-val-loss model is written here)")
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--tok", default=TOKENIZER_PATH)
    ap.add_argument("--data", default=DATA_ROOT)
    ap.add_argument("--standard", action="store_true",
                    help="A/B: train plain LM objective (no distillation framing)")
    args = ap.parse_args()

    TOKENIZER_PATH, DATA_ROOT = args.tok, args.data
    torch.manual_seed(args.seed)
    tok = build_tokenizer()
    model = HMN(VOCAB_SIZE, args.dim, args.state, args.layers, n_experts=16, top_k=2,
                n_mem_cells=8, mem_top_k=4, memory_interval=2)
    print(f"params: {sum(p.numel() for p in model.parameters()):,} | "
          f"mode={'standard' if args.standard else 'distill'} | steps={args.steps}",
          flush=True)

    train = StreamingPairs("train", tok, args.bs, seed=args.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    eos = tok.token_to_id("</s>")
    ignore = lossf.ignore_index
    best = 1e9

    gen = train.batches()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        X, Y = next(gen)
        opt.zero_grad()
        logits = model(X)
        loss = lossf(logits.reshape(-1, VOCAB_SIZE), Y.reshape(-1))
        aux = model.moe_aux_loss()
        (loss + aux).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if args.sched == "cosine":
            sched.step()
        if step % args.eval_every == 0:
            sc = eval_protocol(model, tok, "val", n=15, seed=args.seed, max_new=128)
            aloss = answer_loss(model, tok, "val", n=15, seed=args.seed)
            el = time.time() - t0
            print(f"step {step:5d} loss={loss.item():.3f} aux={aux.item():.3f} "
                  f"ans_loss={aloss:.2f} syntax={sc['syntax']:.2f} api={sc['api']:.2f} "
                  f"run_exact={sc['run_exact']:.2f} [{el:.0f}s]", flush=True)
            if args.save and loss.item() < best:
                best = loss.item()
                torch.save(model.state_dict(), args.save)
                print(f"  saved checkpoint (loss {best:.3f})", flush=True)


if __name__ == "__main__":
    main()
