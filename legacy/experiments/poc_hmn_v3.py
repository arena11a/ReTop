"""PoC hmn_v3 vs v2 on the slot-copy ceiling (distill_design attempt-3 dead-end).

Task A (slot-copy): "<s><|user|>pip install {pkg}<|assistant|>pip install {pkg}</s>"
  train packages: 60 seen; VAL packages: 40 UNSEEN (content matches prompt exactly).
  v2 measured: 0/40 exact (emits dominant train pkg). v3 must >= 85% via register copy.
Task B (recall regression): pairs in context -> exact answer via memory lane.

Metric = exact token-sequence equality of the assistant response.
"""
import argparse, os, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from hmn_v3 import HMN3, HMN3_NoReg
from hmn_v2 import HMN

VOCAB = 3190
TOKENIZER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "retop_tokenizer.json")
EOS = "</s>"
ASI = "<|assistant|>"

PKGS_SEEN = [f"pkg{i:03d}" for i in range(60)]
PKGS_UNSEEN = [f"pkg{i:03d}" for i in range(60, 100)]  # 40 holdout
KEYS = [f"k{i}" for i in range(50)]


def build_tok():
    return Tokenizer.from_file(TOKENIZER_PATH)


def ids_of(tok, s):
    return tok.encode(s).ids


def eos_id(tok):
    return tok.token_to_id(EOS)


def asi_id(tok):
    return ids_of(tok, ASI)[0]


def make_slot_batch(tok, pkgs, bs, seed):
    rng = random.Random(seed)
    X, Y = [], []
    for _ in range(bs):
        p = rng.choice(pkgs)
        user, gold = f"pip install {p}", f"pip install {p}"
        ids = ids_of(tok, f"<s><|user|>{user}<|assistant|>{gold}</s>")
        targets = ids[1:] + [eos_id(tok)]
        asi = asi_id(tok)
        a = ids.index(asi)
        for i in range(a):
            targets[i] = -100
        # force full-length equality later via padding in collate
        X.append(ids); Y.append(targets)
    T = max(len(x) for x in X)
    Xb = torch.full((bs, T), eos_id(tok), dtype=torch.long)
    Yb = torch.full((bs, T), -100, dtype=torch.long)
    for j, (x, y) in enumerate(zip(X, Y)):
        Xb[j, :len(x)] = torch.tensor(x); Yb[j, :len(y)] = torch.tensor(y)
    return Xb, Yb


def make_recall_batch(tok, bs, seed):
    rng = random.Random(seed)
    X, Y = [], []
    for _ in range(bs):
        n = rng.randint(3, 8)
        pairs = [(rng.choice(KEYS), rng.randint(0, 9999)) for _ in range(n)]
        q = rng.choice(pairs)
        seq = " ".join(f"{k} {v}" for k, v in pairs)
        user, gold = f"START {seq} ? {q[0]}", str(q[1])
        ids = ids_of(tok, f"<s><|user|>{user}<|assistant|>{gold}</s>")
        targets = ids[1:] + [eos_id(tok)]
        asi = asi_id(tok); a = ids.index(asi)
        for i in range(a):
            targets[i] = -100
        X.append(ids); Y.append(targets)
    T = max(len(x) for x in X)
    Xb = torch.full((bs, T), eos_id(tok), dtype=torch.long)
    Yb = torch.full((bs, T), -100, dtype=torch.long)
    for j, (x, y) in enumerate(zip(X, Y)):
        Xb[j, :len(x)] = torch.tensor(x); Yb[j, :len(y)] = torch.tensor(y)
    return Xb, Yb


def decode(model, tok, prompt, max_new=64, gate_report=False):
    ids = list(prompt)
    last_gate = None
    with torch.no_grad():
        for _ in range(max_new):
            inp = torch.tensor([ids])
            if gate_report:
                out = model(inp, return_gate=True)
                if isinstance(out, tuple):
                    logits, g = out
                else:
                    logits, g = out["logits"], out["g"].squeeze(-1)
                last_gate = g[0, -1].item()
                nxt = logits[0, -1].argmax(-1).item()
            else:
                out = model(inp)
                logits = out if isinstance(out, torch.Tensor) else out["logits"]
                nxt = logits[0, -1].argmax(-1).item()
            if nxt == eos_id(tok):
                break
            ids.append(nxt)
    return tok.decode(ids[len(prompt):]), last_gate


def eval_slots(model, tok, pkgs, seed=0, gate_report=True):
    model.eval()
    ok = 0; total = 0; gates = []
    for p in pkgs:
        prompt = ids_of(tok, f"<s><|user|>pip install {p}<|assistant|>")
        out, g = decode(model, tok, prompt, gate_report=gate_report)
        total += 1
        ok += int(out.strip() == f"pip install {p}")
        if g is not None:
            gates.append(g)
    model.train()
    return ok / max(1, total), (sum(gates) / len(gates) if gates else 0.0)


def eval_recall(model, tok, n=50, seed=0):
    model.eval(); ok = 0
    for _ in range(n):
        X, _ = make_recall_batch(tok, 1, seed=_ * 7)
        prompt = X[0].tolist()
        asi = asi_id(tok); a = prompt.index(asi)
        gold = tok.decode(prompt[a+1:])
        gold = gold.split("</s>")[0].strip()
        prompt_ids = prompt[:a+1]
        out, _ = decode(model, tok, prompt_ids, max_new=16)
        ok += int(out.strip() == gold)
    model.train()
    return ok / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gate-bias", type=float, default=-2.0)
    ap.add_argument("--no-reg", action="store_true", help="ablate: softmax head only")
    ap.add_argument("--v2-base", action="store_true",
                    help="baseline: plain HMN (v2) coupling+MoE backbone, LM head")
    ap.add_argument("--mix-recall", type=float, default=0.3,
                    help="fraction of steps that train on recall task")
    args = ap.parse_args()

    torch.manual_seed(0)
    tok = build_tok()
    if args.v2_base:
        model = HMN(VOCAB, args.dim, 8, args.layers, n_experts=16, top_k=2,
                    n_mem_cells=8, mem_top_k=4, memory_interval=2)
    elif args.no_reg:
        model = HMN3_NoReg(VOCAB, args.dim, 8, args.layers)
    else:
        model = HMN3(VOCAB, dim=args.dim, state_dim=8, n_layers=args.layers,
                     gate_bias=args.gate_bias, use_moe=False, asi_id=asi_id(tok))
    print(f"params: {sum(p.numel() for p in model.parameters()):,} "
          f"| model={'v2-base' if args.v2_base else ('NoReg' if args.no_reg else 'v3-register')}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    best_seen = 0.0
    t0 = time.time()

    for step in range(1, args.steps + 1):
        r = random.Random(step)
        if r.random() < args.mix_recall:
            X, Y = make_recall_batch(tok, args.bs, step)
        else:
            X, Y = make_slot_batch(tok, PKGS_SEEN, args.bs, step)
        opt.zero_grad()
        out = model(X)
        logits = out if isinstance(out, torch.Tensor) else out["logits"]
        loss = lossf(logits.reshape(-1, VOCAB), Y.reshape(-1))
        if isinstance(out, dict) and "copy_dist" in out:
            # aux: shape the register to point at the correct context token
            aux = lossf(out["copy_dist"].reshape(-1, VOCAB), Y.reshape(-1))
            loss = loss + 1.0 * aux
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % 300 in (0, 299) or step == args.steps:
            gate_report = not args.v2_base
            s_acc, _g = eval_slots(model, tok, PKGS_SEEN, seed=7,
                                   gate_report=gate_report)
            u_acc, ug = eval_slots(model, tok, PKGS_UNSEEN, seed=9,
                                   gate_report=gate_report)
            if args.mix_recall > 0:
                r_acc = eval_recall(model, tok, n=25)
            else:
                r_acc = float("nan")
            tl = time.time() - t0
            print(f"step {step:5d} loss={loss.item():.3f} "
                  f"seen={s_acc:.3f} unseen={u_acc:.3f} (gate_seen={_g:.2f} gate_unseen={ug:.2f}) "
                  f"recall={r_acc:.3f} [{tl:.0f}s]", flush=True)
            best_seen = max(best_seen, s_acc)
            torch.save(model.state_dict(), f"hmn_poc_{'v2' if args.v2_base else ('nr' if args.no_reg else 'v3')}.pt")


if __name__ == "__main__":
    main()