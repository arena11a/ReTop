"""retop — one-command build/train/chat for the ReTop (HMN v3) architecture.

The vision: YOU bring data + hardware. retop designs the architecture, the
tokenizer, the training recipe, and the inference loop — no architecture
knowledge required. Machine specs are auto-detected (CPU/RAM/CUDA) and mapped to
a verified config (see docs/hmn_v3_design.md + task findings for what works).

Three commands:

  # 1. train a tokenizer from a raw-text corpus (streaming, low-RAM)
  python retop.py tok --data my_corpus.txt --out tok.json --vocab 4000

  # 2. train a model. Data = .jsonl chat pairs or .txt corpus.
  python retop.py train --data my_chat.jsonl --tok tok.json --out myai.pt
  #    --spec auto: pick config from hardware.  --spec small|medium|large|xl

  # 3. chat with the trained model (reads the config sidecar myai.pt.json)
  python retop.py chat --checkpoint myai.pt --interactive
  python retop.py chat --checkpoint myai.pt --prompt "pip install numpy"

Verified recipe baked in (2026-08-12, from the v1/v2 findings):
  - constant LR always (cosine collapses the small-model LR prematurely)
  - loss masked to the ASSISTANT response region for chat data (prompt tokens
    starve the answer head — measured 14.7 nat answer loss under full loss)
  - dual-head decoder (gen + hard-copy) for slot-copy that softmax cannot do
  - reversible blocks + streaming chunked data -> trains on just a few GB RAM
"""
import argparse
import json
import os
import random
import sys
import time

import torch
import torch.nn as nn

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from hmn import HMN3, HMN3_NoReg, HMN3AttentionWR
from hmn.checkpoint import load_compat
from hmn.config import HMNConfig, create_model
from hmn.recipe import (decode_v33, loss_v33, make_chat_ids, make_chat_targets,
                        resolve_device, seed_guardrail)

ROOT = os.path.dirname(os.path.abspath(__file__))
EOS = "</s>"
UNK = "<unk>"
PAD = "<pad>"
BOS = "<s>"
USER = "<|user|>"
ASSIST = "<|assistant|>"

SPECIAL_TOKENS = [BOS, EOS, UNK, PAD, USER, ASSIST]

# spec name -> hyperparams. dim/layers = task2-verified region (D64-128);
# cells -> not parametrized here (v3 uses in-context register). bs/seq/steps
# scale with the machine so "auto" is genuinely usable on a weak box.
SPECS = {
    "tiny":       dict(dim=64,  layers=3, moe=False, bs=8,  seq=64,  steps=2500, gate_bias=-1.0),
    "small":      dict(dim=96,  layers=3, moe=False, bs=8,  seq=96,  steps=4000, gate_bias=-1.0),
    "medium":     dict(dim=144, layers=4, moe=True,  bs=8,  seq=128, steps=6000, gate_bias=-1.0),
    "large":      dict(dim=192, layers=6, moe=True,  bs=12, seq=160, steps=8000, gate_bias=0.0),
    "xl":         dict(dim=256, layers=8, moe=True,  bs=16, seq=192, steps=10000, gate_bias=0.0),
    "attn-tiny":  dict(dim=64,  layers=2, moe=False, bs=8,  seq=64,  steps=2500, gate_bias=-1.0, variant="attention"),
    "attn-small": dict(dim=96,  layers=3, moe=False, bs=8,  seq=96,  steps=4000, gate_bias=-1.0, variant="attention"),
    "attn-medium":dict(dim=128, layers=3, moe=False, bs=8,  seq=128, steps=6000, gate_bias=-1.0, variant="attention"),
    "attn-large": dict(dim=256, layers=4, moe=False, bs=8,  seq=192, steps=8000, gate_bias=0.0, variant="attention"),
}


def detect_spec():
    """Pick a spec string from the machine. Pure heuristics, no guarantees —
    only reads counts, conservatively low so even the auto pick runs on CPU."""
    cpus = os.cpu_count() or 4
    mem_gb = 0.0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    mem_gb = int(line.split()[1]) / 1024 / 1024
                    break
    except OSError:
        pass
    cuda = torch.cuda.is_available()
    mem_gb = mem_gb if mem_gb else 4.0
    if cuda:
        return "xl" if mem_gb >= 16 else "large"
    if mem_gb >= 32 or cpus >= 16:
        return "large"
    if mem_gb >= 12 or cpus >= 8:
        return "medium"
    if mem_gb >= 6:
        return "small"
    return "tiny"


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------

def build_tokenizer(data, out, vocab, special_tokens=SPECIAL_TOKENS):
    if os.path.isdir(data):
        files = [os.path.join(data, f) for f in sorted(os.listdir(data))
                 if f.endswith((".txt", ".jsonl", ".text"))]
    else:
        files = [data]
    for f in files:
        if not os.path.exists(f):
            raise SystemExit(f"data file not found: {f}")

    tok = Tokenizer(models.BPE(unk_token=UNK))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel(add_prefix_space=True)
    trainer = trainers.BpeTrainer(vocab_size=vocab, special_tokens=special_tokens,
                                  min_frequency=2, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train(files, trainer)
    tok.save(out)
    print(f"tokenizer: vocab={tok.get_vocab_size()} -> {out}", flush=True)
    return tok


# ---------------------------------------------------------------------------
# data: streaming, two formats
# ---------------------------------------------------------------------------

def iter_lines(data):
    if os.path.isdir(data):
        names = [os.path.join(data, f) for f in sorted(os.listdir(data))
                 if f.endswith((".jsonl", ".txt"))]
    else:
        names = [data]
    for name in names:
        with open(name, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


class Dataset:
    """Streams records and encodes them on the fly; train/val split by a
    deterministic hash so re-runs with the same --seed are reproducible.

    .jsonl chat record: {"messages":[{role:user,content},{role:assistant,content}]}
      -> encode <s><|user|>..<|assistant|>..</s>, mask loss to assistant region
    .jsonl text record / .txt file:
      -> plain-LM blocks of `seq` tokens (streamed, never fully in RAM)
    """

    def __init__(self, data, tok, seq, val_frac=0.05, seed=0):
        self.tok = tok
        self.seq = seq
        self.val_frac = val_frac
        self.seed = seed
        self.eos = tok.token_to_id(EOS)
        self.data = data
        if not os.path.isdir(data) and not os.path.exists(data):
            raise SystemExit(f"--data not found: {data}")

    def _split(self, key):
        h = hash((self.seed, key)) % 1000
        return "val" if h < self.val_frac * 1000 else "train"

    def records(self, split):
        """(X, Y, Yc, G) tensors, padded to batch max length, response-masked.

        Chat records also carry identity-register labels (Yc = copy targets,
        G = gate target) computed by hmn/recipe.make_chat_targets so loss_v33
        can train the copy lane; plain-text records are all-ignore for the
        copy channel (blend CE only).
        """
        batch_x, batch_y, batch_yc, batch_g = [], [], [], []
        asid = self.tok.token_to_id(ASSIST)
        for line in iter_lines(self.data):
            try:
                rec = json.loads(line)
            except ValueError:
                rec = None
            if isinstance(rec, dict) and "messages" in rec:
                msgs = rec["messages"]
                if len(msgs) < 2 or msgs[1]["role"] != "assistant":
                    continue
                user, gold = msgs[0]["content"], msgs[1]["content"]
                if self._split(user[:40]) != split:
                    continue
                ids = make_chat_ids(self.tok, user, gold)
                y, yc, gt = make_chat_targets(ids, asid, self.eos)
                batch_x.append(ids)
                batch_y.append(y)
                batch_yc.append(yc)
                batch_g.append(gt)
            else:
                # plain text: tokenize one line -> block tokens
                if self._split(line[:60]) != split:
                    continue
                ids = self.tok.encode(line).ids[:self.seq]
                y = ids[1:] + [self.eos]
                batch_x.append(ids)
                batch_y.append(y)
                batch_yc.append([-100] * len(y))
                batch_g.append([-1.0] * len(y))
            if len(batch_x) >= 32:
                yield from self._emit(batch_x, batch_y, batch_yc, batch_g)
                batch_x, batch_y, batch_yc, batch_g = [], [], [], []
        if batch_x:
            yield from self._emit(batch_x, batch_y, batch_yc, batch_g)

    def _emit(self, xs, ys, ycs, gs):
        T = max(len(x) for x in xs)
        Xb = torch.full((len(xs), T), self.eos, dtype=torch.long)
        Yb = torch.full((len(xs), T), -100, dtype=torch.long)
        YcB = torch.full((len(xs), T), -100, dtype=torch.long)
        Gb = torch.full((len(xs), T), -1.0, dtype=torch.float)
        for j, (x, y, yc, g) in enumerate(zip(xs, ys, ycs, gs)):
            Xb[j, :len(x)] = torch.tensor(x)
            Yb[j, :len(y)] = torch.tensor(y)
            YcB[j, :len(yc)] = torch.tensor(yc)
            Gb[j, :len(g)] = torch.tensor(g)
        yield Xb, Yb, YcB, Gb

    def batches(self, split, bs, rng):
        buf = list(self.records(split))
        while True:
            rng.shuffle(buf)
            for i in range(0, len(buf), bs):
                yield buf[i]


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def build_model(arch, cfg, tok):
    asi = tok.token_to_id(ASSIST) if tok.token_to_id(ASSIST) is not None else None
    variant = cfg.get("variant", "ssm")
    if arch == "plain":
        return HMN3_NoReg(cfg["vocab"], dim=cfg["dim"], state_dim=8,
                          n_layers=cfg["layers"])
    if variant == "attention":
        return HMN3AttentionWR(cfg["vocab"], dim=cfg["dim"],
                               n_layers=cfg["layers"],
                               use_moe=cfg["moe"],
                               gate_bias=cfg["gate_bias"], asi_id=asi)
    return HMN3(cfg["vocab"], dim=cfg["dim"], state_dim=8,
                n_layers=cfg["layers"], use_moe=cfg["moe"],
                gate_bias=cfg["gate_bias"], asi_id=asi, aux_copy=True,
                exact_blend=cfg.get("exact_blend", False))


def train(args, cfg):
    seed_guardrail(args.seed)
    tok = Tokenizer.from_file(args.tok)
    vocab = tok.get_vocab_size()
    cfg.update(vocab=vocab, arch=args.arch, seed=args.seed, spec=args.spec,
               exact_blend=getattr(args, "exact_blend", False),
               tokenizer=os.path.abspath(args.tok))
    model = build_model(args.arch, cfg, tok)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"arch={args.arch} spec={args.spec} params={n_params:,} "
          f"vocab={vocab} dim={cfg['dim']} layers={cfg['layers']} "
          f"moe={cfg['moe']} bs={cfg['bs']} seq={cfg['seq']} steps={cfg['steps']}",
          flush=True)

    data = Dataset(args.data, tok, seq=cfg["seq"], val_frac=0.05, seed=args.seed)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = random.Random(args.seed)

    ev = args.eval_every or max(100, cfg["steps"] // 20)
    best = float("inf")
    t0 = time.time()
    for step in range(1, cfg["steps"] + 1):
        X, Y, Yc, G = next(iter(data.batches("train", cfg["bs"], rng)))
        opt.zero_grad()
        out = model(X)
        if isinstance(out, dict):
            # v3 dual-head: hmn/recipe.loss_v33 = blend CE + gen CE (gen rows
            # only) + manual -log p copy CE. NOT CE(copy_dist, Y): copy_dist is
            # already a probability and CE would log-softmax it again, pinning
            # the loss at ~ln(VOCAB) with no gradient.
            loss, l_blend, l_gen, l_copy = loss_v33(out, Y, Yc, G, lossf=lossf,
                                                     w_copy=args.w_copy)
            if hasattr(model, 'moe_aux_loss'):
                loss = loss + model.moe_aux_loss()
        else:
            loss = lossf(out.reshape(-1, vocab), Y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % ev == 0 or step == cfg["steps"]:
            vl, vt = 0.0, 0
            vgen = data.batches("val", cfg["bs"], random.Random(step))
            for _ in range(64):
                X, Y, _Yc, _G = next(vgen, (None, None, None, None))
                if X is None:
                    break
                with torch.no_grad():
                    out = model(X)
                    # v6 M1-B: stats path has no blended logits — val metric
                    # falls back to the gen head (same scale, log-probs).
                    logits = out if isinstance(out, torch.Tensor) \
                        else out.get("logits", out["gen_logits"])
                    m = Y != -100
                    if m.sum() == 0:
                        continue
                    vl += lossf(logits[m].reshape(-1, vocab),
                                Y[m].reshape(-1)).item() * int(m.sum())
                    vt += int(m.sum())
            print(f"step {step:5d} train={loss.item():.3f} val={vl/vt:.3f} "
                  f"lr={opt.param_groups[0]['lr']:.1e} [{time.time()-t0:.0f}s]",
                  flush=True)
            if vl / vt < best:
                best = vl / vt
                torch.save(model.state_dict(), args.out)
                with open(args.out + ".json", "w") as f:
                    json.dump(cfg, f, indent=2)
                print(f"  * best val {best:.3f} -> {args.out}", flush=True)
    print(f"DONE. best val CE {best:.3f} (checkpoint: {args.out})", flush=True)


# ---------------------------------------------------------------------------
# chat / inference
# ---------------------------------------------------------------------------

def chat(args):
    side = args.checkpoint + ".json"
    if os.path.exists(side):
        with open(side) as f:
            cfg = json.load(f)
    else:
        raise SystemExit(f"no config sidecar {side} — retrain with retop.py train")
    tok = Tokenizer.from_file(cfg["tokenizer"])
    model = build_model(cfg.get("arch", "v3"), cfg, tok)
    dev = resolve_device(args.device)
    load_compat(model, args.checkpoint, device=dev)
    model.to(dev).eval()
    print(f"loaded {cfg.get('arch', 'v3')} ({sum(p.numel() for p in model.parameters()):,} "
          f"params, spec={cfg.get('spec')}) from {args.checkpoint}", flush=True)
    if args.interactive:
        while True:
            try:
                user = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user:
                continue
            prompt = make_chat_ids(tok, user)
            print(decode_v33(model, tok, prompt, args.max_new, device=dev)[0])
    else:
        if not args.prompt:
            raise SystemExit("need --prompt or --interactive")
        prompt = make_chat_ids(tok, args.prompt)
        print(decode_v33(model, tok, prompt, args.max_new, device=dev)[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("tok", help="train a BPE tokenizer from raw text")
    p.add_argument("--data", required=True, help=".txt corpus or dir of .txt")
    p.add_argument("--out", required=True)
    p.add_argument("--vocab", type=int, default=4000)
    p.set_defaults(fn=build_tokenizer)

    p = sub.add_parser("train", help="train a model (auto architecture)")
    p.add_argument("--data", required=True, help=".jsonl chat/text or .txt corpus")
    p.add_argument("--tok", required=True, help="tokenizer from `retop tok`")
    p.add_argument("--out", default="model.pt", help="checkpoint path")
    p.add_argument("--spec", default="auto",
                   choices=["auto"] + list(SPECS),
                   help="architecture size (auto = from this machine)")
    p.add_argument("--arch", default="v3", choices=["v3", "plain", "attention"],
                   help="v3 = dual-head copy+gen (default), plain = softmax-only, "
                        "attention = v7 attention-WR variant")
    p.add_argument("--steps", type=int, default=None, help="override spec steps")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--w-copy", type=float, default=1.0)
    p.add_argument("--exact-blend", action="store_true",
                   help="v6 M1-B: legacy dense blended-logits oracle instead of "
                        "the IRStats index path")
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=train, eval_every=None)

    p = sub.add_parser("chat", help="talk to a trained checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--max-new", type=int, default=96)
    p.add_argument("--prompt", default=None, help="one-shot; or --interactive")
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None,
                   help="compute device (auto-detect default; see resolve_device)")
    p.set_defaults(fn=chat)

    args = ap.parse_args()
    if args.cmd == "train":
        cfg = dict(SPECS[args.spec if args.spec != "auto" else detect_spec()])
        if args.steps:
            cfg["steps"] = args.steps
        args.spec = args.spec if args.spec != "auto" else detect_spec()
        train(args, cfg)
    elif args.cmd == "tok":
        build_tokenizer(args.data, args.out, args.vocab)
    else:
        chat(args)


if __name__ == "__main__":
    main()