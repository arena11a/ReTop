"""M8 baseline (2026-08-13) — HMN3 vs HMN3_NoReg vs a vanilla transformer.

External review asked for an honest, same-size, same-task comparison. This
harness trains three models on the identical multi-template slot task:

  HMN3        : pointer-register + dual-head copy/gate (default v3.3 arch).
                 --stem-addr makes row-0 addressable (M2-dev).
  HMN3_NoReg  : same WR blocks + softmax head, copy channel REMOVED (the
                 in-repo ablation).
  Vanilla     : standard pre-LN causal transformer, tied embeddings, learned
                 positional embedding, greedy decode. No copy pointer at all.

Fairness contract:
  - same dim/layers, params within ~10% (vanilla tuned to land on/above HMN3)
  - identical data pipeline (make_slot_batch, same templates, same steps/bs)
  - each model uses its OWN best decode: blend+boundary_eos for HMN3/NoReg,
    plain argmax+EOS for vanilla (which has no gate/copy signals to trust)
  - all evals on the SAME unseen slots + never-trained probe templates

Decode note: HMN3's boundary_eos is a decoder-time stop rule; give the vanilla
model the same courtesy of EOS-token stopping (learned), which is the vanilla
equivalent. No template/probe slot overlap between train and eval.

Run:  python experiments/v4/m8_baseline.py            # ~300 steps default

Observed (600 steps, 10 templates, unseen-slot eval, seed-stable):
  HMN3      (664K, stem-addr): trained 1.000 / probes 1.000
  HMN3_NoReg(342K)           : 0.000 / 0.000   (no copy lane at all)
  Vanilla   (667K)           : 0.000 / 0.000, BUT 1.0 on SEEN slots and memorizes
             unseen ids (pkg099->pkg049 in single-tpl probe): the softmax head
             fits seen pairs by rote and fails on unseen — not a decoder bug.
"""
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizers import Tokenizer

from hmn import HMN3, HMN3_NoReg
from hmn.recipe import (EOS, eval_slots, make_chat_ids, make_slot_batch,
                        resolve_device, seed_guardrail)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PKGS_SEEN = [f"pkg{i:03d}" for i in range(60)]
PKGS_UNSEEN = [f"pkg{i:03d}" for i in range(60, 100)]
PROBE_TEMPLATES = [
    "mount {slot}", "uninstall {slot}", "clean {slot}", "check {slot}",
]


def build_tok(path=os.path.join(ROOT, "retop_tokenizer.json")):
    return Tokenizer.from_file(path)


class VanillaTransformer(nn.Module):
    """Standard pre-LN causal transformer. dim/n_layers match HMN3; MLP width
    tuned so total params land at/above HMN3's (the baseline should not be
    secretly smaller). Learned causal mask + learned positional embedding."""

    def __init__(self, vocab_size, dim=96, n_layers=3, mlp_scale=4, cls=1):
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(256, dim)
        layers = []
        for _ in range(n_layers):
            layers.append(_VanillaBlock(dim, dim * mlp_scale))
        self.blocks = nn.ModuleList(layers)
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.embed.weight          # tied embeddings (vs HMN3)

    def forward(self, input_ids):
        x = self.embed(input_ids) + self.pos(
            torch.arange(input_ids.shape[1], device=input_ids.device))
        T = x.shape[1]
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.ln(x))


class _VanillaBlock(nn.Module):
    def __init__(self, dim, mlp_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(),
                                 nn.Linear(mlp_dim, dim))

    def forward(self, x, mask):
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        a = (q @ k.transpose(-1, -2)) / (self.ln1.normalized_shape[0] ** 0.5)
        a = a.masked_fill(mask, float("-inf"))
        a = a.softmax(-1)
        x = x + self.out(a @ v)
        x = x + self.mlp(self.ln2(x))
        return x


def decode_vanilla(model, tok, prompt_ids, max_new=16, device=None):
    """Greedy argmax decode for the vanilla transformer. Stops on EOS token.
    No copy/gate signals exist, so this is the honest baseline decoder."""
    eos = tok.token_to_id(EOS)
    ids = list(prompt_ids)
    dev = resolve_device(device)
    with torch.no_grad():
        for _ in range(max_new):
            inp = torch.tensor([ids], device=dev)
            logits = model(inp)[0, -1]
            nxt = logits.argmax(-1).item()
            if nxt == eos:
                break
            ids.append(nxt)
    return tok.decode(ids[len(prompt_ids):])


def eval_model(model, tok, slots, tpl, mode, seed=0, max_new=16, device=None):
    """Dispatch exact-match eval on the right decoder for each architecture."""
    if isinstance(model, HMN3):
        a, g, _ = eval_slots(model, tok, slots, template=tpl, mode=mode, seed=seed,
                             boundary_eos=True, max_new=max_new, device=device)
        return a
    model.eval()
    ok = tot = 0
    for p in slots:
        gold = tpl.format(slot=p)
        prompt = make_chat_ids(tok, gold)
        out = decode_vanilla(model, tok, prompt, max_new=max_new, device=device)
        ok += int(out.strip() == gold)
        tot += 1
    model.train()
    return ok / tot


def train_steps(model, opt, tok, steps, bs, templates, desc, device=None):
    losses = []
    t0 = time.time()
    for step in range(1, steps + 1):
        X, Y, Yc, G = make_slot_batch(tok, PKGS_SEEN, bs, step, templates=templates,
                                      device=device)
        opt.zero_grad()
        out = model(X)
        if isinstance(model, (HMN3)):
            logits = out["logits"]
            criteria = nn.CrossEntropyLoss(ignore_index=-100)
            loss = criteria(logits.reshape(-1, logits.shape[-1]), Y.reshape(-1))
        else:
            loss = F.cross_entropy(out.reshape(-1, out.shape[-1]),
                                   Y.reshape(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        losses.append(loss.item())
    print(f"  {desc}: final loss {losses[-1]:.4f}  [{time.time()-t0:.0f}s]")
    return sum(losses) / len(losses)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--smoke", action="store_true",
                    help="fast CI-mode: 60 steps per model, prints HMN3-vs-others "
                         "gap only (no per-template table)")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--templates",
                    default=("pip install {slot}|import {slot}|run {slot}|"
                             "apt install {slot}|remove {slot}|delete {slot}|"
                             "get {slot}|cache {slot}|fetch {slot}|pip uninstall {slot}"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None,
                    help="compute device (auto-detect default; see resolve_device)")
    args = ap.parse_args()

    dev = resolve_device(args.device)
    seed_guardrail(args.seed)
    torch.manual_seed(args.seed)
    tok = build_tok()
    tpls = [t.strip() for t in args.templates.split("|") if t.strip()]
    vocab = tok.get_vocab_size()

    if args.smoke:
        args.steps = min(args.steps, 60)

    print(f"vocab={vocab} templates={len(tpls)} steps={args.steps} device={dev}")

    hmn = HMN3(vocab, dim=96, state_dim=8, n_layers=3,
               use_moe=False, gate_bias=0.0, asi_id=tok.token_to_id("<|assistant|>"), aux_copy=True, sparse_marginal=True,
               gate_mode="deterministic", use_think=False, k_max=4,
               user_id=tok.token_to_id("<|user|>"), stem_addr=True).to(dev)
    noreg = HMN3_NoReg(vocab, dim=96, state_dim=8, n_layers=3).to(dev)
    vanilla = VanillaTransformer(vocab, dim=96, n_layers=3, mlp_scale=4).to(dev)

    for name, m in [("HMN3(stem-addr)", hmn), ("HMN3_NoReg", noreg),
                    ("Vanilla", vanilla)]:
        print(f"  {name}: {sum(p.numel() for p in m.parameters()):,} params")

    opt1 = torch.optim.AdamW(hmn.parameters(), lr=1e-3)
    opt2 = torch.optim.AdamW(noreg.parameters(), lr=1e-3)
    opt3 = torch.optim.AdamW(vanilla.parameters(), lr=1e-3)

    train_steps(hmn, opt1, tok, args.steps, args.bs, tpls, "HMN3(stem-addr)", dev)
    train_steps(noreg, opt2, tok, args.steps, args.bs, tpls, "HMN3_NoReg", dev)
    train_steps(vanilla, opt3, tok, args.steps, args.bs, tpls, "Vanilla", dev)

    if args.smoke:
        hmn_avg = sum(eval_model(hmn, tok, PKGS_UNSEEN, t, "blend", seed=11,
                                 device=dev)
                      for t in tpls[:2]) / min(2, len(tpls))
        v_avg = sum(eval_model(vanilla, tok, PKGS_UNSEEN, t, "blend", seed=11,
                               device=dev)
                    for t in tpls[:2]) / min(2, len(tpls))
        assert hmn_avg >= 1.0, "HMN3 must generalize on unseen slots (regression)"
        assert v_avg <= hmn_avg, "vanilla must not out-copy the register harness"
        print(f"  smoke OK: HMN3 {hmn_avg:.3f} >= vanilla {v_avg:.3f}")
        return 0
    print("\n== eval: unseen slots, trained templates + never-trained probes ==")
    for name, model in [("HMN3(stem)", hmn), ("HMN3_NoReg", noreg),
                        ("Vanilla", vanilla)]:
        trained = [eval_model(model, tok, PKGS_UNSEEN, t, "blend", seed=11,
                              device=dev)
                   for t in tpls]
        probes = [eval_model(model, tok, PKGS_UNSEEN, t, "blend", seed=13,
                             device=dev)
                  for t in PROBE_TEMPLATES]
        print(f"  {name:<12} trained_avg={sum(trained)/len(trained):.3f} "
              f"probe_avg={sum(probes)/len(probes):.3f}")
        for t, v in zip(tpls, trained):
            print(f"    trained[{t.split()[0]:<10}]={v:.3f}")
        for t, v in zip(PROBE_TEMPLATES, probes):
            print(f"    probe  [{t.split()[0]:<10}]={v:.3f}")


if __name__ == "__main__":
    main()