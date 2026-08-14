"""M12d micro-pointer-register (2026-08-14) — can an autoregressive COLUMN
POINTER learn the reorder/swap task?

Chain of M12 findings:
  M12   full HMN 1200 steps, swap task: loss plateau ~4.0, unseen 0.000
  M12c  + 'swap:' marker: identical plateau -> not a conditioning problem
  M12b  copy lane re-emits ~70% of swapped rows given a correct START column,
        walls at row 0 content-initiation + the fragment seam.

M12b implies the ONLY missing mechanism is a START-COLUMN pointer that the
gate/anchor can open at row 0 (and at the seam). This probe isolates that
question with the smallest possible realization: an autoregressive pointer
network over prompt columns.

  state   = TransformerEncoder(prompt) + decode over emitted-so-far
  ptr_t   = softmax( Decoder(h, prev_token) @ prompt_keybook )
  emit_t  = ids[ptr_argmax]                 # pure content-addressable copy

RESULT (2026-08-14, 400 steps, seed-42, bs16): training loss 0.0001 (perfect
  fit), seen pairs decode OK, but unseen_acc 0.000 with HTML-visible leakage:
  unseen 'deploy lib003 and fetch pkg061' -> '... lib063 ... pkg001...',
  digit-borrowing straight from the TRAINING slots. Teacher-forced gate trace:
  rows 1-9 gate 1.0 (copy), but at the SEAM row (the ' and fetch' step) the
  gate drops to 0.0 and the GEN head re-emits memorized digits; copy resumes
  gates-1.0 only after the seam.

CONCLUSION: a bare content-addressable pointer with the exact copy-marginal
output DOES NOT fix reorder. The model satisfies the training distribution by
MEMORIZING slot digits and using the gen head at the seam (gate fallback) —
the very behavior HMN shows. The mechanism needed is the one M12b named:
keep the gate OPEN across the fragment seam and re-point the pointer
content-addressably (start-col into the next fragment), i.e. the block that
must generalize is the SEAM RE-SEED, which both HMN and this miniature fail
to learn. M12 therefore is NOT addressable by any isolated pointer; it needs
the gate/anchor to carry an explicit "fragment n start" signal through the
seam. This confirms and hardens M12b's decomposition with an independent
architecture.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import random

import torch
import torch.nn as nn
from tokenizers import Tokenizer

from hmn.recipe import seed_guardrail

TOKENIZER = os.path.join(ROOT, "retop_tokenizer.json")
A_TRAIN = [f"pkg{i:03d}" for i in range(0, 40)]
B_TRAIN = [f"lib{i:03d}" for i in range(40, 80)]
A_UNSEEN = [f"pkg{i:03d}" for i in range(60, 80)]
B_UNSEEN = [f"lib{i:03d}" for i in range(0, 20)]
EOS = 1  # </s>


class PointerNet(nn.Module):
    """Autoregressive pointer over prompt columns.

    prompt: (B, Tin) ids. Answer generated stepwise: at step t the decoder
    consumes the last emitted token, cross-attends the prompt encoder, and
    outputs a column distribution over prompt positions; the emitted token is
    ids[argmax col]. A head-token head (vocab) handles <|assistant|> seeding
    and any non-copied token (e.g. EOS).
    """

    def __init__(self, vocab, d=96, n_layers=2, max_ans=32):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(1, 256, d) * 0.02)
        # plain-matmul encoder blocks (portable; torch CPU SDPA dispatch is
        # flaky on this sandbox -> intermittent illegal-instruction, so avoid
        # nn.TransformerEncoder which uses it).
        self.enc_q = nn.Linear(d, d)
        self.enc_k = nn.Linear(d, d)
        self.enc_v = nn.Linear(d, d)
        self.enc_o = nn.Linear(d, d)
        self.enc_ff = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(),
                                    nn.Linear(2 * d, d))
        self.n_enc = n_layers
        # plain-matmul decoder attention (portable; torch CPU SDPA dispatch is
        # flaky on this sandbox -> intermittent illegal-instruction, so avoid
        # TransformerDecoderLayer which uses it).
        self.dec_q = nn.Linear(d, d)
        self.dec_k = nn.Linear(d, d)
        self.dec_v = nn.Linear(d, d)
        self.dec_o = nn.Linear(d, d)
        self.dec_ff = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(),
                                    nn.Linear(2 * d, d))
        self.q = nn.Linear(d, d)
        self.col_head = nn.Linear(d, 1)   # keybook bias per prompt column
        self.head = nn.Linear(d, vocab)   # for <|assistant|>/EOS/non-copy
        self.gate = nn.Linear(d, 1)       # learned copy-vs-gen weight per row

    def _dec_attn(self, t, mem):
        """t:(B,Tq,d) queries attend mem:(B,Tm,d) -> (B,Tq,d)."""
        q = self.dec_q(t); k = self.dec_k(mem); v = self.dec_v(mem)
        a = q @ k.transpose(1, 2) / (self.d ** 0.5)
        a = torch.softmax(a, dim=-1)
        return self.dec_o(a @ v)

    def _dec_block(self, t, mem):
        h = self._dec_attn(t, mem) + t
        return self.dec_ff(h) + h

    def _enc_block(self, e):
        q = self.enc_q(e); k = self.enc_k(e); v = self.enc_v(e)
        a = torch.softmax(q @ k.transpose(1, 2) / (self.d ** 0.5), dim=-1)
        h = self.enc_o(a @ v) + e
        return self.enc_ff(h) + h

    def encode_prompt(self, x):
        e = self.emb(x) + self.pos[:, :x.shape[1]]
        for _ in range(self.n_enc):
            e = self._enc_block(e)
        return e

    def forward(self, x, target=None):
        """Training: teacher-forced decode over gold answer columns."""
        B, Tin = x.shape
        key = self.encode_prompt(x)                 # (B, Tin, d)
        keys = self.col_head(key).squeeze(-1)       # (B, Tin)
        if target is None:
            return self.greedy(x, key, keys)
        # build decoder input = [assistant_id] + target[:-1]
        asid = 5
        t_in = torch.cat([torch.full((B, 1), asid), target[:, :-1]], 1)
        te = self.emb(t_in) + self.pos[:, :t_in.shape[1]]
        h = self._dec_block(te, key)                    # (B, Tans, d)
        q = self.q(h)
        ptr = torch.softmax((q @ key.transpose(1, 2) + keys.unsqueeze(1))
                            / (self.d ** 0.5), dim=-1)   # (B, Tans, Tin)
        head = self.head(h)                         # (B, Tans, V)
        # copy-marginal: p(col-matching gold) = sum over prompt cols with
        # the target token. Build a full vocab copy distribution by scattering.
        copy = torch.zeros_like(head)
        xg = x.unsqueeze(1).expand(B, ptr.shape[1], -1)   # (B, Tans, Tin)
        copy = copy.scatter_add(-1, xg, ptr)              # (B, Tans, V)
        g = torch.sigmoid(self.gate(h))                   # (B, Tans, 1)
        blend = g * copy + (1 - g) * torch.softmax(head, -1)
        return ptr, head, blend, g

    def greedy(self, x, key=None, keys=None, max_ans=32):
        B, Tin = x.shape
        if key is None:
            key = self.encode_prompt(x)
            keys = self.col_head(key).squeeze(-1)
        out_toks = []
        cur = torch.full((B, 1), 5)                  # <|assistant|>
        for _ in range(max_ans):
            te = self.emb(cur) + self.pos[:, :cur.shape[1]]
            h = self._dec_block(te, key)
            q = self.q(h[:, -1:])
            ptr = torch.softmax((q @ key.transpose(1, 2) + keys.unsqueeze(1))
                                / (self.d ** 0.5), dim=-1)
            head = self.head(h[:, -1:])
            xg = x.unsqueeze(1).expand(B, 1, -1)     # full prompt cols
            copy = torch.zeros_like(head)
            copy = copy.scatter_add(-1, xg, ptr)
            g = torch.sigmoid(self.gate(h[:, -1:]))
            blend = g * copy + (1 - g) * torch.softmax(head, -1)
            tok = int(blend.argmax(-1)[0, 0])
            out_toks.append(tok)
            cur = torch.cat([cur, torch.tensor([[tok]])], 1)
            if tok == EOS:
                break
        return out_toks


def make_batch(tok, a_slots, b_slots, bs, seed):
    rng = random.Random(seed)
    eos = tok.token_to_id("</s>")
    asid = tok.token_to_id("<|assistant|>")
    X, Y = [], []
    for _ in range(bs):
        a = rng.choice(a_slots)
        b = rng.choice(b_slots)
        user = f"fetch {a} and deploy {b}"
        gold = f"deploy {b} and fetch {a}"
        ids = tok.encode(user).ids + [asid]
        y = tok.encode(gold).ids + [eos]
        X.append(ids); Y.append(y)
    Tin = max(len(x) for x in X)
    Tans = max(len(y) for y in Y)
    Xb = torch.full((bs, Tin), 1, dtype=torch.long)
    Yb = torch.full((bs, Tans), -100, dtype=torch.long)
    for j in range(len(X)):
        Xb[j, :len(X[j])] = torch.tensor(X[j])
        Yb[j, :len(Y[j])] = torch.tensor(Y[j])
    return Xb, Yb


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    tok = Tokenizer.from_file(TOKENIZER)
    seed_guardrail(42)
    torch.manual_seed(42)
    m = PointerNet(tok.get_vocab_size())
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    print(f"pointer-net micro (autoregressive column pointer), steps={steps}")
    for step in range(1, steps + 1):
        X, Y = make_batch(tok, A_TRAIN, B_TRAIN, 16, step)
        opt.zero_grad()
        ptr, head, blend, g = m(X, target=Y)
        ce = lossf(torch.log(blend.clamp(min=1e-9)).reshape(-1, tok.get_vocab_size()),
                   Y.reshape(-1))
        loss = ce
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        opt.step()
        if step % 100 == 0 or step == steps:
            rng = random.Random(step)
            ok = tot = 0
            with torch.no_grad():
                for _ in A_UNSEEN:
                    a = rng.choice(A_UNSEEN)
                    b = rng.choice(B_UNSEEN)
                    user = f"fetch {a} and deploy {b}"
                    gold = f"deploy {b} and fetch {a}"
                    ids = tok.encode(user).ids + [tok.token_to_id("<|assistant|>")]
                    Xb = torch.tensor([ids])
                    out = m.greedy(Xb, max_ans=len(tok.encode(gold).ids) + 2)
                    text = tok.decode(out)
                    # cut at EOS if present
                    if "</s>" in text:
                        text = text.split("</s>")[0]
                    tot += 1
                    ok += int(text == gold)
            print(f"  step {step:4d} loss={loss.item():.3f} unseen_acc={ok/tot:.3f}",
                  flush=True)
    print("done")


if __name__ == "__main__":
    main()