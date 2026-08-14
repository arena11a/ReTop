"""M12b sandbox (2026-08-14) — is reorder learnable with a START-column pointer?

The M12 finding: the swap gold ("deploy lib055 and fetch pkg028") is a
CONTIGUOUS SUBSTRING of the prompt ("fetch pkg028 and deploy lib055"). The
identity lane's next-token chaining therefore needs only a START column to
anchor to — it does NOT need to "flip attention sinks". The only missing piece
is choosing a start column at row 0 that is NOT the user's first token.

The rigid stem-addr anchor forces c = u + (t-a), so row 0 always points at the
user's first column (u). For the swap that column holds "fetch" not "deploy".

RESULT — fresh 11/16 and 11/18: with the CORRECT continuation fed as input
forward, the copy lane re-emits ~70% of reordered rows EXACTLY (gate 0.997).
Gate diagnostics decompose the misses into THREE distinct micro-walls:

  gold = de p lo y l i b 0 55 and  fetch p k g 02 8
  row0 gate=0.000 MISS      <- start: gate REFUSES to open a copy for the
  row1 gate=0.003 MISS         reordered head (expects ASI-copy or EOS)
  row2 gate=0.997 chained   <- 'deploy lib055' chains fine once past start
  ...
  row8 gate=0.997 chained
  row9 gate=0.003 MISS      <- FRAGMENT SEAM: the ' and' gate closes, and
  row10 gate=0.997 wrong-col  even when open (row10) it points at the wrong
  row11 gate=0.003 MISS        column (' de' not ' fetch') at the seam
  row12 gate=0.997 chained  <- 'pkg028' digits chain fine again

CONCLUSION (refines M12): the register CAN copy reordered contiguous text —
the "flip attention sinks" framing was too pessimistic. The wall decomposes
into (a) row 0 content-initiation: gen has no mechanism and the gate won't
open a copy for a non-anchored head token, and (b) fragment seams: the gate
is local (per-prev-token) and has no way to RESTART the pointer at a
reordered fragment head, so it closes or points wrong exactly at ' and',
then re-opens past it. This points to a concrete mechanism: the gate/anchor
needs an explicit "restart at fragment n" capability (a positional/structural
marker), not a two-sink composition. M12 remains open but is now well-scoped.

Sandbox only: reads a trained ckpt, constructs a custom decode loop inline,
does NOT modify HMN3 / recipe dicts.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import torch
from tokenizers import Tokenizer

from hmn import HMN3
from hmn.recipe import make_chat_ids, resolve_device, seed_guardrail
import train_v3 as tv

TOKENIZER = os.path.join(ROOT, "retop_tokenizer.json")


def main():
    tok = Tokenizer.from_file(TOKENIZER)
    dev = resolve_device()
    m = HMN3(tok.get_vocab_size(), dim=96, state_dim=8, n_layers=3,
             use_moe=False, gate_bias=0.0, asi_id=tv.asi_id(tok),
             keys_proj=False, aux_copy=True, sparse_marginal=True,
             gate_mode="deterministic", use_think=False, k_max=4,
             user_id=tok.token_to_id("<|user|>"), stem_addr=False)
    m.load_state_dict(torch.load("/tmp/opencode/m4_c2b_nothink.pt",
                                 map_location=dev, weights_only=False))
    m.to(dev).eval()
    seed_guardrail(42)

    cases = [
        ("fetch pkg028 and deploy lib055",
         "deploy lib055 and fetch pkg028", "deploy"),
        ("deploy lib005 and fetch pkg019",
         "fetch pkg019 and deploy lib005", "fetch"),
    ]
    # pointer check: at ANSWER rows, does copy_dist's argmax at row t equal
    # gold token t (i.e. can the register re-emit that EXACT column)?
    for user, gold, seed_tok in cases:
        ids_full = make_chat_ids(tok, user)
        gold_toks = tok.encode(gold).ids
        asid = tok.token_to_id("<|assistant|>")
        a = ids_full.index(asid)
        # emulate the decode: prefix = prompt + already-emitted gold prefix
        n_emit = len(gold_toks)
        seq = ids_full + gold_toks
        with torch.no_grad():
            d = m(torch.tensor([seq], device=dev))
        cp = d["copy_dist"][0]           # (T+ngold, V)
        # copy argmax at each answer row, compared to gold_toks
        hits = []
        for t in range(n_emit):
            row = a + t
            pred = int(cp[row].argmax())
            hits.append(pred == gold_toks[t])
            print(f"  row{t}: copy_argmax={tok.decode([pred])!r} "
                  f"gold={tok.decode([gold_toks[t]])!r} match={hits[-1]}")
        print(f"user={user!r}  copy-argmax-along-gold: "
              f"{sum(hits)}/{n_emit} {'CHAINABLE' if all(hits) else 'NOT chainable'}\n")


if __name__ == "__main__":
    main()