"""M12 long-horizon probe — is the reorder FAILURE structural or just slow?

The 240-step probe (experiments/v4/m12_reorder_probe.py) showed unseen 0.000
for a swap transform (user = "fetch {a} and deploy {b}", gold = swapped).
Hypothesis: row 0 must be GEN-emitted ("deploy"), a skill the echo-only gen
head never had to develop (row 0 was always ASI-anchored copy or EOS). Test by
training 1200 steps and watching the unseen curve.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "v4"))

import torch
from tokenizers import Tokenizer

from hmn import HMN3
from hmn.recipe import seed_guardrail

import m12_reorder_probe as m12

TOKENIZER = os.path.join(ROOT, "retop_tokenizer.json")


def main():
    tok = Tokenizer.from_file(TOKENIZER)
    seed_guardrail(42)
    torch.manual_seed(42)
    m = HMN3(tok.get_vocab_size(), dim=96, state_dim=8, n_layers=3,
             use_moe=False, gate_bias=0.0, asi_id=tok.token_to_id("<|assistant|>"), aux_copy=True, sparse_marginal=True,
             gate_mode="deterministic", use_think=False, k_max=4,
             user_id=tok.token_to_id("<|user|>"), stem_addr=False)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    lossf = torch.nn.CrossEntropyLoss(ignore_index=-100)
    for step in range(1, 1201):
        X, Y = m12.make_reorder_batch(tok, m12.A_TRAIN, m12.B_TRAIN, 16, step)
        opt.zero_grad()
        out = m(X)["logits"]
        loss = lossf(out.reshape(-1, out.shape[-1]), Y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        opt.step()
        if step % 200 == 0:
            m.eval()
            acc = m12.eval_reorder(m, tok, m12.A_UNSEEN, m12.B_UNSEEN, seed=step)
            print(f"step {step:4d} loss={loss.item():.3f} unseen_acc={acc:.3f}",
                  flush=True)
            m.train()
    print("done", flush=True)


if __name__ == "__main__":
    main()