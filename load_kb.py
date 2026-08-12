"""Load the Python+build knowledge base (Step 3) into the episodic memory bank.

The KB entries are static factual snippets that should NOT live in the model's
weights. We seed the DifferentiableEpisodicMemory cells with their embeddings:
  - key   = mean embedding of the entry's "key" phrase + topic
  - value = mean embedding of the entry's "text"
using the model's own tokenizer + embedding table (so retrieval is compatible
with the token representation the model actually sees). Cells are then frozen
(non-trainable) so training cannot overwrite the curated knowledge.

Usage:
    from distill_kb import load_kb_into_memory
    load_kb_into_memory(model.memory, model.embed, tok, freeze=True)
"""
import json
import torch
from distill_kb import KB


def embed_phrase(embed, tok, phrase):
    ids = tok.encode(phrase).ids
    if not ids:
        return torch.zeros(embed.weight.shape[1])
    vecs = embed(torch.tensor([ids]))            # (1, T, dim)
    return vecs.squeeze(0).mean(dim=0)           # (dim,)


def load_kb_into_memory(memory, embed, tok, freeze=True):
    n = len(KB)
    D = embed.weight.shape[1]
    keys = torch.stack([embed_phrase(embed, tok, f"{e['topic']} {e['key']}") for e in KB])
    vals = torch.stack([embed_phrase(embed, tok, e["text"]) for e in KB])
    # pad to n_cells if KB < cells, else truncate to cells
    n_cells = memory.n_cells
    if n > n_cells:
        keys = keys[:n_cells]; vals = vals[:n_cells]; n = n_cells
    elif n < n_cells:
        keys = torch.cat([keys, torch.randn(n_cells - n, D) * 0.1])
        vals = torch.cat([vals, torch.randn(n_cells - n, D) * 0.1])
    with torch.no_grad():
        memory.cell_keys.copy_(keys)
        memory.cell_values.copy_(vals)
    if freeze:
        memory.cell_keys.requires_grad_(False)
        memory.cell_values.requires_grad_(False)
    return n


if __name__ == "__main__":
    import sys, torch
    sys.path.insert(0, "/home/yonoob/projects/ReTop")
    from tokenizers import Tokenizer
    from hmn_v2 import HMN
    tok = Tokenizer.from_file("/home/yonoob/projects/ReTop/retop_tokenizer.json")
    m = HMN(3190, dim=64, state_dim=8, n_layers=2, n_mem_cells=256)
    n = load_kb_into_memory(m.memory, m.embed, tok, freeze=False)
    # sanity: query each KB key, check the closest cell is itself
    with torch.no_grad():
        k = m.memory.cell_keys
        ok = 0
        for e in KB[:20]:
            q = embed_phrase(m.embed, tok, f"{e['topic']} {e['key']}")
            best = (k @ q / (k.norm(dim=-1) * q.norm() + 1e-8)).argmax().item()
            if KB[best]["id"] == e["id"]:
                ok += 1
    print(f"loaded {n} KB entries into {m.memory.n_cells} cells; "
          f"retrieval self-hit {ok}/20")
