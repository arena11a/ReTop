"""HMN v3 — Helix Register-Network (hmn_v3_design.md).

The core new contribution vs v2 / vs vanilla single-pass LLMs:

  Dual-Head Decoder with Identity-Register copy channel.

  - Working Register (WR): existing reversible coupling backbone (+MoE) for
    contextual abstraction (reused from hmn_v2).
  - Identity Register (IR): a RAW-token lane. During a forward, it captures the
    token identity of every distinct token seen in the sequence (rows = token
    embeddings, content-addressable by row similarity).
  - Dual Head: generate-logits (softmax head) ⊕ copy-logits (similarity of a WR
    query against IR rows, masked to the token set actually present in context).
    Final logits = gen + gate*x.copy, so when the WR learns to emit a query that
    matches the identity of a token present in the context, the argmax can pick
    that EXACT token — unblocking hard slot-copy that v2's pure-softmax head could
    never do (val slot-copy 0/40).

Reuses verified v2 ingredients: Pre-LN coupling blocks (reconstruction-safe),
helper moe_aux_loss hook. Latent thinking buffer kept as a separate flag
(--thinking-buffer wiring added in the trainer/experiment).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from hmn.v2 import HelixCouplingBlock, ReversibleFunction, SparseConditionalCompute, DifferentiableEpisodicMemory


class IdentityRegister(nn.Module):
    """Literal token lane. Register rows = raw token embeddings of distinct
    tokens present in the context. Copy scores are a SOFT attention-marginal
    over context positions (pointer-style): copy_logit(token t) =
    sum_j a_j * 1[id_j == t] where a = softmax(beta * sim(key(h), embed_j)).
    Fully differentiable in training (the router learns which position to
    attend -> can emit an EXACT context token on eval). Returns:
      - copy_logits over vocab (bounded floor -30 elsewhere)
      - retrieved context vector (same attention blend) for the gate

    Addressing / Payload split (v3 innovation):
      ADDRESS in meaning-space = contextual WR states (h). Context distinguishes
      duplicate tokens ("pkg061's p" vs "pip's p" — identical raw vectors, so
      content-attention on raw embeds cannot localize them; task2 held the
      opposite lesson but that was for write-then-read storage consistency,
      not single-pass copy).
      PAYLOAD in token-space = the raw token id at the addressed position, so
      the emitted token is EXACTLY the context token (pointer-generator).
    """

    def __init__(self, dim, beta_init=8.0, asi_id=None):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        self.asi_id = asi_id
        self.key_proj = nn.Linear(dim, dim)

    def forward(self, ctx_h, ids, query):
        # ctx_h: (B, T, dim) CONTEXTUAL WR states (addressing keys)
        # ids:   (B, T) raw token ids (payload)
        # query: (B, T, dim) contextual query (WR state at query position)
        B, T, D = ctx_h.shape
        beta = self.beta.abs() + 1.0
        qk = F.normalize(self.key_proj(query), dim=-1)         # (B,T,D)
        ek = F.normalize(ctx_h, dim=-1)                        # (B,T,D) contextual addr
        sim = (qk @ ek.transpose(-1, -2)) * beta               # (B,T,T) prices
        # causal + configurable copy-set (see below)
        mask = torch.triu(torch.ones(T, T, device=query.device, dtype=torch.bool), 1)
        # COPY-SET: legal copy targets = PROMPT-region tokens (before the
        # <|assistant|> boundary). Self-copy of generated tokens would let the
        # model loop its own output (observed in PoC). Slot values / recall
        # answers live in the prompt, so nothing is lost.
        if self.asi_id is not None and (ids == self.asi_id).any():
            idx = (ids == self.asi_id).float().argmax(-1, keepdim=True)  # (B,1), -1 if absent
            have = (ids == self.asi_id).any(-1, keepdim=True)            # (B,1)
            bound = torch.where(have, idx, torch.full_like(idx, T)).long()
            tgt = torch.arange(T, device=query.device).unsqueeze(0)      # (1,T)
            legal = (tgt.unsqueeze(1) < bound.unsqueeze(-1))             # (B,T,T)
            mask = mask.unsqueeze(0).expand(B, T, T) | ~legal
        sim = sim.masked_fill(mask, float("-inf"))
        a = sim.softmax(-1)                                    # position attention
        # token mass = attention summed over token id (sums to ~1 per position);
        # tokens absent from the context stay at exactly 0. pure PROBABILITY scale
        # (no logit floor — the decoder blends on probabilities).
        mass = torch.zeros(B, T, self.vocab, device=query.device)
        mass = mass.scatter_add(-1, ids.unsqueeze(1).expand(B, T, T), a)  # (B,T,V)
        # context vector = attended blend of CONTEXTUAL states (gate's memory read)
        ctx = (a.unsqueeze(-1) * ctx_h.unsqueeze(1)).sum(2)    # (B,T,D)
        return mass, ctx

    def set_vocab(self, v):
        self.vocab = v


class DualHeadDecoder(nn.Module):
    """gen ⊕ copy. Pointer-generator style: final distribution is a convex blend
      p = (1-g)*softmax(gen) + g*copy_dist
    returned as log p. copy_dist = register attention mass over vocab (an exact
    context token can get ~1.0 mass -> hard copy achievable). g learned per
    token from [h, ctx]. This keeps both paths on the SAME probability scale, so
    the copy path actually gets shaped during training (with a gen head that
    could otherwise memorize seen slots)."""

    def __init__(self, dim, vocab, tie_embed=None, gate_bias=0.0):
        super().__init__()
        self.gen = nn.Linear(dim, vocab, bias=False)
        self.gate = nn.Linear(dim * 2, 1)
        self.gate_bias = gate_bias
        if tie_embed is not None:
            self.gen.weight = tie_embed

    def forward(self, h, copy_logits, ctx, eps=1e-8):
        gen = torch.log_softmax(self.gen(h), -1)
        # copy mass already sums ~1 over vocab; renormalize to be safe
        copy_dist = F.normalize(copy_logits, p=1, dim=-1)
        # clamp gate logit: keep the blend in a regime where BOTH paths keep
        # receiving gradient (g exactly 1 murders the gen path -> model regresses
        # to copying the prompt forever, observed in PoC).
        gl = self.gate(torch.cat([h, ctx], -1)) + self.gate_bias
        g = torch.sigmoid(gl.clamp(-3.0, 3.0))
        p = (1 - g) * gen.exp() + g * copy_dist
        return torch.log(p.clamp(min=eps)), g


class LatentThinkingBuffer(nn.Module):
    """Adaptive deliberation: re-run the WR block sequence over the last hidden
    state, refining it in latent space without decoding to text tokens. Stops
    when the top-1 output confidence stops increasing (convergence) or K_max
    reached. Zero new vocab cost — pure compute scaling with input difficulty."""

    def __init__(self, dim, k_max, thr=0.02):
        super().__init__()
        self.k_max = k_max
        self.thr = thr
        self.adapt = nn.Linear(dim, dim)

    def forward(self, h, block_fn):
        prev_conf = None
        hk = h
        for k in range(self.k_max):
            hk = block_fn(hk)
            hk = hk + self.adapt(hk)
            conf = torch.softmax(hk[..., 0, :], -1).max(-1).values
            if prev_conf is not None and ((conf - prev_conf).abs().mean() < self.thr):
                break
            prev_conf = conf
        return hk


class HMN3(nn.Module):
    """Full v3: embed -> WR (L coupling blocks) -> IR (raw-id lane) ->
    dual head (gated copy⊕gen). Optionally LatentThinkingBuffer before the head
    (compute-scaling deliberation, enabled via use_think).
    If aux_copy>0, forward also returns copy_dist (register attention mass);
    the trainer adds CE(copy_dist, Y) so the register is shaped DIRECTLY to
    point at the correct context token — otherwise a gen head that memorizes
    seen slots starves the copy path of gradient (seen=learn, unseen=0
    deadlock observed in the PoC).
    """

    def __init__(self, vocab_size, dim=96, state_dim=8, n_layers=3, n_experts=16,
                 top_k=2, use_moe=False, use_think=False, k_max=4, tie_weights=True,
                 gate_bias=0.0, aux_copy=True, asi_id=None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList(
            [HelixCouplingBlock(dim, state_dim) for _ in range(n_layers)]
        )
        self.moe_off = nn.Identity()
        self.use_moe = use_moe
        self.moe_list = nn.ModuleList([
            SparseConditionalCompute(dim, n_experts, top_k) if use_moe else None
            for _ in range(n_layers)])
        self.ir = IdentityRegister(dim, asi_id=asi_id)
        self.ir.set_vocab(vocab_size)
        self.dual = DualHeadDecoder(dim, vocab_size,
                                    tie_embed=self.embed.weight if tie_weights else None,
                                    gate_bias=gate_bias)
        self.use_think = use_think
        self.think = LatentThinkingBuffer(dim, k_max) if use_think else None
        self.aux_copy = aux_copy

    def moe_aux_loss(self):
        if not self.use_moe:
            return torch.tensor(0.0)
        return sum(m.last_aux_loss * m.aux_coef for m in self.moe_list if m is not None)

    def blocks_apply(self, x=None):
        def _block_fn(h):
            for i, blk in enumerate(self.blocks):
                h = ReversibleFunction.apply(h, [blk])
                if self.moe_list[i] is not None:
                    h = h + self.moe_list[i](h)
            return h
        return _block_fn

    def forward(self, input_ids, return_gate=False):
        ids = input_ids
        x = self.embed(ids)
        block_fn = self.blocks_apply()
        h = block_fn(x)
        if self.use_think:
            h = self.think(h, block_fn)
        copy_logits, ctx = self.ir(h, ids, h)
        logits, g = self.dual(h, copy_logits, ctx)
        if self.aux_copy:
            copy_dist = F.normalize(copy_logits.clamp(min=0.0), p=1, dim=-1)
            d = {"logits": logits, "g": g, "copy_dist": copy_dist}
            return d
        if return_gate:
            return logits, g
        return logits


class HMN3_NoReg(nn.Module):
    """Ablation: same WR + dual-head but copy channel disabled (pure softmax).
    Control for measuring what the register itself contributes."""

    def __init__(self, vocab_size, dim=96, state_dim=8, n_layers=3, tie_weights=True):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList(
            [HelixCouplingBlock(dim, state_dim) for _ in range(n_layers)])
        self.head = nn.Linear(dim, vocab_size, bias=False)
        if tie_weights:
            self.head.weight = self.embed.weight

    def forward(self, input_ids):
        h = self.embed(input_ids)
        for blk in self.blocks:
            h = ReversibleFunction.apply(h, [blk])
        return self.head(h)

    def moe_aux_loss(self):
        return torch.tensor(0.0)