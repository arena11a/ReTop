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
    """Literal token lane (v3.3 — "next-token lookup"). 

    v3.1 used a contextual query vs context keys: the COPY channel then had to
    decode 'which token do I want next' through a single linear map — impossible
    per-position routing (measured 2026-08-12: ptr CE frozen at 2.08, copy CE
    frozen at 7.27 for 1500 steps; attention never sharpened past ~uniform;
    seen-only memorization, unseen 0/40). v3.2 (raw keys, same query) fixed
    nothing — the blocker is the QUERY, not the keys.

    v3.3 makes the register a pure identity lookup:
      query  = RAW embedding of the CURRENT token ids[t] (keys, unshifted)
      keys   = RAW embeddings of the prompt region (identity lane)
      a      = softmax(beta * cos(query, key_j))   <- self-match is EXACT (=1)
      payload= the token at position j+1 (what FOLLOWS the matched token)

    => 'the token right after the last thing I emitted/positioned'. For
    slot-copy ("pip install pkg061" -> answer "pip install pkg061") the chain
    is: seed=pip -> self-match prompt 'pip' -> copy 'install'; seed=install ->
    copy 'pkg061' UNSEEN token. The gen head emits the stable first answer
    token and EOS (both gutter rows: seed=ASI unreachable, payload=ASI != EOS);
    the gate learns when the lookup is reliable (tends ON in the mirror body,
    OFF at gutter rows). Corrected 2026-08-13: the seed MUST be ids[t], not
    ids[t-1] — the prev-shift made the register answer the already-seen token.

    This mirrors the verified v2 lever (task2 #10): addressing that is
    IDENTITY-based (raw embed self-match) hit 97-99% recall; contextual
    addressing stayed 51-62%. Identity lookup generalizes to unseen tokens
    because no knowledge about the token is needed — only its position.
    """

    def __init__(self, dim, beta_init=30.0, asi_id=None, keys_proj=False, eos_id=1):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        self.asi_id = asi_id
        self.eos_id = eos_id
        self.key_proj = nn.Linear(dim, dim)
        self.keys_proj = nn.Linear(dim, dim) if keys_proj else None

    def _attn(self, keys, ids):
        """Shared identity-lane attention (v4 refactor). Returns everything the
        dense and sparse paths need WITHOUT building the (B, T, V) copy mass:
          a        (B, T, T) position attention over prompt columns
          nxt      (B, T) payload token id per column (0 for the last column)
          n_legal  (B, T) # of legal copy columns per row
          ctx      (B, T, D) attended raw-key blend (gate's memory read)
          behind   (B, T) bool rows at/before the ASI boundary (forced gen)
          mass_same (B, T) sum of attention over same-token-id columns (gate)
          mask     (B, T, T) bool legal-column mask (needed by position-gather)
        """
        B, T, D = keys.shape
        beta = self.beta.abs() + 1.0
        qk = F.normalize(keys, dim=-1)                              # (B,T,D) ids[t]
        ek = qk                                                     # identity addrs
        sim = (qk @ ek.transpose(-1, -2)) * beta                    # (B,T,T)
        mask = torch.triu(torch.ones(T, T, device=keys.device, dtype=torch.bool), 1)
        if self.asi_id is not None and (ids == self.asi_id).any():
            idx = (ids == self.asi_id).float().argmax(-1, keepdim=True)  # (B,1)
            have = (ids == self.asi_id).any(-1, keepdim=True)            # (B,1)
            bound = torch.where(have, idx, torch.full_like(idx, T)).long()
            legal = (torch.arange(T, device=keys.device).unsqueeze(0).unsqueeze(1)
                     < bound.unsqueeze(-1))                              # (B,1,T)
            mask = mask.unsqueeze(0).expand(B, T, T) | ~legal
        # payload per column: ids[j+1]; last column has no successor
        nxt_col = torch.cat([ids[:, 1:], torch.full_like(ids[:, :1], -1)], dim=1)
        bad_payload = (nxt_col == self.asi_id) | (nxt_col == self.eos_id) | (nxt_col == -1)
        mask = mask | bad_payload.unsqueeze(1).expand(B, T, T)
        sim = sim.masked_fill(mask, float("-inf"))
        a = sim.softmax(-1)                                    # position attention
        n_legal = (~mask).sum(-1)                               # (B,T)
        nxt = torch.cat([ids[:, 1:], torch.zeros_like(ids[:, :1])], dim=1)
        ctx = (a.unsqueeze(-1) * keys.unsqueeze(1)).sum(2)     # (B,T,D) raw blend
        same = (ids.unsqueeze(1) == ids.unsqueeze(2))          # (B,T,T)
        mass_same = (a * same.float()).sum(-1)                 # (B,T) 0..1
        behind = torch.zeros(B, T, dtype=torch.bool, device=keys.device)
        if self.asi_id is not None and (ids == self.asi_id).any():
            bound2 = torch.where(have, idx, torch.full_like(idx, T)).squeeze(-1).unsqueeze(1)
            behind = torch.arange(T, device=keys.device).unsqueeze(0) <= bound2
        return a, nxt, n_legal, ctx, behind, mass_same, mask

    def forward(self, keys, ids, query, return_attn=False):
        # keys:  (B, T, dim) RAW token embeddings (identity lane, addressing)
        # ids:   (B, T) raw token ids (payload)
        # query: (B, T, dim) — v3.3: IGNORED for addressing (contextual query
        #        was the v3.1 blocker); kept in signature for API compat.
        B, T, D = keys.shape
        a, nxt, n_legal, ctx, behind, mass_same, _ = self._attn(keys, ids)
        # PAYLOAD = token at position j+1 (masked-off columns carry 0 weight,
        # so their zeroed payload index never scatters).
        mass = torch.zeros(B, T, self.vocab, device=keys.device)
        mass = mass.scatter_add(-1, nxt.unsqueeze(1).expand(B, T, T), a)  # (B,T,V)

        if return_attn:
            return mass, ctx, a, n_legal, behind, mass_same
        return mass, ctx, n_legal, behind, mass_same

    def sparse_forward(self, keys, ids, query):
        """v4 sparse copy-marginal: return position attention + payload ONLY —
        no (B, T, V) copy-mass tensor is materialized. Copy probabilities are
        recovered on demand with copy_prob_sparse() (position gather). For long
        contexts this removes the O(T·V) memory that grows linearly with T (V =
        3190 fixed) and is the base for template-general copy (M2+).
        """
        a, nxt, n_legal, ctx, behind, mass_same, mask = self._attn(keys, ids)
        return a, nxt, n_legal, ctx, behind, mass_same, mask

    def set_vocab(self, v):
        self.vocab = v


class DualHeadDecoder(nn.Module):
    """gen ⊕ copy. Pointer-generator style: final distribution is a convex blend
      p = (1-g)*softmax(gen) + g*copy_dist
    returned as log p. copy_dist = register attention mass over vocab (an exact
    context token can get ~1.0 mass -> hard copy achievable).

    g is now DETERMINISTIC (v3.3-final, measured 2026-08-13): the copy lane is
    trustworthy iff the query's attention mass lands on a column holding the
    SAME token id as the seed (an exact string duplicate in the prompt). We sum
    attention over same-token-id columns:
        gate_mass = sum_j a_j * [ids[j] == ids[t]]
        g = sigmoid(tau * (gate_mass - thr))
    An exact in-prompt twin gives ~0.99 (on); when the seed's only twin is the
    ASI/EOS-boundary column (excluded by the payload mask) that column carries
    zero softmax mass, gate_mass collapses toward 0, and gen must emit
    (EOS/first token). Top-1 mass and entropy gates were both tried and failed:
    at a boundary row attention falls back 'medium-sharp' on a WRONG token and
    both signals stay high, so decode looped the slot value forever.
    A learned gate on [h, ctx] (v3.1-v3.3) collapsed to ~0.05 — same outcome
    via a different mechanism (see identity-register postmortem).
    """

    def __init__(self, dim, vocab, tie_embed=None, gate_bias=0.0):
        super().__init__()
        # gen takes [h, gate_mass, behind] (dim+2): the two extra channels are a
        # DIRECT boundary signal. At the last answer row gate_mass collapses to
        # ~0 while behind=0, so gen can learn a reliable "emit EOS" rule there
        # instead of a per-slot lottery (pkg094 'rude' bug, 2026-08-13). At the
        # first-answer row behind=1 -> 'pip'. Without these gen is a bare
        # Linear(h) and ~1-2/40 unseen slots flip the EOS decision per seed.
        self.gen = nn.Linear(dim + 2, vocab, bias=False)
        self.gate_bias = gate_bias
        self.tau = nn.Parameter(torch.tensor(12.0))
        if tie_embed is not None:
            # tie only the h-cols (embed is dim-wide); the two extra boundary
            # channels stay free.
            self.gen.weight.data[:, :dim] = tie_embed.detach()

    def forward(self, h, copy_logits, ctx, attn=None, n_legal=None, behind=None,
                gate_mass=None, eps=1e-8, sparse=False, nxt=None):
        if gate_mass is not None:
            b = behind.unsqueeze(-1).float()
            gm = gate_mass.unsqueeze(-1)
            gen = torch.log_softmax(self.gen(torch.cat([h, gm, b], -1)), -1)
            # deterministic same-token-id gate (see class docstring).
            # threshold 0.5 fixed: exact twin ~0.99, boundary ~0.
            g = torch.sigmoid((self.tau * (gate_mass.unsqueeze(-1) - 0.5))
                              .clamp(-6.0, 6.0))
            g = g * (1.0 - b)                      # prompt/ASI rows: gen
        else:
            # legacy path (no attn provided): fall back to a static open gate
            gen = torch.log_softmax(self.gen(
                torch.cat([h, torch.zeros_like(h[..., :1]), torch.zeros_like(h[..., :1])], -1)), -1)
            g = torch.sigmoid(torch.full_like(gen[..., :1], self.gate_bias).clamp(-3, 3))
        if sparse:
            # v4 sparse copy-marginal: forward does NOT materialize (B,T,V).
            # We still need a (B,T,V) copy dist for the BLEND, but it is built
            # from position attention via a gather that skips the masked columns
            # (they carry zero attention). This keeps blend/decode semantics
            # identical to the dense path while the LOSS can read p_copy purely
            # from (attn, nxt) without ever expanding to vocab width.
            B, T = nxt.shape
            copy = torch.zeros(B, T, copy_logits.shape[-1] if copy_logits is not None
                               else self.gen.out_features, device=nxt.device)
            copy = copy.scatter_add(-1, nxt.unsqueeze(1).expand(B, T, T), attn)
            copy_dist = F.normalize(copy.clamp(min=0.0), p=1, dim=-1)
            p = (1 - g) * gen.exp() + g * copy_dist
            return torch.log(p.clamp(min=eps)), g, copy_dist
        # copy mass already sums ~1 over vocab; renormalize to be safe
        copy_dist = F.normalize(copy_logits, p=1, dim=-1)
        p = (1 - g) * gen.exp() + g * copy_dist
        return torch.log(p.clamp(min=eps)), g

    def gen_probs(self, h, gate_mass, behind):
        """Conditioned gen distribution for external logging/aux CE. Mirrors
        forward() so HMN3.forward's gen_logits uses the SAME [h,gm,behind]
        input the blend does (otherwise the aux CE trains a different head)."""
        b = behind.unsqueeze(-1).float()
        gm = gate_mass.unsqueeze(-1)
        return torch.log_softmax(self.gen(torch.cat([h, gm, b], -1)), -1)


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
                 gate_bias=0.0, aux_copy=True, asi_id=None, keys_proj=False,
                 sparse_marginal=False):
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
        self.ir = IdentityRegister(dim, asi_id=asi_id, keys_proj=keys_proj)
        self.ir.set_vocab(vocab_size)
        self.dual = DualHeadDecoder(dim, vocab_size,
                                    tie_embed=self.embed.weight if tie_weights else None,
                                    gate_bias=gate_bias)
        self.use_think = use_think
        self.think = LatentThinkingBuffer(dim, k_max) if use_think else None
        self.aux_copy = aux_copy
        self.sparse_marginal = sparse_marginal

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

    def forward(self, input_ids, return_gate=False, return_attn=False):
        ids = input_ids
        x = self.embed(ids)                       # raw token lane (IR keys)
        block_fn = self.blocks_apply()
        h = block_fn(x)
        if self.use_think:
            h = self.think(h, block_fn)
        # v3.3-final: the deterministic gate is read off the register attention,
        # so always compute it (cheap) and hand it to the dual head.
        if self.sparse_marginal:
            a, nxt, n_legal, ctx, behind, gate_mass, _ = \
                self.ir.sparse_forward(x, ids, h)
            logits, g, copy_dist = self.dual(h, None, ctx, attn=a, n_legal=n_legal,
                                             behind=behind, gate_mass=gate_mass,
                                             sparse=True, nxt=nxt)
            attn = a
        else:
            copy_logits, ctx, attn, n_legal, behind, gate_mass = \
                self.ir(x, ids, h, return_attn=True)
            logits, g = self.dual(h, copy_logits, ctx, attn=attn, n_legal=n_legal,
                                  behind=behind, gate_mass=gate_mass)
            copy_dist = F.normalize(copy_logits.clamp(min=0.0), p=1, dim=-1)
        gen_logits = self.dual.gen_probs(h, gate_mass, behind)
        d = {"logits": logits, "g": g, "copy_dist": copy_dist, "attn": attn,
             "n_legal": n_legal, "gen_logits": gen_logits}
        if self.sparse_marginal:
            d["nxt"] = nxt
        if return_gate:
            return logits, g
        return d


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