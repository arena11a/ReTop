"""HMN v2 (Helix Memory Network) — rebuilt from Helix-Memory-Network-v2.1.md pseudocode 10.1-10.3.

Components:
  - SelectiveSSM (Pre-LN, Mamba-style selective scan, causal over time)
  - HelixCouplingBlock (reversible coupling F1/F2, Pre-LN inside)
  - ReversibleFunction (custom autograd, no intermediate activation storage)
  - SparseConditionalCompute (Product-Key MoE + aux load-balancing loss)
  - DifferentiableEpisodicMemory (key_proj(prev)/val_proj(cur) split, soft DNC write/read)
  - HMN (embed + L reversible coupling blocks + MoE + memory + head)

Source of truth: Helix-Memory-Network-v2.1.md (pseudocode 10.1-10.3) + validated results
in section 18 (Pre-LN default, aux load-balancing loss, key/val projection split).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelectiveSSM(nn.Module):
    """Mamba-style selective SSM, causal scan over time axis.
    v2.0: Pre-LN (LayerNorm inside, before in_proj) is the default.
    v2.2: chunked parallel scan (roadmap #5) — recurrence h_t = dA_t*h_{t-1} + dB_t
    is a diagonal (elementwise) linear recurrence, solved with an exact two-phase
    scan (no log-domain clamping, bit-stable under arbitrary decay):
        phase 1: per chunk, vectorized loop over chunk positions computes
                 cumulative products P_j and zero-input forced response F_j
        phase 2: sequential loop over chunks connects chunk states
        h_j = P_j * h_in + F_j
    Python loop drops from T to (chunk_size + ceil(T/chunk_size)) steps;
    within-chunk work is vectorized across (B, n_chunks, dim, state)."""

    def __init__(self, dim, state_dim, prenorm=True, chunk_size=16):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.chunk_size = chunk_size
        self.ln = nn.LayerNorm(dim) if prenorm else None
        self.in_proj = nn.Linear(dim, state_dim * 2)  # B, C (input-dependent)
        self.delta_proj = nn.Linear(dim, dim)         # dt (input-dependent, per dim)
        self.A_log = nn.Parameter(torch.randn(dim, state_dim))
        self.D = nn.Parameter(torch.randn(dim))
        self.out_proj = nn.Linear(dim, dim)

    def _chunked_scan(self, log_da, db):
        """Exact two-phase chunked scan for h_t = exp(log_da_t)*h_{t-1} + db_t.
        log_da, db: (B, T, dim, state). Returns h: (B, T, dim, state).

        Phase 1 (vectorized loop over chunk_size positions, parallel across all
        chunks): cumulative products P and zero-input forced response F.
        Phase 2 (sequential loop over n_chunks): propagate chunk states.
        h = P * h_in + F reconstructs the full sequential scan exactly."""
        B, T, D, S = log_da.shape
        C = self.chunk_size
        n_chunks = (T + C - 1) // C
        T_pad = n_chunks * C
        if T_pad > T:
            pad = T_pad - T
            log_da = F.pad(log_da, (0, 0, 0, 0, 0, pad), value=0.0)   # dA=1 in padding
            db = F.pad(db, (0, 0, 0, 0, 0, pad), value=0.0)          # dB=0 in padding
        dA = torch.exp(log_da).reshape(B, n_chunks, C, D, S)
        dB = db.reshape(B, n_chunks, C, D, S)
        P = torch.cumprod(dA, dim=2)                                 # within-chunk products
        forced = torch.empty(B, n_chunks, C, D, S, device=log_da.device)
        f = torch.zeros(B, n_chunks, D, S, device=log_da.device)
        for j in range(C):                                           # phase 1 (C steps)
            f = f * dA[:, :, j] + dB[:, :, j]
            forced[:, :, j] = f
        A_chunk = P[:, :, -1]                                        # total chunk product
        B_chunk = forced[:, :, -1]                                   # forced end state
        h_in = torch.zeros(B, D, S, device=log_da.device)
        h_ins = []
        for c in range(n_chunks):                                    # phase 2 (n_chunks steps)
            h_ins.append(h_in)
            h_in = A_chunk[:, c] * h_in + B_chunk[:, c]
        h_ins = torch.stack(h_ins, dim=1)                            # (B, nc, D, S)
        h = P * h_ins.unsqueeze(2) + forced                          # (B, nc, C, D, S)
        h = h.reshape(B, T_pad, D, S)[:, :T]
        return h

    def forward(self, x):
        # x: (B, T, dim)
        if self.ln is not None:
            x = self.ln(x)
        B, T, D = x.shape
        dt = F.softplus(self.delta_proj(x)).unsqueeze(-1)            # (B, T, dim, 1)
        Bm, Cm = self.in_proj(x).chunk(2, dim=-1)                    # (B, T, state) each
        A = -torch.exp(self.A_log)                                   # (dim, state)
        log_da = A.unsqueeze(0).unsqueeze(0) * dt                    # (B, T, dim, state)
        db = dt * Bm.unsqueeze(-2)                                   # (B, T, dim, state)
        h = self._chunked_scan(log_da, db)
        y = (h * Cm.unsqueeze(-2)).sum(-1) + self.D * x              # (B, T, dim)
        return self.out_proj(y)


class HelixCouplingBlock(nn.Module):
    """One reversible coupling layer (Pre-LN inside F1/F2)."""

    def __init__(self, dim, state_dim, prenorm=True):
        super().__init__()
        half = dim // 2
        self.F1 = SelectiveSSM(half, state_dim, prenorm=prenorm)
        self.F2 = SelectiveSSM(half, state_dim, prenorm=prenorm)

    def forward(self, h):
        h_a, h_b = h.chunk(2, dim=-1)
        h_a_new = h_a + self.F1(h_b)
        h_b_new = h_b + self.F2(h_a_new)
        return torch.cat([h_a_new, h_b_new], dim=-1)

    def inverse(self, h_new):
        h_a_new, h_b_new = h_new.chunk(2, dim=-1)
        h_b = h_b_new - self.F2(h_a_new)
        h_a = h_a_new - self.F1(h_b)
        return torch.cat([h_a, h_b], dim=-1)


class ReversibleFunction(torch.autograd.Function):
    """Custom autograd that does not store intermediate activations;
    gradients computed via backward reconstruction."""

    @staticmethod
    def forward(ctx, x, blocks):
        ctx.blocks = blocks
        with torch.no_grad():
            h = x
            for block in blocks:
                h = block.forward(h)
        ctx.save_for_backward(h)
        return h

    @staticmethod
    def backward(ctx, grad_output):
        (h,) = ctx.saved_tensors
        blocks = ctx.blocks
        grad = grad_output
        for block in reversed(blocks):
            with torch.no_grad():
                h_prev = block.inverse(h)
            with torch.enable_grad():
                h_prev_ = h_prev.detach().requires_grad_(True)
                h_rec = block.forward(h_prev_)
                grads = torch.autograd.grad(h_rec, [h_prev_] + list(block.parameters()),
                                            grad_outputs=grad)
            grad = grads[0]
            for p, g in zip(block.parameters(), grads[1:]):
                if p.grad is None:
                    p.grad = g
                else:
                    p.grad = p.grad + g
            h = h_prev
        return grad, None


class SparseConditionalCompute(nn.Module):
    """Product-Key MoE-lite with aux load-balancing loss (v2.0).
    v2.3: full straight-through estimator (roadmap #7) — forward keeps hard top-K
    (verified behavior); backward adds a soft top-K path (softmax over all candidate
    scores) so gradient flows through the discrete selection to query/keys, letting
    the router learn WHICH experts to pick, not just how to weight the chosen ones."""

    def __init__(self, dim, n_experts, top_k, key_dim=16, aux_coef=0.1, ste=True):
        super().__init__()
        self.n_experts = n_experts
        self.n_sub = int(n_experts ** 0.5)
        self.top_k = top_k
        self.aux_coef = aux_coef
        self.ste = ste
        self.query_proj = nn.Linear(dim, key_dim)
        self.keys_1 = nn.Parameter(torch.randn(self.n_sub, key_dim // 2) * 0.1)
        self.keys_2 = nn.Parameter(torch.randn(self.n_sub, key_dim // 2) * 0.1)
        self.values = nn.Embedding(n_experts, dim)
        self.last_aux_loss = torch.tensor(0.0)

    def forward(self, x):
        # x: (B, T, dim)
        q = self.query_proj(x)
        q1, q2 = q.chunk(2, dim=-1)
        score1 = q1 @ self.keys_1.T                      # (B, T, n_sub)
        score2 = q2 @ self.keys_2.T
        k = int(math.ceil(self.top_k ** 0.5))
        top1_val, top1_idx = score1.topk(k, dim=-1)
        top2_val, top2_idx = score2.topk(k, dim=-1)
        combined_scores = top1_val.unsqueeze(-1) + top2_val.unsqueeze(-2)   # (B,T,k,k)
        combined_idx = top1_idx.unsqueeze(-1) * self.n_sub + top2_idx.unsqueeze(-2)
        flat_scores = combined_scores.flatten(-2)
        flat_idx = combined_idx.flatten(-2)
        final_val, final_pos = flat_scores.topk(self.top_k, dim=-1)
        final_idx = torch.gather(flat_idx, -1, final_pos)                  # (B,T,top_k)
        weights = final_val.softmax(dim=-1)
        expert_vals = self.values(final_idx)
        out = (weights.unsqueeze(-1) * expert_vals).sum(dim=-2)
        if self.training and self.ste:
            # full STE: soft top-K over ALL candidate scores; forward stays hard
            # (soft - soft.detach() == 0), backward routes gradient to every
            # candidate score -> router learns the discrete selection itself.
            soft_all = flat_scores.softmax(dim=-1)                    # (B,T,k*k)
            all_vals = self.values(flat_idx).detach()                 # values frozen in STE term
            ste_contrib = (soft_all - soft_all.detach()).unsqueeze(-1) * all_vals
            out = out + ste_contrib.sum(dim=-2)
        self.last_aux_loss = self._load_balance(final_idx, weights)
        return out

    def _load_balance(self, idx, gate):
        B = idx.size(0) * idx.size(1)
        frac_selected = torch.zeros(self.n_experts, device=idx.device)
        frac_selected.index_add_(0, idx.reshape(-1),
                                 torch.ones(idx.numel(), device=idx.device) / B)
        frac_gate = torch.zeros(self.n_experts, device=idx.device)
        frac_gate.index_add_(0, idx.reshape(-1), gate.reshape(-1) / B)
        balance = self.n_experts * torch.sum(frac_selected * frac_gate)
        return (balance - 1.0) / (self.n_experts - 1.0)


class DifferentiableEpisodicMemory(nn.Module):
    """Soft DNC-style episodic memory.
    v2.0: key from PREVIOUS hidden, value (erase/add) from CURRENT hidden."""

    def __init__(self, dim, n_cells, top_k, beta_init=10.0, usage_decay=False,
                 combined=False, exempt_combined=False, n_pairs=None):
        super().__init__()
        self.n_cells = n_cells
        self.top_k = top_k
        self.key_proj = nn.Linear(dim, dim)
        self.val_proj = nn.Linear(dim, dim * 2 + 1)     # erase / add / write-strength
        self.read_proj = nn.Linear(dim, dim + 1)
        self.out_proj = nn.Linear(dim, dim)
        self.gate_proj = nn.Linear(dim * 2, dim)
        # v2.1 fix: initialize memory cells from random keys/values (not zeros) so
        # content-addressing has non-zero similarity at step 0 (avoids frozen uniform
        # write weights -> gradient stall in the write path).
        self.cell_keys = nn.Parameter(torch.randn(self.n_cells, dim) * 0.1)
        self.cell_values = nn.Parameter(torch.randn(self.n_cells, dim) * 0.1)
        # learnable sharpening temperature for content addressing (DNC-style beta).
        # v2.2: beta_init=30 verified (sharper attention lifts recall 77%->89%).
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        # v2.2: usage_decay — scale write strength by (1 - cumulative usage) so cells
        # already heavily written get less overwrite. Verified synergy with high beta.
        self.usage_decay = usage_decay
        # v2.2: combined write for multi-token values (design B, "query once"):
        # at the LAST token of a value, write the FULL value (e.g. tens+ones) into the
        # cell addressed by the pair's KEY token (2 tokens back), instead of writing
        # one association per value token. Enables single-read retrieval of the value.
        self.combined = combined
        # v2.2: safety margin — skip usage_decay for combined writes (write full strength).
        # NOTE: verified NOT required (attribution test: off-by-one fix alone reaches
        # 93-99%). Kept only as neutral safety margin.
        self.exempt_combined = exempt_combined
        self.n_pairs = n_pairs     # number of key/value pairs in the sequence ([k,t,o] triples)
        if combined:
            self.comb_val = nn.Linear(dim * 2, dim)     # concat(tens_h, ones_h) -> full-value vector

    def forward(self, x):
        # x: (B, T, dim) — read+write over time with persistent cell state per batch item
        B, T, D = x.shape
        keys = self.cell_keys.unsqueeze(0).expand(B, -1, -1)
        values = self.cell_values.unsqueeze(0).expand(B, -1, -1)
        beta = self.beta.abs() + 1.0
        usage = torch.zeros(B, self.n_cells, device=x.device)
        reads = []
        prev_h = torch.zeros(B, D, device=x.device)
        prev_prev_h = torch.zeros(B, D, device=x.device)
        for t in range(T):
            h = x[:, t]
            q = self.read_proj(h)[..., :D]
            sim = (q.unsqueeze(1) @ keys.transpose(-1, -2)).squeeze(1) / (D ** 0.5)  # (B, cells)
            w = (sim * beta).softmax(dim=-1)
            m = (w.unsqueeze(-1) * values).sum(dim=-2)
            gate = torch.sigmoid(self.gate_proj(torch.cat([h, m], dim=-1)))
            reads.append(gate * h + (1 - gate) * m)
            v = self.val_proj(h)
            erase = torch.sigmoid(v[..., :D])
            add = v[..., D:2 * D]
            strength = torch.sigmoid(v[..., 2 * D:]).squeeze(-1)          # (B,)
            # v2.2 combined write: last token of a value = positions of the ONES digit
            # in a [k, t, o] triple. IMPORTANT (root cause): bound is t < 3*n_pairs,
            # NOT 2*n_pairs — pairs near the end (t >= 2*n_pairs) must also get the
            # combined write or their value's later digits are never stored (off-by-one
            # bug found 2026-08-07: q=5,6 ones-digit recall collapsed to ~7-13%).
            is_val_last = (self.combined and t >= 2 and t % 3 == 2
                           and (self.n_pairs is None or t < 3 * self.n_pairs))
            if is_val_last:
                # write key = the pair's KEY token (2 back), value = full [tens+ones].
                wk = F.normalize(self.key_proj(prev_prev_h), dim=-1)
                combined_val = self.comb_val(torch.cat([prev_h, h], dim=-1))
            else:
                wk = F.normalize(self.key_proj(prev_h), dim=-1)
                combined_val = None
            wsim = (wk.unsqueeze(1) @ keys.transpose(-1, -2)).squeeze(1) / (D ** 0.5)
            write_w = (wsim * beta).softmax(dim=-1) * strength.unsqueeze(-1)
            if self.usage_decay and not (self.exempt_combined and is_val_last):
                write_w = write_w * (1 - usage)
            if is_val_last:
                # full replace of the key's cell with the combined value (erase ~ 1).
                values = values * (1 - write_w.unsqueeze(-1)) \
                    + write_w.unsqueeze(-1) * combined_val.unsqueeze(1)
            else:
                keys = keys * (1 - write_w.unsqueeze(-1)) + wk.unsqueeze(1) * write_w.unsqueeze(-1)
                values = values * (1 - write_w.unsqueeze(-1) * erase.unsqueeze(1)) \
                    + write_w.unsqueeze(-1) * add.unsqueeze(1)
            usage = usage + write_w
            prev_prev_h = prev_h
            prev_h = h
        return torch.stack(reads, 1)


class HMN(nn.Module):
    """Full HMN v2: embed -> L reversible coupling blocks -> MoE -> memory -> head."""

    def __init__(self, vocab_size, dim, state_dim, n_layers, n_experts=16, top_k=2,
                 n_mem_cells=8, mem_top_k=4, memory_interval=1, aux_coef=0.1,
                 prenorm=True, tie_weights=True):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList(
            [HelixCouplingBlock(dim, state_dim, prenorm=prenorm) for _ in range(n_layers)]
        )
        self.moe = SparseConditionalCompute(dim, n_experts, top_k, aux_coef=aux_coef)
        self.memory = DifferentiableEpisodicMemory(dim, n_mem_cells, mem_top_k)
        self.memory_interval = memory_interval
        self.head = nn.Linear(dim, vocab_size, bias=False)
        if tie_weights:
            self.head.weight = self.embed.weight

    def forward(self, input_ids):
        x = self.embed(input_ids)                       # (B, T, dim)
        for i, block in enumerate(self.blocks):
            x = ReversibleFunction.apply(x, [block])    # coupling (reversible) per layer
            x = x + self.moe(x)                         # product-key MoE per layer
            if i % self.memory_interval == 0:
                x = self.memory(x)                  # episodic memory every interval (gate blend replace)
        return self.head(x)

    def moe_aux_loss(self):
        return self.moe.last_aux_loss * self.moe.aux_coef


class HMN_Option1(nn.Module):
    """HMN v2.2 Option-1 architecture (verified 2026-08-07, task2_findings.md #10-11).

    Memory reads RAW token embeddings (parallel branch) instead of backbone hidden
    states. Content addressing keeps token identity (backbone SSM makes hidden states
    contextual -> breaks content addressing -> serial integration stuck at 3-6%).
    Backbone (reversible coupling, no MoE) + memory branch are combined at the head
    via a learnable gate:  z = g*backbone + (1-g)*memory_read.

    Verified config (single-token 8 pairs/50 keys, random-query): 97-99%.
    Multi-token (2-token values 0-99, design B "query once" + combined write): 94-97%.
      - use_multi_head=True + n_digits=2 -> two digit heads (tens, ones)
      - memory(combined=True) writes full value at value-last token (off-by-one fixed)

    NOTE: exempt_combined is a NEUTRAL safety margin only — attribution test showed
    the off-by-one fix (t<3*n_pairs) alone reaches 93-99%.
    """

    def __init__(self, vocab_size, dim=64, state_dim=8, n_layers=2, n_mem_cells=256,
                 mem_top_k=4, memory_interval=1, beta_init=30.0, usage_decay=True,
                 combined=False, exempt_combined=False, n_pairs=None,
                 use_multi_head=False, n_digits=2, tie_weights=True):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList(
            [HelixCouplingBlock(dim, state_dim) for _ in range(n_layers)]
        )
        self.memory = DifferentiableEpisodicMemory(
            dim, n_mem_cells, mem_top_k,
            beta_init=beta_init, usage_decay=usage_decay,
            combined=combined, exempt_combined=exempt_combined, n_pairs=n_pairs,
        )
        self.memory_interval = memory_interval
        self.gate = nn.Linear(dim * 2, dim)
        self.use_multi_head = use_multi_head
        if use_multi_head:
            self.heads = nn.ModuleList([nn.Linear(dim, 10, bias=False) for _ in range(n_digits)])
        else:
            self.head = nn.Linear(dim, vocab_size, bias=False)
            if tie_weights:
                self.head.weight = self.embed.weight

    def forward(self, input_ids):
        h = self.embed(input_ids)                       # (B, T, dim)
        b = h
        for block in self.blocks:
            b = ReversibleFunction.apply(b, [block])    # coupling backbone (no MoE)
        m = self.memory(h)                              # memory on RAW embeddings (parallel)
        g = torch.sigmoid(self.gate(torch.cat([b, m], dim=-1)))
        z = g * b + (1 - g) * m
        if self.use_multi_head:
            return [hd(z) for hd in self.heads]
        return self.head(z)

    def moe_aux_loss(self):
        return torch.tensor(0.0)
