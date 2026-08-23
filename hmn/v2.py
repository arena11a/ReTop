"""HMN v2 (Helix Memory Network) — rebuilt from Helix-Memory-Network-v2.1.md pseudocode 10.1-10.3.

Core building blocks kept in the package (v6 slim-down):

  - SelectiveSSM (Pre-LN, Mamba-style selective scan, causal over time,
    chunked closed-form parallel scan)
  - HelixCouplingBlock (reversible coupling F1/F2, Pre-LN inside)
  - ReversibleFunction (custom autograd, no intermediate activation storage)
  - SparseConditionalCompute (Product-Key MoE + aux load-balancing loss)

The v2-era full models (HMN, HMN_Option1, DifferentiableEpisodicMemory) were
removed from the package in v6 — they are preserved under git tag `v3.3` and
in the local legacy archive. HMN3 consumes only the four classes above.
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

    def __init__(self, dim, state_dim, prenorm=True, chunk_size=8):
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

        v2.4: closed-form per chunk (fast on CPU — autograd-cheap).
        Within a chunk the recurrence is solved exactly in closed form:

            h_t = exp(L_t) * h_in  +  exp(L_t) * S_t
            L_t = cumsum(log_da)[t]            (log of running product)
            S_t = cumsum(db_t * exp(-L_t))[t]  (log-domain forced response)

        so the OLD per-position python loop + in-place writes (which materialized
        CopySlices/SelectBackward nodes and made backward ~8x slower than the
        already-cheap recompute pass) collapse to 4 vectorized ops. Phase 2 keeps
        the sequential loop over n_chunks only (chunk states connect exactly).
        Purity: float32 throughout — |L| <= chunk*|clamp| = 72 < 88 (exp range),
        and e^-72 > f32 subnormal floor, so no double conversion needed.
        clamp=-9 caps the per-step decay rate at e^-9 ~ 0.0001 (vs underflow-to-0
        in the old cumprod form — a hard reset that now becomes a strong decay;
        irrelevant for trained scales, keeps exp() in range)."""
        B, T, D, S = log_da.shape
        C = self.chunk_size
        n_chunks = (T + C - 1) // C
        T_pad = n_chunks * C
        if T_pad > T:
            pad = T_pad - T
            log_da = F.pad(log_da, (0, 0, 0, 0, 0, pad), value=0.0)   # dA=1 in padding
            db = F.pad(db, (0, 0, 0, 0, 0, pad), value=0.0)          # dB=0 in padding
        log_da = log_da.clamp(min=-9.0)
        L = torch.cumsum(log_da.reshape(B, n_chunks, C, D, S), dim=2)  # (B,nc,C,D,S)
        eL_inv = torch.exp(-L)
        Sf = torch.cumsum(db.reshape(B, n_chunks, C, D, S) * eL_inv, dim=2)
        f = torch.exp(L) * Sf                     # zero-input forced response
        A_chunk = torch.exp(L[:, :, -1, :, :])    # total chunk product (B,nc,D,S)
        B_chunk = f[:, :, -1, :, :]               # chunk forced end state (B,nc,D,S)
        h_in = torch.zeros(B, D, S, device=log_da.device)
        h_ins = []
        for c in range(n_chunks):                # phase 2 (n_chunks steps)
            h_ins.append(h_in)
            h_in = A_chunk[:, c] * h_in + B_chunk[:, c]
        h_ins = torch.stack(h_ins, dim=1)         # (B, nc, D, S)
        h = torch.exp(L) * h_ins.unsqueeze(2) + f  # (B, nc, C, D, S)
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
