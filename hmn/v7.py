"""HMN v7 — Attention-WR variant + MoE routing improvements.

v7 M8: Attention-WR replaces the reversible SelectiveSSM coupling with a
standard transformer block (Pre-LN + RoPE + RMSNorm + SwiGLU-MoE). The IR
+ DualHeadDecoder are shared — only the WR changes.

Key design decisions:
  - RMSNorm (faster than LayerNorm, same quality)
  - RoPE for position encoding (proven at scale)
  - SwiGLU FFN (gated activation, proven at scale)
  - Optional MoE on the FFN (same SparseConditionalCompute)
  - Gradient checkpointing (attention is NOT reversible)
  - Same output shape (B, T, dim) as SSM-WR — IR + head unchanged

v7 M9: MoE routing improvements (Switch Transformer load-balancing, noisy gates).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from hmn.v2 import SparseConditionalCompute


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (faster than LayerNorm)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for attention."""

    def __init__(self, dim, max_seq_len=8192):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, q, k, offset=0):
        seq_len = q.shape[2]
        if seq_len + offset > self.cos_cached.shape[0]:
            self._build_cache(seq_len + offset + 1024)
        # q,k: (B, n_heads, T, head_dim) — RoPE applied per-head
        half = q.shape[-1] // 2
        cos = self.cos_cached[offset:offset + seq_len, :half]  # (T, head_dim//2)
        sin = self.sin_cached[offset:offset + seq_len, :half]
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, half)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)
        q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
        k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
        return q_rot, k_rot


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network (gated activation)."""

    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class AttentionBlock(nn.Module):
    """Transformer block: Pre-LN + Multi-Head Attention (RoPE) + SwiGLU FFN.

    This replaces HelixCouplingBlock for the attention-WR variant.
    NOT reversible — uses gradient checkpointing for memory efficiency.
    """

    def __init__(self, dim, n_heads=None, head_dim=None, dropout=0.0,
                 max_seq_len=8192):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads or max(1, dim // 64)
        self.head_dim = head_dim or (dim // self.n_heads)
        assert self.n_heads * self.head_dim == dim

        # Pre-LN for attention
        self.ln1 = RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)

        # RoPE
        self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len)

        # Pre-LN for FFN
        self.ln2 = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, dropout=dropout)

    def forward(self, x, offset=0):
        B, T, D = x.shape

        # Attention with Pre-LN
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE
        q, k = self.rope(q, k, offset=offset)

        # Scaled dot-product attention (causal mask)
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Attention output
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        out = self.out_proj(out)
        x = x + out

        # FFN with Pre-LN
        x = x + self.ffn(self.ln2(x))

        return x


class HMN3AttentionWR(nn.Module):
    """HMN3 with attention-WR variant.

    Same interface as HMN3: embed → WR (L AttentionBlocks) → IR → DualHeadDecoder.
    The IR + DualHeadDecoder are shared — only the WR changes.

    Key differences from HMN3 (SSM-WR):
      - AttentionBlock instead of HelixCouplingBlock
      - Gradient checkpointing instead of ReversibleFunction
      - Same output shape (B, T, dim)
      - Same IR + DualHeadDecoder interface
    """

    def __init__(self, vocab_size, dim=96, n_layers=3, n_heads=None,
                 n_experts=16, top_k=2, use_moe=False, tie_weights=True,
                 gate_bias=0.0, asi_id=None, user_id=None,
                 stem_addr=False, seam_addr=False, max_run=16,
                 max_seq_len=8192, use_checkpoint=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        # Embedding
        self.embed = nn.Embedding(vocab_size, dim)

        # Attention-WR blocks
        self.blocks = nn.ModuleList([
            AttentionBlock(dim, n_heads=n_heads, max_seq_len=max_seq_len)
            for _ in range(n_layers)
        ])

        # MoE (optional, same as SSM-WR)
        self.use_moe = use_moe
        self.moe_list = nn.ModuleList([
            SparseConditionalCompute(dim, n_experts, top_k) if use_moe else None
            for _ in range(n_layers)
        ])

        # IR + DualHeadDecoder (shared with SSM-WR)
        from hmn.v3 import IdentityRegister, DualHeadDecoder, SeedPointer
        self.ir = IdentityRegister(dim, asi_id=asi_id,
                                   user_id=user_id, stem_addr=stem_addr)
        self.ir.set_vocab(vocab_size)
        self.dual = DualHeadDecoder(dim, vocab_size,
                                    tie_embed=self.embed.weight if tie_weights else None,
                                    gate_bias=gate_bias)
        self.seam_addr = seam_addr
        self.seed_ptr = SeedPointer(dim, max_run=max_run) if seam_addr else None
        self.sparse_marginal = False
        self.aux_copy = False
        self.exact_blend = False

    def moe_aux_loss(self):
        if not self.use_moe:
            return torch.tensor(0.0)
        return sum(m.last_aux_loss * m.aux_coef for m in self.moe_list if m is not None)

    def blocks_apply(self, x):
        """Apply attention blocks with optional gradient checkpointing."""
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint and self.training:
                h = grad_checkpoint(blk, x, use_reentrant=False)
            else:
                h = blk(x)
            if self.moe_list[i] is not None:
                h = h + self.moe_list[i](h)
            x = h
        return x

    def forward(self, input_ids, return_gate=False, return_attn=False,
                seam_anchor=None, exact_blend=None):
        ids = input_ids
        x = self.embed(ids)

        # Attention-WR (NOT reversible — gradient checkpointing)
        h = self.blocks_apply(x)

        # IR + DualHeadDecoder (same as HMN3)
        st = self.ir.stats(x, ids, seam_anchor=seam_anchor)
        gen_logits, g = self.dual.gate_and_gen(h, st.mass_same, st.behind,
                                               n_legal=st.n_legal)
        d = {"gen_logits": gen_logits, "g": g, "stats": st,
             "n_legal": st.n_legal}

        if self.seed_ptr is not None:
            B, T = ids.shape
            bound = torch.full((B,), T, dtype=torch.long, device=ids.device)
            if self.ir.asi_id is not None and (ids == self.ir.asi_id).any():
                idx = (ids == self.ir.asi_id).float().argmax(-1)
                have = (ids == self.ir.asi_id).any(-1)
                bound = torch.where(have, idx, torch.full_like(idx, T)).long()
            ptr_logits, len_logits = self.seed_ptr(h, x, torch.clamp(bound - 1, min=0))
            d["ptr_logits"] = ptr_logits
            d["len_logits"] = len_logits

        return d
