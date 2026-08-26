"""HMN configuration and model factory (v7).

Easy model creation:
    from hmn import HMNConfig, create_model

    cfg = HMNConfig.from_preset("cpu-small")
    model = create_model(cfg, asi_id=tok.token_to_id("<|assistant|>"))

    # or directly:
    model = create_model("gpu-small", asi_id=5)
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HMNConfig:
    """Unified configuration for all HMN model variants.

    Attributes:
        vocab_size: vocabulary size (default 3190 for retop tokenizer)
        dim: model hidden dimension
        state_dim: SSM state dimension (SSM-WR only)
        n_layers: number of backbone layers
        n_experts: number of MoE experts (when use_moe=True)
        top_k: top-k routing (when use_moe=True)
        use_moe: enable Mixture-of-Experts
        use_think: enable LatentThinkingBuffer
        k_max: max thinking steps (when use_think=True)
        tie_weights: tie embedding and output weights
        gate_bias: initial gate bias (lower = more gen-heavy)
        asi_id: assistant token id (required for training)
        user_id: user token id (for stem-addressing)
        gate_mode: "deterministic" or "relative" (learned gate)
        stem_addr: enable stem-addressing (v4)
        seam_addr: enable omega-seam (v5, for reorder tasks)
        max_run: max run length for seam (when seam_addr=True)
        exact_blend: use dense oracle path (debug/parity only)
        variant: "ssm" or "attention" (v7)
        n_heads: attention heads (attention-WR only, auto if None)
        max_seq_len: max sequence length for RoPE (attention-WR only)
        use_checkpoint: gradient checkpointing (attention-WR only)
        dropout: dropout rate (attention-WR only)
    """
    vocab_size: int = 3190
    dim: int = 96
    state_dim: int = 8
    n_layers: int = 3
    n_experts: int = 16
    top_k: int = 2
    use_moe: bool = False
    use_think: bool = False
    k_max: int = 4
    tie_weights: bool = True
    gate_bias: float = 0.0
    asi_id: Optional[int] = None
    user_id: Optional[int] = None
    gate_mode: str = "deterministic"
    stem_addr: bool = False
    seam_addr: bool = False
    max_run: int = 16
    exact_blend: bool = False
    # v7 attention-WR
    variant: str = "ssm"
    n_heads: Optional[int] = None
    max_seq_len: int = 8192
    use_checkpoint: bool = True
    dropout: float = 0.0

    # --- presets ---

    PRESETS = {
        "cpu-small": {"dim": 48, "state_dim": 8, "n_layers": 2, "variant": "ssm"},
        "gpu-small": {"dim": 64, "state_dim": 8, "n_layers": 2, "variant": "ssm"},
        "gpu-medium": {"dim": 128, "state_dim": 16, "n_layers": 3, "variant": "ssm"},
        "gpu-large": {"dim": 256, "state_dim": 32, "n_layers": 4, "variant": "ssm"},
        "attn-small": {"dim": 64, "n_layers": 2, "variant": "attention"},
        "attn-medium": {"dim": 128, "n_layers": 3, "variant": "attention"},
        "attn-large": {"dim": 256, "n_layers": 4, "variant": "attention"},
    }

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "HMNConfig":
        """Create config from a named preset, with optional overrides.

        Presets: cpu-small, gpu-small, gpu-medium, gpu-large,
                 attn-small, attn-medium, attn-large.
        """
        if name not in cls.PRESETS:
            raise ValueError(f"Unknown preset {name!r}. "
                             f"Available: {list(cls.PRESETS.keys())}")
        return cls(**cls.PRESETS[name], **overrides)

    def param_count_estimate(self) -> int:
        """Rough parameter count (exact depends on MoE/seam config)."""
        d, V, L = self.dim, self.vocab_size, self.n_layers
        embed = V * d
        if self.variant == "attention":
            # AttentionBlock: QKV (3d^2) + out (d^2) + SwiGLU (2*d*4d=8d^2)
            # + 2x RMSNorm (2d) = 12d^2 + 2d per layer
            backbone = L * (12 * d * d + 2 * d)
        else:
            # HelixCouplingBlock: 4 coupling params per layer
            backbone = L * (4 * d * d + 4 * d * self.state_dim)
        head = d * V  # gen head (tied = 0 extra)
        ir = 1  # beta param
        moe = 0
        if self.use_moe:
            n_sub = int(self.n_experts ** 0.5)
            moe = L * (d * 16 + n_sub * 16 + self.n_experts * d)
        return embed + backbone + head + ir + moe


def create_model(config, asi_id=None, user_id=None, **overrides):
    """Factory function: create an HMN model from config.

    Args:
        config: HMNConfig instance, or a preset name string.
        asi_id: assistant token id (overrides config.asi_id).
        user_id: user token id (overrides config.user_id).
        **overrides: override any config field.

    Returns:
        HMN3 or HMN3AttentionWR instance.
    """
    from hmn.v3 import HMN3
    from hmn.v7 import HMN3AttentionWR

    if isinstance(config, str):
        config = HMNConfig.from_preset(config)
    elif isinstance(config, dict):
        config = HMNConfig(**config)

    # Apply overrides
    cfg_dict = {
        "vocab_size": config.vocab_size,
        "dim": config.dim,
        "state_dim": config.state_dim,
        "n_layers": config.n_layers,
        "n_experts": config.n_experts,
        "top_k": config.top_k,
        "use_moe": config.use_moe,
        "use_think": config.use_think,
        "k_max": config.k_max,
        "tie_weights": config.tie_weights,
        "gate_bias": config.gate_bias,
        "asi_id": asi_id if asi_id is not None else config.asi_id,
        "user_id": user_id if user_id is not None else config.user_id,
        "gate_mode": config.gate_mode,
        "stem_addr": config.stem_addr,
        "seam_addr": config.seam_addr,
        "max_run": config.max_run,
        "exact_blend": config.exact_blend,
    }
    for k, v in overrides.items():
        cfg_dict[k] = v

    if config.variant == "attention":
        return HMN3AttentionWR(
            vocab_size=cfg_dict["vocab_size"],
            dim=cfg_dict["dim"],
            n_layers=cfg_dict["n_layers"],
            n_heads=config.n_heads,
            n_experts=cfg_dict["n_experts"],
            top_k=cfg_dict["top_k"],
            use_moe=cfg_dict["use_moe"],
            tie_weights=cfg_dict["tie_weights"],
            gate_bias=cfg_dict["gate_bias"],
            asi_id=cfg_dict["asi_id"],
            user_id=cfg_dict["user_id"],
            stem_addr=cfg_dict["stem_addr"],
            seam_addr=cfg_dict["seam_addr"],
            max_run=cfg_dict["max_run"],
            max_seq_len=config.max_seq_len,
            use_checkpoint=config.use_checkpoint,
            dropout=config.dropout,
        )
    else:
        return HMN3(**cfg_dict)
