"""Helix Memory Network (HMN) — public API.

v2  : reversible coupling backbone (SelectiveSSM) + product-key MoE +
      DifferentiableEpisodicMemory. Verified 2026-08-07: single-token recall 97-99%,
2-token recall 94-97% (multi-head).
      recall 97-99% (single-token), 94-97% (multi-token), on random-query eval.
v3  : dual-head decoder with Identity Register (hard copy lane) + optional latent
      thinking buffer. PoC stage — the register solves slot-copy that a softmax
      head alone cannot (see docs/hmn_v3_design.md).
"""

from hmn.v2 import (
    DifferentiableEpisodicMemory,
    HelixCouplingBlock,
    HMN,
    HMN_Option1,
    ReversibleFunction,
    SelectiveSSM,
    SparseConditionalCompute,
)
from hmn.v3 import (
    DualHeadDecoder,
    HMN3,
    HMN3_NoReg,
    IdentityRegister,
    LatentThinkingBuffer,
    RelativeGate,
    SeedPointer,
)

__all__ = [
    "DifferentiableEpisodicMemory",
    "HelixCouplingBlock",
    "HMN",
    "HMN_Option1",
    "ReversibleFunction",
    "SelectiveSSM",
    "SparseConditionalCompute",
    "DualHeadDecoder",
    "HMN3",
    "HMN3_NoReg",
    "IdentityRegister",
    "LatentThinkingBuffer",
    "RelativeGate",
    "SeedPointer",
]

__version__ = "0.5.0"
