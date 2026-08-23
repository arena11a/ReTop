"""Helix Memory Network (HMN) — public API.

Core building blocks (v6 slim-down):
  backbone  : SelectiveSSM + HelixCouplingBlock + ReversibleFunction +
              SparseConditionalCompute (from hmn/v2.py)
  v3        : IdentityRegister, DualHeadDecoder, SeedPointer, HMN3 — the
              dual-register decoder with seam/stem run anchoring
  checkpoint: load_compat shim for pre-v6 checkpoints (dead keys stripped)

The full v2-era models (HMN, HMN_Option1, DifferentiableEpisodicMemory) were
removed in v6; they live under git tag `v3.3` and the local legacy archive.
"""

from hmn.checkpoint import load_compat
from hmn.v2 import (
    HelixCouplingBlock,
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
    "HelixCouplingBlock",
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
    "load_compat",
]

__version__ = "0.5.0"
