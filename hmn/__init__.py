"""Helix Memory Network (HMN) — public API.

Core building blocks (v6 slim-down):
  backbone  : SelectiveSSM + HelixCouplingBlock + ReversibleFunction +
              SparseConditionalCompute (from hmn/v2.py)
  v3        : IdentityRegister, DualHeadDecoder, SeedPointer, HMN3 — the
              dual-register decoder with seam/stem run anchoring
  packing   : v6 M5 sequence packing (doc-masked padding, per-doc
              position ids, pack/unpack round trip)
  checkpoint: load_compat shim for pre-v6 checkpoints (dead keys stripped)

The full v2-era models (HMN, HMN_Option1, DifferentiableEpisodicMemory) were
removed in v6; they live under git tag `v3.3` and the local legacy archive.
"""

from hmn.checkpoint import load_compat
from hmn.packing import (doc_masked_padding, doc_position_ids,
                         pack_sequences, unpack_outputs)
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
    "doc_masked_padding",
    "doc_position_ids",
    "pack_sequences",
    "unpack_outputs",
]

__version__ = "0.6.0"
