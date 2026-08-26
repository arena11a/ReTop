"""Helix Memory Network (HMN) — public API.

Core building blocks (v9):
  backbone  : SelectiveSSM + HelixCouplingBlock + ReversibleFunction +
              SparseConditionalCompute (from hmn/v2.py)
  v3        : IdentityRegister, DualHeadDecoder, SeedPointer, HMN3 — the
              dual-register decoder with seam/stem run anchoring
  v7        : HMN3AttentionWR (attention-WR variant), RMSNorm, RoPE, SwiGLU,
              SparseConditionalComputeV2 (improved MoE routing)
  v9        : Trainer (production training loop with AMP, grad accumulation,
              LR schedule, checkpoint resume, early stopping)
  packing   : v6 M5 sequence packing (doc-masked padding, per-doc
              position ids, pack/unpack round trip)
  streaming : v6 M6 streaming data pipeline (jsonl reader, bounded shuffle)
  eval      : evaluation harness (slot-copy, chain, reorder, permutation)
  checkpoint: load_compat shim for pre-v6 checkpoints (dead keys stripped)
"""

from hmn.checkpoint import load_compat
from hmn.config import HMNConfig, create_model
from hmn.packing import (doc_masked_padding, doc_position_ids,
                         pack_sequences, unpack_outputs, pack_batch)
from hmn.streaming import (StreamJsonlReader, BoundedBufferShuffle,
                           ChatIdConverter, InfiniteStreamDataset, pad_collate)
from hmn.trainer import Trainer, TrainerConfig
from hmn.v2 import (
    HelixCouplingBlock,
    ReversibleFunction,
    SelectiveSSM,
    SparseConditionalCompute,
)
from hmn.v3 import (
    AttentionSeedPointer,
    DualHeadDecoder,
    HMN3,
    HMN3_NoReg,
    IdentityRegister,
    LatentThinkingBuffer,
    RelativeGate,
    SeedPointer,
)
from hmn.v7 import (
    HMN3AttentionWR,
    SparseConditionalComputeV2,
    RMSNorm,
    RotaryPositionEmbedding,
    SwiGLUFFN,
    AttentionBlock,
)

__all__ = [
    # config & factory
    "HMNConfig",
    "create_model",
    # v2 backbone
    "HelixCouplingBlock",
    "ReversibleFunction",
    "SelectiveSSM",
    "SparseConditionalCompute",
    # v3 dual-register decoder
    "AttentionSeedPointer",
    "DualHeadDecoder",
    "HMN3",
    "HMN3_NoReg",
    "IdentityRegister",
    "LatentThinkingBuffer",
    "RelativeGate",
    "SeedPointer",
    # v7 attention-WR + improved MoE
    "HMN3AttentionWR",
    "SparseConditionalComputeV2",
    "RMSNorm",
    "RotaryPositionEmbedding",
    "SwiGLUFFN",
    "AttentionBlock",
    # v9 trainer
    "Trainer",
    "TrainerConfig",
    # packing
    "pack_sequences",
    "doc_masked_padding",
    "doc_position_ids",
    "unpack_outputs",
    "pack_batch",
    # streaming
    "StreamJsonlReader",
    "BoundedBufferShuffle",
    "ChatIdConverter",
    "InfiniteStreamDataset",
    "pad_collate",
    # checkpoint
    "load_compat",
]

__version__ = "0.9.4"
