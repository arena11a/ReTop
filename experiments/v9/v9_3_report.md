# v9.3 Multi-Family Training Results

**Date**: 2026-08-27
**Status**: COMPLETE — multi-family training alone is NOT sufficient

## Experiment

Trained on multiple verb families simultaneously, evaluated on unseen verb families.

### Setup
- Model: D=128, L=3, attention variant with seam_addr + attn_ptr
- Training: 3000 steps, bs=8, lr=3e-4
- Train families: fetch/deploy, load/unload, check/clean, search/find, start/stop, push/pull
- Unseen families: read/write, open/close

### Results

| Config | Seen | Unseen Slot | Unseen Verb |
|--------|------|-------------|-------------|
| 1 family (baseline) | 0.800 | 0.000 | 0.000 |
| 2 families | 0.800 | 0.100 | 0.000 |
| 4 families | 0.800 | 0.100 | 0.025 |
| 6 families (3000 steps) | 0.900 | 0.000 | 0.000 |

### Key Finding

**Multi-family training does NOT solve the ptr3 plateau.** Even with 6 training families, the model still scores 0.000 on unseen verb families.

## Root Cause (Refined)

The model learns **verb-specific token patterns** rather than abstract structural operations. Even when trained on multiple families, it memorizes:
- "fetch" → copy position X
- "deploy" → copy position Y
- "load" → copy position X (different tokens, same pattern)

But it cannot generalize to:
- "read" → ??? (never seen before)

## Implications for v9.4+

Multi-family training alone is insufficient. Need to address:

1. **Structural tokenization** (v9.4): Separate verb tokens from slot tokens
2. **Verb normalization** (v9.5): Replace all verbs with generic tokens during training
3. **Contrastive learning** (v9.6): Learn verb-agnostic representations
4. **Pointer supervision** (v9.7): Train pointer to predict structural positions, not token positions

## Recommendation

**Do NOT rely on data augmentation alone.** The model's architecture or training objective must be changed to learn abstract operations.
