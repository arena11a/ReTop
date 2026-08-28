# v9.2 Probe Report: ptr3 Plateau Root Cause

**Date**: 2026-08-27
**Status**: COMPLETE — root cause identified

## Executive Summary

The ptr3 plateau (~0.7 unseen) is **NOT a capacity problem**. The model **cannot generalize to unseen verb families** because it memorizes specific verb tokens instead of learning the abstract "swap two segments" structural operation. This is a **training signal problem**, not a data or capacity problem.

## Key Findings

### Probe A: Capacity (D=64/128/256)

| Config | Seen | Unseen Slot | Unseen Verb | Time |
|--------|------|-------------|-------------|------|
| D=64 L=2 | 0.900 | **0.900** | 0.000 | 140s |
| D=128 L=3 | 0.800 | 0.000 | 0.000 | 190s |
| D=256 L=3 | 0.600 | 0.100 | 0.000 | 421s |

**Bigger models are WORSE.** D=256 underperforms D=64 on both seen and unseen. The model is overfitting to specific tokens rather than learning structure.

### Probe B: Signal (Per-Family Unseen Eval)

| Verb Pair | Accuracy |
|-----------|----------|
| load/unload | 0.000 |
| check/clean | 0.000 |
| search/find | 0.000 |
| start/stop | 0.000 |

**Cross-verb generalization is 0.000.** The model completely fails to transfer to unseen verb pairs, even with the same slot pattern.

### Probe C: Seen-Unseen Gap

| Seen | Unseen Slot | Unseen Verb | Gap |
|------|-------------|-------------|-----|
| 0.500 | 0.000 | 0.000 | **0.500** |

Massive generalization gap. The model memorizes training distribution without learning transferable structure.

### Probe D: Data Scaling

| Pairs | Seen | Unseen Slot | Unseen Verb |
|-------|------|-------------|-------------|
| 10 | 1.000 | 0.600 | 0.000 |
| 20 | 1.000 | 0.100 | 0.000 |
| 40 | 1.000 | 0.000 | 0.000 |

**More data makes slot generalization WORSE.** With 10 pairs the model generalizes to unseen slots (0.600), but with 40 pairs it drops to 0.000. The model overfits to specific tokens as data increases.

## Root Cause Analysis

1. **Verb tokens are not structurally separated.** The model sees "fetch", "deploy", "load", "unload" as arbitrary tokens, not as structural markers that define a transformation pattern.

2. **The seam pointer learns position, not structure.** The pointer CE loss trains the model to predict "copy from position X", but it doesn't learn that "X is the second slot" or "the pattern is: copy B first, then A".

3. **Cross-entropy loss doesn't penalize wrong structure.** If the model outputs "fetch pkg050 and deploy lib070" instead of "deploy lib070 and fetch pkg050", the loss is the same as any other wrong answer — there's no structural prior.

## Implications for v9.3-v9.7

The ptr3 plateau is fundamentally about **learning abstract operations**. Solutions must address:

1. **Structural tokenization** (v9.3): Separate verb tokens from slot tokens so the model can learn verb-agnostic operations
2. **Pointer supervision** (v9.4): Train the pointer to predict structural positions, not just token positions
3. **Data augmentation** (v9.5): Use multiple verb families in training to force structural learning
4. **Architectural priors** (v9.6): Add explicit structural attention or position encoding

## Recommendation

**Do NOT scale model size.** The problem is not capacity — bigger models make it worse. Focus on:
- Training on multiple verb families simultaneously (data augmentation)
- Adding structural position encoding to the seam mechanism
- Using contrastive loss to learn verb-agnostic representations
