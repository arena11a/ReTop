# v9 Summary: Understanding-First Research

**Date**: 2026-08-27
**Philosophy**: Inspired by YouTube video where 440-param human-designed AI beat LLM-designed AIs

## v9.1: Configurable Architecture (M22) ✓
- All hardcoded params moved to `HMNConfig` with backward-compatible defaults
- New `Trainer` class with AMP, grad accumulation, LR schedule, checkpoint resume
- 95+ tests pass

## v9.2: Failure Probes — ptr3 Plateau Root Cause ✓

**Root cause: The model memorizes verb tokens instead of learning abstract swap structure.**

| Probe | Finding |
|-------|---------|
| A: Capacity | Bigger models are WORSE (D=256 < D=64) |
| B: Signal | Cross-verb generalization = 0.000 across all families |
| C: Gap | Massive generalization gap (seen=0.500, unseen=0.000) |
| D: Data | More data makes slot generalization WORSE (overfitting) |

## v9.3: Multi-Family Training ✓
- Trained on 1/2/4/6/8 verb families simultaneously
- Unseen verb generalization still 0.000 even with 8 families
- **Conclusion**: Multi-family training alone is NOT sufficient

## v9.3: Speed — torch.compile ✓
- Added `use_compile` + `compile_mode` to TrainerConfig and HMNConfig
- Trainer._setup_compile(): automatic model compilation
- Benchmark results (CPU):
  - D=64: 1.21x speedup
  - D=128: 0.79x (slower due to graph breaks)
  - D=256: 1.16x speedup
- Graph breaks from IRStats data-dependent branching (fundamental)
- CPU ceiling: ~1.2x with current architecture

## v9.4: Verb Normalization ✓
- Replaced verbs with generic A/B markers during training
- Unseen verb generalization: 0.075 (up from 0.000)
- **Conclusion**: Slight improvement but not enough

## v9.4b: Structural Decoding (decode_rotate) ✓
- Used rule-based structural rotation decoding
- Still 0/5 on all unseen verb families
- **Conclusion**: Structural decoding also depends on verb-specific tokens

## Key Insights

1. **The ptr3 plateau is a TRAINING SIGNAL problem**, not capacity/data
2. **Bigger models make it worse** — more capacity = more overfitting to tokens
3. **Multi-family training doesn't help** — model still memorizes verb-specific patterns
4. **All decoding strategies fail** on unseen verbs — they depend on token-level patterns

## Fundamental Issue

The tokenizer splits verbs into sub-tokens that are NOT reusable across verb families:
- `fetch` = [74, 386, 527] (3 tokens)
- `load` = [1964] (1 token)
- `read` = [1725] (1 token)

These are completely different token patterns. The model cannot learn that "fetch" and "load" are structurally equivalent.

## Recommendations for v10+

1. **Custom tokenizer with verb-class tokens**: Add `<VERB_A>`, `<VERB_B>` tokens
2. **Two-stage approach**: Detect verb class → apply learned transformation
3. **Pre-training on structural patterns**: Train on abstract patterns before specific verbs
4. **Contrastive learning**: Learn verb-agnostic representations
5. **Skip neural swap**: Use purely structural decoding with verb-agnostic segment detection

## Files

- `experiments/v9/probe_all.py` — all-in-one probe script
- `experiments/v9/probe_report.md` — v9.2 detailed report
- `experiments/v9/v9_3_multi_family.py` — multi-family training experiment
- `experiments/v9/v9_3_report.md` — v9.3 report
- `experiments/v9/v9_4_verb_norm.py` — verb normalization experiment
