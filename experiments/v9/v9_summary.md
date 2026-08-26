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
| D: Data | More data makes slot generalization WORSE |

## v9.3: Speed — torch.compile ✓
- Added `use_compile` + `compile_mode` to TrainerConfig and HMNConfig
- Benchmark results (CPU): D=64 1.21x, D=128 0.79x, D=256 1.16x (avg ~1.1x)
- Graph breaks from IRStats data-dependent branching (fundamental)

## v9.4: Head-to-Head — Attention vs SSM ✓
- Slot-copy: both 1.000 (SSM with 25-46% fewer params)
- Reorder: SSM 0.900 vs attention 0.700 at D=64
- **Key finding**: SSM is the better default for ReTop

## v9.5: ptr3 Breakout — Multi-task + Curriculum ✓
- Baseline: 0.000 unseen verb
- Multi-task (50/50): 0.000 unseen verb
- Curriculum: 0.000 unseen verb
- **Conclusion**: Multi-task/curriculum does NOT help — ptr3 is fundamental

## v9.6: Perm Generalization — Neural Seeder ✓
- Improved SeedPointer with positional encoding
- Tested on unseen verb families
- **Finding**: Positional encoding helps slightly but not enough

## Key Insights

1. **The ptr3 plateau is a TRAINING SIGNAL problem**, not capacity/data
2. **Bigger models make it worse** — more capacity = more overfitting to tokens
3. **Multi-family training doesn't help** — model still memorizes verb-specific patterns
4. **All decoding strategies fail** on unseen verbs — they depend on token-level patterns
5. **SSM is better than attention** for ReTop's task profile (25-46% fewer params)

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
- `experiments/v9/v9_3_benchmark.py` — torch.compile benchmark
- `experiments/v9/v9_4_head_to_head.py` — attention vs SSM comparison
- `experiments/v9/v9_5_ptr3_breakout.py` — multi-task + curriculum experiment
