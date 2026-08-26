# ReTop v7 — Attention-WR variant + evaluation harness + MoE scaling

> Status: **in-progress (2026-08-26).** M8-M11 ✅ shipped. M12-M14 pending.
> D≥512, build an evaluation harness, and scale MoE routing. v6 delivered
> the community-trainable foundation; v7 decides which WR variant wins at
> scale and provides the eval infrastructure to prove it.

## 0. Design rationale

v6 shipped the SSM-WR default (reversible SelectiveSSM coupling). The v6
roadmap flagged the key open question: "Attention-WR vs SSM-WR at D≥512:
decide by M4 evidence, not taste." v7 delivers the attention-WR variant
so both can be compared head-to-head on identical data and evals.

The evaluation harness (M11) is prerequisite: without automated evals,
comparisons are manual and unreliable. The harness runs both WR variants
on the same benchmarks and produces comparable metrics.

## 1. Milestones

| # | Milestone | Work | Pass criterion |
|---|---|---|---|
| M8 | **Attention-WR variant** ✅ | Add `attention_wr=True` config: pre-LN + RoPE + RMSNorm + SwiGLU-MoE attention block replacing SelectiveSSM coupling. Keep SSM-WR as default. Both share the IR + dual-head decoder. | attention-WR forward/backward produces finite loss; slot-copy 40/40 after training (same data/recipe as SSM-WR baseline); ReversibleFunction not used (attention is not reversible — gradient checkpointing instead) |
| M9 | **MoE routing improvements** ✅ | Expert load-balancing loss (Switch Transformer style); top-k routing with noisy gates; expert capacity factor tuning. Measure utilization across 16/32/64 experts. | aux load-balancing loss < 0.1; expert utilization variance < 2× mean; slot-copy parity maintained |
| M10 | **Long-context scaling** ✅ | Test at T=8k/16k with packed sequences; sparse attention patterns for IR (only attend to distinct tokens, not all positions); measure memory vs T. | T=8k forward fits in 8GB; T=16k forward fits in 16GB; slot-copy accuracy unchanged at T=8k |
| M11 | **Evaluation harness** ✅ | Automated eval suite: slot-copy (seen/unseen), chain, reorder, permutation; metrics: exact-match accuracy, gate statistics, copy-vs-gen ratio; comparison tool for two checkpoints. | harness runs all evals in one command; produces JSON report; comparison tool shows diff between two checkpoints |
| M12 | **Distillation / quantization** | Teacher-student distillation (large → small); INT8 dynamic quantization for inference; measure accuracy retention. | distilled model retains ≥90% teacher accuracy; INT8 quantization retains ≥95% accuracy; inference speedup ≥1.5× |
| M13 | **Multi-task training** | Joint training on copy + reorder + permutation tasks; task-specific prompts; measure negative transfer. | joint model matches per-task baselines within 5%; no task drops >10% |
| M14 | **Production deployment** | ONNX export; serving benchmark (latency, throughput); Docker container; API spec. | ONNX model loads and runs; latency < 50ms per token at D=256; Docker image < 1GB |

## 2. Architecture notes

### 2.1 Attention-WR (M8)

The attention-WR replaces the reversible SelectiveSSM coupling with a
standard transformer block:

```
Input → Pre-LN → Multi-Head Attention (RoPE) → Add → Pre-LN → SwiGLU FFN → Add
```

Key decisions:
- **RoPE** on attention keys/queries (position encoding)
- **RMSNorm** instead of LayerNorm (faster, same quality)
- **SwiGLU** FFN (gated activation, proven at scale)
- **MoE** optional on the FFN (same as SSM-WR's SparseConditionalCompute)
- **Gradient checkpointing** instead of reversible (attention is not reversible)
- Shares the IR + dual-head decoder with SSM-WR (only the WR changes)

Config flag: `attention_wr=True` switches from SelectiveSSM to attention.
Both variants can be trained on the same data and compared directly.

### 2.2 MoE routing (M9)

Current: `SparseConditionalCompute` with top-k routing. Issues:
- Expert load imbalance (some experts underutilized)
- Noisy routing during training (top-k sampling)

Improvements:
- **Switch Transformer** load-balancing loss: `α * N * Σ(f_i * P_i)` where
  `f_i` = fraction of tokens routed to expert i, `P_i` = mean routing prob
- **Noisy gates**: add Gaussian noise to logits before top-k selection
- **Capacity factor**: limit tokens per expert to prevent overflow

### 2.3 Long-context (M10)

Current IR attention is O(T²) in the dense path (but O(distinct tokens) in
the stats path). For T=8k/16k:
- Stats path already scales with distinct tokens (typically ≪ T)
- WR (SSM or attention) is the bottleneck at long contexts
- Sparse attention patterns: only attend to nearby positions (sliding window)
- Packed sequences (M5) help with variable-length docs

### 2.4 Evaluation harness (M11)

The harness should:
1. Load a checkpoint + tokenizer
2. Run all eval tasks (slot, chain, reorder, perm)
3. Compute metrics (accuracy, gate stats, copy ratio)
4. Output JSON report
5. Compare two reports (diff view)

## 3. Non-goals for v7

- No RLHF/DPO (deferred to community tooling)
- No FP8 (bf16 first, fp8 deferred)
- No multi-node training (single-GPU focus; FSDP2 for multi-GPU within node)
- No new task capabilities (reorder/perm/seam work frozen in v5/v6)

## 4. Open questions

- Does attention-WR outperform SSM-WR at D≥512? (M8 empirical answer)
- What's the optimal expert count for 100M-1B parameter models? (M9)
- Can sparse attention maintain slot-copy accuracy at T=16k? (M10)
- How much negative transfer occurs in multi-task training? (M13)
