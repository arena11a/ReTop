# v6 M7 — Kernel design notes: fused dual-head ⊕ optional SelectiveSSM scan

> Status: **design + compiler probe (2026-08-26).** This environment has NO
> Triton (torch 2.13.0+cpu, no CUDA): everything kernel-side below is a
> DESIGN contract, deliberately not code. The only executed evidence is the
> torch.compile probe (`experiments/v6/m7_compile_probe.py`, results in §3).
> Roadmap pass criterion (M7): kernel numerics ≤1e-3 vs eager AND ≥1.3×
> decode speedup at D768 — §6 flags an honest risk against the second half.

## 0. Where M7 plugs into post-M1 code

After M1 the register lane is SPARSE: copy mass lives in per-group payload
histogram runs (`IRStats.h_key/h_bnd/h_cnt/h_denom`, `hmn/v3.py:380-423`),
not in a dense (B,T,V) tensor. Consumers today:

| consumer | location | width |
|---|---|---|
| gen head (log-softmax over `[h,gm,b]`) | `DualHeadDecoder.gate_and_gen`, hmn/v3.py:538 | (B,T,V) slab |
| deterministic gate g | hmn/v3.py:551 | (B,T,1) scalar/row |
| full-vocab blended logits (HF emission) | `hf.py:147-162` | **dense (B,T,V) `p_copy` + blend** |
| train loss | `recipe.py:758` / `hf.py:176` | per-target logaddexp, no slabs |
| decode argmax | `recipe.py:blend_argmax` | {gen-argmax} ∪ P_t support only |

The only remaining DENSE (B,T,V)-wide work is logits emission (HF
`forward(labels=None)`): it builds `genp` (log_softmax), then a dense
`p_copy` via a **host-side python loop** over group runs (`hf.py:148-157`),
then `log(clamp)` + `logaddexp`. That is exactly the "three separate
(B,T,V) materializations" of roadmap §2.6. Train loss and decode were
already de-slabbled in M1-B/C and need NO kernel.

## 1. Fused dual-head kernel: gen log-softmax ⊕ copy ⊕ gate blend

### 1.1 Contract

    inputs:
      gen_logits (B,T,V)   fp32 (bf16 accepted, upcast on load) — GEMM stays OUTSIDE
      g          (B,T,1)   fp32 — computed OUTSIDE (see 1.4)
      run tables           h_key (sorted int64 flat keys), per-row run bounds
                           lo/hi (B,T) int32 precomputed via searchsorted on
                           col_grp*V (same keys as hf.py:141-145, hoisted to
                           torch ops before launch), h_cnt/denom → frac
      forced/anchor        forced (B,T) bool + anchor_pay (B,T) (v3.py:374-379)
    output:
      logp_blend (B,T,V)   out[t,v] = logaddexp( log1p(-g_t) + gen_lp[t,v],
                                             (log g_t + log frac_t[v]) if v ∈ P_t else -inf )
      with gen_lp = log_softmax(gen_logits) over v, P_t = own-group payload set,
      frac = h_cnt / denom (forced rows: anchored payload at frac=1.0).

### 1.2 Layout of the sparse copy support (the actual design problem)

C_g = |P_t| ≤ distinct payloads of the row's token group — empirically tens
at trained scales (M1 histograms). Three options, recommendation last:

- **A. Padded CSR** `(B·T, R_max)` payload/fraction matrices: static shapes,
  trivial kernel; wasted lanes on tail-heavy max_run; recompile per shape
  bucket. Simplest, most wasteful.
- **B. Two-kernel fixup**: pass 1 writes `log1p(-g)+gen_lp` everywhere;
  pass 2 applies logaddexp corrections ONLY at run entries
  (O(total histogram entries), no dense copy slab ever exists). Robust to
  huge C_g; two launches.
- **C. Single-pass scalar-bounded loop (RECOMMENDED)**: grid = (B·T);
  program loads run bounds lo,hi as SCALARS, sweeps V in BLOCK_V tiles —
  sweep 1 computes online max/lse (standard two-pass online softmax),
  sweep 2 recomputes tile logits, subtracts lse, and for each tile tests
  which of its C_g payloads falls in range (C_g comparisons per tile, cheap)
  writing `logaddexp(base, corr)` there instead of base. One launch, no
  atomics, no padding; falls back to B only if some C_g pathology (>~1k
  distinct payloads/group) ever appears — assertable host-side from
  `h_cnt` stats.

### 1.3 Edge cases (all pinned by existing eager semantics)

- empty support (lo==hi) or g==0 rows (prompt/behind folded into g at
  v3.py:553): pure base path, uniform per-program branch.
- g→1: replicate the CONSUMER clamps exactly — emission clamps
  `g∈[1e-12, 1-1e-7]` (hf.py:160); loss clamps only the top
  (recipe.py:758). The kernel targets emission first; document which clamp
  it bakes in per call site rather than silently picking one.
- forced rows: anchor payload at frac=1.0 (copy_mode_p semantics, v3.py:428).
- eps-floor difference: the legacy dense oracle floors with
  `log(p.clamp(min=1e-8))` (hmn/v3.py:597) → deep-tail values read -18.42;
  true logaddexp does not floor. Bounded discrepancy, only where p<1e-8;
  argmax unaffected; tier-B behavioral suites arbitrate (§5).

### 1.4 Numerics decisions

- Gate stays OUTSIDE the kernel in an fp32 island: τ≈12 sigmoid saturates
  bf16 (roadmap §2.3, M3 lesson). Kernel receives fp32 g.
- `log1p(-g)` not `log(1-g)`; fp32 accumulation throughout; online-softmax
  running max with rescale.
- Backward = custom autograd.Function: Triton forward + EAGER recompute
  under autograd for backward (house style — ReversibleFunction recomputes;
  M3 requires recompute precision match). Fused backward deferred until
  profiling demands it.

### 1.5 What it does NOT replace

loss_v33 per-target logaddexp and blend_argmax stay as-is — they are
already O(support) post-M1. The kernel serves full-vocab emission
(HF packaging, eval perplexity) and any future dense-logit consumer.

## 2. Optional: fused SelectiveSSM scan

Current chain (`SelectiveSSM.forward` + `_chunked_scan`, hmn/v2.py:92-104):
LN → in_proj/delta_proj → softplus → dt; A=-exp(A_log); elementwise
log_da=A·dt, db=dt·Bm; chunked closed-form scan materializes L, eL_inv, Sf,
f (+pad copies) — six-ish (B,T,D_h,S) intermediates — then y=(h*Cm).sum +
D·x → out_proj.

Why it matters at gpu-large (m4_scale_specs.py:58: D=768/L=12/**S=64**):
each F-block's scan tensors are (1,T,384,64) ≈ 101 MB fp32 at T=1024, ~a
dozen slab traversals per call, 24 calls per forward (L12 × F1/F2) — see
§6 for the byte count. This, not the head, is the largest DRAM-traffic
item at V=3190.

Design boundary:
- fuse PHASE 1 only: elementwise pre-chain + pad-free within-chunk cumsum +
  closed-form response into ONE kernel emitting h and chunk-boundary states;
- keep phase 2 (chunk connector, n_chunks sequential steps on tiny (B,D_h,S)
  states) in torch — or raise chunk_size; a persistent/grid-sync variant is
  explicitly not v1.
- purity constraints carry over verbatim (v2.py docstring): fp32 only,
  `log_da.clamp(min=-9)`, |L| ≤ chunk·9 < 88, e^-72 above subnormal floor.
- backward: scan adjoint is nontrivial → same forward-fused/eager-backward
  strategy as §1.4 initially; ReversibleFunction recompute interplay must be
  recompute-consistent (same caveat family as the M5 FSDP unknown).
- DECISION GATE before any Triton here: probe `torch.compile(ssm._chunked_scan)`
  first — if inductor recovers most traffic, skip the hand kernel. NOT yet
  probed (this doc's probe covers the head/model level only).

## 3. torch.compile probe results (executed 2026-08-26)

Env: torch 2.13.0+cpu, cuda=False, triton ABSENT, 4 cpus / 2 torch threads.
Probe availability checks both pass:
`hasattr(torch,'compile') == True`; trivial compile round-trip OK.
Script: `experiments/v6/m7_compile_probe.py --full`.

| config | shape | eager ms | compiled ms | speedup | max\|Δ\|gen_lp | max\|Δ\|g |
|---|---|---|---|---|---|---|
| guardrail-std | D96/L3/B2/T128/V3190 | 424.29 | 249.21 | **1.70×** | 6.1e-05 | 0.0 |
| D768-ish | D768/L2/B1/T64/V3190 | 446.68 | 345.58 | **1.29×** | 3.7e-04 | 0.0 |

Findings:
- Whole-model compile wins 1.29–1.70× on CPU (inductor-cpp) come mostly
  from the WR block chains; numeric drift 6e-05…3.7e-04 — already inside
  the M7 1e-3 tolerance, encouraging for the kernel gate.
- Head region `gate_and_gen` = **12% of forward** at toy scale; compiled
  alone it is 1.06× (small cfg) but **0.71× — SLOWER — at D768-ish**: the
  compiler does NOT win the Linear+[gm,b]-concat+log_softmax fusion in the
  target regime. Hand fusion has real headroom (CPU-cpp evidence; GPU may
  differ).
- Graph break #1 (whole-model trace): `G = int(grp_sorted.max()) + 1`
  (hmn/v3.py:336) — data-dependent scalar extraction inside IRStats.
  Dynamo suggests `TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1`; that changes
  device-sync semantics and must be re-probed before relying on it.
- Honest boundary: CPU backend ≠ Inductor-Triton; GPU conclusions here are
  prioritization signals, not benchmarks.

## 4. Kernel numerics: verifying ≤1e-3 vs eager

Three tiers, landing as tests in `test_hmn.py` (pattern:
`test_v6_m1a_index_stats`):

- **Tier A (unit)**: synthetic gen_logits incl. extremes (±80, subnormals),
  random small histograms, g ∈ {0, 1, ε, 1-ε}, forced rows, empty-support
  rows. Reference = eager composition using the SAME formula
  (log1p/logaddexp/clamps of §1.1). Assert max|Δ| ≤ 1e-3 absolute in
  log-space AND ≤1e-6 relative where |ref| > 5. dtypes: fp32, and
  bf16-in/fp32-compute.
- **Tier B (behavioral parity)**: slot_v4/chain_v4 batches through the full
  forward with the fused emission vs stats-path eager: gen_logits/g
  unchanged (kernel touches emission only), blended argmax identical on
  every answer row, eval accs identical, guardrails green — same acceptance
  style as M1.
- **Tier C (precision interplay)**: repeat A/B under autocast-bf16 with the
  fp32 islands intact (M3 harness reuse); same tolerances vs fp32 eager.

Enumerated divergence sources (expected magnitude): ULP of tl.exp/tl.log
(~1e-7), reduction-order differences in the row lse (~1e-6 rel),
online-rescale error (~1e-7), the documented eps-floor semantic difference
(§1.3, bounded, argmax-neutral), log1p precision near g→1 (mitigated by the
consumer clamp). Acceptance = all tiers green + CI green; any tolerance
failure defaults to the eager path (kernel behind a config flag, oracle
retained per house law).

## 5. Performance model at D768

Setup: gpu-large preset — D=768, L=12, state_dim S=64, V=3190
(m4_scale_specs.py:58); fp32 head slabs; A100-class effective HBM
~1.5–2.0 TB/s; N = B·T = 4096 tokens prefill.

Head stage, slab = N·V·4B:

| V | slab size | eager emission traffic (§0 op walk: genp W; p_copy zeros+W; clamp.log R+W; logaddexp 2R+1W) | fused (R logits + W out) | saved |
|---|---|---|---|---|
| 3190 | 52 MB | ~7 slabs ≈ 366 MB | 2 slabs ≈ 104 MB | **~260 MB → 130–175 µs/forward** |
| 32768 (community tok) | 537 MB | ~7 slabs ≈ 3.8 GB | 2 slabs ≈ 1.1 GB | **~2.7 GB → 1.3–1.8 ms/forward** |

Plus: removes the host-side python loop over groups (hf.py:148) — a
launch/sync tax that grows with G.

SSM scan stage at gpu-large (B=1, T=1024, D_h=384, S=64): scan intermediate
slab = 101 MB; unfused phase-1 ≈ 11–13 traversals ≈ 1.1–1.3 GB/call; fused
phase-1 ≈ 2 traversals → save ~1 GB/call × 24 calls ≈ **~24 GB/forward ≈
12–16 ms** (upper bound — assumes nothing is already fused; measure first,
per the §2 decision gate).

Prioritization consequence (the useful output of this section): at
V=3190 the BYTES live in the SSM scan, not the head — order of attack is
(1) `torch.compile(_chunked_scan)` probe, (2) fused head (smallest
correctness surface, unblocks HF emission), (3) fused SSM phase 1. At
community vocabularies (V≥16k) the head fusion becomes co-equal.

Decode regime (T=1) honesty: head bytes collapse to the weight read
(770·V·4B ≈ 9.8 MB @V=3190) and step time is launch-latency + WR-scan
bound; the fused head saves ~4 launches (~tens of µs). The roadmap's
"≥1.3× end-to-end DECODE speedup at D768" is therefore AT RISK from this
kernel alone — recommend either amending the criterion to prefill/train
throughput or scoping the decode win to a captured/compiled whole-step
graph. Decision recorded here, to be settled when a GPU env exists.

## 6. Rollout checklist

- [x] torch.compile probes (availability + HMN3 forward) — §3, script committed
- [ ] re-probe with `TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` (graph-break lever)
- [ ] `torch.compile(_chunked_scan)` probe (SSM decision gate, §2)
- [ ] Tier A unit test skeleton against an eager reference implementation of §1.1
- [ ] Triton env available → implement fused head (option C), tiers A/B/C
- [ ] fused SSM phase 1 (only if the compile probe leaves ≥2× traffic on the table)
- [ ] settle the decode-criterion question (§5) with a GPU measurement
