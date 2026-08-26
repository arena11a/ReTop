# v6 M7 — Kernels: fused dual-head design notes + torch.compile probe

> Status: **design + probe (2026-08-26, branch v6).** Roadmap item M7
> (`docs/v6_scaling_roadmap.md` §1 table): *"Triton fused dual-head (gen
> softmax ⊕ copy ⊕ gate blend in one kernel)"* with pass criteria *kernel
> numerics match eager ≤1e-3* and *end-to-end ≥1.3× decode speedup at D768*.
>
> **Honest boundary:** this environment is CPU-only (torch 2.13.0+cpu, no
> CUDA driver, `import triton` fails). No Triton kernel is committed or can
> be executed here — by instruction this milestone delivers DESIGN NOTES +
> the torch.compile probe only. Every number below was measured in this
> session (§3); every performance figure in §5 is an analytical model, not a
> GPU measurement. GPU validation is an explicit follow-up (§6).

## 0. Why the kernel still pays after M1 (scope honesty)

M1 removed the `(B,T,V)` blend from the DEFAULT train-loss and decode-argmax
paths (per-target exact `logaddexp` in `loss_v33`, `hmn/recipe.py:758`;
candidate-set argmax bound in `blend_argmax`, `hmn/recipe.py:671`). The fused
kernel therefore targets the consumers that still want a full-width blended
distribution:

1. **HF integration** — `hmn/hf.py:161` materializes the full-vocab blend for
   `generate()` compatibility.
2. **Oracle/debug paths** — `--exact-blend` (`hmn/v3.py:801-809`) exists
   precisely because the dense math remains the reference.
3. **Full-distribution consumers** — perplexity eval, KD, sampling with copy
   mass, and any future tokenizer growth (community 32k–128k vocabs put
   `(B,T,V)` cost back in scope linearly).
4. **Decode latency** — candidate scoring currently runs host-side Python
   (`payloads_of` slices + the `blend_argmax` loop); a device-side fused
   scorer removes the host sync.

The dense eager reference being fused (bit-faithful M1-A oracle,
`DualHeadDecoder.forward` sparse/dense branch, `hmn/v3.py:594-597`):

```
gen       = log_softmax(head(cat([h, gm, b], -1)))   # (B,T,V)
copy_dist = F.normalize(copy_logits, p=1, dim=-1)     # (B,T,V)
p         = (1 - g) * gen.exp() + g * copy_dist
out       = log(p.clamp(min=eps))
```

Semantic contract (identical value computed per element, cf. `loss_v33`):
`out[v] = logaddexp( log(1-g) + ls[v], log(g) + log c[v] )` with `ls =
log_softmax(gen_logits)` and `c` the copy distribution.

## 1. Fused dual-head kernel

### 1.1 Input/output contract

| operand | shape | source |
|---|---|---|
| `gen_logits` | `(B,T,V)` fp32 (bf16 allowed post-M3, fp32 island inside) | `dual.gen` head output |
| `g` | `(B,T,1)` fp32 | deterministic or RelativeGate (`v3.py:551-553`, `v3.py:627-639`) |
| copy support | **sparse**, NOT dense: per-row group id `col_grp[b,t]` (`v3.py:339`) into the group payload runs `(h_y, h_cnt)` + `denom` (`v3.py:389-397`); fractions `cnt/denom` | `IRStats` |
| forced anchors | `(B,T)` long, `-100` none; anchored payload at p=1.0 one-hot (`v3.py:374-379`, `424-429`) | `IRStats.anchor_pay` |
| **out** | `(B,T,V)` fp32 blended log-probs | kernel |

Key structural fact: after M1's twins-uniform snap the copy lane has support
only on the seed group's payloads `P` (typically `|P| ≪ V`). The kernel must
NOT take a dense `(B,T,V)` copy input — that would reintroduce the
materialization M1 removed.

### 1.2 Algorithm (design, not code)

One program per `(b,t)` row; `V` walked in `BLOCK_V` lanes (V=3190 → 4 tiles
at 1024):

1. **Sweep 1 (softmax + stash):** stream `gen_logits` tiles once; maintain
   running `(m, s)` online-softmax state; write `e_v = exp(z_v − m_run)`
   (rescaled on `m` updates) into an **SMEM buffer of `V` fp32** = 12.8 KB at
   V=3190 (fits the 48 KB static budget alongside staging). This makes the
   pass truly "one pass over HBM", per the roadmap phrasing.
2. **Histogram stage:** zero-init a second SMEM region (or reuse after
   barrier), scatter the row-group's `P` fractions (`cnt/denom`) — `P` loads
   are a few dozen bytes. Forced rows skip this: `c_v = [v == anchor]`.
3. **Sweep 2 (blend, SMEM-resident):** per lane,
   `out_v = logaddexp(log1p(-g) + (log e_v − log s_final),
   log(g) + log c_v)`, then the exact eager clamp semantics
   `out_v = max-blend … log(max(p_v, eps))`. Lanes beyond V tail-masked.
   `logaddexp` implemented as `max + log1p(exp(−|Δ|))` to mirror
   `torch.logaddexp`.
4. Single `(B,T,V)` HBM write of `out`.

Fallback for V > smem budget (≥32k vocabs): drop the SMEM stash, do TWO HBM
passes over `gen_logits` (re-read in sweep 2), or Option B: per-lane binary
search into the sorted unique histogram keys (`h_key` layout already sorted
by construction, `v3.py:386-390`) — `log2|P|` probes, no smem. Decision by
measurement at target vocab; default Option A/smem for V ≤ 16k.

### 1.3 Backward (cheap for a pleasant reason)

Post M1-B, copy fractions are integer-count-derived **constants w.r.t.
parameters** (raw-id histograms) — gradient flows only through `gen_logits`
and `g`. With `p_v = (1−g)e^{ls_v} + g·c_v`:

```
r_v      = grad_out_v · (1−g) / p_v                      (c_v = 0 rows: p=(1−g)e^{ls})
grad_z_u = e^{ls_u} · ( r_u − Σ_v r_v e^{ls_v} )          # softmax Jacobian
grad_g   = Σ_v grad_out_v · (c_v − e^{ls_v}) / p_v
```

Phase 1: `torch.autograd.Function` with fused forward, saving `(e, s, p)`
and computing backward with standard tensor ops. Phase 2: fused backward
kernel. **Determinism is mandatory**: `ReversibleFunction.backward`
(`hmn/v2.py:143-163`) RE-RUNS block forwards during backward, so any
atomics/nondeterministic reduction inside a fused kernel used in blocks
would desynchronize the reversible reconstruction. No atomics anywhere.

### 1.4 Edge cases (each maps to a test in §4)

- `g ∈ {0, 1}`: clamp exactly like `loss_v33` — `log1p(-g.clamp(max=1-1e-7))`,
  `log(g.clamp(min=1e-12))` (`recipe.py:758-759`); deterministic gate is born
  in [0.0025, 0.9975] but the learned gate is unbounded.
- Empty payload group (`P = ∅`): `c_v ≡ 0`, copy term `-inf`,
  `logaddexp` degenerates to the gen lane — never NaN.
- Forced/stem/seam rows: one-hot copy at `anchor_pay`, `g` irrelevant
  (matches dense one-hot attention semantics, `v3.py:424-429`).
- `p < eps` floor rows: reproduce `log(max(p, eps))` exactly (reachable when
  g→1 and target ∉ P).
- `V % BLOCK_V ≠ 0`: tail masking; `-inf` masked logits flow through as
  `exp(-inf)=0`.
- ASI/EOS illegal payloads are already excluded upstream (`col_ok`,
  `v3.py:324-325`) — kernel trusts the histogram.

### 1.5 Decode variant

At B=T=1 the whole problem is 3190 elements — launch-latency-bound, not
bandwidth-bound. Two-step plan: (a) standalone kernel captured in a CUDA
graph together with the rest of the step; (b) later, epilogue fusion into
the head GEMM (persistent kernel consumes accumulator tiles directly),
removing one full logits write+read and a launch. Phase (b) is where the
≥1.3× decode criterion most plausibly lands at V=3190 (see §5).

## 2. Optional: fused SelectiveSSM forward (`hmn/v2.py:22-104`)

Current `_chunked_scan` materializes ≥8 `(B,T,D/2,S)` fp32 intermediates
(`log_da` (+clamped copy), `db`, `L`, `eL_inv`, `Sf`, `f`, `A_chunk`,
`B_chunk`, `h_ins`, `h`). This is the LARGEST bandwidth term in training —
see §5 — which is why the roadmap marks it optional-but-attractive.

Kernel shape (mirrors the existing exact two-phase closed form, `v2.py:47-90`):
grid `(B·n_chunks, D/(BLOCK_D))`; each program walks its chunk's `C=8`
positions sequentially with the state vector in registers, emitting `h`
tiles; cross-chunk carries `A_chunk/B_chunk` go through a small workspace
`(B, n_chunks, D, S)`; phase 2 connects chunks exactly as today.

Numerics contract to replicate BIT-CLOSELY (all already documented in code):
`clamp(log_da, min=-9.0)` applied BEFORE the cumsum; padding uses
`dA=1, dB=0`; identical op order in the closed form; fp32 purity bounds
`|L| ≤ 72 < 88` hold unchanged (`v2.py:56-66`).

Interop risks: (i) called twice per step under reversible recompute — same
determinism requirement as §1.3; (ii) prior art exists (`mamba_ssm`
`selective_scan_*`, causal-conv1d kernels) — adopt-vs-write must be decided
by measuring those FIRST on A100 against this exact recurrence; (iii) the M4
open question "attention-WR vs SSM-WR at D≥512" gates the effort: if WR
switches to attention, FA kernels replace this entirely. **Recommendation:
measure `mamba_ssm` + torch.compile-on-scan before writing anything; the
compile probe below suggests inductor may already fuse much of the
elementwise scaffolding around the small phase-2 loop.**

## 3. torch.compile probe — MEASURED RESULTS (2026-08-26)

Environment: torch **2.13.0+cpu**, Python 3.12, Linux x86_64, **CPU-only**
(no CUDA driver; `import triton` → ModuleNotFoundError). Inductor ran its
CPU C++/OpenMP backend. Probe script (condensed, fully self-contained):

```python
import time, torch, torch.nn.functional as F
from hmn.v3 import HMN3                       # repo root on sys.path

m  = HMN3(vocab_size=3190, dim=96, state_dim=8, n_layers=3, use_moe=False,
          gate_bias=-1.0, asi_id=5).eval()
ids = torch.randint(6, 2000, (4, 512)); ids[:, 256] = 5
def bench(fn, x, warmup=3, iters=10):
    for _ in range(warmup): fn(x)
    t0 = time.perf_counter()
    for _ in range(iters): fn(x)
    return (time.perf_counter() - t0) / iters * 1e3
with torch.no_grad():
    ms_eager = bench(m, ids)
    cm = torch.compile(m, dynamic=False)
    t0 = time.perf_counter(); cm(ids); compile_s = time.perf_counter()-t0
    ms_comp = bench(cm, ids, warmup=2)
print(ms_eager, ms_comp, ms_eager/ms_comp, compile_s)

# dual-head blend region (the kernel target), exact v3.py:594-597 math
def blend(z, cm_, g, eps=1e-8):
    gen = torch.log_softmax(z, -1)
    cd  = F.normalize(cm_, p=1, dim=-1)
    return torch.log(((1 - g) * gen.exp() + g * cd).clamp(min=eps))
z  = torch.randn(8, 1024, 3190) * 4
cm_= torch.zeros(8, 1024, 3190).scatter_add_(
        -1, torch.randint(0, 3190, (8, 1024, 16)), torch.rand(8, 1024, 16))
g  = torch.rand(8, 1024, 1)
```

Results (wall-clock, `no_grad`, eval mode):

| probe | config | eager | compiled | speedup | 1st-call compile | max abs diff |
|---|---|---|---|---|---|---|
| `hasattr(torch,'compile')` | — | — | **True** | — | — | — |
| identity lambda compiles | — | — | **works** | — | — | — |
| `HMN3.forward` | cpu-small D96/L3/S8, B4 T512 | 935.40 ms | 756.52 ms | **1.24×** | 153.1 s | gen 9.2e-05, gate **0.0** |
| `HMN3.forward` | gpu-large SLICE D768/S64/**L2**, B1 T384 | 3165.40 ms | 2754.40 ms | **1.15×** | 53.5 s | gen 9.2e-04, gate **0.0** |
| blend region (§ target) | `(8,1024,3190)` fp32 (105 MB/tensor) | 836.34 ms | 805.57 ms | **1.04×** | 1.1 s | **1.5e-05** |

Graph-break census (`torch._dynamo.explain`): **14 breaks per forward** in
both configs. First break: `Tensor.item()` on `int(grp_sorted.max())`
(`hmn/v3.py:337`) — i.e. the IR index build; root causes are the
data-dependent scalars and control flow (`sel.numel()`, `fm.any()`,
searchsorted-derived Python ints, dynamic shapes from sorts) in `IRStats`.

Findings, stated plainly:

1. Whole-forward compile is a free **1.15–1.24× on CPU**, capped by the 14
   IRStats graph breaks (WR blocks + head do get fused). Cheap follow-ups:
   `torch._dynamo.config.capture_scalar_outputs = True`, and/or deliberately
   excluding `IRStats` from compilation (it is index bookkeeping, not FLOPs).
2. The blend region compiles to only **1.04× on CPU** — the chain is
   DDR-bandwidth-bound and eager already streams it near optimally. **This
   does NOT predict the GPU case**: the Triton kernel's value on A100/H100 is
   eliminating multiple ~100 MB-class HBM round trips (§5), which a CPU probe
   cannot exhibit. The CPU probe validates NUMERICS (1.5e-05 ≪ 1e-3 ✓), not
   the speedup thesis.
3. Parity of the compiled D768 whole-forward is **9.2e-04** — inside the M7
   ≤1e-3 budget but uncomfortably close (inductor GEMM/reduction reordering).
   On GPU, validate against the dense oracle with TF32 disabled and re-check.
4. Compile cost 53–153 s per graph: fine for training jobs, unacceptable
   cold-start per decode request → precompile/AOT-cache in serving.

Roadmap reconciliation: §2.6 scheduled this probe "in M4"; it is delivered
here alongside M7 as instructed — M4 records may reference this section.

## 4. Kernel numerics: verifying ≤1e-3 vs eager

**Reference definition:** the dense oracle (`--exact-blend`, bit-faithful
pre-M1 math per M1-A/B) IS the eager ground truth. Verification ladder:

1. **Unit:** randomized `gen_logits`/`g`/histograms (incl. realistic
   skew: logits scaled ×4) → assert `max|kernel − DualHeadDecoder.dense| ≤
   1e-3`. Expected fp32 residual ~1e-6; the budget is spent on `tl.exp`/
   libdevice ULP differences and online-vs-two-pass reduction order.
2. **Adversarial:** g→{0,1⁻}; empty payload groups; forced-anchor rows;
   constructed `p ≈ eps` floor rows; `V % BLOCK_V ≠ 0` tails; `-inf` masked
   columns; bf16-input mode compared against a **bf16** eager reference
   (declared separately — never mix references).
3. **Properties:** `logsumexp(out, dim=-1) == 0 ± 1e-3` per row (both lanes
   normalized ⇒ blend is normalized); monotonicity of `argmax out` in `g`
   toward the copy support.
4. **Behavioral:** guardrail suites through a kernel-backed decode switch —
   same gate as M1-B: slot_v4/chain_v4 40/40 (gate 0.775), chain/reorder
   1.000, v5 M3/M4 suites green.
5. **Determinism:** repeated runs bitwise-equal (required by reversible
   recompute, §1.3/§2).
6. **GPU acceptance adds:** on-device eager-vs-kernel diff DISTRIBUTION
   (p50/p99), not just max; TF32 off for the fp32 comparison.

## 5. Performance model @ D768 (analytical, not measured)

Spec constants (`experiments/v6/m4_scale_specs.py:58-62`): V=3190, D=768,
L=12, S=64, MoE top-k2. Canonical train shape B=8, T=2048; fp32 = 4 B/elem.
A100 HBM ≈ 1.55 TB/s (H100 ≈ 3.35 TB/s).

**(a) Dual-head stage, training shape.** One `(B,T,V)` tensor =
8·2048·3190·4 B = **208.9 MB**. Eager chain traversals (read+write per op):
GEMM write 209; `log_softmax` 418; `.exp()` 418; `F.normalize` 418; blend
627; `clamp` 418; `log` 418 ⇒ ≈ **2.93 GB** HBM traffic. Fused kernel
(one-pass, §1.2): GEMM write 209 + read 209 + write 209 ≈ **0.63 GB**
(⇒ ~4.6× stage traffic cut; ~14× if the head GEMM epilogue absorbs the
kernel entirely). Time: ~1.9 ms → ~0.41 ms per forward at A100 bandwidth.
**Honest framing:** against a full L=12 step this stage is single-digit % of
step time — dual-head fusion primarily serves the §0 consumers
(HF/eval/sampling), not overall train throughput.

**(b) Decode, B=T=1.** Per-tensor 12.8 KB — entirely launch/sync-bound
(~µs-scale op overhead × dozens of ops, plus the host-side candidate loop in
`blend_argmax`). The ≥1.3× decode criterion at V=3190 decomposes roughly:
CUDA-graph/compile capture of the whole step (largest lever, §3 finding 4),
device-side candidate scoring removing host syncs, epilogue fusion saving a
launch. At community vocab sizes (32k+: 128 KB/tensor, ~7 traversals ≈ 1.8 MB
per row per beam) bandwidth starts to matter even at T=1 — the kernel's
margin grows with V.

**(c) SelectiveSSM stage, training shape — the dominant term.** Per
half-SSM call at D/2=384, S=64: one `(B,T,384,64)` tensor = 8·2048·384·64·4 B
= **1.61 GB**; `_chunked_scan` writes ≥8 of them (plus comparable reads) ⇒
order **25 GB traffic per SSM call**, ×2 halves ×L=12 layers ⇒ hundreds of
GB per forward — dwarfing (a). Fused scan: streaming tiles + carry workspace
`(B,n_chunks,384,64)` = 201 MB ⇒ ~10× cut on the dominant term.
**Priority order for actual kernel-writing effort: (1) SSM scan fusion,
(2) decode/epilogue path, (3) standalone dual-head blend kernel.**

**(d) Roofline sanity.** The blend math is ~12 flops/elem over ~8 B/elem of
traffic ⇒ arithmetic intensity ≈ 1.5 flops/B vs the A100 ridge point
(~200:1): deeply memory-bound. Confirms the optimization strategy is
*minimize traversals*, not FLOPs — exactly what fusion buys.

## 6. Exit criteria for the kernel-writing phase (follow-up)

Blocked on: GPU runner access; M4 open question (attention-WR vs SSM-WR).
Acceptance: §4 ladder green; measured stage speedups within ±30 % of the §5
model; **end-to-end decode ≥1.3× at D768 on A100** (roadmap gate). Landing
pattern mirrors M1: kernel behind a flag (`dual.fused_kernel="triton"|None`),
dense oracle path retained for parity.

### References

- Dense blend reference: `hmn/v3.py:556-597` (`DualHeadDecoder`)
- Stats/copy-lane structures: `hmn/v3.py:286-455` (`IRStats`)
- Per-target blend CE / candidate bound: `hmn/recipe.py:666-767`
- Chunked SSM scan + purity notes: `hmn/v2.py:22-104`
- Reversible recompute (determinism constraint): `hmn/v2.py:129-163`
- HF blend consumer: `hmn/hf.py:150-180`
- D768 spec constants: `experiments/v6/m4_scale_specs.py:42-63`
- Roadmap item: `docs/v6_scaling_roadmap.md` §1 (M7 row), §2.6
