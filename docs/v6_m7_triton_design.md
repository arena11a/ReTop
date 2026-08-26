# v6 M7 — Kernel design notes: fused dual-head + optional SSM scan, torch.compile probe

> Status: **design notes + probe (2026-08-26).** No Triton source exists yet —
> deliberately: the repo law bans fake benchmarks, and this environment is
> CPU-only (`torch 2.13.0+cpu`), so any "kernel result" written today would be
> fiction. What IS real here: the eager dataflow audit (§1), the measured
> `torch.compile` probe (§3), the numeric verification protocol (§4), and the
> bandwidth model (§5). The roadmap gates — *numerics ≤ 1e-3*, *≥1.3× decode
> at D768* — stay OPEN until an A100/H100 runner executes them.
>
> Probe script (committed, rerunnable): `experiments/v6/m7_compile_probe.py`.

## 0. Ground truth this design is anchored to

- M1-B/C froze the semantics the kernel must reproduce EXACTLY at answer rows:
  snapped copy lane with support only on the seed group's payloads
  (`IRStats`, `hmn/v3.py:286`; blend argmax bound proof, `hmn/recipe.py:671`).
  Kernels chase semantics, not the reverse — same rule that sequenced M1 → M7.
- gpu-large spec (`experiments/v6/m4_scale_specs.py:58`): V=3190, D=768,
  L=12, state=64, MoE top-k2. All §5 numbers use it.
- fp32-island discipline (M3): losses/gate math stays fp32 regardless of
  activation dtype — the kernel inherits this (fp32 accumulators mandatory).

## 1. Fused dual-head kernel (gen log-softmax ⊕ copy ⊕ gate blend, one pass)

### 1.1 Eager dataflow today (what we are collapsing)

Stats path per forward (`hmn/v3.py:783-790`):

```
gen_logits = log_softmax(W_gen · [h | gm | b])        # (B,T,V)  v3.py:545
g          = sigmoid(clamp(tau·(gate_mass − 0.5), ±6))·(1−behind)   # v3.py:551-553
```

Consumers split three ways:

| consumer | reads | (B,T,V) materialized? |
|---|---|---|
| train CE `loss_v33` (`recipe.py:706`) | `gen_logits.gather(y)` ⊕ `st.prob_at(y)` via scalar `logaddexp` | NO (M1-C) |
| HF export `_blended_logprobs` (`hmn/hf.py:114-162`) | builds `p_copy` (zeros+scatter), `lp_c=log(p_copy)`, `out=logaddexp(...)` | YES — three tensors |
| decode `decode_v33`/`blend_argmax` (`recipe.py:671`) | `{gen argmax} ∪ payloads_of(t)` scored exactly | NO |

The dense-oracle path (`DualHeadDecoder.forward`, `v3.py:594-597`) additionally
materializes `copy_dist`, `gen.exp()`, `p`, `log(p)` — four more round-trips.
The fused kernel targets the `_blended_logprobs` contract (full-vocab blended
log-probs) and the decode epilogue; training keeps the M1-C gather form,
which the kernel also serves by emitting `blended[y]` rows on demand
(same kernel, V-tile masked to the target column).

### 1.2 Kernel I/O contract

Inputs (all prebuilt by existing torch code — nothing data-dependent crosses
the kernel boundary):

| operand | shape / dtype | source |
|---|---|---|
| `h` | (B,T,D) bf16/fp32 | WR output |
| `gate_mass`, `behind` | (B,T) fp32 / bool | `st.mass_same`, `st.behind` |
| `W_gen` | (V, D+2) bf16 | `dual.gen.weight` ([h,gm,b] layout, tie_embed respected upstream) |
| `tau` | scalar fp32 | `dual.tau` (learned; relative-gate variant → precompute g on host instead, see below) |
| copy CSR | `row_ptr` (B·T+1) i32, `pay_ids` (Σ\|P\|) i32, `pay_fr` (Σ\|P\|) f32 | derived host-side in one `torch.diff/cumsum` from the already-sorted `(h_key,h_bnd,h_cnt,h_y)` histograms + `denom` |
| forced anchors | `forced` (B,T) i64, `anchor_pay` (B,T) i64 | `st.forced`, `st.anchor_pay` |

Output: `log_p` (B,T,V) fp32 — the ONLY vocab-wide tensor allocated.

Row-run ranges (`row_ptr`) are a pure re-index of `IRStats`' flat histogram:
rows of group g inherit g's `[lo,hi)` run slice (`_run_slice`,
`v3.py:431-441`) — O(B·T) searchsorted on the host, no new semantics.

### 1.3 Single-pass math per row t

Non-normative pseudocode (math spec, not Triton):

```text
x_t   = concat(h[t], gate_mass[t], behind[t])            # D+2, loaded once
# phase A — online logsumexp while streaming the V dimension:
m     = -inf; s = 0
for V-tile: z = W_gen[tile] @ x_t                        # fp32 acc
            m_new = max(m, max(z));  s = s·exp(m−m_new) + Σexp(z−m_new);  m = m_new
lse   = m + log(s)
# phase B — second stream over V (weights resident in L2 from phase A):
for V-tile: gen_lp = z − lse
            if forced[t] ≥ 0:  lp = (y == anchor_pay[t]) ? 0 : −inf
            else:
              c = pay_fr[row_ptr[t]..row_ptr[t+1]] lookup for y ∈ P_t   # SMEM map
              lp = logaddexp(log(1−g) + gen_lp,
                             y ∈ P_t ? log(g) + log(c) : −inf)   # snapped lane
            write log_p[t, tile] = lp
```

Two facts make this genuinely one-pass-cheap rather than a rewrite of the
dense scatter:

1. **Out-of-support rows are free**: post-M1 snap, `p_copy(y)=0` for
   y ∉ P_t, so `logaddexp` degenerates to `log(1−g)+gen_lp` — no copy term,
   no scatter, no renormalization pass.
2. **Forced rows** are a one-hot select on `anchor_pay` (mass 1.0), matching
   `v3.py:424-429` exactly.

The deterministic gate (`sigmoid(clamp(τ·(gm−0.5),±6))`, times `(1−behind)`)
is computed inline per row in fp32 — τ≈12 saturation is exactly why the clamp
MUST be reproduced inside the kernel, not approximated by the sigmoid's
natural saturation. If `gate_mode="relative"` (`RelativeGate`, `v3.py:600`),
g is produced by the small MLP on host/torch first and passed in as (B,T);
the kernel takes g as an input. One kernel, two gate front-ends.

### 1.4 Tiling sketch

- Grid: `(ceil(T/T_tile), ceil(V/V_tile))`, e.g. T_tile=8 rows/program,
  V_tile=512 columns. Each program holds ≤8 row vectors of D+2 = 770 fp32 in
  registers/smem (~24 KB) and streams its V-tile of `W_gen`.
- Phase A/B two-stream trick: at T·V=2048·3190 the whole `W_gen` (bf16,
  4.9 MB) sits in L2 after the first stream, so the second stream is L2-bandwidth,
  not HBM. Alternative single-stream variant: keep running (m,s) AND stash
  per-tile partial z in smem when V_tile·T_tile fits (512·8·4B = 16 KB) —
  preferred; falls back to two-stream at larger tiles.
- Decode specialization (B=T=1): grid collapses to V-tiles only;
  additionally score just `{gen_argmax} ∪ P` for argmax mode
  (`blend_argmax` semantics) without writing any (1,V) tensor.

### 1.5 Memory accounting — the "avoids 3 separate (B,T,V)" claim, quantified

Eager `_blended_logprobs` chain, counted as full (B,T,V)-sized R/W passes
(fp32 islands): logits write 1, log_softmax R+W 2, `p_copy` zero-init 1,
clamp+log R+W 2, logaddexp R(genp)+R(lp_c)+W(out) 3 ⇒ **9 passes**, with 4
tensors live simultaneously at peak.

Fused kernel: `h` read O(B·T·D) + `W_gen` read (L2-resident) + sparse copy
runs + **1 write** ⇒ ~1.6 vocab-sized passes equivalent, 1 live tensor.

Per forward, B=1, T=2048, fp32:

| V | one (B,T,V) | eager chain (≈9 passes) | fused (≈1.6 equiv.) | peak live tensors |
|---|---|---|---|---|
| 3190 (gpu-large) | 26.1 MB | ≈235 MB | ≈42 MB | 4 → 1 |
| 32 768 | 268 MB | ≈2.4 GB | ≈430 MB | 4 → 1 |
| 131 072 | 1.07 GB | ≈9.6 GB | ≈1.5 GB | 4.3 GB → 1.07 GB |

At the shipped V=3190 the absolute saving is modest (~190 MB/fwd @T=2048);
the design pays off at community vocab scale (§5) and at decode-launch level
(§1.4/§5.3). Stated plainly so nobody expects a 2× end-to-end win from this
kernel alone at V=3190.

## 2. Optional: fused SelectiveSSM forward

Motivation first — this is the BIGGER bandwidth fish. The chunked closed-form
scan (`_chunked_scan`, `hmn/v2.py:47-90`) materializes per call: padded
`log_da`,`db`; `L`; `eL_inv`; `Sf`; `f`; `A_chunk`,`B_chunk`,`h_ins`,`h` —
each (B,T,D,S) fp32. At D=768, S=64, T=2048, B=1 one such tensor is
**402.7 MB**; peak ~5 concurrent ⇒ ~2 GB transient allocations and ~8 passes
(~3.2 GB traffic) per `SelectiveSSM.forward`. Each coupling layer calls it
twice (F1/F2 at D/2) × 12 layers ⇒ tens of GB of intermediate traffic per
training forward. The recurrence itself is trivial arithmetic; the cost is
entirely tensor round-trips.

Design (two kernels, preserving v2.4's closed-form exactly):

- **Phase 1 (per-chunk)**: grid over (b, chunk, d_slice). In-program
  sequential scan over C=8 positions computes `L=cumsum(log_da)`,
  `Sf=cumsum(db·e^{−L})`, `f=e^L·Sf`, and emits chunk aggregates
  `(A_chunk,B_chunk)` — all in registers/SMEM. With d_slice=64, a position
  tile is 64×64×4B = 16 KB; C=8 positions ≈ 128 KB — borderline vs 164 KB
  A100 SMEM ⇒ loop d_slices inside the program (state slice resident,
  positions streamed). Only `f`, `A_chunk`, `B_chunk` go to HBM.
- **Phase 2 (inter-chunk)**: exact sequential connect `h_in ← A_c·h_in+B_c`
  over n_chunks=T/8. Two options, decided by measurement: (a) persistent
  kernel with cooperative grid sync (one launch, n_chunks in-kernel steps);
  (b) keep the tiny host loop of `torch` scalar-tensor ops as today
  (n_chunks=256 launches of (D,S)-sized work is launch-bound — option (a)
  exists precisely to kill this).
- **Output fusion**: contract `y = Σ_s h·C_m + D⊙x` inside phase 1's epilogue
  so `h` (B,T,D,S) is NEVER written — output drops to (B,T,D). This alone
  deletes ~half the phase-1 traffic; combined, projected ~25–50× traffic cut
  per call (exact number is a measurement, not a claim).
- Numerics invariants carried verbatim from `v2.py`: `clamp(min=−9)` on
  `log_da` BEFORE cumsum (purity note: |L| ≤ 72 < 88 fp32 exp range);
  padding rows dA=1/dB=0; fp32 throughout regardless of AMP mode.
- Backward: recompute-in-backward matches the `ReversibleFunction` pattern
  (`v2.py:129`) — phase-1 recompute from saved `(log_da, db)` inputs only;
  no (B,T,D,S) activations stored. Interop risk with FSDP (M5 open question)
  applies here too; sequence: land forward numerics gate first.

## 3. torch.compile probe — MEASURED (2026-08-26, this machine)

Environment: `torch 2.13.0+cpu`, CUDA: none, 4 cores (torch threads=2),
cpu-small spec (dim=96/L=3/state=8/V=3190), stats path, eval, B=1 T=20,
fixed shapes, `dynamic=False`, median of 3×100-call batches.
Script: `experiments/v6/m7_compile_probe.py`.

```
hasattr(torch, 'compile') -> True
torch.compile(lambda x: x) -> compiles & runs ("compile works")
```

HMN3 forward, eager vs compiled:

| metric | value |
|---|---|
| eager | 18–38 ms/fwd (CPU wall noise across sessions) |
| compiled | 9–25 ms/fwd |
| **speedup** | **1.26× – 2.80× across 3 sessions; typical ≈1.5×** |
| first-call compile | 14–77 s |
| max \|Δ gen_logits\| compiled vs eager | **5.341e-05** (identical every session) |
| max \|Δ g\| | **0.0** (bitwise) |

Both deltas ≤ 1e-3 ⇒ the compiled graph is numerically safe to build on.

Dynamo findings that shape the whole M7 plan:

1. **Graph break at `IRStats.__init__`** — `G = int(grp_sorted.max()) + 1`
   (`v3.py:336`) is a data-dependent scalar; dynamo splits the graph around
   IRStats and compiles embed/WR/head separately. That break is currently
   LOAD-BEARING.
2. **Do NOT set `TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1`**: verified failure —
   `InductorError: LoweringException: GuardOnDataDependentSymNode: Could not
   guard on Eq(u0 + 1, 1)`. Compilation aborts outright. Beyond the guard,
   capturing through IRStats would also key graphs on G (group count varies
   per batch) → recompile churn. Conclusion: **compile scope = WR block stack
   + dual-head epilogue wrapper; never IRStats internals.**
3. Strategy consequence for Triton: on GPU, Inductor would already fuse the
   pointwise tail (softmax/exp/logaddexp) into its own Triton — but it cannot
   fuse ACROSS the cuBLAS GEMM boundary, nor turn the dense `p_copy`
   scatter+log into the sparse CSR read of §1.3. Those two things are the hand
   kernel's entire reason to exist. If Triton is unavailable, a respectable
   fallback is `torch.compile(DualHeadDecoder epilogue)` + accepting the
   extra passes (numbers in §1.5 shrink but don't vanish).

## 4. Kernel numerics: verifying ≤1e-3 against eager

Reference oracle: `HMN3(exact_blend=True)` — the dense dual-head path M1
deliberately kept bit-faithful — plus `IRStats.prob_at/payloads_of` for the
snapped lane. Compare in **log space** (max-abs on blended log-probs): the
loss consumes log-probs, and log-space diffs absorb exp/rounding noise where
probability-space diffs would amplify near-zero masses.

Protocol (mirrors M1-A/M1-C conventions):

1. **Primary gate**: `max |logp_kernel − logp_eager| ≤ 1e-3` over ALL rows
   (prompt rows included — they're declared approximate in IRStats ctx, but
   the BLEND lane is exact there too under the snap; assert it).
2. **Property checks** (each independently asserted):
   - `exp(log_p).sum(-1) ∈ [1±1e-3]` per row;
   - forced rows: argmax == `anchor_pay` exactly; mass at anchor ∈ [1±1e-6];
   - rows with empty payload groups reduce to `log(1−g)+gen_lp` (pure gen)
     within 1e-6 of the closed form;
   - out-of-support columns match `log(1−g)+gen_lp` to ≤1e-6 (this pins the
     snap semantics, not just aggregate error);
   - gate saturation: rows with `|τ·(gm−0.5)| > 6` produce g identical to the
     clamped eager value (bitwise target; else ≤1e-7).
3. **Adversarial input sweep**: β ∈ {1, 12, 31, 100}; duplicate-heavy prompts
   (whole-sequence same id); ASI-boundary rows; single-column prompts;
   T not divisible by chunk/tile sizes; G=1 degenerate sequences.
4. **Gate is held to a TIGHTER bar than the blend** (≤1e-6, ideally bitwise
   on the clamped path): g multiplies the entire distribution, and M1-B
   measured gate diff 0.0 between paths — regress that and parity suites lie.
5. **dtype matrix**: fp32 end-to-end; bf16 activations with fp32 accumulation
   (island rule); bf16 weights. Never bf16 accumulation in the lse.
6. **Randomized trials**: 300 seeded random (ids, β, τ, forced-mask) cases vs
   oracle — same shape as M1-C's brute-force validation.
7. **End-to-end behavioral gate**: slot_v4 + chain_v4 suites through a
   `kernel_blend=True` flag (pattern copied from `exact_blend`) must hold
   40/40 with identical gate traces before the flag defaults anywhere.

Honest boundary: Triton targets NVIDIA GPUs; this repo's CI is CPU-only.
Steps 1–6 run first against a **pure-torch emulation of the kernel's exact
reduction order** (two-stream online-lse of §1.3) — that validates the ALGORITHM
and the tolerance budget without hardware. Steps on real hardware (ptxas
behavior, tf32 defaults, denormals) need the GPU runner; the milestone's
numeric gate closes only there. This is the same CPU-honesty M4 used.

## 5. Performance model @ D768

Roofline framing: A100-80GB ≈ 1.94 TB/s HBM2e peak (~1.6 TB/s achievable),
~161 bf16-FLOP/byte ridge. The head epilogue at prefill shapes is GEMM-bound
(AI ≈ 2T ≫ ridge at T≥512) — fusion buys the elementwise tails, NOT the GEMM;
at decode (T=1, AI≈2) it is purely memory/latency-bound — fusion buys
everything. So the roadmap's *decode* speedup criterion is the right one to
chase, and prefill wins should be stated in bytes, not speedup.

### 5.1 Head epilogue traffic per forward (B=1, T=2048, fp32 islands, bf16 weights)

From §1.5 table: eager ≈235 MB vs fused ≈42 MB at V=3190 ⇒ ~0.12 ms saved at
1.6 TB/s on a step whose WR+MoE compute dominates ⇒ **low single-digit %
end-to-end at V=3190 prefill**. At V=131072: 9.6 GB vs 1.5 GB ⇒ ~5 ms saved —
material. Peak-allocation relief (4→1 live vocab tensors) matters for
batch-size headroom at every V: 4.3 GB → 1.07 GB transient at V=131k.

### 5.2 SSM scan traffic per training forward (L=12, F1/F2, T=2048)

Eager: ~(6–8)×402.7 MB × 24 calls ≈ **60–77 GB** intermediate traffic
(≈35–45 ms of pure HBM time at 1.6 TB/s, before counting allocator pressure).
Fused (§2 design): ~34 MB per half-D call ⇒ <2 GB total ⇒ **~30–50× traffic
cut on the WR stack**. This — not the head — is where a prefill-visible
speedup lives, and why §2 is "optional" only in priority, not in value.

### 5.3 Decode step (B=1, T=1, D768/L12)

- Bandwidth view: active-weight read dominates (multi-100 MB); head `W_gen`
  is 4.9 MB of it ⇒ fusing the head changes almost nothing byte-wise at
  V=3190.
- Latency view: eager epilogue ≈ 15–20 kernel launches (channel-concat, GEMV,
  log_softmax, clamp/sigmoid chain, scatter/normalize, exp/mul-add/log,
  logaddexp, topk) ≈ 70–140 µs of launch/gap overhead; fused = 2 launches
  (head kernel + candidate scorer). Whether that reaches the roadmap's ≥1.3×
  depends on the rest of the step: the SSM decode path still runs a host
  chunk loop and IRStats does per-step sort/searchsorted with `.item()` syncs.
- Therefore the ≥1.3× decode criterion is credible ONLY as a COMBINED result:
  head kernel (launch cuts) + §2 scan kernel (removes the chunk-loop stalls)
  + optionally CUDA-graphing the whole decode step. Head-alone at V=3190 will
  not clear it; at V≥32k it plausibly could on bandwidth alone. Measured,
  not assumed, when hardware lands.

## 6. Milestone status

| M7 gate (roadmap line 100) | status |
|---|---|
| kernel numerics match eager ≤1e-3 | probe-level PASS for `torch.compile` (5.3e-05 / 0.0); Triton gate OPEN (needs GPU runner, §4 protocol locked) |
| end-to-end ≥1.3× decode @ D768 | OPEN — analysis says combined-kernel result (§5.3); measurement pending |
| design notes + probe | THIS DOC — done |
