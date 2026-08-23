# ReTop v6 — HMN as a trainable foundation architecture (1B–7B+, GPU-native)

> Status: **roadmap (2026-08-23).** Goal: turn HMN from a CPU research model
> into a foundation architecture the community can train on A100/H100 clusters
> — a "RAG-native LLM": token-exact grounding from the prompt without an
> external vector DB. Every milestone keeps the repo law: explicit metric +
> threshold, guardrails green, honest boundaries stated.

## 0. The key structural insight (why scaling HMN is easier than it looks)

The Identity Register's attention is **exact self-match on raw token ids**
(query = keys = raw embeddings, `keys_proj` default OFF). `cos(q,k) = 1` iff
two columns hold the SAME token id; softmax over identical similarities is
uniform among them. Therefore the `(B,T,T)` similarity tensor and the
`(B,T,V)` copy-mass scatter are a wasteful implementation of what is
semantically an **inverted index: token_id → list of prompt positions**.

Consequences (v6-M1 builds on this):
- IR lookup can be rewritten as a per-sequence hash map with EXACT parity to
  the dense path (uniform weight over same-id columns reproduces the softmax;
  `mass_same`, `n_legal`, payload chains all read off the index).
- stem-addr / seam forcing already bypass the tensor path with direct plans —
  they get simpler under the index form.
- Long-context cost drops from O(T²)+O(T·V) to O(total distinct tokens).

## 1. Milestones

### M1 implementation spec (locked 2026-08-23, after empirical probe)

Probe result (canonical slot seq, β≈31): dense attention rows carry ~9
non-zero columns — cross-id ε-mass EXISTS (`e^{β·cos}` for cos≈0.2–0.5 is
small but non-zero). Therefore bit-parity is impossible by construction;
**behavioral parity** (all guardrail suites + eval accs) is the gate, and the
semantic change "cross-id ε-mass dropped; twins-uniform snap" must be stated
in the release notes.

Phased commits (each keeps CI green):
- **A. intra-dense wins** ✅ 2026-08-23: the second `(B,T,T)` tensor (`same`
  bool matrix for `mass_same`) and the `n_legal` O(T²) reduction are replaced
  by index-derived computations inside the existing dense flow (`hmn/v3.py`):
  `n_legal` is a prefix-sum over a per-column legality vector (the mask's only
  row dependence is `j <= t`, so a cumsum reproduces it EXACTLY); `mass_same`
  uses the twins-uniform identity — same-id columns carry bitwise-identical
  pre-softmax scores (`cos=1` on identical embeddings), so
  `mass_same[t] = c_t · exp(beta − lse_t)` with `c_t` = legal twins with
  `j <= t` read off a stable sort + composite-key searchsorted inverted index.
  No consumer contract changes; dense-vs-sparse parity still 0.000e+00, all
  guardrails green with identical metrics (40/40 gate 0.775, chain/reorder
  1.000). Measured stats-section speedup: **7× @ T=512, 72× @ T=2048,
  153× @ T=4096** (B=4, CPU); removes the `(B,T,T)` bool + its fp32 product
  (~335 MB transient at T=4096). Unit test `test_v6_m1a_index_stats` pins the
  brute-force parity (tolerance 1e-4 = FP summation noise only).
  Also fixed: `experiments/v4/m1_sparse_parity.py` now actually uses
  `load_compat` (imported-but-unused since M0.5 → strict load crashed on the
  legacy checkpoint).
- **B. consumer migration**: introduce `IRStats` container (CSR grouped by
  value-id: legal member positions ascending + cumcounts; per-group payload
  histograms via flattened (b,gid,payload) sort) replacing raw `a` in:
  `loss_v33` copy branch (`stats.prob_at(Yc)`), blend CE via exact
  `logaddexp(log(1-g)+gen_logp[y], log g+copy_logp[y])`, gate inputs,
  ctx (segment-mean), decode copy-argmax (per-group payload MODE table).
  `out["logits"]` dense-shape becomes oracle-only (`--exact-blend`);
  trainers/decode consume the stats API. Tests asserting dense shapes are
  updated with documented rationale.
- **C. decode candidate scheme**: blend argmax over {gen top-k ∪ copy mode}
  proven exact via probability bounds; (B,T,V) survives only in
  `(B,T,V_gen_head)` which is the model output itself (irreducible, §6 note).

| # | Milestone | Work | Pass criterion |
|---|---|---|---|
| M0 | Freeze | branch v6 from main @ ed3ef40 | CI green (v3.3/v4/v5 gates) |
| M0.5 | **Core slim-down** ✅ 2026-08-23 | v2-era models removed (tag v3.3 preserves), dead `key_proj/keys_proj` dropped, `load_compat` shim in all loaders, extras `[hf]/[scale]`, infer v3-only | suites green on slim core |
| M1 | **IR as inverted index** | replace `_attn` tensors w/ per-sequence index; keep dense path as oracle flag | bit-parity vs dense on slot/chain/perm suites (loss+decode); memory: no (B,T,T)/(B,T,V) allocations at T=4k |
| M2 | HF packaging | `HMN3Config`/`HMNForCausalLM` (PreTrainedModel), save/from_pretrained, CausalLMOutput; tokenizer round-trip | `SFTTrainer` trains 100 steps on toy data without custom loop; outputs identical to native path |
| M3 | Precision ladder | AMP BF16 harness; audit loss_v33 / gate BCE / SeedPointer in fp32-island style (softmax/log in fp32); reversible-backward recompute under bf16 | train parity vs FP32 within tolerance on 600-step run; no NaN over 5 seeds |
| M4 | Scale spec + smoke | new specs (`gpu-large`: D768/L12 MoE-SwiGLU top-k2 …); single-GPU A100 smoke train | loss ↓ monotonically 1k steps at D768; tokens/s reported; ckpt resumable |
| M5 | Distributed | sequence packing (doc-masked, position_ids) + FSDP2/DeepSpeed wrap — verify ReversibleFunction autograd compat first | 2×GPU smoke: linear-ish throughput scaling ≥1.6×; eval parity with single-GPU |
| M6 | Streaming data | HF Hub streaming (fineweb/slimpajama-class) → chat-id format; bounded buffer shuffle; billion-token budget accounting | 24h stream w/o RAM growth; val split stable across restarts |
| M7 | Kernels | Triton fused dual-head (gen softmax ⊕ copy ⊕ gate blend in one kernel); optional FA/Mamba kernels for WR | kernel numerics match eager ≤1e-3; end-to-end ≥1.3× decode speedup at D768 |

Order rationale: M1 first because every later item multiplies its cost —
kernels, packing, and distributed all shrink once the register stops
materializing T²/T·V. M2 early so community tooling (TRL/SFTTrainer) works
before scale experiments.

## 2. Architecture notes per proposal (grounded in current code)

### 2.1 Identity Register O(1) lookup (M1)
Current: `IdentityRegister._attn` builds `sim = qk @ ek.T` (B,T,T) then dense
scatter to (B,T,V) (`hmn/v3.py`). Index form:
```
index[token_id] -> [positions...]        # built once per forward from ids
payload(j) = ids[j+1]                    # unchanged
copy_mass(v) = uniform(index[v])         # == masked softmax exactly
mass_same(t) = count(index[ids[t]] ∩ legal) / len(index[ids[t]])
```
Seam/stem plans apply directly as anchor overrides (no attention to patch).
Caveats to verify in M1: `keys_proj=True` variant breaks exact-match (either
drop the option or keep a dense fallback path behind the old flag);
beta sharpening is irrelevant when sims are identical (document).

### 2.2 Working Register modernization (M4)
Decision point, not a given: WR today is reversible SelectiveSSM coupling
(`hmn/v2.py`) — recurrence IS its position mechanism, so RoPE applies only if
we add an attention-WR variant (the M8 vanilla baseline is that starting
point). Proposal: keep SSM-WR default; ship an attention-WR variant
(pre-LN + RoPE + RMSNorm + SwiGLU-MoE) behind config so both can be compared
at scale instead of assuming one wins.

### 2.3 Precision (M3)
Risks specific to this codebase: manual `-log p` copy CE must gather in fp32
(`clamp(min=1e-9)` underflows bf16); gate `sigmoid(tau*(m-0.5))` with tau≈12
saturates in bf16; `ReversibleFunction.backward` recomputes blocks — recompute
precision must match forward or reversibility drifts. Mitigation pattern:
fp32 islands around losses/gate + bf16 elsewhere, verified by M3's parity test.

### 2.4 Distributed (M5)
Unknowns flagged honestly: `ReversibleFunction` uses a custom autograd Function
with saved-tensor reconstruction — FSDP hook interop is unproven; if it
conflicts, fallback is gradient-checkpointing-style sharding per block.
Sequence packing must respect the answer-region masks (Y/Yc/G triple) —
packing is per-document with loss masks carried through, not naive concat.

### 2.5 Data (M6)
`gen_chat.py` stays as the GUARDRAIL corpus generator (deterministic, verified
answers). Community-scale training adds a streaming reader targeting the
chat-id format of `hmn/recipe.make_chat_ids`; identity-register labels are
computed on-the-fly per document, so no precomputed label storage is needed.

### 2.6 Kernels (M7)
Fused dual-head kernel = gen log-softmax + copy distribution + gate blend in
one pass (removes three (B,T,V) round-trips). Only started after M1 freezes
semantics — kernels chase a moving target otherwise. torch.compile probe runs
in M4 to catch free wins early.

## 3. Non-goals for v6

- No new task capabilities (reorder/seam work is done and frozen in v5 tags).
- No RLHF/DPO implementation — M2 only guarantees compatibility via TRL.
- No FP8 (bf16 first; fp8 deferred until bf16 parity is boring).

## 4. Open questions

- Does the inverted-index IR change anything for the ptr3 plateau (~0.7)?
  Hypothesis: no (seam seeding is orthogonal), but M1's parity suite should
  include seam tasks to confirm.
- Attention-WR vs SSM-WR at D≥512: decide by M4 evidence, not taste.
- License/packaging: PyPI release alongside M2 (name `retop-hmn`).
