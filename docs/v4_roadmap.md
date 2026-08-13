# ReTop v4 — longer, smarter, template-general (roadmap)

> Status: **in development** (branch `v4`). v3.3 (`hmn_v33.pt`) is the frozen,
> guardrail-verified baseline; all v4 work is additive behind an `arch=v4`
> opt-in so v3.3 behavior never changes.

---

## 1. Direction (decided 2026-08-13)

- **Budget** : GPU first-class — `detect_spec` grows for strong machines; CPU
  4 GB still works (tiny/small).
- **Priority** : *smart before long* — get reasoning / multi-step thinking
  right before chasing long context.
- **Copy generalization** : redesign the Identity Register (sparse marginal +
  relative gate) AND train it on many templates — **not** architecture alone.
  Why: the 0/40 on `import/run/apt {slot}` is not gate-only. The gen head can
  only *open* the copy chain for templates it has seen (`pip`); without
  multi-template training there is nothing to light the copy lane with.
- **Thinking** : prove the existing 1-pass latent-depth buffer first
  (`LatentThinkingBuffer`, `hmn/v3.py`) — if it wins, grow it into a
  latent-slot variant.
- **UX** : extend the existing GUI first; OpenAI-style server later, once the
  core is complete.

---

## 2. The problem, measured

v3.3 verified slot-copy, 200/200 (5 seeds x 40 unseen) — but:

| Probe | v3.3 |
|---|---|
| unseen `pip install {slot}` (blend/hard/structural) | **40/40** |
| 4-/5-digit slots | ~37/40 (pointer leaks) |
| alphanumeric `pkgA12`, `xy12` | 30/30 |
| templates `import / run / apt install` | **0/40** (gate lexicon-bound) |
| repeated-digit `pkg333..pkg999` | 3/9 (copy lane loops) |

v4 goals on the same probes: `import/run/apt` -> >= 36/40, and new long /
multi-step tasks that v3.3 cannot express at all.

---

## 3. Architecture v4 (components)

### A. IR — sparse copy-marginal
`copy_mass` is currently `(B, T, V)` via `scatter_add` (`hmn/v3.py`).
Replace with a per-sequence unique-token table -> ``(B, T, V_ct)`` where
`V_ct <= T`. Decode/loss gather back to vocab ids. No behavior change in v3.3 —
this is a memory/long-context enabler and the base for generalization.

Files: `hmn/v3.py`, `hmn/recipe.py` (loss_v33 / decode_v33 use `copy_dist`).

### B. IR — relative gate
Current: deterministic `sigmoid(tau * (gate_mass - 0.5))` (`hmn/v3.py`,
DualHeadDecoder). v3.1's *learned* gate collapsed to ~0.05 because it used copy
confidence alone.

v4: `g = sigmoid(MLP([h, gate_mass, gen_margin, behind, n_legal]))` — copy when
`(exact twin exists) AND (gen-head is not confident)`. `gen_margin` = gen
top-1 minus top-2 logit delta, the missing counter-signal.

Files: `hmn/v3.py`.

### C. Multi-template curriculum
`hmn/recipe.py::make_slot_batch` accepts `templates=[...]`; train_v3.py samples
per batch. Gen learns to open the copy chain for many verbs; the gate sees
copy-conditions across templates.

Files: `hmn/recipe.py`, `train_v3.py`, `gen_slots.py`.

### D. Thinking (smart first) — two phases
- **D1 (prove)**: wire `use_think` through retop (`retop.py build_model` does
  not pass it today); train on multi-step data; on/off ablation.
- **D2 (grow)**: latent-slot deliberation state in the decode loop, only at
  hard rows (conditional; only if D1 wins).

Files: `retop.py`, `train_v3.py`, `hmn/v3.py`, `hmn/recipe.py`.

### E. WR scale on GPU
New specs in `retop.py::SPECS` for strong boxes (`silver` D384/L6,
`gold` D512+/L8, MoE on), validated — MoE is unexplored at high D.

### F. Data (`gen_chat.py`, additive domains)
- `multi_step` : 2-3 step reasoning, answers computed in the generator.
- `code` : deterministic Python-stub Q&A (function write/answer computed here).

### G. Long context (later — gets cheaper for free from A)
Not chased early. Comes after smart.

---

## 4. Milestones

| Phase | Milestone | Work | Files | Pass criterion |
|---|---|---|---|---|
| M0 | Baseline + doc | branch v4, this doc, verify v3.3 | `docs/` | 40/40 + ALL TESTS |
| M1 | Sparse marginal | IR mass `(B,T,V)->` position-gather; loss reads (attn,nxt) | `hmn/v3.py`, `recipe.py` | **DONE**: parity 0.000e+00, 40/40 kept |
| M2 | Relative gate | learned gate w/ gen_margin | `hmn/v3.py` | template probes up |
| M3 | Multi-template | curriculum batch | `recipe.py`, `train_v3.py`, `gen_slots.py` | unseen templates >= 36/40; original still 40/40 |
| M4 | Think D1 | wire + validate on multi-step | `retop.py`, `train_v3.py`, `gen_chat.py` | thinking > no-thinking on multi-step |
| M5 | GPU spec | silver/gold + MoE@large | `retop.py` | val good, no crash |
| M6 | Think D2 | latent-slot (if M4 wins) | `hmn/v3.py`, `recipe.py` | beats D1 |
| M7 | GUI v4 | toggles + v4 verify matrix | `retop_gui.py` | think/spec toggle usable |
| M8 | Docs v4 | design doc, v4 guardrail, README | `docs/`, `experiments/verified/` | v4 guardrail pass + v3.3 still pass |
| later | Server | streaming + OpenAI-style endpoint | new | — |

Order M1 -> M4 is "smart before long": sparse marginal is done first because it
is the base of the new IR (and shortens context cost as a side effect).

---

## 5. Safety rules (no regressions)

- `experiments/verified/slot_v33_seed42.py` must keep passing 40/40 at every
  milestone. `hmn_v33.pt` is read-only.
- `test_hmn.py` runs after every change.
- v4 lives behind `arch=v4`; v3.3 remains the default. No silent behavior
  change.
- Every "smarter" claim needs an explicit metric + threshold, never vibes.

---

## 6. Open questions while building

- **M1 finding (2026-08-13)**: the true long-context bottleneck is NOT the
  `(B,T,V)` copy mass — it is the register attention `sim = qk @ ek.T`
  (`(B,T,T)`, explodes at T > ~1.5k on 4 GB) AND the dual-head logits
  `(B,T,V)` (the model output itself). M1 eliminated the copy-path duplicate
  allocations and made the copy CE read only (attn, nxt); long-context must
  tackle `(B,T,T)` (chunked/approximate attention) next, and `(B,T,V)` logits
  are irreducible for a full-vocab head.
- Whether `V_ct` marginal breaks the `copy_dist` renormalization identity used
  by `loss_v33` (must re-check the `-log p_target` gather) — resolved: M1 uses
  the position-gather formulation which is bit-identical, no renormalization
  drift.
- MoE gain at high D: enable only if it measurably helps val on GPU.
- Multi-template curriculum order: fixed order vs temperature-sampled mix.