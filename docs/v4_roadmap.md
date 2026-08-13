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
| M3 | Multi-template | curriculum batch | `recipe.py`, `train_v3.py`, `gen_slots.py` | **PARTIAL**: trained templates -> 4/4 at 1.000 (import/run/apt went 0/40 -> 40/40); never-seen probes still 0.000 -> M2 relative gate is REQUIRED, confirmed |
| M2 | Relative gate | learned gate w/ gen_margin | `hmn/v3.py` | structure DONE; **finding**: gate learns & doesn't collapse (0.4-0.78 vs v3.1's 0.05) but row-0 subword off-by-one is the real unseen-template blocker -> M3b (see §6) |
| M3b | _new_ Template pooling | train on many verbs, probe held-out verbs | `train_v3.py` | 10 trained verbs -> all 1.000; held-out verb (mount/uninstall/clean/check) still 0.000 at gate 0.001 |
| M4 | Think D1 | wire + validate on multi-step | `hmn/v3.py`, `recipe.py`, `train_v3.py` | **DONE**: new 2-slot chain task (disjoint pkg/lib families); think 0.925 vs no-think 0.900 (300+500 steps, stable); think also halves fails 13->6/60 and removes the empty-output gate-collapse class -> thinking wins on multi-step. Residual: EOS-loop at g~0.96 both modes (decode-level, not think) |
| M4b | _new_ | EOS reliability on chain task | `recipe.py` (decode) | **DONE**: cycle guard (2-token window, decoder-time only). no-think 0.900->0.925, think 0.925->0.950; guardrails bit-identical. Residual model-level fails (not decode): `lib0` subtoken truncate + no-think row-0 gate collapse -> M2-dev |
| M5 | GPU spec | silver/gold + MoE@large | `retop.py` | val good, no crash |
| M6 | Think D2 | latent-slot (if M4 wins) | `hmn/v3.py`, `recipe.py` | beats D1 |
| M7 | GUI v4 | toggles + v4 verify matrix | `retop_gui.py` | think/spec toggle usable |
| M8 | **Baseline** | HMN vs vanilla transformer & HMN3_NoReg, same size, same task | `experiments/` | quantify dual-register edge, honest README |
| M9 | **CI** | GitHub Actions: guardrail + tests on every push | `.github/workflows/` | green on push |
| M10 | Docs v4 | design doc, v4 guardrail, README | `docs/`, `experiments/verified/` | v4 guardrail pass + v3.3 still pass |
| later | Server | streaming + OpenAI-style endpoint | new | — |

Order M1 -> M3 -> M2 -> M4 is "smart before long": sparse marginal is done
first because it is the base of the new IR. Multi-template (M3) runs BEFORE the
relative gate (M2) — the gate generalization needs varied training data first,
otherwise the learned gate trains on a single-template distribution and learns
to be template-bound again (the v3.1 failure mode). M8/M9 (baseline + CI) were
added from external review feedback.

## 4b. External review feedback (2026-08-13) — how it maps

| feedback | disposition |
|---|---|
| scope too narrow (slot-copy only) | M4 multi-step + F code domain |
| gate template-bound | M3 multi-template (before M2) |
| attention-mass tensor doesn't scale | M1 reduced copy-path duplicates; `(B,T,T)` sim identified as the true long-context work — recorded in §6 |
| no CI | **M9 new** |
| no baseline comparison (transformer same size) | **M8 new**: HMN3 vs HMN3_NoReg (in-repo ablation) vs vanilla transformer |
| README claims "softmax cannot" | soften wording after M8 has numbers; note HMN3_NoReg IS the in-repo proof |

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
- **M2/M3b finding (2026-08-13)**: the real unseen-template blocker is NOT the
  gate (the relative gate learns and stays open on probes, unlike deterministic
  gate at 0.001) — it is the **first answer token**: the copy lane is a
  next-token lookup (payload = ids[j+1]), and on the seed row the query
  self-matches the FIRST subtoken of the template verb ('re' in 'remove'),
  emitting the NEXT subtoken ('mo'). So row-0 can only be produced by the gen
  head, and gen is lexically bound to seen verbs no matter how many templates
  are pooled. Options to unblock: (a) make the register addressable by the
  whole matched stem, not just one token id; (b) train the gen head on a
  template-start distribution that is open-vocabulary. Both are M2-dev follow-
  ups; the immediate lever already measured is verb pooling helping every
  SEEN verb but none held out (template-bound copy persists).
- **M4 finding (2026-08-13)**: thinking helps the 2-slot chain task ONLY after
  the two slot families have DISJOINT token id sets. The original
  pkgA/pkgB pairing shared 'p','k','g' + digit subtokens, so the second copy
  chain was a 50% coin-flip *before* any thinking could matter (both modes
  stuck at 0.000). With pkg/lib disjoint families the task separates: think
  0.925 vs no-think 0.900, and think eliminates the no-think empty-output
  failures (gate collapsed to 0.001 at row 0 -> ''). Remaining shared failure:
  EOS-loop — the gate stays ~0.96 after the final token because "and deploy
  lib012" leaves the same-token-id mass intact, so neither mode emits EOS
  inside max_new (decode-plumbing, M4b). Takeaway: multi-step is a REGISTER-
  ADDRESSING problem first; thinking amplifies what the register can already
  bind.
- **M4b finding (2026-08-13)**: the chain EOS-loop root cause is structural:
  after the final true token (e.g. '9' of lib039) the seed's twin points at
  ' and' (start of the template segment) -> copy lane replays ' and deploy
  lib039' forever; gate stays ~0.997 so the old g<0.5 boundary rule never
  fires. Player plot: a 1-token (prev,next) cycle detector is a false
  positive (segments legitimately share tokens like '0'/'9'), but a 2-token
  window is unique per true answer and breaks the loop deterministically.
- **M2-dev design probe (2026-08-13)**: measured row-0 attention on 30
  template/slot combos: the ASI row's attention does NOT look at the template
  start (it peaks on the slot digits, 30/30 mismatch with the payload the
  answer needs). Row-0 copy cannot be enabled by just un-masking loss rows —
  the attention was never trained to anchor on the template prefix. Two viable
  designs: (i) stem-addressing: row-0 query = functional of the template
  prefix `ids[2:a]` (the whole pre-slot region) instead of raw `ids[a]`,
  making the register intrinsically address "the answer's first segment";
  (ii) anchor-copy: force row-0 attention onto the columns whose payload chain
  completes the template prefix, supervised with a dedicated aux CE. (i) is the
  user-chosen path (Stem-addressing). Must stay code-guarded behind a flag so
  v3.3/parity is bit-identical until parity re-check.
- **M2-dev result (2026-08-13)**: stem-addressing implemented and validated
  under `--stem-addr` (`hmn/v3.py`, `hmn/recipe.py`, `train_v3.py`). Anchor the
  ASI row's copy attention onto the USER column -> row-0 payload becomes the
  template's first token (deterministic gate opens, no more forced gen). M3b's
  never-seen probes went from 0.000 to 1.000 for mount/uninstall/clean after
  600 steps — the first template-general row-0 fix. `check` (BPE c,he,c,k)
  still failed: the seed 'c' matched BOTH col2 and col4, tying raw-identity
  attention -> wrong payload. Fixed by extending the anchor beyond row 0 keyed
  by stem POSITION (answer row a+i -> user col u+i); the repeated-subtoken tie
  is resolved because the anchor picks the unique positional column, not the
  ambiguous identity. Result: all 10 trained + 4/4 probes (mount/uninstall/
  clean/check) at 1.000, gate 0.89. Default OFF -> v3.3/parity untouched
  (test_hmn + M1 parity + slot_v33_seed42 all green).
- **M2-dev -> M4 chain (2026-08-13)**: stem-addr applied to the two-slot chain
  task (`fetch {a} and deploy {b}`, no-think). The row-0 gate collapse that kept
  no-think at 0.925 unseen is removed by the anchored row-0 (copy 'fetch'
  instead of a collapsed gen). 600 steps -> unseen blend/hard **1.000**, robust
  across eval seeds 9/11/13/17/21. think was 0.950; stem-addr now matches/exceeds
  it WITHOUT the latent buffer — the anchor is a strictly cheaper fix for the
  row-0 addressing problem (compute scaling buys less than fixing the address).
  Residual fails are model-level (lib0 subtoken truncation comes from the
  same-lookups, no-think row-0 gate collapse), not decoder plumbing.