# ReTop v5 — omega-seam: from echo to transform (roadmap)

> Status: **M1 PASSED (2026-08-22).** The reorder/transform wall of v4 (M12,
> open since 2026-08-14) is closed by seam re-seeding: unseen reorder went
> **0.000 → 1.000**. All v5 work is additive behind `HMN3(seam_addr=True)` —
> default OFF keeps every v3.3/v4 guardrail bit-identical.

## 1. Direction

Maps the "Ω-Coder / AWC" architecture vision onto this repo's measured
culture (small generator + strong verifier + executable skill memory):

| Ω-Coder module | ReTop realization | Status |
|---|---|---|
| Tiny generator | HMN3 (~675K params, CPU-trainable) | shipped |
| Verifier-in-the-loop | decode rules (`boundary_eos`/`pos_eos`/`cycle_break`) + guardrails + CI | shipped |
| Compiled skill (anchor) | stem-addr `c = u + (t−a)` | shipped (v4 M2-dev) |
| **Fragment-run composition** | **SeedPointer + run echo (`seam_addr`)** | **v5 M1** |
| Skill library + retrieval | generalize anchors into reusable skills | open (M2) |

Design principle that fell out of M1: the register fixes CONTENT within a
run; the SeedPointer fixes RUN BOUNDARIES (which column a fragment starts
from, how long it lasts). Echo = one run; reorder = many runs.

## 2. Mechanism (v5 M1)

- `hmn/v3.py::SeedPointer` — at a seam row predicts (a) the prompt column the
  next copy-run's payload chain starts from, (b) the run length.
  Supervised by ptr CE + len CE against gold structure (`recipe.seam_losses`).
- `IdentityRegister._attn(seam_anchor=...)` — forced positional echo per row:
  anchor col c advances +1 inside a run (generalized stem-addr); forced rows
  open the deterministic gate (`mass_same = 1.0`).
- `make_reorder_ids` — the swapped gold is assembled FROM prompt token ids
  (`Gt = U[i_u+1:] + [U[i_u]] + U[:i_u]`) so every answer row has an EXACT
  identity twin (ByteLevel space-prefix variants solved at the data layer).
- `decode_v33(seam=True)` — greedy run machine: emit forced payload while the
  run lasts; on exhaustion read SeedPointer argmax for the next run;
  terminate via `pos_eos` (a permutation preserves |answer| == |user|).

Honest boundary kept: gold text is the canonical decoding of the swapped
token sequence — mid-answer run starts reuse the prompt's byte-level token
variants, so one boundary space differs from naive string reordering
(`"… and fetch …"` decodes `"… andfetch …"`). Exact-match compares against
the canonical decoding. Mechanical content is exact.

## 3. Results (2026-08-22)

Task: user `fetch {a} and deploy {b}` → gold `deploy {b} and fetch {a}`;
train pkg000–039 / lib040–079, eval unseen pkg060–079 / lib000–019
(disjoint families, same split as m12 probes). D96/L3/gate_bias −1.0,
bs16, lr1e-3, CPU.

| Milestone | Config | Result |
|---|---|---|
| M12 wall (all prior configs, 240–1200 steps ×4 probes) | no mechanism | **0.000** everywhere |
| M0 control (this script, blend-CE only, 240 steps) | seam OFF | seen 0.000, unseen **0.000** — wall reproduced |
| M1c hypothesis test (600 steps) | seam ON | seen **1.000**, unseen **1.000** (gate ~0.90), stable step 100→600; ptr CE 0.01, len CE 0.00 |
| M1d robustness sweep | seeds 42/43/44 | unseen reorder **1.000 / 1.000 / 0.950** (mean 0.983; worst case 19/20) |
| Flag-OFF parity (no-regression guarantee) | `test_hmn.py` + `slot_v33_seed42.py` | ALL TESTS + ALL GUARDS PASSED (bit-identical) |

Pass criterion met: unseen ≥ 0.60 (actual 1.00).

## 4. Files

- `hmn/v3.py`: `SeedPointer`, `_attn(seam_anchor)`, `HMN3(seam_addr, max_run)`
- `hmn/recipe.py`: `make_reorder_ids`, `reorder_anchors`, `make_reorder_batch`,
  `seam_losses`, `eval_reorders`, `decode_v33(seam=True)`
- `experiments/v5/omega_seam.py`: M0/M1 pipeline (baseline, train, sweep)
- `test_hmn.py`: parity (anchor=None bit-identical), anchor-payload match,
  decode mechanics

## 5. M2 — joint multi-task (2026-08-22, PASSED)

One model, two tasks, two anchor machineries coexisting:
echo chain via stem-addr (no anchors), reorder via seam anchors. The
stem-addr block now skips when a `seam_anchor` tensor is present (otherwise
it fired on reorder sequences and corrupted their non-echo rows).

`python experiments/v5/omega_seam.py --task joint --steps 800 --save omega_joint.pt`

| Task | unseen result |
|---|---|
| Echo chain (`fetch {a} and deploy {b}` → same, pos_eos) | **1.000** |
| Reorder (swapped gold) seen / unseen | **1.000 / 1.000** |

Both at 1.00 from step 100 through 800; guardrails still green.

## 6. M3 — run generalization (2026-08-22, PASSED with a lesson)

Generalized the machinery to N-segment rotations (`make_perm_ids` /
`perm_anchors` / `make_perm_batch`: user "p1 and p2 ... and pn" → gold
rotate-left; anchors derived structurally for any N).

**Stage 1 — zero-shot MISS.** The single-template M1/M2 ckpt scored 0/20 on
every probe (unseen verbs/families, 3-seg rotation, repeated digits): a
SeedPointer trained on ONE template is lexicon/geometry-bound — the v4-M3
lesson repeating at the pointer level.

**Stage 2 — curriculum (joint3) + measured wall.** Mixed verb/family pools +
2- and 3-seg batches (1200+2400 steps). Echo chain 1.000, swap 1.000 held;
teacher-forced p3 answer lane is EXACT (copy-argmax==gold 1.000, gate 0.996)
but per-seam ptr-hit plateaus at **0.55→0.70 over 2400 steps** — the learned
per-seam seeder does not converge cross-template.

**Stage 3 — structural cycling closes it.** Third instance of the repo's
core lesson (learned guess loses to structural rule, after boundary_eos/M6):
`decode_rotate` lets SeedPointer seed RUN 0 only; segment boundaries are
parsed from the prompt's separator token and walked cyclically. Result
(omega_cur2.pt, zero retraining after stage 2):

| Probe | result |
|---|---|
| P1 swap, seen shape, unseen slots (neural seam decode) | **20/20** |
| P2 HELD-OUT verbs open/close + held-out family rel | **20/20** |
| P3 3-seg rotation mixed verbs | **20/20** |
| P4 4-seg rotation — beyond max trained N=3 | **20/20** |
| P5 repeated-digit slots | **20/20** |

Guardrails after all changes: `test_hmn.py` ALL TESTS PASSED,
`slot_v33_seed42.py` ALL GUARDS PASSED (flag-OFF bit-identical).

Scope note (honest): rotate decoding declares the ROTATION family at decode
time (as v4 evals declared templates); fully-neural arbitrary-permutation
seeding stays open.

## 7. M4 — executable skill library (2026-08-22, PASSED)

`hmn/skills.py`: the Ω-Coder "Skill Distiller" as runnable code.

- **Skill** = a run RECIPE (echo / rotate) — knowledge outside weights.
- **Distill** = draw one fresh instance, COVERAGE-gate (plan length must
  equal gold length — caught a real off-by-one: echo's sum(seg_lens)
  excluded separators and false-greened the old verifier), then
  teacher-force verify (copy-argmax==gold AND gates open on every planned
  row) BEFORE storing. Evidence before acceptance.
- **Retrieval** = slot-invariant fingerprint (n_parts, per-segment first
  token ids). Shared verbs across families (echo vs swap both fetch/deploy)
  → ambiguous status ESCALATES instead of guessing; caller passes family
  hint (mission-board constraint analog).
- **Execute** = greedy seam walk of the retrieved plan on unseen slots;
  unknown fingerprints fall back to rotate(k0=ptr), reported in meta.

Result (`experiments/v5/m4_skills.py`, omega_cur2.pt): distill 4/4 verified,
execution **10/10 on every trained family via library hits** (echo, swap,
rot3, rot4-len-gen) and **10/10 held-out verbs+family via fallback**
(rotate generalizes). Ambiguity guard demonstrably escalates.
Guardrails after all changes: ALL TESTS + ALL GUARDS PASSED.

## 8. Open items

- Arbitrary permutations without declaring family AND without shared-verb
  hints: needs a converging per-seam seeder (neural ptr3 plateaus ~0.7,
  §6 stage 2) or richer fingerprints.
- Lift neural ptr3 > 0.7 (capacity vs signal question, open).
- Skill VERSIONING + success-rate tracking per entry (Ω-Coder Module 7
  full form).
