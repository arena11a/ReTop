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
| M4b | _new_ | EOS reliability on chain task | `recipe.py` (decode) | **DONE**: cycle guard (2-token window, decoder-time only). no-think 0.900->0.925, think 0.925->0.950; guardrails bit-identical. Residual model-level fails (not decode): `lib0` subtoken truncate + no-think row-0 gate collapse -> M2-dev. BOTH closed by M2-dev stem-addr (row-0) + M6 pos_eos (termination): chain hard 0.948 -> 1.000 on 5 seeds |
| M5 | GPU spec | silver/gold + MoE@large | `retop.py` | val good, no crash |
| M6 | Think D2 | latent-slot (if M4 wins) | `hmn/v3.py`, `recipe.py` | beats D1 |
| M7 | GUI v4 | toggles + v4 verify matrix | `retop_gui.py` | think/spec toggle usable |
| M8 | **Baseline** | HMN vs vanilla transformer & HMN3_NoReg, same size, same task | `experiments/v4/m8_baseline.py` | **DONE (2026-08-13)**: HMN3 664K = 1.000/1.000 (trained/probe); vanilla 667K = 0.000 (fits SEEN slots 1.0 by rote, memorizes unseen ids pkg099->pkg049); HMN3_NoReg 342K = 0.000. Copy pointer is required — softmax cannot re-emit unseen prompt tokens |
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
- **M8 finding (2026-08-13)**: same-size baseline closes the loop on the
  original "softmax cannot" claim with real numbers. vanilla transformer
  (667K, same dim/layers as HMN3 664K, tied embed, pre-LN, greedy argmax): hits
  1.0 on SEEN slots but 0.0 on UNSEEN slots and 0.0 on every probe template —
  it rote-fits the training pairs (unseen pkg099 was decoded as memorized
  pkg049). HMN3_NoReg (342K, register removed) = 0.0 everywhere. HMN3 +
  stem-addr = 1.0 on trained AND never-seen probes. The copy lane is not an
  optimization nicety: it is the only component that can re-emit an unseen
  prompt token.
- **M6 finding (2026-08-13)**: after stem-addr, think adds ~0 on the tasks it
  was built for (chain: stem-only hard 0.948 == stem+think 0.948; blend both
  1.000). The latent buffer was compensating for row-0 addressing, which the
  deterministic anchor now does cheaper. M6 is NOT 'think v2' — the correct
  turn is to solve the residual the anchor does NOT cover. Two closed:
  (a) repeated-subtoken slot (pkg333): content was already correct (anchor)
  but decode never knew WHEN the answer ends — gate stays ~0.93 (seed '3'
  still has a twin) so boundary_eos cannot fire. (b) chain long-EOS loop, same
  shape. Fix = pos_eos: in the echo task len(answer) == len(user tokens) ==
  len(prompt)-3 is known a priori, so the decoder forces EOS exactly there.
  Result: repeated-digit 0.333 -> **1.000** (hard+blend), chain hard 0.948 ->
  **1.000** on all seeds. Deterministic decoder-time rule, default OFF, same
  family as boundary_eos/cycle_break. design principle: the pointer fixes
  CONTENT, the length-bound fixes TERMINATION. README can now state this with evidence.
- **M6 chain on UNSEEN FORMATS (2026-08-13)**: anchor+pos_eos is format-agnostic
  — chain with 4-digit pkgs, alnum libs, and repeated-digit slots (pkg99999)
  all decode 1.000 (hard+blend) without retraining. Along the way found and
  fixed a REAL decode bug: `cycle_break` false-positives on repeated identical
  output tokens (prev==cand, e.g. the '9' run in pkg99999) — the (prev,next)
  window that is "unique per true answer" breaks when the answer itself reuses
  a token consecutively. Fix: ignore self-pairs (prev==cand) in the cycle
  detector; the true EOS-loop replays a multi-token segment ('and deploy
  lib039…') whose transitions are non-self, so M4b detection is preserved.
  Verified: 5-seed chain + repeated-digit chain + slot_v4 all 1.000.
  Residual fails are model-level (lib0 subtoken truncation comes from the
  same-lookups, no-think row-0 gate collapse), not decoder plumbing.
- **M6 honest boundary (2026-08-13)**: anchor + pos_eos assume an ECHO task
  (user == gold). Quantified the wall: a no-echo transform (user = "fetch pkg028
  and deploy lib055", gold = reordered "deploy lib055 and fetch pkg028") fails
  hard — gate 0.003, empty output — because (a) the identity lane is a
  one-token-next-copy (no reorder), (b) anchor forces the positional echo, and
  (c) pos_eos length still matches but the content never appears. This is a
  DESIGN boundary, not a bug: the register copies what the prompt says, not a
  computed transform. A reorder/transform task needs the gen head to assemble
  copied pieces (pointer + gen compose), which the current gate never learns
  because every v4 task is echo. Stated in README as the honest limit; remains
  open (M12 candidate).
- **M12 probe (2026-08-14, experiments/v4/m12_reorder_probe.py)**: trained the
  swap task 240 steps x2 (bs16) to test whether gen+pointer can COMPOSE a
  reorder when every gold token exists verbatim in the prompt. stem-addr OFF:
  loss 4.29->4.19, unseen 0.000 (deterministic, never learns). stem-addr ON:
  loss 16.3 flat, unseen 0.000 — the anchor actively forces the USER's first
  token ("fetch") as answer row 0, wrong for the swap. CONCLUSION: the wall is
  not row-0 seeding — it is that sparse identity attention cannot FLIP two
  attention sinks (swap two copied fragments). Row-0 gen is necessary but not
  sufficient; M12 requires a compositional mechanism the register lane lacks.
  Reorder/transform remains OPEN.
- **M12 probe, long-horizon confirmation (2026-08-14,
  experiments/v4/m12_reorder_long.py)**: 1200 steps stem-addr OFF. loss
  4.33->3.97 (echo chain hits ~0.003; the plateau is decisive), unseen_acc
  0.000 at every 200-step checkpoint. => the reorder wall is STRUCTURAL, not a
  training-horizon artifact. Root cause narrowed: the gen head cannot SEED a
  content token at row 0 when the first gold token has no identity anchor —
  echo tasks never require content-initiation (row 0 is ASI-anchored copy or
  EOS), so the gen lane has no mechanism for it (can only emit EOS / continue
  a copy in blend). M12 needs a SEEDED POINTER mechanism: let the gen head
  initiate a copy into the register, i.e. a pointer register rather than a
  next-token identity only.
- **M12b read-only gate decomposition (2026-08-14,
  experiments/v4/m12b_substring_chain.py)**: with the CORRECT continuation fed
  forward on the trained chain ckpt, the copy lane re-emits ~70% of reordered
  rows EXACTLY (11/16, 11/18) at gate 0.997 — so the "flip two attention
  sinks" framing in M12 was TOO PESSIMISTIC: reordered contiguous text IS
  copyable once a row starts. The wall decomposes into two micro-mechanisms:
  (a) row 0 content-initiation (gate 0.000 refuses to open for the reordered
  head; gen has no content seed) and (b) fragment seams — per-prev-token
  gate closes (0.003) or points wrong exactly at ' and' (the seam between
  swapped fragments), then re-opens past it. So M12's concrete mechanism is
  not "composition" but an explicit RESTART: the anchor needs a
  positional/structural way to start a copy at fragment n (seeded pointer),
  and the gate needs the counterpart. Well-scoped; remains OPEN.
- **M12c marker probe (2026-08-14, experiments/v4/m12c_marker_probe.py)**:
  added an explicit "swap:" structural marker to the reorder template so the
  model gets a loud signal that the answer is reordered. 400 steps, seed-42:
  loss 3.99->3.80, unseen 0.000 — statistically IDENTICAL to M12 without the
  marker (3.97 plateau at 1200 steps). => the row-0 content-initiation wall is
  in GEN LANE CAPACITY, not the conditioning signal; template/marker diversity
  on the SAME architecture cannot unlock it. Closes the "signal conditioning"
  branch of M12 — addressing a reorder head requires a MECHANISM change
  (a gate-openable seeded pointer), not louder input features.
- **M6 chain length generalization (2026-08-14)**: existing 2-slot stem-addr
  ckpt, zero retraining, on UNSEEN 3-slot chains (fetch {a} and deploy {b} and
  stop {c}, both seen + unseen ids): pos_eos=True -> 1.000 (30/30), pos_eos=False
  -> 0.833. The failures are the length-boundary recursion (out keeps
  re-emitting "and deploy {b} and..." past the answer), i.e. exactly the case
  pos_eos was built for; note the subtoken leak in one failure
  ("stop lib03021" — '21' continued from lib021). Anchor+pos_eos compose to
  arbitrary template length (T-dependent), so slot-chain capacity is length
  GENERALIZING, not just template-generalizing. The anchor's c = u + (t-a) is
  T-invariant by construction.
- **M6 length generalization, stress (2026-08-14, follow-up)**: same ckpt on
  3- and 4-slot chains with an UNSEEN verb ("open") and unseen id families
  (bin{...}). pos_eos=True -> 1.000 for BOTH lengths, in hard AND blend modes
  (3-slot blend pe=False 0.700, 4-slot blend pe=False 0.850, all failures the
  same post-answer recursion). So capacity is length-generalizing up to at
  least 4 slots and robust to unseen verbs/families — the recursion failure
  without pos_eos is purely a termination signal, never a content error.
- **M6 chain fully closed (2026-08-14)**: pos_eos=True lifts the last chain
  gap — the 0.950/0.948 hard/blend on the standard 2-slot benchmark — to 1.000
  for BOTH the think and nothink checkpoints (seed 9, unseen a/b pools, both
  modes). Combined with the length/generalization findings above, the chain
  task is now 1.000 everywhere pos_eos is on; the think/no-think difference
  that M4b documented is fully subsumed by the decoder-side termination fix.
  avg_gate 0.948 unchanged (content copy is correct; only termination was off).
- **M2-dev slot completeness under blend (2026-08-14)**: the v4 slot task was
  validated in hard mode (slot_v4 guardrail, 18/18). Re-checked the DEPLOYMENT
  path (blend mode, the only mode that uses gen tokens) across the full set:
  10 trained templates 1.000, 4 probes (unseen template shapes) 1.000, 4-row
  generalization matrix (incl. pkg12345/pkgg99999 repeated digits,
  version42 alnum, pkg006600 double-pad) 1.000 — all under pos_eos=True. So
  slot is now 1.000 in hard AND blend, trained AND probe, matrix AND chain.