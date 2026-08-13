# HMN v3.3 — Identity Register slot-copy (final)

> Status: **validated 2026-08-13.** 40/40 unseen exact on the canonical task,
> guardrail: `experiments/verified/slot_v33_seed42.py`. This document is the
> from-the-failure-history account: what broke in v3.0→v3.2, what v3.3 does
> instead, the measured numbers, and the honest boundaries of the mechanism.

---

## 1. The problem that motivated v3

The v2 family (reversible coupling + MoE + episodic memory) recalled key→value
tokens at 97–99% (verified single-token recall), but hit a hard wall on
**slot-copy**:

```
user:   pip install pkg042
answer: pip install pkg042     ← must re-emit the EXACT literal
```

with *unseen* slots (`pkg060..pkg099`, never in training) the softmax head scored
**0/40 exact**. The mechanism failure is structural, not a matter of scale:

> A softmax head is a distribution over a fixed vocabulary. Re-emitting an unseen
> token like the digits `04`/`2` of `pkg042` requires that token's shared
> embeddings to have been pushed by gradient — which never happened for the exact
> digit sequence, because distinct 3-digit suffixes are combinatorially rare.
> The head can only emit a *blurred average* of "some pkg-suffix", never the
> exact literal (measured: 100% wrong on val in the distillation attempt 3).

**v3's thesis:** separate the answer into two channels and let a learned gate
route per token — *copy the prompt's exact tokens*, *generate the rest*.

---

## 2. The architecture that works (v3.3-final)

```
input_ids
   │
   ├─▶ [Embedding] ──────────▶ Identity Register (IR)  "literal lane"
   │                              │
   │   [WR: L× SelectiveSSM]      │
   │         │                    │
   │         ▼                    ▼
   │   ctx state h_t ──▶ gate/select ──▶ copy_dist over {tokens in prompt}
   │
   └─▶ [DualHead]  gen_logits (WR, softmax)  ⊕  copy_dist (IR, p(·|prompt))
                          g = gate(h_t)          final = (1-g)·gen + g·copy
```

- **Working Register (WR):** the v2 backbone. Produces contextual hidden states
  for generation.
- **Identity Register (IR):** raw-embedding lane, *not* contextualized by the
  backbone (that was the v2 finding: contextualization destroys content-address,
  14% → 99% switch). The copy distribution = attention over prompt token *ids*
  (v3.x `HMN3` in `hmn/v3.py`).
- **Pointer / gate supervision (v3.3):** for each answer target, the IR must
  attend to the prompt position whose *next* token equals the target — a
  "what follows the token I just emitted" bigram lookup.
  `make_chat_targets` (in `hmn/recipe.py`) derives, per sequence:
  - `Y` — shifted targets for the blended head, answer region only;
  - `Yc` — copy-channel targets, EOS and the first answer token masked off (the
    register cannot copy either — EOS isn't in the prompt, and the first answer
    token's seed is `<|assistant|>`, also absent);
  - `G` — `1.0` if the target has an exact copyable twin in the prompt, `0.0`
    otherwise. This triple is the whole recipe's input.

### Loss — `loss_v33` (single source of truth)

```
L = CE(blend_logits, Y)                       # final answer correctness
  + w_copy · CE(gen_logits, Y)[G == 0 rows]   # gen owns ONLY generate rows
  + w_copy · mean(−log p_copy(target))        # manual -log p on copy_dist
```

Three traps this encoding was designed to avoid (each was a real bug, §4).

### Decode — `decode_v33`, `boundary_eos`

- `blend`: `argmax((1-g)·gen + g·copy)`.
- `hard`: if `g > 0.5`, `argmax(copy)`, else blend.
- `boundary_eos` (structural rule): in slot-copy every answer token after the
  first has an exact prompt twin ⇒ its gate ≈1. The *only* row where the gate
  can collapse mid-answer is the final one — and that row must emit EOS. So:
  **when the gate drops below threshold after step 0, force EOS.** This turns a
  per-slot lottery (~1–2 fails/seed) into a deterministic boundary.

`hmn/recipe.py` centralizes `make_chat_ids`, `make_chat_targets`,
`make_slot_batch`, `loss_v33`, `decode_v33`, `eval_slots`. `train_v3.py`,
`retop.py`, `infer.py` all call these — the loss math can no longer drift across
files.

---

## 3. Verified numbers

Config: D96/L3, state_dim 8, gate_bias −1.0, `w_copy` 1.0, batch 8, lr 3e-4,
1400 steps, seed 0, CPU (~7 min). Train = `pip install pkg000..059`; eval =
`pkg060..099`, zero overlap (in-process split).

Run `python experiments/verified/slot_v33_seed42.py`.

| Eval (40 unseen) | exact |
|---|---|
| blend + boundary_eos | 40/40 |
| hard (gate>0.5 → copy) | 40/40 |
| structural variant `pip install -r {slot}` (never trained) | 40/40 |
| reproducibility sweep 5 seeds × 40 | 200/200 |

## 4. The failure history (why each choice is a choice)

### 4.1 The learned gate must stay deterministic
Tried (v3.0–v3.1 phase): entropy-based gate, soft top-k gate, gate = max same-
token-id copy mass. All collapsed mid-training — the copy mass for *any* token
can look confident under a soft distribution, so the gate learned to trust the
generate head everywhere instead, and slot-copy silently reverted to softmax
behavior. **Resolution:** the gate is trained with a hard target (`G ∈ {0,1}`)
and combined deterministically with `make_chat_targets`; decode uses the
*structurally-defined* rule, not a "confidence the network invented".

### 4.2 CE-on-probs — the frozen-loss bug (v3.1→v3.3, most expensive)
`copy_dist` is already a probability distribution (sums to 1). Using
`CrossEntropyLoss(copy_dist, Y)` re-log-softmaxes it and pins the loss at
~ln(VOCAB)=7.07 with a dead gradient. This was the cause of every "frozen"
training curve since v3.1. **Resolution:** manual `−log p_target` via gather;
`test_hmn.py::test_recipe_copy_ce_is_manual_logp` is a regression guard (it
fails if anyone reintroduces CE-on-probs).

### 4.3 gen CE must be masked to generate rows
Training the gen head on *copy* rows makes it memorize the slot digits, so at
the boundary row it fires the same copied token instead of EOS (`pkg060→'OST'`,
`pkg066→'6'` loop). **Resolution:** `G != 0` rows are masked out of the gen CE;
the gen head owns exactly the first answer token, the EOS, and absent targets.

### 4.4 The decode boundary is a rule, not a guess
Relying on the gen head to realize "the last copied token is followed by EOS"
per-slot failed ~1–2 slots per seed. **Resolution:** `boundary_eos` (a faithful
encoding of the *task structure*): the register gate provably drops at the final
row, so force EOS there. This removed the last nondeterminism in exact-match.

### 4.5 Special-token / encoding bugs
The tokenizer's BPE *can* split concatenated special tokens
(`"<|assistant|>"` → 3 ids when glued to text). `make_chat_ids` encodes each
part separately and joins ids — the invariant `test_recipe_chat_ids_no_special_split`
keeps every chat special at exactly one id.

## 5. Measured generalization & known limitations

The Identity Register generalizes over slot **values** (unseen 3-digit packages,
alphanumeric `pkgA12`, `xy12`) — but it is **not** yet a general copy operation:

| Probe | result | explanation |
|---|---|---|
| 4-/5-digit slots | ~37/40 | longer copy chains mostly exact; occasionally the pointer leaks |
| alphanumeric slots | 30/30 (`pkgL12` & lowercase `xy12` styles) | register is digit/template-agnostic in mechanics |
| `import/run/apt install {pkg}` templates | **0/40** | the gate is lexicon-bound to the trained template; the model never learned "copy the prompt tail" as a context-free op |
| repeated-digit slots `pkg333..999` | ~3/9 | the copy lane loops (gate stays ≥0.93 copying the same token); `boundary_eos` cannot fire because the gate never drops |

Section 5 is updated from the guardrail — run
`slot_v33_seed42.py` to reprint; the assert-guards cover only the six numbers in
§3, the probes are printed not asserted.

## 6. Limitations & future work

- `(B, T, vocab)` copy-mass tensor per forward: fine for toy lengths, must be
  reworked (marginalize copy over present-token ids) before scaling context.
- Template-general gating (4.5/5) is the single biggest open problem — a
  context-free trigger "my next token is verbatim somewhere in the prompt"
  would make the copy lane a general operation.
- Repeated-token loops (5) need a copy-chain consistency check at decode
  (e.g. abort copy when the target was already emitted and gate stays maximal).
- Latent Thinking Buffer (`HMN3 use_think`, `k_max`) exists in the code but is
  not part of the verified slot-copy result; unvalidated for this task.

## 7. Reproducing

`train_v3.py` builds its slot data in-process (PKGS_SEEN 60 / PKGS_UNSEEN 40,
deterministic per seed) — it does **not** read a `slots.jsonl` file. To get the
canonical run, use the default seed and a fresh save path:

```bash
python train_v3.py --steps 1400 --arch v31 --save my_slots.pt   # ~7 min CPU
python experiments/verified/slot_v33_seed42.py --checkpoint my_slots.pt   # 40/40
python test_hmn.py                                        # recipe guardrails
```

`gen_slots.py` is for writing slot-copy `.jsonl` files consumed by
`retop.py train` and for inspecting the seen/unseen split — not required for
this pipeline.