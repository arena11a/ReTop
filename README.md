# ReTop — Helix Memory Network (HMN)

**ReTop** is a small, from-scratch, CPU-trainable language model built on the
**Helix Memory Network (HMN)** architecture: a dual-register decoder that keeps
two lanes separate — one that *thinks* and one that *copies verbatim*.

Its headline result: re-emitting the **exact literal of an UNSEEN token** from
the prompt, at **40/40** on the canonical slot-copy eval — something a
single-pass softmax decoder structurally cannot do — trained in ~15 minutes on a
4-core CPU with ~4 GB RAM.

```
pip install pkg042   ──▶   pip install pkg042     (40/40 on pkg060..pkg099,
                                                  none seen at train time)
```

Datasets are streamed and chunked; nothing is loaded fully into memory.

---

## The architecture in one picture

Standard LLMs are single-pass probabilistic generators: one forward pass →
softmax → sample. A softmax head outputs a *distribution over the vocabulary*,
so it can only emit a blurred average of "some token like the one I saw" — it
can never re-emit a token exactly if that token never appeared in training.

**HMN's answer is two registers instead of one blurred state:**

```
input ──▶ [Working Register]  ─ contextual state h_t ─┐
         │  (reversible SelectiveSSM + MoE, thinks)   │
         └─▶ [Identity Register] ─ raw-token copy ────┤
              (literal lane, content-addressable,     │
               NOT blurred by contextualization)      │
                                                      ▼
         [Dual-Head Decoder]  final = (1−g)·gen ⊕ g·copy
                              g = gate(h_t)  (learned, {0,1} target)
```

- **Working Register (WR)** — the contextual backbone that builds
  abstractions for *reasoning / generation*.
- **Identity Register (IR)** — a raw token-id lane. It stores the prompt's
  exact tokens and yields a *copy distribution* over them, deliberately kept out
  of the contextualization path (contextualization destroys content-address:
  raw-embed read 99% vs backbone read 14% was the measured switch).
- **Dual-Head Decoder + gate** — per output token a learned gate `g` blends the
  softmax *generate* head with the *copy* head. Copy rows are supervised with a
  hard `{0,1}` target (a probabilistic gate collapsed during training — the
  v3.0–v3.1 failure history).

The full design, motivation, failure history and boundaries live in
[`docs/hmn_v3_design.md`](docs/hmn_v3_design.md).

---

## Verified results

### v3.3 — slot-copy (2026-08-13, `experiments/verified/slot_v33_seed42.py`)

Train `pkg000..pkg059` (60 seen), eval `pkg060..pkg099` (40 unseen).
Config: D96/L3, gate_bias −1.0, 1400 steps, batch 8, lr 3e-4, CPU.

| Eval | Exact |
|---|---|
| Unseen slots, blend decode + boundary rule | **40/40** |
| Unseen slots, hard decode (gate>0.5 → copy argmax) | **40/40** |
| Structural variant `pip install -r {pkg}` (never trained) | **40/40** |
| 5 seeds × 40 unseen (reproducibility sweep) | **200/200** |

Generalization — measured, honest boundaries (§5 of `docs/hmn_v3_design.md`):

| Probe | Result | Meaning |
|---|---|---|
| 4- & 5-digit slots (`pkg9000..`) | 37/40 | longer chains mostly copy, occasionally leak |
| Alphanumeric slots (`pkgA12`, `xy12`) | 30/30 | register is not digit-specific |
| Templates `import / run / apt install` | 0/40 | gate is lexicon-bound to the trained template — NOT a general "copy prompt tail" op |
| Repeated-digit slots (`pkg333..pkg999`) | 3/9 | copy lane loops (gate stays ≥0.93), boundary rule can't fire |

---

## Quickstart — running in ~5 minutes

Requires Python 3.9+ (tested 3.12), CPU-only PyTorch, `tokenizers`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install tokenizers
pip install -e .            # optional: import hmn from anywhere
```

### 1. Verify the shipped checkpoint (guardrail, ~1 min CPU)

```bash
python experiments/verified/slot_v33_seed42.py
#   [PASS] unseen blend 'pip install {slot}': 1.000 (40/40, gate 0.775)
#   [PASS] unseen hard (gate>0.5 -> copy argmax): 1.000
#   [PASS] structural variant 'pip install -r {slot}': 1.000
#   ALL GUARDS PASSED
```

### 2. Train your own slot-copy model

```bash
python gen_slots.py --out data/slots.jsonl --n-seen 600 --n-unseen 400 --seed 0
python train_v3.py --steps 1400 --arch v31 --save hmn_v33.pt    # ~15 min CPU
```

### 3. Chat with a checkpoint

```bash
python infer.py --checkpoint hmn_v33.pt --arch v3 --dim 96 --layers 3 \
                --gate-bias -1.0 --prompt "pip install pkg042"
python infer.py --checkpoint hmn_v33.pt --arch v3 --dim 96 --layers 3 \
                --gate-bias -1.0 --interactive
```

> Checkpoints are `model.state_dict()`; `infer.py` prints a clear error if the
> `--arch` / hyperparameters don't match.

### 4. Web UI (Gradio)

Four tabs: **DATA** (generate/upload slot data), **TRAIN** (live-loss console,
stop button, SVG chart), **CHAT** (load checkpoint + decode), **VERIFY**
(guardrail report). Training streams stdout as a subprocess, so the recipe never
drifts from the CLI.

```bash
pip install "gradio>=4"
retop-gui                          # opens http://127.0.0.1:7860
```

---

## Training on your own data

### Generate bilingual-free corpora with `gen_chat.py`

`gen_chat.py` streams combinatorial chat-pairs up to a target token budget
(~4 chars/token heuristic), deterministic per `--seed`:

```bash
python gen_chat.py --domain english       --target-tokens 10_000_000 --out data/english_10m.jsonl
python gen_chat.py --domain math_general  --target-tokens 30_000_000 --out data/math_general_30m.jsonl
python gen_chat.py --domain math_complex  --target-tokens 60_000_000 --out data/math_complex_60m.jsonl
python gen_chat.py --domain math_advanced --target-tokens 120_000_000 --out data/math_advanced_120m.jsonl
```

| domain | token budget | content |
|---|---|---|
| `english` | 10M / 50M / 100M | general knowledge, trivia, how-to dialogue |
| `math_general` | 30M | arithmetic, distance, sharing, %, rectangles |
| `math_complex` | 60M | fractions, ratios, linear equations, triangle area |
| `math_advanced` | 120M | powers, sqrt/cuberoot, circles, quadratics, compound % |

> The `data/` directory is git-ignored — regenerate it yourself with the
> commands above (each corpus is seed-deterministic). Math answers are computed
> in the generator, so the corpora are internally consistent.

### Train & chat via `retop.py` (auto config)

```bash
python retop.py tok --data my_corpus.txt --out tok.json --vocab 4000
python retop.py train --data data/english_10m.jsonl --tok tok.json --out myai.pt
python retop.py chat --checkpoint myai.pt --interactive
```

`retop.py` auto-detects the machine (CPU/RAM/CUDA) and picks a verified spec.
Data formats (slot-copy, chat pairs, plain text) are documented in
[`docs/data_prep.md`](docs/data_prep.md).

---

## Project layout

```
hmn/                        core package
  __init__.py               HMN, HMN_Option1, HMN3, HMN3_NoReg
  v2.py                     SelectiveSSM, coupling, MoE, episodic memory
  v3.py                     IdentityRegister, DualHeadDecoder, HMN3
  recipe.py                 v3.3 recipe: make_chat_ids, loss_v33, decode_v33,
                            make_slot_batch, eval_slots  ← single source of truth
retop.py                    one-command tok / train / chat (writes config sidecar)
train_v3.py                 v3 slot-copy trainer (uses hmn/recipe)
gen_slots.py                slot-copy dataset generator (deterministic seen/unseen)
gen_chat.py                 streaming EN/Math chat corpus generator (data/)
infer.py                    load checkpoint + tokenizer, generate
retop_gui.py                4-tab Gradio UI (DATA/TRAIN/CHAT/VERIFY)
test_hmn.py                 model tests incl. recipe guardrails
hmn_v33.pt[.json]           verified checkpoint + recipe sidecar
experiments/verified/slot_v33_seed42.py   one-command 40/40 guardrail
docs/                       design + data-prep (hmn_v3_design.md, data_prep.md)
data/                       training corpora (git-ignored, from gen_*.py)
retop_tokenizer.json        the tokenizer (vocab 3190)
```

---

## Architecture notes

- **Reversible coupling** (`ReversibleFunction`) recomputes activations during
  backward instead of storing them — memory-efficient by design, which is why
  everything runs in ~4 GB RAM.
- **The v3.3 recipe** (`hmn/recipe.py`) is the single source of truth for all
  entry points. Historically the same loss math drifted across three files and
  the copy-CE bug — feeding a *probability distribution* into
  `CrossEntropyLoss`, which log-softmaxes it again and pins the loss at
  ~ln(VOCAB) — kept returning. `loss_v33` uses a manual `-log p_target`
  (guarded by a regression test).
- **`boundary_eos`** is a structural decode rule: in slot-copy every answer
  token after the first has an exact twin in the prompt (gate ≈1); the only row
  where the gate can collapse is the final one, which *must* emit EOS. Forcing
  EOS on a low gate makes the boundary deterministic (removed the ~1-2 fails/seed
  lottery).

### Known issues / limitations
- `HMN3` builds a `(B, T, vocab)` attention-mass tensor per forward — fine for
  the toy sequence lengths, don't scale to long contexts without reworking the
  copy-marginal.
- Copy-gate generalizes over slot *values* but not over *templates* (§2 of the
  design doc) — the Identity Register is not yet a general copy operation.
- Repeated-identical tokens inside a slot let the copy lane loop (gate stays
  high); the boundary rule cannot fire. See `docs/hmn_v3_design.md` §5.

---

## Reproducibility

Every generator and trainer takes a `--seed`; slot data, splits, and decodes are
deterministic (greedy). The seed-42 eval guardrail
(`experiments/verified/slot_v33_seed42.py`) either passes 40/40 or fails loudly.

## Tests

```bash
python test_hmn.py        # model + recipe guardrails: copy CE is -log p,
                          # no CE-on-probs, custom loss_v33 decode, chat ids
```

## License

Apache-2.0 — see [`LICENSE`](LICENSE).