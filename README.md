# ReTop — Helix Memory Network (HMN)

A small, from-scratch, CPU-trainable language model whose key trick is an
**Identity Register**: a raw-token copy lane gated against a normal softmax
generate head. It breaks the classic single-pass decoder's hard ceiling —
*re-emitting the exact literal of an UNSEEN token from the prompt* — at 100% on
the canonical slot-copy eval, trained in ~15 minutes on a CPU.

```
pip install pkg042   ──▶   pip install pkg042     (40/40 on pkg060..pkg099,
                                                  none seen at train time)
```

Everything runs on a 4-core CPU with ~4 GB RAM. Datasets are streamed and
chunked, never loaded fully into memory.

---

## Why a split-channel architecture?

Standard LLMs are single-pass probabilistic generators: one forward → softmax →
sample. This repo's own experiments verified three hard ceilings of that design
(`legacy/docs/task1_findings.md`, `legacy/docs/task2_findings.md`,
`legacy/docs/distill_design.md`):

| Ceiling (verified in this repo) | Evidence |
|---|---|
| Content-addressing of memory is destroyed by contextualization (SSM/MoE) | raw-embed read 62% vs backbone read 14% → +β30 +usage-decay 99% |
| Slot-copy is impossible from a softmax head | `pip install {pkg}` = 0/40 exact on unseen slots (attempt 3, distill_design) |
| Multi-step "thinking" costs extra output tokens | distillation needed rule+copy+verify but single-pass can't |

**HMN's answer:** two registers instead of one blurred state.

- **Working Register (WR)** — reversible coupling backbone (SelectiveSSM) + MoE.
  Builds contextual abstractions for *reasoning / generation*.
- **Identity Register (IR)** — a raw-token lane, content-addressable by the
  contextual state but payload is the exact token id → hard copy.
- **Dual-Head Decoder** — `gen ⊕ copy`, a learnable gate `g` blends a softmax
  head with a copy distribution over the token ids actually in context.

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

### v2 (earlier, 2026-08-07)

| Task | Result | Source |
|---|---|---|
| Recall, single-token, 8 pairs/50 keys | 97–99% | `legacy/docs/task2_findings.md` |
| Recall, 2-token values, multi-head | 94–97% | `legacy/docs/task2_findings.md` |
| Distill template curriculum (unseen slots) | syntax 70–91%, API 89–95% | `legacy/docs/distill_design.md` |

---

## Project layout

```
hmn/                        core package
  __init__.py               HMN, HMN_Option1, HMN3, HMN3_NoReg
  v2.py                     SelectiveSSM, coupling, MoE, episodic memory
  v3.py                     IdentityRegister, DualHeadDecoder, HMN3
  recipe.py                 v3.3 recipe: make_chat_ids, loss_v33, decode_v33,
                            make_slot_batch, eval_slots   ← single source of truth
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
legacy/                     v1/v2 research: findings docs, old generators,
                            the switch experiments (kept for history)
retop_tokenizer.json        the tokenizer (vocab 3190)
```

---

## Requirements & install

- Python 3.9+ (tested 3.12), CPU-only PyTorch, `tokenizers`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install tokenizers
pip install -e .            # optional: import hmn from anywhere
```

## Quickstart

### 1. Verify the shipped checkpoint (guardrail, ~1 min CPU)

```bash
python experiments/verified/slot_v33_seed42.py
#   [PASS] unseen blend …  1.000
#   [PASS] unseen hard  …  1.000
#   [PASS] structural   …  1.000
#   ALL GUARDS PASSED
```

### 2. Train your own slot-copy model

```bash
python gen_slots.py --out data/slots.jsonl --n-seen 600 --n-unseen 400 --seed 0
python train_v3.py --steps 1400 --arch v31 --save hmn_v33.pt    # ~15 min CPU
```

### 3. Train via retop (auto config + tokenizer)

```bash
python retop.py tok --data my_corpus.txt --out tok.json --vocab 4000
python retop.py train --data my_chat.jsonl --tok tok.json --out myai.pt
python retop.py chat --checkpoint myai.pt --interactive
```

Data formats (slot-copy, chat pairs, plain text) are documented in
`docs/data_prep.md`.

### 4. Chat with a checkpoint

```bash
python infer.py --checkpoint hmn_v33.pt --arch v3 --dim 96 --layers 3 \
                --gate-bias -1.0 --prompt "pip install pkg042"
python infer.py --checkpoint hmn_v33.pt --arch v3 --dim 96 --layers 3 \
                --gate-bias -1.0 --prompt "pip install pkg042"
python infer.py --checkpoint hmn_v33.pt --arch v3 --dim 96 --layers 3 \
                --gate-bias -1.0 --interactive
```

> Checkpoints are `model.state_dict()`; `infer.py` prints a clear error if the
> `--arch` / hyperparameters don't match.

### 5. Web UI (Gradio)

Four tabs: **DATA** (generate/upload slot data), **TRAIN** (live-loss console,
stop button, SVG chart), **CHAT** (load checkpoint + decode), **VERIFY**
(guardrail report). Training runs as a subprocess that streams stdout, so the
recipe never drifts from the CLI.

```bash
pip install "gradio>=4"
retop-gui                          # opens http://127.0.0.1:7860
```

### 6. Tests

```bash
python test_hmn.py        # model + recipe guardrails: copy CE is -log p,
                          # no CE-on-probs, custom loss_v33 decode, chat ids
```

---

## Architecture notes

- **Reversible coupling** (`ReversibleFunction`) recomputes activations during
  backward instead of storing them — memory-efficient by design.
- **The v3.3 recipe** (`hmn/recipe.py`) is the single source of truth for all
  entry points. Historically the same loss math drifted across three files and
  the copy-CE bug — feeding a *probability distribution* into `CrossEntropyLoss`,
  which log-softmaxes it again and pins the loss at ~ln(VOCAB) — kept returning.
  `loss_v33` uses a manual `-log p_target` (guarded by a regression test).
- **`boundary_eos`** is a structural decode rule: in slot-copy, every answer
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

## Reproducibility

Every generator and trainer takes a `--seed`; slot data, splits, and decodes are
deterministic (greedy). The seed-42 eval guardrail
(`experiments/verified/slot_v33_seed42.py`) either passes 40/40 or fails loudly.

## License

Apache-2.0 (applied via `pyproject.toml`).