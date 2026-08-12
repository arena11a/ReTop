# ReTop — Helix Memory Network (HMN)

A small, from-scratch, CPU-trainable language model architecture designed to break
the ceilings of a vanilla single-pass softmax decoder: **hard slot-copy**,
**content-addressable recall**, and **adaptive latent thinking** — without needing
a GPU or a big machine to train.

Everything here runs on a 4-core CPU with ~4 GB RAM. The datasets are streamed and
chunked, so generation never loads the full dataset into memory.

---

## Why a new architecture?

Standard LLMs are single-pass probabilistic generators: read context → one forward →
softmax → sample. This project's own experiments (in `docs/`) verified three hard
ceilings of that design:

| Ceiling (verified in this repo) | Evidence |
|---|---|
| Contextualization (SSM/MoE) destroys content-address of memory | `docs/task2_findings.md` #3 |
| Slot-copy is impossible — a softmax head cannot re-emit a token from the prompt exactly | `distill_design.md` attempt 3: `pip install {pkg}` 0/40 exact on unseen slots |
| "Thinking" costs extra text tokens (chain-of-thought) | distillation needed "rule+copy+verify" but single-pass can't |

**HMN's answer:** split the job into channels.

- **Working Register (WR)** — reversible coupling backbone (SelectiveSSM) + product-key
  MoE. Builds contextual abstractions for *reasoning / generation*.
- **Identity Register (IR, v3)** — a raw-token lane. Content-addressable by the *contextual*
  state, but payload is the *exact* token id → hard copy.
- **Dual-Head Decoder (v3)** — `gen ⊕ copy`, a learnable gate `g` blends a softmax head
  with a copy distribution over the token set actually present in context.
- **Latent Thinking Buffer (v3)** — re-run the backbone in latent space K times, stopping
  when confidence converges; compute scales with difficulty, zero extra vocab cost.

---

## Verified results (v2, 2026-08-07, random-query eval)

| Task | Result | Config | Source |
|---|---|---|---|
| Recall, single-token, 8 pairs/50 keys | **97–99%** | D64/L2/256c gate-blend, β30, usage-decay | `docs/task2_findings.md` #10 |
| Recall, 2-token values 0–99 (multi-head) | **94–97%** | same + combined write, off-by-one fixed | `docs/task2_findings.md` #11–12 |
| Distill template curriculum (unseen slots) | syntax 70–91%, API 89–95%, shell-exact 72% | D128/L6, 2000 steps | `docs/distill_design.md` |

> **v3 is at PoC stage.** The dual-head register is designed to solve the 0/40
> slot-copy ceiling, but has not yet been validated end-to-end. See
> `docs/hmn_v3_design.md` (open questions) — this is the active research line.

The numbers above were all earned the hard way: failures, root causes, attribution
tests, and a seed-42 guardrail are documented in `docs/task1_findings.md` and
`docs/task2_findings.md`. Claims in the older design doc that could **not** be
reproduced are explicitly marked `UNVERIFIED` there.

---

## Project layout

```
hmn/                     public package (importable, installable)
  __init__.py            HMN, HMN_Option1, HMN3, HMN3_NoReg, blocks
  v2.py                  SelectiveSSM, coupling, MoE, episodic memory, HMN
  v3.py                  IdentityRegister, DualHeadDecoder, LatentThinkingBuffer, HMN3
hmn_v2.py, hmn_v3.py     backward-compat shims (legacy scripts)
train_arithmetic.py      Task 1: arithmetic curriculum trainer
train_recall.py          Task 2: key-value recall trainer
train_distill.py         distillation SFT trainer (python/build domain)
gen_arithmetic.py        streaming arithmetic dataset generator (stages A–D)
gen_recall.py            streaming key-value recall dataset generator
gen_distill_python.py    free-text distillation pairs generator
gen_distill_templates.py template-curriculum distillation generator
infer.py                 load a checkpoint + tokenizer, generate (one-shot / REPL)
test_arithmetic.py       data-generator unit tests
test_recall.py           recall data-generator unit tests
test_hmn.py              model tests: forward/backward, reversibility, gradients
experiments/             research scripts (option sweeps, v3 PoC) — historical
docs/                    design + findings (incl. verified result tables)
retop_tokenizer.json     the tokenizer (vocab 3190)
```

---

## Requirements & install

- Python 3.9+ (tested 3.12), CPU-only PyTorch is fine, `tokenizers`
- Verified on: 4 cores, 4.6 GB RAM, ~1.7 s/step (D64/L4, batch 8)

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install tokenizers
pip install -e .          # optional: import hmn from anywhere
```

## Quickstart

### 1. Generate a dataset

```bash
python gen_recall.py --train 20000 --val 3000 --seed 42
```

### 2. Train on CPU

```bash
# recall task (verified config lands ~94–99%)
python train_recall.py --steps 1500 --dim 64 --layers 4 --mem-cells 256 --bs 8
```

All trainers stream chunks from `hmn_data/` (git-ignored, regenerable) and report
held-out eval at intervals. Every script takes `--tok <path>` / `--data <dir>` to
override the default repo-relative paths.

### 3. Use a trained model

```bash
# interactive
python infer.py --checkpoint ckpt.pt --interactive
# one-shot (must pass the same hyperparameters used at train time)
python infer.py --checkpoint ckpt.pt --arch v2 --dim 64 --layers 4 --mem-cells 256 --prompt "pip install numpy"
```

> Checkpoints are `model.state_dict()` — the CLI prints a clear error if the
> `--arch`/hyperparameters don't match the checkpoint.

### 4. Run the tests

```bash
python test_hmn.py        # model: forward/backward, reversibility, gradient check
python test_arithmetic.py # data generators
python test_recall.py     # data generators
```

---

## Architecture notes

- **Reversible coupling** (`ReversibleFunction`) recomputes activations during
  backward instead of storing them — memory-efficient by design (also what makes
  the latent thinking loop cheap: activations reconstruct for free).
- **Memory reads raw token embeddings** (`HMN_Option1`) rather than backbone hidden
  states. The backbone makes hidden states contextual, which breaks content
  addressing; raw embeddings preserve token identity. This was the single biggest
  measured win (14% → 62% → 99%, `docs/task2_findings.md`).
- **β sharpening (β≈30) + usage decay** were verified synergistic for recall
  (81–89% individually → 99% together).

### Known issues
- `DifferentiableEpisodicMemory.out_proj` is defined but not used in `forward`
  (dead parameters, ~no effect on results).
- v3's `IdentityRegister` allocates a `(B, T, vocab)` attention-mass tensor per
  forward — fine for the toy seq lengths here, but don't scale to long contexts
  without reworking the copy-marginal.

## Reproducibility

Every generator and trainer takes a `--seed`. Eval breakdowns are reported by
bucket (e.g. `n_pairs`, `query_frac`) so a "high overall" that is actually a
positional shortcut is visible — exactly how the false 100% in the original doc
was caught (`docs/task2_findings.md` #7).

## License

Apache-2.0 (TODO: pick & apply before publishing).
