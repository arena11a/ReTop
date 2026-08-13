# Distillation Design — Domain: Python + Build (HMN's own stack)

**Phase 3 goal:** "smart model + easy train + low spec" via (1) distillation from a
teacher, (2) episodic memory holding specialist knowledge outside parameters,
(3) narrow coding domain.

## Step 1 decisions

- **Domain (C):** Python + build system — exactly what HMN itself uses to train
  (venv, pip, Gradle, pyproject, error debugging). Chosen because:
  - existing data can seed it (retop_dataset_python5m.jsonl), 
  - teacher (me) can author verified-correct pairs with zero API cost,
  - narrow, testable eval (AST-parses, runs, API usage exact).
- **Teacher:** the assistant (me) — no API key, no Ollama model, 2.6GB free RAM.
  Distillation = supervised fine-tune on my authored (prompt, response) pairs.
  This is "knowledge distillation" in the soft sense: high-precision target
  distributions from a capable source, into a small student (HMN dim64-128).
- **Hardware guardrails:** 4.6GB RAM, 2 threads, CPU only. Dataset is chunked
  jsonl (streaming, same as gen_arithmetic). Student = HMN dim64/2L or dim96/3L.

## Top-20 topics (core curriculum)

| # | Topic | Example prompt |
|---|-------|----------------|
| 1 | venv creation/activation | "Create a venv and install torch" |
| 2 | pip install / requirements | "Freeze my deps to requirements.txt" |
| 3 | python vs python3, shebang | "Why does 'python' fail but 'python3' work?" |
| 4 | Gradle build basics | "Run gradle build and explain --stacktrace" |
| 5 | pyproject.toml / setuptools | "Make my package pip-installable" |
| 6 | file IO (open/read/write) | "Read a file line by line safely" |
| 7 | path handling (pathlib/os.path) | "Join paths cross-platform" |
| 8 | string/format (f-strings) | "Format a float to 2 decimals" |
| 9 | list/dict/set comprehensions | "Build a dict from a list" |
| 10 | functions (args/kwargs/defaults) | "Write a function with *args" |
| 11 | error handling (try/except) | "Catch FileNotFoundError cleanly" |
| 12 | debugging (print/pdb/logging) | "Debug a script with pdb" |
| 13 | imports/modules | "Import from a sibling module" |
| 14 | classes/properties/dunder | "Define a class with __repr__" |
| 15 | stdlib: itertools/collections | "Count occurrences with Counter" |
| 16 | typing/type hints | "Type a function returning Optional[str]" |
| 17 | subprocess/exec | "Run a shell command from Python" |
| 18 | pytest testing | "Write a pytest test + fixture" |
| 19 | common errors/fixes | "Why ModuleNotFoundError? Fix it" |
| 20 | torch basics (HMN stack) | "Create an nn.Linear and call it" |

## Eval protocol (Step 5 metric, not just loss)

Per val sample, score 3 dimensions:
1. **syntax score**: `ast.parse(assistant)` succeeds → 1, else 0.
2. **API usage accuracy**: required API tokens (gold set stored in `meta.api`)
   all present in response → 1, else 0.
3. **run score** (subset, "runnable" samples where `meta.run` is executable and
   safe — no IO/network): execute in subprocess with timeout, compare exact
   stdout to `meta.expected_stdout` → 1, else 0.
Aggregate: report per-topic averages + overall. Compare student vs teacher on
val (teacher = ground truth since I authored; student trained HMN).

## Dataset format

Same as existing pipeline:
```
{"messages": [{"role":"user","content":...},{"role":"assistant","content":...}],
 "meta": {"topic": int, "api": ["open","splitlines"], "run": "...", "expected_stdout": "..."}}
```
Split 80/20 by topic + key (deterministic seed), chunked jsonl streaming.

## Step 5 RESULTS (updated): the pivot

**Attempt 1 — 70 free-text pairs (`distill_python/`): FAILED (val 0%).**
Three bugs were found & fixed during investigation:
1. Eval `max_new=32` truncated 15/17 val responses (they are 44-61 tokens) → API/run
   were artificially 0 even for a correct model. Fixed → 128.
2. Cosine LR collapsed to ~0 by step 1200 → memorization stalled at 6.5 nats.
   Constant LR keeps descending → now default.
3. Loss covered prompt+response; the ~40-token prompt dominated gradients. The TRUE
   answer-region CE was 14.7 nats (WORSE than random ln3190=8.07) while logged loss
   read 1.9. Fixed with `encode_pair_masked` (response-only targets).

Even after fixes the HMN core (56K non-embedding params at dim64; 249K at dim128/L6)
ONLY MEMORIZED the 70 pairs (train resp loss →1.0-2.1, val resp loss 12-13, val
syntax/API/run all 0). 70 free-text pairs ≈ 2800 distinct target tokens exceeds the
generalization budget of a 651K-param student.

**Attempt 2 (template curriculum, `distill_templates/`): WORKS.**
Fill-in-the-blank families (pip install/upgrade/freeze, venv, pathlib mkdir/exists,
read_file, fstring pad/round, list/dict comprehensions, Counter, import) with many
slot paraphrases; val = UNSEEN slots, zero answer overlap with train.
Deterministic split (md5, 4/5→train), 131 train / 34 val, 12 families.
dim128/L6 + constant LR + response-masked loss, 2000 steps:

  step  2000 loss=0.03 | val: python-syntax 21/23 (91%), api 39/41 (95%),
                       |      shell-exact 13/18 (72%)

vs teacher baseline synt/API = 100%. Remaining failures are SLOT-CONTENT errors
(model emits a dominant train slot, e.g. `pip install numpy` instead of `six`),
which is the expected small-student limit; the RULE transfers correctly.

**Attempt 3 (expanded curriculum, `gen_distill_templates.py` v2/v3): 4.5x more data.**
Bumps: 125 packages (was 25), 15 files, 19 dirs, 4 paraphrases/template (was 1),
+5 families (write_file, fstring_align, int_add, dict_from_list, expanded counter).
Deterministic md5 split; v3 = 1202 train / 269 val, 16 families.
dim128/L6 + constant LR + response-masked loss, 2600 steps (best-loss ckpt @ step 1800):

  val (all 269): python-syntax 93/133 (70%), shell-exact   0/136,
                 api          239/269 (89%)
  train: exact response reproduction 9/40; val: 0/40   <- rule, not memorization

Key comparisons vs v1 (25-pkg):
  - counter (was 0/12 api) -> 19/20 api after +11 candidates
  - dict_from_list api (was 0/8, wrong meta "dict") -> 7/8 after meta fix
  - pip_install/upgrade: `pip install <pkg>` pattern PERFECT on all 120; only the
    pkg slot is miscopied (emits a dominant train pkg like click/scipy/mypy).

CONCLUSION (both attempts): at 651K params the HMN student learns RULES and
DISTRIBUTIONS well (API 85-95%, python-syntax 70-90% on unseen slots) but cannot
exact-COPY an arbitrary slot token from the prompt; train exact-match is also only
9/40. Slot-copy needs either a weight/bigram attention readout or much higher
capacity -- documented as the small-student limit, not a training bug.
