# Data preparation

`retop.py train` and `train_v3.py` share the same loss/recipe, but their **data
inputs differ**: `train_v3.py` builds its slot-copy data in-process from
hardcoded seen/unseen package lists (no file read); `retop.py train` reads the
on-disk formats below. All records are encoded on the fly (streaming), so files
can be arbitrarily large on low-RAM machines.

## 1. Slot-copy chat pairs (.jsonl) — the v3.3 native format

The format the Identity Register was designed for: the assistant answer is the
exact literal copy of the user prompt.

```json
{"messages": [{"role": "user", "content": "pip install pkg042"},
              {"role": "assistant", "content": "pip install pkg042"}]}
```

`hmn/recipe.make_chat_targets` derives the three supervision labels from one such
record (see `docs/hmn_v3_design.md` §2):

| label | meaning | masked off |
|---|---|---|
| `Y`  | shifted targets, all answer tokens | prompt region |
| `Yc` | copy-channel targets (the register literal) | EOS and the first answer token |
| `G`  | 1.0 if the target exists in the prompt, 0.0 otherwise | — |

Because the answer mirrors the prompt, every answer token after the first has an
exact twin in the prompt — that is what makes the copy lane trainable.

### Generating slot-copy data

```bash
python gen_slots.py --out data/slots.jsonl \
    --n-seen 600 --n-unseen 400 --kind pkg3 \
    --template "pip install {slot}" --seed 0
```

Writes `data/slots.jsonl` (600 seen records), `data/slots-val.jsonl` (400 unseen),
and `data/slots.jsonl.meta.json`. The `--kind` slot vocabularies:

| kind | values | note |
|---|---|---|
| `pkg3` | `pkg000..pkg999` | canonical (matches `hmn_v33.pt`) |
| `pkg4` / `pkg5` | longer slots | generalization probes |
| `alnum` | `pkgA00..pkgJ99` | mixed letters+digits |
| `repeat` | `pkg111..pkg999` | repeated-token slots (known limitation) |

> **The seen/unseen invariant (non-negotiable).** Both files come from ONE
> deterministic shuffled stream sliced by count, so a slot value can never
> appear in both sets. The v3.3 accuracy claims (40/40 unseen) are only
> meaningful if the eval slots were never in the training file. Verify your own
> split the same way: `set(seen) & set(unseen) == ∅`.

## 1b. Distributed chat corpora (`gen_chat.py`) — English & Math

`gen_chat.py` streams combinatorial chat-pairs up to a target token budget
(~4 chars/token heuristic). Deterministic per `--seed`; the sidecar
`<out>.meta.json` records `records`, `chars`, and the approx token count.

```bash
python gen_chat.py --domain english       --target-tokens 10_000_000  --out data/english_10m.jsonl
python gen_chat.py --domain math_general  --target-tokens 30_000_000  --out data/math_general_30m.jsonl
python gen_chat.py --domain math_complex  --target-tokens 60_000_000  --out data/math_complex_60m.jsonl
python gen_chat.py --domain math_advanced --target-tokens 120_000_000 --out data/math_advanced_120m.jsonl
```

Shipped corpora (do not commit — `data/` is git-ignored):

| file | tokens (approx) | content |
|---|---|---|
| `data/english_10m.jsonl` | 10M | general knowledge / trivia / how-to dialogue |
| `data/english_50m.jsonl` | 50M | same, larger |
| `data/english_100m.jsonl` | 100M | same, largest |
| `data/math_general_30m.jsonl` | 30M | arithmetic, distance, sharing, %, rectangles |
| `data/math_complex_60m.jsonl` | 60M | fractions, ratios, linear equations, triangle area |
| `data/math_advanced_120m.jsonl` | 120M | powers, sqrt/cuberoot, circles, quadratics, compound % |

Math answers are computed in the generator (no hand-written labels), so the
corpora are internally consistent. Train with `retop.py train --data <file>`.

## 2. General chat pairs (.jsonl)

```json
{"messages": [{"role": "user", "content": "what is 2+2"},
              {"role": "assistant", "content": "4"}]}
```

- The second message must be the assistant's. The loss is masked to the
  assistant region (the prompt is context, not target) — this is required, a
  full-sequence loss starves the answer head (measured 14.7 nat answer loss).
- The copy labels are still computed: any answer token whose exact twin exists
  in the prompt is a copy row, everything else is a generate row. So the v3.3
  recipe applies unchanged to general chat.
- The train/val split is by a deterministic hash of the first 40 chars of the
  user text (see `Dataset._split`), so re-runs with the same `--seed` are
  reproducible. For a hard disjoint split (e.g. by topic) pre-split the files
  and point `--data` at separate directories.

## 3. Plain text (.txt or .jsonl text records)

```text
from sympy import Symbol, integrate
...
```

Tokenized into `--seq`-token blocks; the copy channel is all-ignore, so the
model degrades to a plain (dual-head) LM. This is the low-effort format: point
`retop.py train --data` at any corpus.

## Format vs. recipe capability

| format | copy lane | gen lane | best for |
|---|---|---|---|
| slot-copy chat pairs | yes (mirror prompt) | first token + EOS | verified slot-copy eval |
| general chat pairs | yes (twin-dependent) | everything else | dialogue |
| plain text | no | everything | pretraining / warm start |
