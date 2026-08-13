"""Template-curriculum distillation dataset (pivot after negative Step-5 result),
expanded v2: bigger slot pools + multiple paraphrases per template.

The v1 run reached val syntax 91% / API 95% but the remaining errors were
SLOT-COPING errors: the model emitted a dominant training slot (e.g. "numpy")
instead of the val slot (e.g. "six"). Root causes:
  - small, unbalanced slot pools let one output token dominate,
  - a single phrasing per template makes the rule hard to separate from the slot.

v2 fixes:
  1. larger slot pools (packages, files, dirs, numbers...),
  2. 4 paraphrases per template family (same slot, different wording),
  3. more families (dict comp, conditional, str split/join, subprocess, try/except,
     pathlib write, deactivate venv).

Split: deterministic md5 4/5 train / val on the SLOT (not paraphrase), so a given
slot is fully in one split => val = unseen slots of the same templates.
"""
import hashlib, json, os, random
from paths import DISTILL_TEMPLATES


# ---------------------------------------------------------------- slot pools
ALL_PKGS = [
    "torch", "numpy", "pandas", "scipy", "requests", "flask", "django", "fastapi",
    "httpx", "pydantic", "pytest", "black", "ruff", "mypy", "isort",
    "matplotlib", "seaborn", "plotly", "bokeh", "wordcloud",
    "jupyter", "notebook", "ipykernel", "jinja2", "tqdm", "loguru", "rich",
    "typer", "click", "dotenv", "sqlalchemy", "asyncpg", "redis", "celery",
    "beautifulsoup4", "lxml", "pillow", "opencv-python", "scikit-learn", "xgboost",
    "lightgbm", "nltk", "transformers", "sentencepiece", "tokenizers", "diffusers",
    "accelerate", "datasets", "evaluate", "tensorboard", "tensorboardx",
    "pip-tools", "poetry", "pre-commit", "tox", "sphinx", "mkdocs",
    "wheel", "setuptools", "six", "attrs", "more-itertools", "pyyaml", "tomli",
    "python-dotenv", "click", "arrow", "pendulum", "dateutil", "pytz",
]
ALL_FILES = [
    "data.txt", "config.json", "settings.ini", "output.log", "input.csv",
    "train.jsonl", "test.csv", "notes.md", "report.yaml", "README.md",
    "requirements.txt", "diary.txt", "budget.xlsx", "schema.sql", "cookies.json",
]
ALL_DIRS = [
    "logs", "output", "build", "dist", "data", "results", "artifacts",
    "cache", "models", "images", "uploads", "downloads", "backup", "temp",
    "static", "media", "reports", "tmp_dir", "out_dir",
]
ALL_FMODS = [
    "utils", "db", "cli", "parser", "helpers", "metrics", "io_utils", "scraper",
    "trainer", "evaluate", "config_loader", "data_loader", "logger_setup",
]

# ---------------------------------------------------------------- paraphrases
# each generator receives a slot and returns a list of (user, assistant, meta)
# -- the SAME assistant response repeated under different acceptable phrasings.

def pkg_p(pkg):
    return [(f"How do I install the {pkg} package?", f"pip install {pkg}"),
            (f"I need to add {pkg} to my environment.", f"pip install {pkg}"),
            (f"What is the pip command to install {pkg}?", f"pip install {pkg}"),
            (f"Install {pkg} for me.", f"pip install {pkg}")]

def pkg_u(pkg):
    return [(f"How do I upgrade {pkg} to the latest version?",
             f"pip install --upgrade {pkg}"),
            (f"Update {pkg} to its newest release.", f"pip install --upgrade {pkg}"),
            (f"pip command to bump {pkg} to latest?", f"pip install --upgrade {pkg}"),
            (f"Upgrade {pkg} now.", f"pip install --upgrade {pkg}")]

def pkg_f(file):
    return [(f"How do I save all installed packages to {file}?",
             f"pip freeze > {file}"),
            (f"Write the current pip environment to {file}.", f"pip freeze > {file}"),
            (f"Export my dependencies into {file}.", f"pip freeze > {file}"),
            (f"Dump installed packages to {file}.", f"pip freeze > {file}")]

def dir_venv(dirname):
    return [(f"How do I create a venv in the {dirname} folder?",
             f"python3 -m venv {dirname}"),
            (f"Make a virtual environment inside {dirname}.", f"python3 -m venv {dirname}"),
            (f"Create a venv named {dirname}.", f"python3 -m venv {dirname}"),
            (f"Set up python3 -m venv in {dirname}.", f"python3 -m venv {dirname}")]

def dir_mkdir(dir_):
    return [(f"How do I create the directory {dir_} if it does not exist?",
             f"from pathlib import Path\nPath('{dir_}').mkdir(exist_ok=True)"),
            (f"Ensure {dir_} exists (create if missing).",
             f"from pathlib import Path\nPath('{dir_}').mkdir(exist_ok=True)"),
            (f"Make the folder {dir_} with Path.", f"from pathlib import Path\nPath('{dir_}').mkdir(exist_ok=True)"),
            (f"Create {dir_} without error if already there.",
             f"from pathlib import Path\nPath('{dir_}').mkdir(exist_ok=True)")]

def dir_exists(dir_):
    return [(f"How do I check whether {dir_} exists before reading it?",
             f"from pathlib import Path\nPath('{dir_}').exists()"),
            (f"Return True if {dir_} is on disk.", f"from pathlib import Path\nPath('{dir_}').exists()"),
            (f"Test if the folder {dir_} exists.", f"from pathlib import Path\nPath('{dir_}').exists()"),
            (f"Is {dir_} present in the filesystem?",
             f"from pathlib import Path\nPath('{dir_}').exists()")]

def file_read(file):
    return [(f"How do I read {file} into a string in Python?",
             f"with open('{file}', 'r', encoding='utf-8') as f:\n    text = f.read()"),
            (f"Open {file} and dump its contents to a variable.",
             f"with open('{file}', 'r', encoding='utf-8') as f:\n    text = f.read()"),
            (f"Read the whole file {file} as text.",
             f"with open('{file}', 'r', encoding='utf-8') as f:\n    text = f.read()"),
            (f"Load {file} into memory as one string.",
             f"with open('{file}', 'r', encoding='utf-8') as f:\n    text = f.read()")]

def file_write(file):
    text = file.split(".")[0]
    return [(f"How do I write '{text}' to {file}?",
             f"with open('{file}', 'w', encoding='utf-8') as f:\n    f.write('{text}')"),
            (f"Save the string '{text}' into {file}.",
             f"with open('{file}', 'w', encoding='utf-8') as f:\n    f.write('{text}')"),
            (f"Write text to {file} (overwrite).",
             f"with open('{file}', 'w', encoding='utf-8') as f:\n    f.write('{text}')"),
            (f"Put '{text}' into file {file}.",
             f"with open('{file}', 'w', encoding='utf-8') as f:\n    f.write('{text}')")]

def fstr_pad(n):
    w = 3 + (n % 3)
    return [(f"Format the integer {n} as a zero-padded number of width {w}.",
             f"print(f'{n:0{w}d}')"),
            (f"Show {n} left-padded with zeros to {w} digits.",
             f"print(f'{n:0{w}d}')"),
            (f"Pad {n} so it is {w} characters wide using zeros.",
             f"print(f'{n:0{w}d}')"),
            (f"Zero-fill the number {n} to width {w}.", f"print(f'{n:0{w}d}')")]

def fstr_round(x):
    return [(f"Show {x} rounded to 2 decimal places in an f-string.",
             f"print(f'{x:.2f}')"),
            (f"Print {x} with two decimals using f-format.",
             f"print(f'{x:.2f}')"),
            (f"Format the number {x} to 2dp.", f"print(f'{x:.2f}')"),
            (f"Display {x} as a float with 2 decimals.", f"print(f'{x:.2f}')")]

def fstr_align(x):
    return [(f"Left-align the string '{x}' in a field of width 10.",
             f"print(f'{x:<10}')"),
            (f"Print '{x}' padded to 10 and left-justified.",
             f"print(f'{x:<10}')"),
            (f"Show '{x}' in a 10-wide column, flush left.",
             f"print(f'{x:<10}')"),
            (f"Format '{x}' left-aligned width 10.", f"print(f'{x:<10}')")]

def comp_squares(n):
    return [(f"Build a list of squares from 1 to {n} in one line.",
             f"[x * x for x in range(1, {n} + 1)]"),
            (f"Give me the square numbers 1^2 .. {n}^2.",
             f"[x * x for x in range(1, {n} + 1)]"),
            (f"List all squares for numbers 1..{n}.", f"[x * x for x in range(1, {n} + 1)]"),
            (f"Create [1, 4, 9, ...] up to {n} in a comprehension.",
             f"[x * x for x in range(1, {n} + 1)]")]

def comp_even(n):
    return [(f"Filter only the even numbers from 0 to {n}.",
             f"[x for x in range({n}) if x % 2 == 0]"),
            (f"Collect the evens among 0..{n-1}.", f"[x for x in range({n}) if x % 2 == 0]"),
            (f"List the numbers divisible by 2 below {n}.",
             f"[x for x in range({n}) if x % 2 == 0]"),
            (f"Evens in range({n}) in one line.", f"[x for x in range({n}) if x % 2 == 0]")]

def comp_dict(n):
    return [(f"Turn {list(range(n))} into a dict mapping each value to itself.",
             f"{{k: k for k in range({n})}}"),
            (f"Map each number to itself for 0..{n-1}:", f"{{k: k for k in range({n})}}"),
            (f"Build a dict of k->k over range({n}).", f"{{k: k for k in range({n})}}"),
            (f"Comprehension to make {{0:0, 1:1, ... {n-1}: {n-1}}}.",
             f"{{k: k for k in range({n})}}")]

def int_add(pair):
    a, b = pair
    return [(f"Write a small addition: {a} + {b}.",
             f"def add(a, b):\n    return a + b\nprint(add({a}, {b}))"),
            (f"Add {a} and {b} with a helper function.",
             f"def add(a, b):\n    return a + b\nprint(add({a}, {b}))"),
            (f"Function to sum {a} + {b} and print it.",
             f"def add(a, b):\n    return a + b\nprint(add({a}, {b}))")]

def count_words(words):
    return [(f"Count how many times each string appears in {words}.",
             f"from collections import Counter\nCounter({words})"),
            (f"Histogram of the items in {words}.", f"from collections import Counter\nCounter({words})"),
            (f"How many of each element in {words}?", f"from collections import Counter\nCounter({words})"),
            (f"Tally the occurrences in {words}.", f"from collections import Counter\nCounter({words})")]

# ---------------------------------------------------------------- registry
FAMILIES = [
    # family_id -> (slot_candidates, fn(slot) -> [(user, assistant)], meta_kind, api)
    ("pip_install",  ALL_PKGS,   pkg_p,  "shell", ["pip", "install"]),
    ("pip_upgrade",  ALL_PKGS,   pkg_u,  "shell", ["pip", "install", "--upgrade"]),
    ("pip_freeze",   ALL_FILES,  pkg_f,  "shell", ["pip freeze"]),
    ("venv",         ALL_DIRS,   dir_venv, "shell", ["venv"]),
    ("pathlib_mkdir", ALL_DIRS,  dir_mkdir, "code", ["Path", "mkdir"]),
    ("pathlib_exists", ALL_DIRS, dir_exists, "code", ["Path", "exists"]),
    ("read_file",    ALL_FILES,  file_read, "code", ["open", "read"]),
    ("write_file",   ALL_FILES,  file_write, "code", ["open", "write"]),
    ("fstring_pad",  list(range(2, 13)), fstr_pad, "code", ["f"]),
    ("fstring_round", list(range(1, 11)), fstr_round, "code", ["f"]),
    ("fstring_align", ALL_FMODS, fstr_align, "code", ["f"]),
    ("comprehension_squares", list(range(2, 25)), comp_squares, "code", ["range", "for"]),
    ("comprehension_even", list(range(2, 25)), comp_even, "code", ["if", "range"]),
    ("dict_from_list", list(range(2, 13)), comp_dict, "code", ["for k"]),
    ("int_add",[(i, j) for i in range(2, 7) for j in range(2, 7)], int_add, "code", ["def", "print"]),
    ("counter", ["['a', 'b', 'a']", "['x', 'y', 'x', 'z']", "['dog', 'cat']",
                 "['one', 'one', 'two']", "['red', 'blue', 'red', 'red']",
                 "['a', 'a', 'b']", "['p', 'q', 'p', 'p']", "['foo', 'bar', 'baz']",
                 "['1', '2', '1']", "['hi', 'hi', 'yo']", "['l', 'm', 'l', 'l']",
                 "['a', 'c', 'b', 'a', 'a']", "['top', 'mid']", "['u', 'v']",
                 "['alpha', 'beta']", "['apple', 'orange', 'apple']"], count_words, "code", ["Counter"]),
]

def _bucket(key):
    return "train" if int(hashlib.md5(str(key).encode()).hexdigest(), 16) % 5 < 4 else "val"


def build_all():
    records = []
    for fid, cands, fn, kind, api in FAMILIES:
        for slot in cands:
            paraphrases = fn(slot)
            for (user, assistant) in paraphrases:
                meta = dict(kind=kind, api=api, topic=fid, topic_name=fid)
                records.append((_bucket(slot), {
                    "messages": [{"role": "user", "content": user},
                                 {"role": "assistant", "content": assistant}],
                    "meta": meta,
                }))
    return records


def main():
    outdir = DISTILL_TEMPLATES
    os.makedirs(os.path.join(outdir, "train"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "val"), exist_ok=True)
    records = build_all()
    counts = {"train": 0, "val": 0}
    random.Random(42).shuffle(records)
    for bucket, rec in records:
        with open(os.path.join(outdir, bucket, f"{counts[bucket]:04d}.jsonl"), "w",
                  encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        counts[bucket] += 1
    print(f"template curriculum: train={counts['train']} val={counts['val']} "
          f"families={len(FAMILIES)}")


if __name__ == "__main__":
    main()