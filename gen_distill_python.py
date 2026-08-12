"""Distillation dataset generator — teacher = assistant-authored pairs.

Domain: Python + build (HMN's own stack). Each pair is authored by the teacher
(the assistant) with verified-correct responses. Pairs carry a meta dict used by
the eval protocol:
  {"messages":[{"role":"user",...},{"role":"assistant",...}],
   "meta": {"topic": int, "api":[required API tokens], "run":code-if-runnable,
            "expected_stdout":str}}

Split: deterministic 80/20 by (topic, key) with seed. Chunked streaming jsonl,
same format as gen_arithmetic.py so train_distill.py can reuse iter_records.
"""
import argparse, json, os, random, hashlib

TOPICS = [
    "venv", "pip", "python_vs_python3", "gradle_build", "pyproject_setuptools",
    "file_io", "pathlib", "string_format", "comprehensions", "functions",
    "error_handling", "debugging", "imports_modules", "classes_dunder",
    "itertools_collections", "typing_hints", "subprocess", "pytest",
    "common_errors", "torch_basics",
]

# topic index -> list of (prompt, response, api, run, expected_stdout)
# run may be "" meaning "not runnable" (IO/network/syntax-only answer).
CUR = {}


def P(topic, prompt, response, api=(), run="", expected_stdout=""):
    CUR.setdefault(topic, []).append((prompt, response, list(api), run, expected_stdout))


# ---- 1 venv ----
P("venv", "How do I create a virtual environment in a project folder and activate it?",
  "python3 -m venv .venv\nsource .venv/bin/activate\n# Windows: .venv\\Scripts\\activate",
  ["venv", "activate"])
P("venv", "I want my project to use an isolated Python so pip installs don't pollute the system. What's the first command?",
  "python3 -m venv .venv",
  ["venv"])
P("venv", "What does activating a venv change on Linux?",
  "Activation prepends the venv's bin dir to PATH so `python` and `pip` point at .venv/bin/python and .venv/bin/pip.",
  ["PATH", "bin"])
P("venv", "How do I leave a virtual environment?",
  "Run `deactivate`. The shell stops prepending the venv's bin dir to PATH.",
  ["deactivate"])
P("venv", "Why create a venv per project instead of installing everything globally?",
  "Each project gets isolated dependency versions, avoiding conflicts (e.g. one project needs numpy 1.x, another 2.x).",
  ["isolated"])
P("venv", "Create a venv and install torch into it with CPU-only wheels.",
  "python3 -m venv .venv\n.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu",
  ["venv", "pip", "install"])

# ---- 2 pip ----
P("pip", "How do I save all installed packages to a requirements.txt?",
  "pip freeze > requirements.txt",
  ["freeze"])
P("pip", "Install all dependencies listed in requirements.txt.",
  "pip install -r requirements.txt",
  ["install", "-r"])
P("pip", "What's the difference between `pip install package` and `pip install --user package`?",
  "Plain pip installs into the active environment (site-packages). --user installs into a per-user directory, used when the system site-packages is not writable.",
  ["install", "--user"])
P("pip", "How do I upgrade a single package with pip?",
  "pip install --upgrade numpy",
  ["--upgrade"])
P("pip", "I get 'externally-managed-environment' when installing with pip. What's happening?",
  "On modern distros the system Python is managed by the OS package manager. Create a venv (python3 -m venv .venv) and install there instead.",
  ["externally-managed-environment", "venv"])
P("pip", "How do I list currently installed pip packages?",
  "pip list",
  ["list"])

# ---- 3 python vs python3 ----
P("python_vs_python3", "Why does `python` fail but `python3` work on Linux?",
  "Some distros don't provide a `python` symlink (only `python3`) to avoid ambiguity between Python 2/3. Use `python3` or create an alias.",
  ["python3"])
P("python_vs_python3", "What should I put at the top of a Python script so it runs with the right interpreter on Linux?",
  "#!/usr/bin/env python3\n# then chmod +x script.py and run ./script.py",
  ["env", "python3"])
P("python_vs_python3", "How do I check which Python version is active?",
  "python3 --version\n# or in code:\nimport sys; print(sys.version)",
  ["--version"])

# ---- 4 gradle build ----
P("gradle_build", "How do I run a Gradle build and get a full stack trace on failure?",
  "./gradlew build --stacktrace",
  ["gradlew", "build", "--stacktrace"])
P("gradle_build", "What is the difference between `./gradlew build` and `./gradlew assemble`?",
  "assemble only produces the artifacts (jars/apks). build also runs check tasks (unit tests, lint) before/along with assembly.",
  ["build", "assemble"])
P("gradle_build", "How do I clean old Gradle build output before rebuilding?",
  "./gradlew clean build",
  ["clean", "build"])
P("gradle_build", "Why should I commit the Gradle wrapper (gradlew) instead of using a system gradle?",
  "The wrapper pins the exact Gradle version for the project, so everyone builds identically regardless of their installed Gradle.",
  ["wrapper", "gradlew"])
P("gradle_build", "How do I run a single unit test class with Gradle?",
  "./gradlew test --tests 'com.example.MyTest'",
  ["test", "--tests"])

# ---- 5 pyproject/setuptools ----
P("pyproject_setuptools", "Write a minimal pyproject.toml so my package is pip-installable.",
  "[build-system]\nrequires = [\"setuptools>=61\"]\nbuild-backend = \"setuptools.build_meta\"\n\n[project]\nname = \"mypkg\"\nversion = \"0.1.0\"\nrequires-python = \">=3.9\"",
  ["build-system", "setuptools", "[project]"])
P("pyproject_setuptools", "How do I install a local package in editable/development mode?",
  "pip install -e .   # editable install: code changes apply without reinstall",
  ["-e", "install"])
P("pyproject_setuptools", "How do I declare a runtime dependency in pyproject.toml?",
  "[project]\ndependencies = [\n    \"requests>=2.28\",\n    \"numpy>=1.24\",\n]",
  ["dependencies"])
P("pyproject_setuptools", "How do I specify which Python versions my package supports?",
  "[project]\nrequires-python = \">=3.9,<4\"",
  ["requires-python"])

# ---- 6 file io ----
P("file_io", "How do I read a text file line by line without loading it all into memory?",
  "with open('data.txt', 'r', encoding='utf-8') as f:\n    for line in f:\n        print(line.rstrip())",
  ["open", "splitlines"], run="open('data.txt','w').write('line1\\nline2\\n')\nwith open('data.txt') as f:\n    for line in f:\n        print(line.rstrip())", expected_stdout="line1\nline2\n")
P("file_io", "How do I safely write to a file so it is closed even if an error occurs?",
  "with open('out.txt', 'w', encoding='utf-8') as f:\n    f.write('hello\\n')",
  ["open", "with"], run="\n", expected_stdout="")
P("file_io", "How do I append text to an existing file instead of overwriting it?",
  "with open('log.txt', 'a', encoding='utf-8') as f:\n    f.write('new line\\n')",
  ["open", "'a'"], run="\n", expected_stdout="")
P("file_io", "What does opening a file with encoding='utf-8' do and why use it?",
  "It decodes bytes as UTF-8 text on read and encodes text as UTF-8 on write, avoiding mojibake on systems whose default locale differs.",
  ["encoding", "utf-8"])
P("file_io", "How do I read an entire file as one string?",
  "with open('data.txt', 'r', encoding='utf-8') as f:\n    text = f.read()",
  ["read"], run="abc\n", expected_stdout="")

# ---- 7 pathlib ----
P("pathlib", "How do I join paths cross-platform so Windows and Linux both work?",
  "from pathlib import Path\np = Path('data') / 'sub' / 'file.csv'\n# Windows-safe and POSIX-safe",
  ["pathlib", "Path"], run="\n", expected_stdout="")
P("pathlib", "How do I check whether a file exists with pathlib?",
  "from pathlib import Path\nif Path('config.json').exists():\n    print('exists')",
  ["Path", "exists"], run="\n", expected_stdout="")
P("pathlib", "How do I list all .py files in a directory recursively?",
  "from pathlib import Path\nfor p in Path('src').rglob('*.py'):\n    print(p)",
  ["rglob"], run="\n", expected_stdout="")
P("pathlib", "How do I get the file name and parent directory from a path?",
  "from pathlib import Path\np = Path('/a/b/c.txt')\nname = p.name      # c.txt\nparent = p.parent  # /a/b",
  ["Path", "name", "parent"], run="\n", expected_stdout="")

# ---- 8 string format ----
P("string_format", "How do I format a float to exactly 2 decimals in Python?",
  "value = 3.14159\nprint(f'{value:.2f}')   # 3.14",
  ["f-string", ".2f"], run="value = 3.14159\nprint(f'{value:.2f}')\n", expected_stdout="3.14\n")
P("string_format", "How do I pad a number with leading zeros, e.g. 7 -> '007'?",
  "print(f'{7:03d}')   # 007",
  [":03d"], run="print(f'{7:03d}')\n", expected_stdout="007\n")
P("string_format", "What is the difference between f-strings and the old % formatting?",
  "f-strings are readable and evaluated inline: f'{name} is {age}'. The % style is older: '%s is %d' % (name, age). Prefer f-strings in Python 3.",
  ["f-string"], run="\n", expected_stdout="")
P("string_format", "How do I insert a variable into a string and align text?",
  "print(f'{name:<10}|')   # left-align in 10 chars",
  ["f-string", "align"], run="\n", expected_stdout="")

# ---- 9 comprehensions ----
P("comprehensions", "How do I build a dict mapping each number to its square?",
  "squares = {x: x*x for x in range(1, 6)}\n# {1:1, 2:4, 3:9, 4:16, 5:25}",
  ["for x in range"], run="squares = {x: x*x for x in range(1, 6)}\nprint(squares)\n", expected_stdout="{1: 1, 2: 4, 3: 9, 4: 16, 5: 25}\n")
P("comprehensions", "How do I filter even numbers from a list with a comprehension?",
  "nums = [1, 2, 3, 4, 5, 6]\nevens = [n for n in nums if n % 2 == 0]\n# [2, 4, 6]",
  ["for n in nums"], run="nums=[1,2,3,4,5,6]\nprint([n for n in nums if n%2==0])\n", expected_stdout="[2, 4, 6]\n")
P("comprehensions", "How do I convert a list of strings to a list of ints?",
  "vals = ['1', '2', '3']\nints = [int(v) for v in vals]",
  ["for v in"], run="print([int(v) for v in ['1','2','3']])\n", expected_stdout="[1, 2, 3]\n")
P("comprehensions", "How do I make a set of unique lengths from a list of words?",
  "words = ['a', 'bb', 'ccc', 'bb']\nlengths = {len(w) for w in words}\n# {1, 2, 3}",
  ["for w in"], run="words=['a','bb','ccc','bb']\nprint({len(w) for w in words})\n", expected_stdout="{1, 2, 3}\n")

# ---- 10 functions ----
P("functions", "Write a function that accepts a variable number of positional arguments and sums them.",
  "def total(*args):\n    return sum(args)\n\nprint(total(1, 2, 3))   # 6",
  ["*args"], run="def total(*args):\n    return sum(args)\nprint(total(1,2,3))\n", expected_stdout="6\n")
P("functions", "Write a function with a default argument value.",
  "def greet(name, greeting='Hello'):\n    return f'{greeting}, {name}'\n\nprint(greet('Ann'))       # Hello, Ann\nprint(greet('Bob', 'Hi'))  # Hi, Bob",
  ["default"], run="def greet(name, greeting='Hello'):\n    return f'{greeting}, {name}'\nprint(greet('Ann'))\nprint(greet('Bob','Hi'))\n", expected_stdout="Hello, Ann\nHi, Bob\n")
P("functions", "How do I accept arbitrary keyword arguments in a function?",
  "def config(**kwargs):\n    for k, v in kwargs.items():\n        print(k, v)\n\nconfig(lr=1e-3, epochs=10)",
  ["**kwargs"], run="def config(**kwargs):\n    for k,v in kwargs.items():\n        print(k,v)\nconfig(lr=1e-3, epochs=10)\n", expected_stdout="lr 0.001\nepochs 10\n")
P("functions", "What is the difference between *args and **kwargs?",
  "*args collects extra positional arguments into a tuple; **kwargs collects extra keyword arguments into a dict.",
  ["*args", "**kwargs"])

# ---- 11 error handling ----
P("error_handling", "How do I catch FileNotFoundError without crashing?",
  "try:\n    with open('missing.txt') as f:\n        data = f.read()\nexcept FileNotFoundError:\n    print('file not found')",
  ["try", "except", "FileNotFoundError"], run="try:\n    open('missing.txt')\nexcept FileNotFoundError:\n    print('file not found')\n", expected_stdout="file not found\n")
P("error_handling", "How do I catch multiple exception types in one except clause?",
  "try:\n    result = 10 / value\nexcept (ValueError, ZeroDivisionError):\n    print('bad input or division by zero')",
  ["except", "ZeroDivisionError"])
P("error_handling", "What does try/except/finally do, and when does finally run?",
  "finally always runs (even on error or return), usually for cleanup like closing resources: try: ... except: ... finally: f.close()",
  ["finally"])
P("error_handling", "How do I re-raise an exception after logging it?",
  "try:\n    risky()\nexcept Exception as e:\n    print('caught', e)\n    raise",
  ["raise"])
P("error_handling", "What is the difference between `raise` and `raise e` inside an except block?",
  "Bare `raise` re-raises the original exception preserving the traceback; `raise e` re-raises a new traceback rooted at that line. Prefer bare raise.",
  ["raise", "traceback"])

# ---- 12 debugging ----
P("debugging", "How do I start the pdb debugger at a specific line?",
  "import pdb; pdb.set_trace()\n# drops into the interactive debugger; n=next, p expr=print, q=quit",
  ["pdb", "set_trace"])
P("debugging", "How do I print a variable's value while debugging a script?",
  "Add print() lines: print('value =', value, flush=True) — or use the logging module for level-filtered output.",
  ["print"])
P("debugging", "What is the difference between print debugging and the logging module?",
  "logging writes to a configurable stream with levels (DEBUG/INFO/WARNING) and timestamps; print goes straight to stdout with no levels. logging is better for non-trivial programs.",
  ["logging"])
P("debugging", "How do I enable full traceback printing for an unhandled exception?",
  "import traceback\ntry:\n    risky()\nexcept Exception:\n    traceback.print_exc()",
  ["traceback", "print_exc"])

# ---- 13 imports/modules ----
P("imports_modules", "How do I import a module from a sibling folder?",
  "If the folder is a package (has __init__.py), use a relative import:\nfrom .utils import helper\n# or add the parent to sys.path: sys.path.insert(0, os.path.dirname(__file__))",
  ["import", "sys.path"])
P("imports_modules", "What is the difference between `import math` and `from math import sqrt`?",
  "`import math` binds the module name (math.sqrt); `from math import sqrt` binds only sqrt directly. The latter pollutes less but the former keeps the namespace explicit.",
  ["import", "from"])
P("imports_modules", "What does `if __name__ == '__main__':` do?",
  "The `if __name__ == '__main__':` block runs only when the file is executed directly, not when imported. Used to guard test/demo code from running on import.",
  ["__main__"])
P("imports_modules", "Why does importing a local file sometimes fail in Jupyter but work in a script?",
  "Jupyter's working directory is not automatically on sys.path the same way. Add `import sys, os; sys.path.insert(0, os.getcwd())` or run as a script.",
  ["sys.path"])

# ---- 14 classes/dunder ----
P("classes_dunder", "Write a class with a useful __repr__.",
  "class Point:\n    def __init__(self, x, y):\n        self.x, self.y = x, y\n    def __repr__(self):\n        return f'Point({self.x}, {self.y})'\n\nprint(Point(1, 2))   # Point(1, 2)",
  ["__repr__"], run="class Point:\n    def __init__(self,x,y):\n        self.x,self.y=x,y\n    def __repr__(self):\n        return f'Point({self.x}, {self.y})'\nprint(Point(1,2))\n", expected_stdout="Point(1, 2)\n")
P("classes_dunder", "How do I make an object support the len() builtin?",
  "Define __len__ returning an int:\nclass Bag:\n    def __len__(self):\n        return len(self.items)",
  ["__len__"])
P("classes_dunder", "What is the difference between __str__ and __repr__?",
  "__str__ is the human-readable form used by print()/str(); __repr__ is the unambiguous developer form used by repr() and in the REPL. If only one exists, __str__ falls back to __repr__.",
  ["__str__", "__repr__"])
P("classes_dunder", "How do I define a read-only property in a class?",
  "class Circle:\n    def __init__(self, r):\n        self.r = r\n    @property\n    def area(self):\n        return 3.14159 * self.r ** 2",
  ["property"], run="class Circle:\n    def __init__(self,r):\n        self.r=r\n    @property\n    def area(self):\n        return 3.14159*self.r**2\nprint(round(Circle(2).area,2))\n", expected_stdout="12.57\n")

# ---- 15 itertools/collections ----
P("itertools_collections", "How do I count occurrences of items in a list?",
  "from collections import Counter\nitems = ['a', 'b', 'a', 'c', 'a']\ncounts = Counter(items)\n# Counter({'a': 3, 'b': 1, 'c': 1})",
  ["Counter"], run="from collections import Counter\nprint(Counter(['a','b','a','c','a']))\n", expected_stdout="Counter({'a': 3, 'b': 1, 'c': 1})\n")
P("itertools_collections", "How do I get the n most common items with Counter?",
  "from collections import Counter\nc = Counter('aaabbc')\nprint(c.most_common(2))   # [('a', 3), ('b', 2)]",
  ["most_common"], run="from collections import Counter\nprint(Counter('aaabbc').most_common(2))\n", expected_stdout="[('a', 3), ('b', 2)]\n")
P("itertools_collections", "How do I pair adjacent items in a list?",
  "from itertools import pairwise\nlist(pairwise([1, 2, 3, 4]))   # [(1,2), (2,3), (3,4)]",
  ["pairwise"], run="from itertools import pairwise\nprint(list(pairwise([1,2,3,4])))\n", expected_stdout="[(1, 2), (2, 3), (3, 4)]\n")
P("itertools_collections", "How do I chain multiple iterables into one?",
  "from itertools import chain\nlist(chain([1,2], [3,4]))   # [1, 2, 3, 4]",
  ["chain"], run="from itertools import chain\nprint(list(chain([1,2],[3,4])))\n", expected_stdout="[1, 2, 3, 4]\n")

# ---- 16 typing hints ----
P("typing_hints", "Type a function that may return a string or None.",
  "from typing import Optional\n\ndef find_name(key: str) -> Optional[str]:\n    return db.get(key)",
  ["Optional"])
P("typing_hints", "Type a function that takes a list of ints and returns a dict of str->int.",
  "from typing import Dict, List\n\ndef build(data: List[int]) -> Dict[str, int]:\n    return {str(i): i for i in data}",
  ["List", "Dict"])
P("typing_hints", "Do type hints affect runtime behavior?",
  "No. They are for static checkers (mypy/pyright) and documentation; Python ignores them at runtime.",
  ["mypy"])
P("typing_hints", "How do I type a function argument that can be multiple types?",
  "from typing import Union\n\ndef f(x: Union[int, str]) -> str:\n    return str(x)\n# Python 3.10+: x: int | str",
  ["Union"])

# ---- 17 subprocess ----
P("subprocess", "How do I run a shell command from Python and capture its stdout?",
  "import subprocess\nresult = subprocess.run(['ls', '-la'], capture_output=True, text=True)\nprint(result.stdout)",
  ["subprocess", "run", "capture_output"])
P("subprocess", "Why should I pass a list of args instead of a raw string to subprocess.run?",
  "A list avoids shell quoting bugs and injection; subprocess runs the program directly without a shell unless shell=True.",
  ["shell=False"])
P("subprocess", "How do I check the exit code of a subprocess?",
  "result = subprocess.run(['true'], capture_output=True)\nprint(result.returncode)   # 0",
  ["returncode"], run="import subprocess\nr = subprocess.run(['python3','-c','print(1)'],capture_output=True,text=True)\nprint(r.returncode)\n", expected_stdout="0\n")
P("subprocess", "How do I run a command and raise an error if it fails?",
  "import subprocess\nsubprocess.run(['cmd'], check=True)\n# check=True raises CalledProcessError on non-zero exit",
  ["check=True"])

# ---- 18 pytest ----
P("pytest", "Write a simple pytest test for an add function.",
  "def add(a, b):\n    return a + b\n\ndef test_add():\n    assert add(2, 3) == 5",
  ["assert"], run="def add(a,b):\n    return a+b\ndef test_add():\n    assert add(2,3)==5\n", expected_stdout="")
P("pytest", "How do I write a pytest fixture that provides a temp file?",
  "import pytest\n\n@pytest.fixture\ndef data_path(tmp_path):\n    p = tmp_path / 'd.txt'\n    p.write_text('hi')\n    return p\n\ndef test_read(data_path):\n    assert data_path.read_text() == 'hi'",
  ["fixture", "tmp_path"])
P("pytest", "How do I run only the tests in one file?",
  "pytest tests/test_utils.py",
  ["pytest"])
P("pytest", "How do I parametrize a pytest test with multiple inputs?",
  "import pytest\n\n@pytest.mark.parametrize('a,b,expected', [(1,2,3), (2,3,5)])\ndef test_add(a, b, expected):\n    assert a + b == expected",
  ["parametrize"])

# ---- 19 common errors ----
P("common_errors", "I get ModuleNotFoundError: No module named 'requests'. What's the fix?",
  "The package isn't installed in the active environment. Run `pip install requests` (in the right venv).",
  ["pip install"])
P("common_errors", "What does TypeError: 'NoneType' object is not subscriptable mean?",
  "You indexed into a NoneType value (None). Add a check: if value is not None: value[0]. The message means the object is None, not that its type is called NoneType as a real type name.",
  ["NoneType"])
P("common_errors", "What does IndexError: list index out of range mean?",
  "You accessed an index beyond the list length. Check bounds before indexing or use .get() for dicts.",
  ["IndexError"])
P("common_errors", "How do I fix AttributeError: 'dict' object has no attribute 'append'?",
  "dicts have no append. Use a list, or d[key] = value to add a mapping. If you need to collect values, use a list then map.",
  ["dict", "append"])
P("common_errors", "What does the error 'command not found: pytest' mean?",
  "pytest isn't on PATH. Install it in the venv: .venv/bin/pip install pytest, then use .venv/bin/pytest.",
  ["pytest"])

# ---- 20 torch basics ----
P("torch_basics", "Create a linear layer in PyTorch and call it on a batch of vectors.",
  "import torch\nimport torch.nn as nn\n\nlayer = nn.Linear(4, 2)\nx = torch.randn(8, 4)\ny = layer(x)\nprint(y.shape)   # torch.Size([8, 2])",
  ["nn.Linear"], run="\n", expected_stdout="")
P("torch_basics", "How do I compute gradients with autograd?",
  "x = torch.tensor([2.0], requires_grad=True)\ny = x ** 2\ny.backward()\nprint(x.grad)   # tensor([4.])",
  ["requires_grad", "backward"], run="import torch\nx=torch.tensor([2.0],requires_grad=True)\ny=x**2\ny.backward()\nprint(x.grad)\n", expected_stdout="tensor([4.])\n")
P("torch_basics", "How do I create a random tensor with a fixed shape?",
  "import torch\nx = torch.randn(3, 5)   # standard normal, shape (3,5)",
  ["randn"], run="\n", expected_stdout="")
P("torch_basics", "How do I move a tensor and keep gradients?",
  "x = torch.randn(4, requires_grad=True)\nx = x.detach().clone()  # copy without grad history",
  ["detach"])


# ---------------- split + write ----------------

def key_of(topic_idx, prompt):
    return hashlib.sha256(f"{topic_idx}|{prompt}".encode()).hexdigest()[:16]


def split_keys(all_records, val_frac, seed):
    """Deterministic hold-out: sample val_frac of unique (topic, prompt) keys."""
    rng = random.Random(seed)
    keys = sorted({key_of(r["meta"]["topic"], r["messages"][0]["content"]) for r in all_records})
    val_keys = set(rng.sample(keys, int(len(keys) * val_frac)))
    return val_keys


def build_records():
    import ast
    recs = []
    for ti, topic in enumerate(TOPICS):
        for (prompt, response, api, run, expected) in CUR.get(topic, []):
            # classify: code response (parses as Python or is an executable block)
            # vs prose answer. Syntax scoring only applies to code responses.
            kind = "code"
            try:
                ast.parse(response)
            except SyntaxError:
                kind = "prose"
            if run and kind == "prose":
                kind = "code"
            recs.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                "meta": {
                    "topic": ti,
                    "topic_name": topic,
                    "kind": kind,
                    "api": api,
                    "run": run,
                    "expected_stdout": expected,
                },
            })
    return recs


def write_chunked(out_dir, records, chunk_size=500):
    os.makedirs(out_dir, exist_ok=True)
    files, written, idx, chunk = [], 0, 0, []
    for rec in records:
        chunk.append(rec)
        if len(chunk) >= chunk_size:
            fn = os.path.join(out_dir, f"chunk_{idx:05d}.jsonl")
            with open(fn, "w", encoding="utf-8") as f:
                for r in chunk:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            files.append(os.path.basename(fn)); chunk = []; idx += 1
        written += 1
    if chunk:
        fn = os.path.join(out_dir, f"chunk_{idx:05d}.jsonl")
        with open(fn, "w", encoding="utf-8") as f:
            for r in chunk:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        files.append(os.path.basename(fn))
    return files, written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/yonoob/projects/ReTop/hmn_data/distill_python")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.2)
    args = ap.parse_args()

    recs = build_records()
    val_keys = split_keys(recs, args.val_frac, args.seed)
    train = [r for r in recs if key_of(r["meta"]["topic"], r["messages"][0]["content"]) not in val_keys]
    val = [r for r in recs if key_of(r["meta"]["topic"], r["messages"][0]["content"]) in val_keys]

    _, tr_n = write_chunked(os.path.join(args.out, "train"), train)
    _, va_n = write_chunked(os.path.join(args.out, "val"), val)
    runnable = sum(1 for r in recs if r["meta"]["run"])
    api_checked = sum(1 for r in recs if r["meta"]["api"])
    print(f"total={len(recs)} train={tr_n} val={va_n} "
          f"runnable={runnable} api_checked={api_checked}")
    per_topic = {}
    for r in recs:
        per_topic[r["meta"]["topic_name"]] = per_topic.get(r["meta"]["topic_name"], 0) + 1
    print("per-topic:", per_topic)


if __name__ == "__main__":
    main()
