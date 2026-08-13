# Python + Build Episodic Memory Knowledge Base (Step 3)

"""Domain knowledge kept OUTSIDE the model's parameters, to be embedded into the
episodic memory bank (DifferentiableEpisodicMemory) at init. Each entry is a
factual, static API snippet — retrievable by content addressing rather than
memorized as weights.

Layout: JSON array of entries. Each entry:
  {"id": "venv-001", "topic": "venv", "key": "<short name>",
   "text": "<exact code/answer>", "tags": [...]}

The memory bank loads these as (key, value) pairs: key = embedding of the
topic/short name, value = embedding of the text. On query, content addressing
retrieves the closest snippet and the gate blends it into the output.
"""

KB = [
  {"id": "venv-001", "topic": "venv", "key": "create venv",
   "text": "python3 -m venv .venv", "tags": ["venv", "create", "virtualenv"]},
  {"id": "venv-002", "topic": "venv", "key": "activate venv linux",
   "text": "source .venv/bin/activate", "tags": ["venv", "activate", "linux"]},
  {"id": "venv-003", "topic": "venv", "key": "leave venv",
   "text": "deactivate", "tags": ["venv", "deactivate"]},
  {"id": "venv-004", "topic": "venv", "key": "venv why",
   "text": "Isolate per-project dependency versions so they do not conflict.",
   "tags": ["venv", "isolation", "why"]},

  {"id": "pip-001", "topic": "pip", "key": "freeze deps",
   "text": "pip freeze > requirements.txt", "tags": ["pip", "freeze", "requirements"]},
  {"id": "pip-002", "topic": "pip", "key": "install from requirements",
   "text": "pip install -r requirements.txt", "tags": ["pip", "install", "-r"]},
  {"id": "pip-003", "topic": "pip", "key": "upgrade package",
   "text": "pip install --upgrade <pkg>", "tags": ["pip", "upgrade"]},
  {"id": "pip-004", "topic": "pip", "key": "list packages",
   "text": "pip list", "tags": ["pip", "list"]},
  {"id": "pip-005", "topic": "pip", "key": "externally-managed fix",
   "text": "Error means the OS manages system Python. Create a venv: python3 -m venv .venv",
   "tags": ["pip", "externally-managed-environment", "fix"]},

  {"id": "py-001", "topic": "python", "key": "shebang",
   "text": "#!/usr/bin/env python3", "tags": ["shebang", "python3", "executable"]},
  {"id": "py-002", "topic": "python", "key": "check version",
   "text": "python3 --version", "tags": ["version"]},
  {"id": "py-003", "topic": "python", "key": "main guard",
   "text": "if __name__ == '__main__':", "tags": ["__main__", "entrypoint"]},

  {"id": "gradle-001", "topic": "gradle", "key": "build with stacktrace",
   "text": "./gradlew build --stacktrace", "tags": ["gradle", "build", "stacktrace"]},
  {"id": "gradle-002", "topic": "gradle", "key": "clean build",
   "text": "./gradlew clean build", "tags": ["gradle", "clean", "build"]},
  {"id": "gradle-003", "topic": "gradle", "key": "run one test class",
   "text": "./gradlew test --tests 'com.example.MyTest'", "tags": ["gradle", "test"]},
  {"id": "gradle-004", "topic": "gradle", "key": "wrapper why",
   "text": "The wrapper pins the Gradle version so all devs build identically.",
   "tags": ["gradle", "wrapper", "why"]},

  {"id": "pyproj-001", "topic": "pyproject", "key": "minimal pyproject",
   "text": "[build-system]\nrequires = [\"setuptools>=61\"]\nbuild-backend = \"setuptools.build_meta\"\n\n[project]\nname = \"mypkg\"\nversion = \"0.1.0\"\nrequires-python = \">=3.9\"",
   "tags": ["pyproject", "setuptools", "package"]},
  {"id": "pyproj-002", "topic": "pyproject", "key": "editable install",
   "text": "pip install -e .", "tags": ["pip", "-e", "editable", "dev"]},
  {"id": "pyproj-003", "topic": "pyproject", "key": "declare dependencies",
   "text": "[project]\ndependencies = [\"requests>=2.28\", \"numpy>=1.24\"]",
   "tags": ["pyproject", "dependencies"]},

  {"id": "fileio-001", "topic": "file_io", "key": "read line by line",
   "text": "with open('f.txt', encoding='utf-8') as f:\n    for line in f:\n        print(line.rstrip())",
   "tags": ["file", "read", "with"]},
  {"id": "fileio-002", "topic": "file_io", "key": "write safely",
   "text": "with open('out.txt', 'w', encoding='utf-8') as f:\n    f.write('hello\\n')",
   "tags": ["file", "write", "with"]},
  {"id": "fileio-003", "topic": "file_io", "key": "append to file",
   "text": "with open('log.txt', 'a', encoding='utf-8') as f:\n    f.write('line\\n')",
   "tags": ["file", "append", "write"]},

  {"id": "pathlib-001", "topic": "pathlib", "key": "join paths",
   "text": "from pathlib import Path\np = Path('data') / 'sub' / 'f.csv'",
   "tags": ["pathlib", "join", "cross-platform"]},
  {"id": "pathlib-002", "topic": "pathlib", "key": "file exists",
   "text": "Path('config.json').exists()", "tags": ["pathlib", "exists"]},
  {"id": "pathlib-003", "topic": "pathlib", "key": "rglob files",
   "text": "for p in Path('src').rglob('*.py'): print(p)", "tags": ["pathlib", "rglob", "glob"]},

  {"id": "str-001", "topic": "string_format", "key": "float 2 decimals",
   "text": "f'{value:.2f}'", "tags": ["format", "float", "decimals"]},
  {"id": "str-002", "topic": "string_format", "key": "zero pad",
   "text": "f'{7:03d}'  # 007", "tags": ["format", "padding", "zero"]},

  {"id": "comp-001", "topic": "comprehensions", "key": "dict square",
   "text": "squares = {x: x*x for x in range(1, 6)}", "tags": ["comprehension", "dict"]},
  {"id": "comp-002", "topic": "comprehensions", "key": "filter even",
   "text": "evens = [n for n in nums if n % 2 == 0]", "tags": ["comprehension", "filter"]},

  {"id": "func-001", "topic": "functions", "key": "varargs sum",
   "text": "def total(*args):\n    return sum(args)", "tags": ["function", "*args"]},
  {"id": "func-002", "topic": "functions", "key": "default arg",
   "text": "def greet(name, greeting='Hello'): ...", "tags": ["function", "default"]},
  {"id": "func-003", "topic": "functions", "key": "kwargs",
   "text": "def config(**kwargs): ...", "tags": ["function", "**kwargs"]},

  {"id": "err-001", "topic": "error_handling", "key": "catch filenotfound",
   "text": "try:\n    open('missing.txt')\nexcept FileNotFoundError:\n    print('not found')",
   "tags": ["try", "except", "FileNotFoundError"]},
  {"id": "err-002", "topic": "error_handling", "key": "multi except",
   "text": "except (ValueError, ZeroDivisionError):", "tags": ["except", "multiple"]},
  {"id": "err-003", "topic": "error_handling", "key": "reraise",
   "text": "except Exception as e:\n    raise  # preserves traceback", "tags": ["raise", "traceback"]},

  {"id": "debug-001", "topic": "debugging", "key": "pdb breakpoint",
   "text": "import pdb; pdb.set_trace()", "tags": ["pdb", "debug"]},
  {"id": "debug-002", "topic": "debugging", "key": "print traceback",
   "text": "import traceback\ntraceback.print_exc()", "tags": ["traceback", "debug"]},

  {"id": "mod-001", "topic": "imports_modules", "key": "main guard",
   "text": "if __name__ == '__main__':", "tags": ["import", "__main__"]},
  {"id": "mod-002", "topic": "imports_modules", "key": "add to sys.path",
   "text": "import sys, os\nsys.path.insert(0, os.path.dirname(__file__))",
   "tags": ["sys.path", "import", "relative"]},

  {"id": "cls-001", "topic": "classes_dunder", "key": "repr example",
   "text": "class Point:\n    def __repr__(self):\n        return f'Point({self.x}, {self.y})'",
   "tags": ["__repr__", "class"]},
  {"id": "cls-002", "topic": "classes_dunder", "key": "len dunder",
   "text": "def __len__(self):\n    return len(self.items)", "tags": ["__len__", "dunder"]},
  {"id": "cls-003", "topic": "classes_dunder", "key": "property",
   "text": "@property\ndef area(self):\n    return 3.14159 * self.r ** 2",
   "tags": ["property", "read-only"]},

  {"id": "col-001", "topic": "itertools_collections", "key": "counter",
   "text": "from collections import Counter\ncounts = Counter(items)",
   "tags": ["Counter", "count"]},
  {"id": "col-002", "topic": "itertools_collections", "key": "most common",
   "text": "Counter('aaabbc').most_common(2)", "tags": ["most_common"]},
  {"id": "col-003", "topic": "itertools_collections", "key": "pairwise",
   "text": "from itertools import pairwise\nlist(pairwise([1, 2, 3, 4]))",
   "tags": ["pairwise", "adjacent"]},
  {"id": "col-004", "topic": "itertools_collections", "key": "chain",
   "text": "from itertools import chain\nlist(chain([1, 2], [3, 4]))", "tags": ["chain"]},

  {"id": "typing-001", "topic": "typing_hints", "key": "optional return",
   "text": "from typing import Optional\ndef f() -> Optional[str]: ...",
   "tags": ["Optional", "typing"]},
  {"id": "typing-002", "topic": "typing_hints", "key": "union",
   "text": "from typing import Union\ndef f(x: Union[int, str]) -> str: ...",
   "tags": ["Union", "typing"]},
  {"id": "typing-003", "topic": "typing_hints", "key": "dict list hints",
   "text": "from typing import Dict, List\ndef f(data: List[int]) -> Dict[str, int]: ...",
   "tags": ["Dict", "List", "typing"]},

  {"id": "subproc-001", "topic": "subprocess", "key": "run capture",
   "text": "import subprocess\nr = subprocess.run(['ls'], capture_output=True, text=True)\nprint(r.stdout)",
   "tags": ["subprocess", "capture_output"]},
  {"id": "subproc-002", "topic": "subprocess", "key": "check exit",
   "text": "subprocess.run(['cmd'], check=True)  # raises on non-zero",
   "tags": ["subprocess", "check=True"]},

  {"id": "pytest-001", "topic": "pytest", "key": "simple test",
   "text": "def test_add():\n    assert add(2, 3) == 5", "tags": ["pytest", "test", "assert"]},
  {"id": "pytest-002", "topic": "pytest", "key": "fixture tmp file",
   "text": "@pytest.fixture\ndef p(tmp_path):\n    return tmp_path / 'd.txt'",
   "tags": ["pytest", "fixture", "tmp_path"]},
  {"id": "pytest-003", "topic": "pytest", "key": "parametrize",
   "text": "@pytest.mark.parametrize('a,b,e', [(1,2,3), (2,3,5)])",
   "tags": ["pytest", "parametrize"]},

  {"id": "errcom-001", "topic": "common_errors", "key": "modulenotfound fix",
   "text": "ModuleNotFoundError: run pip install <pkg> in the correct venv.",
   "tags": ["ModuleNotFoundError", "fix"]},
  {"id": "errcom-002", "topic": "common_errors", "key": "nonetype subscript",
   "text": "TypeError: NoneType not subscriptable -> check 'if value is not None'.",
   "tags": ["NoneType", "TypeError", "fix"]},
  {"id": "errcom-003", "topic": "common_errors", "key": "indexerror",
   "text": "IndexError: check bounds before indexing.", "tags": ["IndexError", "fix"]},
  {"id": "errcom-004", "topic": "common_errors", "key": "dict no append",
   "text": "AttributeError: dict has no append -> use a list, or d[k] = v.",
   "tags": ["AttributeError", "dict", "fix"]},

  {"id": "torch-001", "topic": "torch_basics", "key": "linear layer",
   "text": "import torch\nimport torch.nn as nn\nlayer = nn.Linear(4, 2)\ny = layer(torch.randn(8, 4))",
   "tags": ["torch", "nn.Linear"]},
  {"id": "torch-002", "topic": "torch_basics", "key": "autograd",
   "text": "x = torch.tensor([2.0], requires_grad=True)\ny = x ** 2\ny.backward()\nprint(x.grad)",
   "tags": ["torch", "autograd", "backward"]},
  {"id": "torch-003", "topic": "torch_basics", "key": "random tensor",
   "text": "x = torch.randn(3, 5)", "tags": ["torch", "randn"]},
  {"id": "torch-004", "topic": "torch_basics", "key": "detach clone",
   "text": "x = x.detach().clone()  # copy without grad history",
   "tags": ["torch", "detach", "grad"]},
]

if __name__ == "__main__":
    import json
    print(json.dumps(KB, ensure_ascii=False, indent=2)[:400])
    print("...")
    print(f"total entries: {len(KB)}")
