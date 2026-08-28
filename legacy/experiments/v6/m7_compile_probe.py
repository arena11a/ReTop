"""v6 M7 — torch.compile probe on HMN3 forward (CPU, honest boundary).

Roadmap item (docs/v6_scaling_roadmap.md M7 / §2.6): run torch.compile on the
HMN3 stats-path forward, report wall-clock delta and numeric drift vs eager.
Design notes live in docs/v6_m7_triton_design.md; no Triton kernels exist yet.

Run: .venv/bin/python experiments/v6/m7_compile_probe.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch

from hmn.v3 import HMN3
from hmn.recipe import make_chat_ids, seed_guardrail, ASSIST
from tokenizers import Tokenizer

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
TOL = 1e-3


def main():
    print(f"torch {torch.__version__}  "
          f"hasattr(torch,'compile')={hasattr(torch, 'compile')}  "
          f"threads={torch.get_num_threads()}")

    tok = Tokenizer.from_file(os.path.join(ROOT, "retop_tokenizer.json"))
    asid = tok.token_to_id(ASSIST)

    # cpu-small spec (M4 presets), fixed shapes so dynamo guards stay stable
    ids = make_chat_ids(tok, "pip install pkg000", "pip install pkg000")
    X = torch.tensor([ids], dtype=torch.long)

    seed_guardrail(0)
    model = HMN3(vocab_size=3190, dim=96, state_dim=8, n_layers=3,
                 n_experts=16, top_k=2, use_moe=False, gate_bias=-1.0,
                 asi_id=asid)
    model.eval()

    def bench(fn):
        """Median over 3 batches of N calls (CPU wall clock is noisy)."""
        reps = []
        with torch.no_grad():
            for _ in range(10):
                fn(X)                             # warmup
            for _ in range(3):
                t0 = time.perf_counter()
                for _ in range(N):
                    out = fn(X)
                reps.append((time.perf_counter() - t0) * 1e3 / N)
        return sorted(reps)[1], out

    N = 100
    eager_ms, out_e = bench(model)

    cmodel = torch.compile(model, dynamic=False)
    with torch.no_grad():
        t0 = time.perf_counter()
        out_c = cmodel(X)                         # first call: compilation
        compile_s = time.perf_counter() - t0
    comp_ms, out_c = bench(cmodel)

    d_gen = (out_e["gen_logits"] - out_c["gen_logits"]).abs().max().item()
    d_g = (out_e["g"] - out_c["g"]).abs().max().item()

    print(f"\n=== v6 M7 torch.compile probe — HMN3 forward (stats path, CPU) ===")
    print(f"spec : dim=96 L=3 state=8 vocab=3190 T={X.shape[1]} B=1 eval")
    print(f"eager     : {eager_ms:8.2f} ms/fwd")
    print(f"compiled  : {comp_ms:8.2f} ms/fwd   (first-call compile {compile_s:.1f}s)")
    print(f"speedup   : {eager_ms / comp_ms:.2f}x")
    print(f"max|dgen| : {d_gen:.3e}")
    print(f"max|dg|   : {d_g:.3e}")
    ok = d_gen <= TOL and d_g <= TOL
    print("NUMERICS:", "PASS (<=1e-3)" if ok else f"FAIL (> {TOL:g})")

    # Known dynamo behavior (docs/v6_m7_triton_design.md §3): IRStats' group
    # count G is data-dependent -> graph break around IRStats; the rest compiles.
    # Do NOT set TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1: Inductor then fails with
    # GuardOnDataDependentSymNode on Eq(u0 + 1, 1).
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
