"""v6 M7 — torch.compile probe on the HMN3 forward (roadmap M4/M7 "free wins").

Roadmap context (docs/v6_scaling_roadmap.md §2.6): "torch.compile probe runs
in M4 to catch free wins early." M7's Triton work must beat whatever the
compiler gives for free, so this probe measures:

  1. availability:  hasattr(torch, 'compile') and a trivial compile round-trip
  2. whole-model:   eager vs torch.compile(HMN3) stats-path forward
                    — wall-clock speedup + numeric parity of gen_logits / g
  3. head region:   DualHeadDecoder.gate_and_gen alone compiled vs eager
                    — this Linear+log_softmax+gate is exactly the region the
                    planned fused dual-head kernel replaces; its share of the
                    model forward tells us the ceiling of kernel-level wins.

Honest boundary (no GPU / no Triton in this environment):
  * CPU-only numbers (torch +cpu build). Inductor-CPU is a DIFFERENT backend
    than Inductor-Triton; GPU conclusions are extrapolations only.
  * The stats path contains data-dependent Python branches (`(ids == asi_id)
    .any()`) → graph breaks are expected; we measure what compiles anyway,
    which is what a drop-in `torch.compile(model)` user would get.

Usage: .venv/bin/python experiments/v6/m7_compile_probe.py [--full]
  default probes the repo-standard guardrail config (dim=96/L=3).
  --full adds a D768-ish config (dim=768, L=2, short T) if RAM allows.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch

from hmn.v3 import HMN3

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
VOCAB = 3190


def bench(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def probe(name, dim, n_layers, B, T, iters=10):
    print(f"\n[probe: {name}]  dim={dim} L={n_layers} B={B} T={T} V={VOCAB}")
    torch.manual_seed(0)
    m = HMN3(VOCAB, dim=dim, state_dim=8, n_layers=n_layers,
             use_moe=False, aux_copy=False, asi_id=None)
    m.eval()
    ids = torch.randint(0, VOCAB, (B, T))

    with torch.no_grad():
        ref = m(ids)

    # ---- parity: compiled vs eager -------------------------------------
    mc = torch.compile(m)
    with torch.no_grad():
        out_c = mc(ids)          # first call = compile (untimed)
        d_gen = (out_c["gen_logits"] - ref["gen_logits"]).abs().max().item()
        d_g = (out_c["g"] - ref["g"]).abs().max().item()
        with_out = out_c["gen_logits"].clone()
        assert torch.equal(with_out, out_c["gen_logits"])
    print(f"  parity gen_logits max|Δ| = {d_gen:.3e}   g max|Δ| = {d_g:.3e}")

    # ---- timing ---------------------------------------------------------
    with torch.no_grad():
        t_eager = bench(lambda: m(ids), iters=iters)
        t_comp = bench(lambda: mc(ids), warmup=1, iters=iters)
    sp = t_eager / t_comp
    print(f"  eager    {t_eager * 1e3:8.2f} ms/iter")
    print(f"  compiled {t_comp * 1e3:8.2f} ms/iter   speedup x{sp:.2f}")

    # ---- head region share (the M7 fused-kernel target) -----------------
    h = torch.randn(B, T, dim)
    gm = torch.rand(B, T)
    behind = torch.zeros(B, T, dtype=torch.bool)
    nl = torch.arange(T).unsqueeze(0).expand(B, T).contiguous()
    with torch.no_grad():
        t_head_e = bench(lambda: m.dual.gate_and_gen(h, gm, behind, n_legal=nl),
                         iters=iters)
    dual_c = torch.compile(m.dual.gate_and_gen)
    with torch.no_grad():
        dual_c(h, gm, behind, n_legal=nl)
        t_head_c = bench(lambda: dual_c(h, gm, behind, n_legal=nl),
                         warmup=1, iters=iters)
    print(f"  head gate_and_gen: eager {t_head_e * 1e3:.2f} ms "
          f"({100 * t_head_e / t_eager:.0f}% of fwd) | compiled "
          f"{t_head_c * 1e3:.2f} ms  x{t_head_e / max(t_head_c, 1e-12):.2f}")
    return {"name": name, "dim": dim, "T": T, "B": B,
            "eager_ms": t_eager * 1e3, "compiled_ms": t_comp * 1e3,
            "speedup": sp, "d_gen": d_gen, "d_g": d_g,
            "head_ms": t_head_e * 1e3, "head_pct": 100 * t_head_e / t_eager}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="also probe a D768-shaped config")
    args = ap.parse_args()

    print("== v6 M7 torch.compile probe ==")
    print(f"torch {torch.__version__} | cuda={torch.cuda.is_available()} "
          f"| threads={torch.get_num_threads()} | cpus={os.cpu_count()}")
    ok1 = hasattr(torch, "compile")
    _ = torch.compile(lambda x: x)(torch.zeros(1))
    print(f"probe1 hasattr(torch,'compile'): {ok1} | probe2 trivial compile: OK")

    results = [probe("guardrail-std", dim=96, n_layers=3, B=2, T=128)]
    if args.full:
        results.append(probe("D768-ish", dim=768, n_layers=2, B=1, T=64))

    print("\n== summary ==")
    print(f"{'config':<14}{'eager ms':>10}{'comp ms':>10}{'speedup':>9}"
          f"{'d_gen':>10}{'d_g':>10}{'head%':>7}")
    for r in results:
        print(f"{r['name']:<14}{r['eager_ms']:>10.2f}{r['compiled_ms']:>10.2f}"
              f"{r['speedup']:>8.2f}x{r['d_gen']:>10.2e}{r['d_g']:>10.2e}"
              f"{r['head_pct']:>6.0f}%")
    print("\n(honest boundary: CPU/inductor-cpp backend; not a Triton/GPU result)")


if __name__ == "__main__":
    main()
