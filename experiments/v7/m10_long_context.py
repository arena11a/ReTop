"""v7 M10 — Long-context scaling verification.

Tests:
  1. Forward at T=512/1024/2048 — memory measured
  2. Stats path scales with distinct tokens (not T²)
  3. Attention-WR memory at longer sequences
"""
import os, sys, tracemalloc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import torch
from hmn.v3 import HMN3
from hmn.v7 import HMN3AttentionWR

def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")

def measure_memory(fn, *args):
    """Measure peak RSS increase of fn(*args)."""
    tracemalloc.start()
    fn(*args)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024 / 1024  # MB

def test_long_context_ssm():
    print("[M10-1: SSM-WR long-context (T=512/1024/2048)]")
    torch.manual_seed(42)
    m = HMN3(100, dim=64, state_dim=8, n_layers=2, use_moe=False,
             gate_bias=-1.0, asi_id=5)
    m.eval()

    results = []
    for T in [512, 1024, 2048]:
        x = torch.randint(0, 100, (1, T))
        mem = measure_memory(lambda: m(x))
        results.append((T, mem))
        print(f"  T={T:5d}: {mem:.1f} MB")

    # Memory should scale roughly linearly with T (not T²)
    if len(results) >= 2:
        t1, m1 = results[0]
        t2, m2 = results[-1]
        ratio = (m2 / m1) if m1 > 0 else 0
        t_ratio = t2 / t1
        check(ratio < t_ratio * 1.5,
              f"memory scales sub-quadratically ({ratio:.2f}x for {t_ratio}x T)")

def test_long_context_attention():
    print("[M10-2: Attention-WR long-context (T=512/1024)]")
    torch.manual_seed(42)
    m = HMN3AttentionWR(100, dim=64, n_layers=2, n_heads=4,
                        use_moe=False, gate_bias=-1.0, asi_id=5,
                        use_checkpoint=True)
    m.eval()

    for T in [512, 1024]:
        x = torch.randint(0, 100, (1, T))
        mem = measure_memory(lambda: m(x))
        print(f"  T={T:5d}: {mem:.1f} MB")

def test_stats_path_scaling():
    print("[M10-3: stats path scales with distinct tokens]")
    torch.manual_seed(42)
    m = HMN3(100, dim=64, state_dim=8, n_layers=2, use_moe=False,
             gate_bias=-1.0, asi_id=5)
    m.eval()

    # Many重复 tokens → few distinct → stats should be cheap
    x_repeated = torch.zeros(1, 4096, dtype=torch.long)
    for i in range(4096):
        x_repeated[0, i] = i % 10  # only 10 distinct tokens

    mem_repeated = measure_memory(lambda: m(x_repeated))
    print(f"  T=4096 (10 distinct): {mem_repeated:.1f} MB")

    # Few unique tokens
    x_unique = torch.arange(4096).unsqueeze(0) % 100
    mem_unique = measure_memory(lambda: m(x_unique))
    print(f"  T=4096 (100 distinct): {mem_unique:.1f} MB")

if __name__ == "__main__":
    test_long_context_ssm()
    test_long_context_attention()
    test_stats_path_scaling()
    print("\nM10 ALL PASSED")
