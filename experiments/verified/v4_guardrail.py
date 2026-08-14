"""v4 guardrail (2026-08-13) — one command, all v4 regression gates.

Backs the README v4 table and the CI job (`docs/v4_roadmap.md` M10). Runs, in
order, and asserts:

  1. test_hmn                — model-level forward/backward/reversibility
  2. M1 sparse parity        — sparse copy-marginal bit-identical to dense
  3. v3.3 seed-42 40/40      — shipped hmn_v33.pt still exact on unseen
  4. M8 baseline smoke       — HMN3 (stem-addr) beats vanilla + NoReg on the
                              same-size slot task at short horizon
  5. M6 pos_eos              — repeated-digit slots 1.000 under pos_eos on a
                              freshly trained stem-addr checkpoint

Every step exits non-zero on failure, so `python
experiments/verified/v4_guardrail.py` fails loudly like CI.

Run: python experiments/verified/v4_guardrail.py
     # ~2-3 min CPU (shortened training horizons vs the milestone runs)
"""
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def run(cmd, cwd=ROOT):
    print(f">> {' '.join(cmd)}", flush=True)
    cp = subprocess.run([sys.executable] + cmd, cwd=cwd)
    if cp.returncode != 0:
        print(f"FAILED ({cp.returncode}): {' '.join(cmd)}")
        sys.exit(cp.returncode)


def main():
    run(["test_hmn.py"])
    run(["experiments/v4/m1_sparse_parity.py"])
    run(["experiments/verified/slot_v33_seed42.py"])
    run(["experiments/verified/slot_v4.py", "60"])   # short-horizon, still asserts 1.0
    run(["experiments/v4/m8_baseline.py", "--smoke", "--steps", "60", "--bs", "8"])
    print("\nV4 GUARDRAIL PASSED")


if __name__ == "__main__":
    main()