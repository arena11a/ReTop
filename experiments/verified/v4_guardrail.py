"""v4 guardrail (2026-08-13) — one command, all v4 regression gates.

Backs the README v4 table and the CI job (`docs/v4_roadmap.md` M10). Runs, in
order, and asserts:

  1. test_hmn                — model-level forward/backward/reversibility
  2. M1 sparse parity        — sparse copy-marginal bit-identical to dense
  3. v3.3 seed-42 40/40      — shipped hmn_v33.pt still exact on unseen
4. M8 baseline smoke       — HMN3 (stem-addr) beats vanilla + NoReg on the
                               same-size slot task at short horizon
    5. slot_v4                 — 10 trained + 4 probes + matrix 1.000 (+ pos_eos)
    6. chain_v4                — 2-slot 5 seeds hard+blend + 3-slot length-gen
                                all 1.000 under pos_eos; 3-slot control < 1.0
                                confirms the termination fix is what closes it

Every step exits non-zero on failure, so `python
experiments/verified/v4_guardrail.py` fails loudly like CI.

Run: python experiments/verified/v4_guardrail.py
     # ~3-4 min CPU (shortened training horizons vs the milestone runs)
     # --device cuda  passes the device down to every sub-step (default auto)
"""
import argparse
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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=None,
                    help="compute device for every sub-step (default auto-detect)")
    args = ap.parse_args()
    dev = [] if not args.device else ["--device", args.device]
    run(["test_hmn.py"])  # pure CPU model-level tests (no device plumbing)
    run(["experiments/v4/m1_sparse_parity.py"] + dev)
    run(["experiments/verified/slot_v33_seed42.py"] + dev)
    run(["experiments/verified/slot_v4.py", "60"] + dev)
    run(["experiments/verified/chain_v4.py", "60"] + dev)
    run(["experiments/v4/m8_baseline.py", "--smoke", "--steps", "60", "--bs", "8"] + dev)
    print("\nV4 GUARDRAIL PASSED")


if __name__ == "__main__":
    main()