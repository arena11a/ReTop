# Docs index

| Document | Status | What it is |
|---|---|---|
| [`hmn_v3_design.md`](hmn_v3_design.md) | **current** (v3.3 baseline) | Dual-register architecture: failure history, loss/decode rules, honest boundaries |
| [`data_prep.md`](data_prep.md) | **current** | Data formats for slot-copy / chat pairs / plain text + generator usage |
| [`v4_roadmap.md`](v4_roadmap.md) | **archived** (completed 2026-08-23) | v4 milestones: stem-addr, pos_eos/cycle_break, device support, M12 reorder wall |
| [`v5_omega_roadmap.md`](v5_omega_roadmap.md) | **current** (M1–M4 shipped; open items in §8) | Seam re-seeding, rotations, skill library — the Ω-Coder mapping |
| [`v6_scaling_roadmap.md`](v6_scaling_roadmap.md) | **active** (branch `v6`) | Foundation-architecture scaling: IR-as-index, HF packaging, BF16, distributed, kernels |

Rule of the repo: one roadmap per major version, frozen when the version
merges to `main`; new work opens a fresh file (`docs/v6_*.md`) rather than
extending archived ones.
