"""Backward-compat shim. Canonical module is hmn.v2 (`from hmn import HMN`).

Kept so legacy scripts (experiments/*) that `from hmn_v2 import HMN` keep working.
"""
from hmn.v2 import *  # noqa: F401,F403
