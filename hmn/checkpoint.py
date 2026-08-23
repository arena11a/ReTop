"""Checkpoint loading with legacy-key tolerance (v6 slim-down shim).

Dead parameters removed from HMN3 in v6 (`ir.key_proj`, `ir.keys_proj`) still
exist inside every pre-v6 checkpoint file. `load_compat` strips those keys
before a strict load so old checkpoints keep working, and reports anything
else unexpected instead of failing silently.
"""
import warnings

import torch

LEGACY_PREFIXES = ("ir.key_proj.", "ir.keys_proj.")


def strip_legacy_keys(sd):
    dropped = [k for k in list(sd) if k.startswith(LEGACY_PREFIXES)]
    for k in dropped:
        del sd[k]
    return sd, dropped


def load_compat(model, path, device=None, map_location="cpu"):
    """torch.load + legacy-key strip + strict load.

    Returns (missing_keys, unexpected_keys) from the strict load; raises on
    any mismatch the way load_state_dict(strict=True) does after stripping
    only the KNOWN dead keys — unknown drift still fails loudly.
    """
    if isinstance(path, str):
        sd = torch.load(path, map_location=device or map_location)
        src = path
    else:
        sd = path
        src = "<state_dict>"
    sd, dropped = strip_legacy_keys(sd)
    if dropped:
        warnings.warn(f"{src}: dropped {len(dropped)} dead v5- keys "
                      f"({', '.join(sorted(set(k.split('.')[-2] for k in dropped)))})",
                      stacklevel=2)
    result = model.load_state_dict(sd)
    return result.missing_keys, result.unexpected_keys
