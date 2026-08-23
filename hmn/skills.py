"""v5 M4 — Executable Skill Library (the Ω-Coder "Skill Distiller" module).

A SKILL is a verified RUN RECIPE: a function that turns a parsed prompt
(segment boundaries + first-seed segment index k0) into an executable plan
[(anchor_col, run_len), ...] for the seam machinery.

Design contract (mirrors repo philosophy):
  - knowledge lives OUTSIDE weights as runnable artifacts: recipes are plain
    inspectable code, not weights;
  - nothing enters the library unverified: distillation requires a SOLVED
    instance (teacher-forced copy-argmax == gold on every planned row, gate
    open) before the recipe is accepted;
  - retrieval is content-addressed by a prompt FINGERPRINT (n_parts +
    segment token-counts) — identity addressing at the task level;
  - ambiguous fingerprints (same key, different family) raise instead of
    guessing — the "escalate" rule of Ω-Coder Module 6.

Recipes shipped:
  echo_recipe    : whole user region as ONE run anchored at the USER column
                   (v4 stem-addr semantics).
  rotate_recipe  : cyclic walk of segments starting at k0, separators as
                   1-token mini-runs (v5-M3 decode_rotate semantics).
"""
import random

import torch

from hmn.recipe import REORDER_AND, _find_all_word


# ---------------------------------------------------------------------------
# prompt parsing
# ---------------------------------------------------------------------------

def parse_prompt(ids, asid, tok, sep=REORDER_AND):
    """Segment structure of the USER region -> dict(bounds, ands, n_parts,
    seg_lens, asi_pos, fp). fp is the SLOT-INVARIANT retrieval fingerprint:
    (n_parts, first-token-id of each segment) — verbs identify the task
    template while slot contents stay wild."""
    asi_pos = ids.index(asid)
    U = ids[2:asi_pos]
    ands = _find_all_word(tok, U, sep)
    bounds = [-1] + ands + [len(U)]
    n_parts = len(ands) + 1
    seg_lens = [bounds[k + 1] - bounds[k] - 1 for k in range(n_parts)]
    seg_first = tuple(U[b + 1] for b in bounds[:-1])
    return {"asi_pos": asi_pos, "U": U, "ands": ands, "bounds": bounds,
            "n_parts": n_parts, "seg_lens": seg_lens, "seg_first": seg_first,
            "fp": (n_parts, seg_first)}


# ---------------------------------------------------------------------------
# recipes: (P, k0) -> plan [(anchor_col, run_len), ...]
# ---------------------------------------------------------------------------

def echo_recipe(P, k0=None):
    """Whole user region as ONE run anchored at the USER column (payload =
    first user token). Requires HMN3(stem_addr=True).
    NOTE: length is len(U) — INCLUDES separators (verbatim echo), unlike
    rotate recipes where separators are separate mini-runs."""
    return [(1, len(P["U"]))] if P["U"] else []


def rotate_recipe(P, k0):
    """Cyclic segment walk starting at segment k0 (rotate-left family).
    Same formulas as perm_anchors / decode_rotate (verified in M3)."""
    b = P["bounds"]
    n = P["n_parts"]
    plan = []
    for si in range(n):
        seg = (k0 + si) % n
        plan.append((b[seg] + 2, b[seg + 1] - b[seg] - 1))
        if si < n - 1:
            plan.append((1 + P["ands"][0], 1))
    return plan


RECIPES = {"echo": echo_recipe, "rotate": rotate_recipe}


def gold_k0(ids, gold_ids, P):
    """First-seed segment index implied by the GOLD's first token."""
    starts = [b + 1 for b in P["bounds"][:-1]]
    hits = [m for m, u in enumerate(P["U"]) if u == gold_ids[0]]
    for m in hits:
        for k, s in enumerate(starts):
            if m == s:
                return k
    for m in hits:
        for k in range(P["n_parts"]):
            if P["bounds"][k] < m <= P["bounds"][k + 1]:
                return k
    return 0


# ---------------------------------------------------------------------------
# library
# ---------------------------------------------------------------------------

class SkillLibrary:
    """Fingerprint-keyed store. One fp may hold SEVERAL families (echo vs
    swap share verbs); resolution: unique -> use it; multiple -> caller must
    pass family hint, else escalate; empty -> open-world fallback."""

    def __init__(self):
        self._skills = {}

    def add(self, family, fp, verified_on):
        bucket = self._skills.setdefault(fp, {})
        prev = bucket.get(family)
        entry = {"family": family, "fp": fp, "verified_on": verified_on}
        bucket[family] = entry
        return entry

    def match(self, fp, family=None):
        """-> (entry_or_None, status) with status in
        {'hit', 'hint', 'ambiguous', 'miss'}."""
        bucket = self._skills.get(fp)
        if not bucket:
            return None, "miss"
        if family is not None and family in bucket:
            return bucket[family], "hit"
        if len(bucket) == 1:
            return next(iter(bucket.values())), \
                "hit" if family is None else "hint"
        return None, "ambiguous"

    def __len__(self):
        return sum(len(b) for b in self._skills.values())

    def items(self):
        return sorted((fp, dict(b)) for fp, b in self._skills.items())


# ---------------------------------------------------------------------------
# verification / distillation
# ---------------------------------------------------------------------------

def verify_plan(model, seq, prompt_len, plan, device):
    """Teacher-forced verification of `plan` on one solved sequence.

    seq = [<s> <|user|> U... <|assistant|> GOLD </s>] (the training-format
    ids). Two gates:
      COVERAGE — planned emissions must total exactly len(gold); a short or
      long plan is rejected before any forward (catches off-by-one
      false-greens: echo's sum(seg_lens) once excluded separators).
      EXACTNESS — with every planned row force-anchored, copy_dist argmax
      == next gold id AND gate > 0.5 on all of them.
    Returns (ok: bool, detail: str, rows_checked: int).
    """
    gold_len = len(seq) - 1 - prompt_len
    if sum(L for _, L in plan) != gold_len:
        return False, (f"coverage {sum(L for _, L in plan)} != gold "
                       f"{gold_len}"), 0
    x = torch.tensor([seq], device=device)
    anch = torch.full((1, len(seq)), -100, dtype=torch.long, device=device)
    t = prompt_len - 1                      # row predicting gold[0]
    for (c, L) in plan:
        for j in range(L):
            if t >= len(seq) - 1:
                return False, "plan overruns sequence", 0
            anch[0, t] = c + j
            t += 1
    with torch.no_grad():
        out = model(x, seam_anchor=anch)
    cd, g = out["copy_dist"][0], out["g"][0]
    rows = 0
    for tt in range(len(seq)):
        c = anch[0, tt].item()
        if c < 0:
            continue
        if cd[tt].argmax(-1).item() != seq[tt + 1]:
            return False, f"row {tt}: copy argmax != gold", rows
        if float(g[tt]) <= 0.5:
            return False, f"row {tt}: gate closed on planned row", rows
        rows += 1
    return True, "all planned rows verified", rows


def execute(lib, model, tok, prompt_ids, device, max_new=64, sep=REORDER_AND,
            family=None):
    """Retrieve by fingerprint and run the stored recipe greedily.

    Resolution: unique family under fp -> use; multiple -> `family` hint or
    escalate; miss -> open-world fallback rotate(k0 = ptr-head segment),
    reported via meta['fallback'].
    Returns (text, meta).
    """
    asid = tok.token_to_id("<|assistant|>")
    eos = tok.token_to_id("</s>")
    P = parse_prompt(list(prompt_ids), asid, tok, sep=sep)
    entry, status = lib.match(P["fp"], family=family)
    fallback = status == "miss"
    if fallback:
        with torch.no_grad():
            out = model(torch.tensor([list(prompt_ids)], device=device))
        c0 = int(out["ptr_logits"][0, -1].argmax(-1).item())
        src = c0 + 1
        k0 = next((kk for kk in range(P["n_parts"])
                   if P["bounds"][kk] + 1 <= src <= P["bounds"][kk + 1]), 0)
        use_family = "rotate"
    else:
        if status == "ambiguous":
            raise ValueError(
                f"ambiguous fingerprint {P['fp']} — pass family hint "
                f"(escalate, do NOT guess)")
        use_family = entry["family"]
        k0 = 0 if use_family != "rotate" else (
            entry.get("k0") or 0)
    plan = RECIPES[use_family](P, k0)
    ids = list(prompt_ids)
    gates = []
    cur_c, cur_t, left = None, None, 0
    pi, in_run = 0, 0
    ans_len = max(0, len(prompt_ids) - 3)
    with torch.no_grad():
        while len(ids) - len(prompt_ids) < min(max_new, ans_len + 8):
            t_idx = len(ids) - len(prompt_ids)
            if t_idx >= ans_len:
                break
            inp = torch.tensor([ids], device=device)
            if left > 0:
                c = cur_c + (t_idx - cur_t)
                anch = torch.full((1, inp.shape[1]), -100,
                                  dtype=torch.long, device=device)
                anch[0, -1] = c
                out = model(inp, seam_anchor=anch)
                nxt = ids[c + 1]
                left -= 1
            else:
                out = model(inp)
                if pi >= len(plan):
                    break
                cur_c, L = plan[pi]
                pi += 1
                nxt = ids[cur_c + 1]
                cur_t = t_idx
                left = L - 1
            gates.append(float(out["g"][0, -1]))
            if nxt in (eos, asid):
                break
            ids.append(nxt)
    text = tok.decode(ids[len(prompt_ids):]).strip()
    meta = {"family": use_family, "fp": P["fp"], "status": status,
            "fallback": fallback,
            "gate_avg": sum(gates) / len(gates) if gates else 0.0}
    return text, meta
