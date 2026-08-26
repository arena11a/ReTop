"""v3.3 training recipe — single source of truth for the slot-copy pipeline.

Every entry point (train_v3.py, retop.py, infer.py) must call THESE functions so
the loss math and the decode rule can never drift apart. Historically the recipe
lived in three places at once and the copy-CE bug (feeding a probability
distribution into CrossEntropyLoss, which log-softmaxes it AGAIN and pins the
loss at ~ln(VOCAB) with no gradient — see docs/hmn_v3_design.md "root causes")
kept resurfacing. Centralizing it here makes that impossible.

Recipe (all measured 2026-08-13, 200/200 = 5 seeds x 40 unseen slots):

  1. make_slot_batch   — teacher-forced slot-copy batch, identity-register labels
  2. loss_v33          — blend CE + gen CE (masked to gen rows) + manual copy CE
  3. decode_v33        — greedy decode with the deterministic boundary_eos rule
  4. eval_slots        — exact-match accuracy on a slot list (+ seed guardrail)

Tokenizers: chat specials are encoded PER-PART and joined (never
encode("<s>..user..<|assistant|>") in one call — the ReTop BPE splits specials
when concatenated with text, yielding wrong prompt ids).
"""
import math
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

BOS = "<s>"
EOS = "</s>"
UNK = "<unk>"
PAD = "<pad>"
USER = "<|user|>"
ASSIST = "<|assistant|>"

SPECIAL_TOKENS = [BOS, EOS, UNK, PAD, USER, ASSIST]


def _pad_batch(seqs, extra_seqs=None, pad_vals=None, device=None):
    """Pad variable-length sequences into a batch tensor.

    seqs: list of list[int] — the main sequences (e.g. token ids)
    extra_seqs: optional list of list — additional per-position arrays
                (targets, gate targets, anchors, etc.)
    pad_vals: dict mapping index to pad value for each array
              {0: eos_id for seqs, 1: -100 for Y, ...}
    device: torch device

    Returns list of tensors: [Xb, Yb, ...] aligned with seqs + extra_seqs.
    """
    if pad_vals is None:
        pad_vals = {}
    dev = resolve_device(device)
    T = max(len(x) for x in seqs)
    all_seqs = [seqs] + ([extra_seqs] if extra_seqs else [])
    result = []
    for si, group in enumerate(all_seqs):
        if group is None:
            continue
        pv = pad_vals.get(si, 0)
        Tg = max(len(x) for x in group) if group else T
        Tg = max(Tg, T)  # all groups share the same T
        xb = torch.full((len(group), Tg), pv,
                         dtype=torch.long if isinstance(pv, int) else torch.float,
                         device=dev)
        for j, x in enumerate(group):
            xb[j, :len(x)] = torch.tensor(x, device=dev,
                                           dtype=xb.dtype)
        result.append(xb)
    return result


def resolve_device(device=None):
    """Pick the compute device for the session.

    Resolution order:
      1. explicit `device` argument (torch.device or str),
      2. the RETOP_DEVICE env var (e.g. RETOP_DEVICE=cuda:0),
      3. auto-detect: cuda -> mps -> cpu (first one available).

    Everything below (batch builders, evals, decode) threads the result so a
    GPU is actually used when present; a CPU-only machine falls back to cpu.
    """
    if device is None:
        device = os.environ.get("RETOP_DEVICE", "auto")
    if device in ("auto", "auto-detect"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


# default chat-template used by make_chat_ids / decode (override via args)
DEFAULT_TEMPLATE = "pip install {slot}"
# v4 M4 multi-step task: the answer runs TWO distinct copy chains in sequence
# ("fetch pkg028 and deploy lib055"). slot A and slot B are drawn from
# DIFFERENT families (pkg-prefix vs lib-prefix, disjoint digit ranges) so
# their token ids never collide — the identity register can then bind each
# segment unambiguously (the original pkgA/pkgB variant shared 'p','k','g'/digit
# subtokens, making the second chain a coin-flip at 50%). The register must
# re-seed mid-answer on "and deploy": the multi-step property.
CHAIN_A_TEMPLATE = "fetch {slot}"
CHAIN_B_TEMPLATE = "and deploy {slot}"
CHAIN_SLOTS_A = [f"pkg{i:03d}" for i in range(0, 40)]       # train
CHAIN_SLOTS_B = [f"lib{i:03d}" for i in range(40, 80)]      # train
CHAIN_SLOTS_A_U = [f"pkg{i:03d}" for i in range(60, 100)]   # eval unseen
CHAIN_SLOTS_B_U = [f"lib{i:03d}" for i in range(0, 40)]     # eval unseen


def make_chat_ids(tok, user, gold=None):
    """Encode <s><|user|>{user}<|assistant|>[{gold}</s>] with PER-PART encoding.

    NOTE (root cause, 2026-08-12): this tokenizer's BPE splits specials when
    they are concatenated with text — encode("<s>..<|assistant|>..</s>") yields
    <|assistant|> -> [869, 21, 5] and </s> -> [869, 21, 1] (NOT single ids 5/1).
    Encoding each part separately and joining ids keeps every special as its
    exact single token, so the answer region / EOS boundary are clean.
    """
    bos, uid = tok.token_to_id(BOS), tok.token_to_id(USER)
    asid, eos = tok.token_to_id(ASSIST), tok.token_to_id(EOS)
    ids = [bos, uid] + tok.encode(user).ids + [asid]
    if gold is not None:
        ids += tok.encode(gold).ids + [eos]
    return ids


def make_chat_targets(ids, asid, eos, stem_row0=False):
    """Per-sequence answer-region labels -> (y, yc, gt).

    y  : shifted targets (all answer tokens incl. EOS) for blend CE
    yc : copy-channel targets; EOS and the FIRST answer token (seed=ASI, absent
         from the prompt) are -100 — the register cannot copy them by design.
         v4 M2-dev stem_row0: with stem-addressing the ASI boundary row's
         attention is anchored onto the USER column, so its payload (the
         template's first token) IS copyable — enable the copy target there.
    gt : 1.0 if the target exists as a copyable prompt payload, 0.0 otherwise
         (EOS / first token / absent). Drives the gen-CE mask and the loss.
    """
    T = len(ids)
    targets = ids[1:] + [eos]
    a = ids.index(asid)                      # first answer position
    prompt = ids[1:a]                        # user region only (excl <s>, asi)
    y, yc, gt = [-100] * T, [-100] * T, [-1.0] * T
    for t in range(a, T):
        y[t] = targets[t]
        if targets[t] == eos:
            yc[t] = -100
            gt[t] = 0.0
            continue
        if t == a and not stem_row0:
            yc[t] = -100
            gt[t] = 0.0
            continue
        yc[t] = targets[t]
        # v3.3 pointer: copy payload = token at prompt position j whose NEXT
        # (j+1) equals the target — the model looks up 'what follows the token
        # I just emitted' (raw-identity match on prev token, exact for unique
        # prompt tokens, holds for UNSEEN slot values).
        hit = next((j for j in range(len(prompt) - 1)
                    if prompt[j + 1] == targets[t]), None)
        if hit is None:
            gt[t] = 0.0                      # target absent from prompt -> gen
        else:
            gt[t] = 1.0
    return y, yc, gt


def make_slot_batch(tok, slots, bs, seed, template=DEFAULT_TEMPLATE,
                    templates=None, stem_row0=False, device=None):
    """Slot-copy batch -> (X, Y, Yc, G).

    X: full teacher-forced <s><|user|>{template}<|assistant|>{template}</s>
    Y: shifted targets, -100 outside answer region (blend CE, incl. EOS)
    Yc: copy-channel targets, EOS/first-token forced -100 (register can't copy)
    G: gate target 1.0/0.0/-1.0 (see _slot_targets) — masks gen CE in loss_v33

    v4 M3: pass `templates=[...]` to train on MANY templates in the same batch
    (per-record rng.choice). This un-locks the gate from a single lexicon:
    the gen head learns to OPEN the copy chain for any trained verb, and the
    copy lane sees copy-conditions across templates. `template=` is kept for
    v3.3 backward compat (single-template behavior unchanged).
    """
    rng = random.Random(seed)
    eos = tok.token_to_id(EOS)
    asid = tok.token_to_id(ASSIST)
    tpls = templates if templates is not None else [template]
    X, Y, YC, G = [], [], [], []
    for _ in range(bs):
        p = rng.choice(slots)
        tpl = rng.choice(tpls)
        user = gold = tpl.format(slot=p)
        ids = make_chat_ids(tok, user, gold)
        y, yc, gt = make_chat_targets(ids, asid, eos, stem_row0=stem_row0)
        X.append(ids); Y.append(y); YC.append(yc); G.append(gt)
    dev = resolve_device(device)
    T = max(len(x) for x in X)
    eos_t = torch.tensor(eos, device=dev)
    Xb = torch.full((bs, T), eos, dtype=torch.long, device=dev)
    Yb = torch.full((bs, T), -100, dtype=torch.long, device=dev)
    YcB = torch.full((bs, T), -100, dtype=torch.long, device=dev)
    Gb = torch.full((bs, T), -1.0, dtype=torch.float, device=dev)
    for j in range(bs):
        Xb[j, :len(X[j])] = torch.tensor(X[j], device=dev)
        Yb[j, :len(Y[j])] = torch.tensor(Y[j], device=dev)
        YcB[j, :len(YC[j])] = torch.tensor(YC[j], device=dev)
        Gb[j, :len(G[j])] = torch.tensor(G[j], device=dev)
    return Xb, Yb, YcB, Gb


def make_slot_chain_batch(tok, bs, seed, stem_row0=False, device=None):
    """v4 M4 multi-step batch -> (X, Y, Yc, G).

    user = "fetch {a} and deploy {b}"   (a ∈ pkg-family, b ∈ lib-family)
    gold = same. a,b re-drawn per record (sampling from the disjoint families)
    so no fixed pairing is memorizable. Two independent copy chains in ONE
    answer: the register must chain "fetch pkg028", then RE-SEED on
    "and deploy lib055" mid-answer. Token sets of a and b are disjoint, so the
    second chain is addressable — the difficulty is the re-seed itself.
    """
    rng = random.Random(seed)
    eos = tok.token_to_id(EOS)
    asid = tok.token_to_id(ASSIST)
    X, Y, YC, G = [], [], [], []
    for _ in range(bs):
        a = rng.choice(CHAIN_SLOTS_A)
        b = rng.choice(CHAIN_SLOTS_B)
        user = f"fetch {a} and deploy {b}"
        gold = user
        ids = make_chat_ids(tok, user, gold)
        y, yc, gt = make_chat_targets(ids, asid, eos, stem_row0=stem_row0)
        X.append(ids); Y.append(y); YC.append(yc); G.append(gt)
    dev = resolve_device(device)
    T = max(len(x) for x in X)
    Xb = torch.full((bs, T), eos, dtype=torch.long, device=dev)
    Yb = torch.full((bs, T), -100, dtype=torch.long, device=dev)
    YcB = torch.full((bs, T), -100, dtype=torch.long, device=dev)
    Gb = torch.full((bs, T), -1.0, dtype=torch.float, device=dev)
    for j in range(bs):
        Xb[j, :len(X[j])] = torch.tensor(X[j], device=dev)
        Yb[j, :len(Y[j])] = torch.tensor(Y[j], device=dev)
        YcB[j, :len(YC[j])] = torch.tensor(YC[j], device=dev)
        Gb[j, :len(G[j])] = torch.tensor(G[j], device=dev)
    return Xb, Yb, YcB, Gb


# ---------------------------------------------------------------------------
# v5 omega-seam: reorder task (the M12 wall) with fragment-run anchoring
# ---------------------------------------------------------------------------

REORDER_AND = "and"


def _find_word(tok, ids_list, word):
    """Index of the first single token decoding exactly to `word` (stripped)."""
    for j, t in enumerate(ids_list):
        if tok.decode([t]).strip() == word:
            return j
    return -1


def make_reorder_ids(tok, a_s, b_s):
    """Build teacher-forced reorder ids from PROMPT token variants.

    ByteLevel BPE gives mid-prompt words their space-prefixed id (' de') while
    an independently encoded gold starts 'de' — a different id with identical
    decoded text. Identity addressing needs EXACT ids, so the swapped answer
    is assembled FROM the user region tokens:  Gt = U[i_u+1:] + [U[i_u]] + U[:i_u]
    (decodes to "deploy {b} and fetch {a}" after strip; every row copyable).
    Returns (ids, asi_pos, i_u).
    """
    bos = tok.token_to_id(BOS)
    uid = tok.token_to_id(USER)
    asid = tok.token_to_id(ASSIST)
    eos = tok.token_to_id(EOS)
    U = list(tok.encode(f"fetch {a_s} and deploy {b_s}").ids)
    i_u = _find_word(tok, U, REORDER_AND)
    if i_u <= 0:
        raise AssertionError("make_reorder_ids: 'and' not found as a single token")
    Gt = U[i_u + 1:] + [U[i_u]] + U[:i_u]
    ids = [bos, uid] + U + [asid] + Gt + [eos]
    return ids, 2 + len(U), i_u


def _find_all_word(tok, ids_list, word):
    return [j for j, t in enumerate(ids_list)
            if tok.decode([t]).strip() == word]


def make_perm_ids(tok, parts, sep=REORDER_AND):
    """v5 M3: N-segment ROTATION ids (generalization of make_reorder_ids).

    user = "p1 sep p2 ... sep pn"; gold = rotate-left [p2..pn, p1], assembled
    FROM prompt token variants so every answer row keeps an exact identity
    twin. Works for arbitrary N >= 2 and arbitrary verb phrases per part.
    Returns (ids, asi_pos, and_positions).
    """
    bos = tok.token_to_id(BOS)
    uid = tok.token_to_id(USER)
    asid = tok.token_to_id(ASSIST)
    eos = tok.token_to_id(EOS)
    user = f" {sep} ".join(parts)
    U = list(tok.encode(user).ids)
    ands = _find_all_word(tok, U, sep)
    if len(ands) != len(parts) - 1:
        raise AssertionError(
            f"make_perm_ids: expected {len(parts) - 1} '{sep}' tokens, got {len(ands)}")
    if any(ands[i] >= ands[i + 1] for i in range(len(ands) - 1)):
        raise AssertionError("make_perm_ids: separators not strictly increasing")
    bounds = [-1] + ands + [len(U)]
    segs = [U[bounds[k] + 1: bounds[k + 1]] for k in range(len(parts))]
    if any(len(s) == 0 for s in segs):
        raise AssertionError("make_perm_ids: empty segment")
    order = list(range(1, len(parts))) + [0]
    Gt = []
    for i, oi in enumerate(order):
        if i > 0:
            Gt.append(U[ands[0]])
        Gt.extend(segs[oi])
    return [bos, uid] + U + [asid] + Gt + [eos], 2 + len(U), ands


def perm_anchors(ids, asid, tok, sep=REORDER_AND):
    """v5 M3: N-run anchors for ids built by make_perm_ids.

    Anchor formulas (derived once, valid for all N):
      segment-token row (segment k, offset j): anchor col = bounds[k] + 2 + j
      separator row:                           anchor col = 1 + ands[0]
    Seam/run extraction is structural (anchor discontinuity), so run count,
    lengths and permutation shape need NO task-side hardcoding.
    Same contract as reorder_anchors: (anchors, seams, runs).
    """
    asi_pos = ids.index(asid)
    U = ids[2:asi_pos]
    Gt = ids[asi_pos + 1:len(ids) - 1]
    ands = _find_all_word(tok, U, sep)
    if not ands:
        raise AssertionError("perm_anchors: no separator token found")
    n_parts = len(ands) + 1
    bounds = [-1] + ands + [len(U)]
    segs = [U[bounds[k] + 1: bounds[k + 1]] for k in range(n_parts)]
    order = list(range(1, n_parts)) + [0]
    exp = []
    for i, oi in enumerate(order):
        if i > 0:
            exp.append(U[ands[0]])
        exp.extend(segs[oi])
    if Gt != exp:
        raise AssertionError("perm_anchors: ids were not built by make_perm_ids")
    cs = []
    for i, oi in enumerate(order):
        if i > 0:
            cs.append(1 + ands[0])
        for j in range(len(segs[oi])):
            cs.append(bounds[oi] + 2 + j)
    n_g = len(Gt)
    T = len(ids)
    anchors = [-100] * T
    seams = [False] * T
    runs = [-100] * T
    seam_rows = [r for r in range(n_g) if r == 0 or cs[r] != cs[r - 1] + 1]
    for si, r0 in enumerate(seam_rows):
        r1 = seam_rows[si + 1] if si + 1 < len(seam_rows) else n_g
        anchors[asi_pos + r0] = cs[r0]
        seams[asi_pos + r0] = True
        runs[asi_pos + r0] = max(0, (r1 - r0) - 1)
    for r in range(n_g):
        if r not in seam_rows:
            anchors[asi_pos + r] = cs[r]
    return anchors, seams, runs


def reorder_anchors(ids, asid, tok):
    """Fragment-run anchors for the reorder task (v5 omega-seam).

    The answer is THREE copy runs over the prompt in swapped order:
      run 0: "{b}-tail"    <- payload chain echoing through U[i_u+1..]
      run 1: "and"         <- column i_u-1's payload (the 'and' token itself)
      run 2: "fetch {a}"   <- payload chain restarting at the USER column
    Anchor semantics match the register: row t copies ids[c+1] when attention
    is forced onto column c. Returns (anchors, seams, runs) aligned to full
    `ids` length:
      anchors[t] anchor column c (-100 where not forced)
      seams[t]   True on run-start rows (SeedPointer is supervised there)
      runs[t]    run-length class target (length-1) on seam rows, else -100
    Raises AssertionError if the swap does not decompose exactly (tokenizer
    drift guard — same invariant style as make_chat_ids).
    """
    asi_pos = ids.index(asid)
    U = ids[2:asi_pos]                       # user region tokens
    Gt = ids[asi_pos + 1:len(ids) - 1]       # gold tokens (strip eos)
    i_u = _find_word(tok, U, REORDER_AND)
    if i_u <= 0:
        raise AssertionError("reorder_anchors: 'and' not found as a single token")
    if Gt != U[i_u + 1:] + [U[i_u]] + U[:i_u]:
        raise AssertionError("reorder_anchors: swap does not decompose into "
                             "contiguous prompt runs (use make_reorder_ids)")
    n_g = len(Gt)
    i_g = len(U) - i_u - 1                   # index of 'and' within Gt
    cs = []
    for r in range(n_g):
        if r < i_g:
            cs.append(2 + i_u + r)           # echo through "deploy {b}"
        elif r == i_g:
            cs.append(1 + i_u)               # payload of col before ' and'
        else:
            cs.append(1 + (r - i_g - 1))     # restart at USER col, echo "{a}"
    T = len(ids)
    anchors = [-100] * T
    seams = [False] * T
    runs = [-100] * T
    seam_rows = [r for r in range(n_g)
                 if r == 0 or cs[r] != cs[r - 1] + 1]
    for si, r0 in enumerate(seam_rows):
        r1 = seam_rows[si + 1] if si + 1 < len(seam_rows) else n_g
        t_row = asi_pos + r0
        anchors[t_row] = cs[r0]
        seams[t_row] = True
        runs[t_row] = max(0, (r1 - r0) - 1)  # class index = length-1
    # non-seam answer rows still need their forced echo anchor
    for r in range(n_g):
        if r not in seam_rows:
            anchors[asi_pos + r] = cs[r]
    return anchors, seams, runs


def make_reorder_batch(tok, a_slots, b_slots, bs, seed, device=None):
    """v5 omega-seam batch -> (X, Y, Yc, G, A, S, R).

    Every gold row is a COPY row (gt=1): each reordered token has an anchored
    prompt twin by construction. Y/Yc/G follow the make_slot_batch contract;
    A/S/R come from reorder_anchors. Padding: eos / -100 / False.
    """
    rng = random.Random(seed)
    eos = tok.token_to_id(EOS)
    asid = tok.token_to_id(ASSIST)
    X, Y, YC, G, AN, S, R = [], [], [], [], [], [], []
    for _ in range(bs):
        a_s = rng.choice(a_slots)
        b_s = rng.choice(b_slots)
        ids, asi_pos, _ = make_reorder_ids(tok, a_s, b_s)
        anchors, seams, runs = reorder_anchors(ids, asid, tok)
        targets = ids[1:] + [eos]
        Tn = len(ids)
        y, yc, gt = [-100] * Tn, [-100] * Tn, [-1.0] * Tn
        for t in range(asi_pos, Tn):
            tgt = targets[t]
            y[t] = tgt
            if tgt == eos:
                yc[t] = -100
                gt[t] = 0.0
            else:
                yc[t] = tgt
                gt[t] = 1.0
        X.append(ids); Y.append(y); YC.append(yc); G.append(gt)
        AN.append(anchors); S.append(seams); R.append(runs)
    dev = resolve_device(device)
    T = max(len(x) for x in X)
    Xb = torch.full((bs, T), eos, dtype=torch.long, device=dev)
    Yb = torch.full((bs, T), -100, dtype=torch.long, device=dev)
    YcB = torch.full((bs, T), -100, dtype=torch.long, device=dev)
    Gb = torch.full((bs, T), -1.0, dtype=torch.float, device=dev)
    Ab = torch.full((bs, T), -100, dtype=torch.long, device=dev)
    Sb = torch.zeros((bs, T), dtype=torch.bool, device=dev)
    Rb = torch.full((bs, T), -100, dtype=torch.long, device=dev)
    for j in range(bs):
        L = len(X[j])
        Xb[j, :L] = torch.tensor(X[j], device=dev)
        Yb[j, :L] = torch.tensor(Y[j], device=dev)
        YcB[j, :L] = torch.tensor(YC[j], device=dev)
        Gb[j, :L] = torch.tensor(G[j], device=dev)
        Ab[j, :L] = torch.tensor(AN[j], device=dev)
        Sb[j, :L] = torch.tensor(S[j], device=dev)
        Rb[j, :L] = torch.tensor(R[j], device=dev)
    return Xb, Yb, YcB, Gb, Ab, Sb, Rb


def make_perm_batch(tok, parts_fn, bs, seed, device=None):
    """v5 M3 batch for arbitrary rotations -> (X, Y, Yc, G, A, S, R).

    parts_fn(rng) returns the user's `parts` list (e.g. ["fetch pkg001",
    "deploy lib042", "stop bin007"]); gold = rotate-left. Same tensor
    contract as make_reorder_batch.
    """
    rng = random.Random(seed)
    eos = tok.token_to_id(EOS)
    asid = tok.token_to_id(ASSIST)
    X, Y, YC, G, AN, S, R = [], [], [], [], [], [], []
    for _ in range(bs):
        ids, asi_pos, _ands = make_perm_ids(tok, parts_fn(rng))
        anchors, seams, runs = perm_anchors(ids, asid, tok)
        targets = ids[1:] + [eos]
        Tn = len(ids)
        y, yc, gt = [-100] * Tn, [-100] * Tn, [-1.0] * Tn
        for t in range(asi_pos, Tn):
            tgt = targets[t]
            y[t] = tgt
            if tgt == eos:
                yc[t] = -100
                gt[t] = 0.0
            else:
                yc[t] = tgt
                gt[t] = 1.0
        X.append(ids); Y.append(y); YC.append(yc); G.append(gt)
        AN.append(anchors); S.append(seams); R.append(runs)
    dev = resolve_device(device)
    T = max(len(x) for x in X)
    Xb = torch.full((bs, T), eos, dtype=torch.long, device=dev)
    Yb = torch.full((bs, T), -100, dtype=torch.long, device=dev)
    YcB = torch.full((bs, T), -100, dtype=torch.long, device=dev)
    Gb = torch.full((bs, T), -1.0, dtype=torch.float, device=dev)
    Ab = torch.full((bs, T), -100, dtype=torch.long, device=dev)
    Sb = torch.zeros((bs, T), dtype=torch.bool, device=dev)
    Rb = torch.full((bs, T), -100, dtype=torch.long, device=dev)
    for j in range(bs):
        L = len(X[j])
        Xb[j, :L] = torch.tensor(X[j], device=dev)
        Yb[j, :L] = torch.tensor(Y[j], device=dev)
        YcB[j, :L] = torch.tensor(YC[j], device=dev)
        Gb[j, :L] = torch.tensor(G[j], device=dev)
        Ab[j, :L] = torch.tensor(AN[j], device=dev)
        Sb[j, :L] = torch.tensor(S[j], device=dev)
        Rb[j, :L] = torch.tensor(R[j], device=dev)
    return Xb, Yb, YcB, Gb, Ab, Sb, Rb


def seam_losses(out, S, R, A):
    """v5 omega-seam aux losses: pointer CE + length CE on seam rows.

    out must contain "ptr_logits"/"len_logits" (HMN3 with seam_addr=True).
    S: bool seam mask; R: run-length class targets (-100 ignore); A: anchor
    columns (-100 ignore) used as the pointer target. Returns (l_ptr, l_len).
    """
    dev = (out.get("gen_logits") if "gen_logits" in out
           else out["logits"]).device
    zero = torch.zeros((), device=dev)
    l_ptr = l_len = zero
    if "ptr_logits" in out and "len_logits" in out and S.any():
        b_i, t_i = S.nonzero(as_tuple=True)
        p = out["ptr_logits"][b_i, t_i]                    # (Ns, T_cols)
        pt = A[b_i, t_i].clamp(min=0)                      # (Ns,) target col
        l_ptr = F.cross_entropy(p.float(), pt)
        ln = out["len_logits"][b_i, t_i]                   # (Ns, max_run)
        lt = R[b_i, t_i].clamp(min=0)                      # class = length-1
        l_len = F.cross_entropy(ln.float(), lt)
    return l_ptr, l_len


def eval_slot_chains(model, tok, a_slots, b_slots, seed=0, mode="blend",
                     max_new=40, boundary_eos=False, cycle_break=False,
                     pos_eos=False, device=None):
    """Exact-match on the two-slot chain task for UNSEEN slot pairs.
    a drawn from `a_slots`, b from `b_slots` (each record re-sampled).
    Returns (accuracy, avg_gate, avg_gen_tokens)."""
    model.eval()
    rng = random.Random(seed)
    ok = tot = 0
    gates, ngen = [], 0
    for _ in a_slots:
        a = rng.choice(a_slots)
        b = rng.choice(b_slots)
        gold = f"fetch {a} and deploy {b}"
        prompt = make_chat_ids(tok, gold)
        out, g, ng = decode_v33(model, tok, prompt, mode=mode, max_new=max_new,
                                boundary_eos=boundary_eos, cycle_break=cycle_break,
                                pos_eos=pos_eos, device=device)
        tot += 1
        ok += int(out.strip() == gold)
        gates.append(g); ngen += ng
    model.train()
    return ok / tot, (sum(gates) / len(gates) if gates else 0.0), ngen / max(1, tot)


def decode_rotate(model, tok, prompt_ids, max_new=48, device=None):
    """v5 M3b: rotation decoding — SeedPointer seeds RUN 0 only; every later
    run follows the structural cyclic order of the prompt's segments.

    Lesson repeated from v4 (boundary_eos/M6) and now measured here
    (omega_cur2 ptr3 plateau 0.64-0.70 @2400 steps): a learned head asked to
    re-seed EVERY seam stays lexicon/geometry-bound, while the answer lane
    itself is already exact (teacher-forced copy-argmax == gold 1.000). So:
    neural proposes the first fragment start; the decoder derives segment
    boundaries (counting the separator token in the USER region) and walks
    them cyclically — rotate-left semantics, T-invariant, no gold access.

    Returns (text, gate_avg, n_seeded).
    """
    ids = list(prompt_ids)
    eos = tok.token_to_id(EOS)
    asid = tok.token_to_id(ASSIST)
    asi_pos = ids.index(asid)
    U = ids[2:asi_pos]
    ands = _find_all_word(tok, U, REORDER_AND)
    if not ands:
        raise AssertionError("decode_rotate: no separator in prompt")
    n_parts = len(ands) + 1
    bounds = [-1] + ands + [len(U)]
    ans_len = max(0, len(prompt_ids) - 3)
    gates = []
    seeded = 0
    plan = None          # list of (anchor_col, run_len) consumed in order
    plan_i = 0
    run_left = 0
    cur = None           # (anchor_base, run_start_t)
    with torch.no_grad():
        while len(ids) - len(prompt_ids) < max_new:
            t_idx = len(ids) - len(prompt_ids)
            if pos_eos_done(t_idx, ans_len):
                break
            inp = torch.tensor([ids], device=device)
            if run_left > 0:
                c = cur[0] + (t_idx - cur[1])
                anch = torch.full((1, inp.shape[1]), -100,
                                  dtype=torch.long, device=device)
                anch[0, -1] = c
                out = model(inp, seam_anchor=anch)
                nxt = ids[c + 1]
                run_left -= 1
            else:
                out = model(inp)
                if plan is None:
                    # FIRST seed: neural proposal -> segment index
                    c0 = int(out["ptr_logits"][0, -1].argmax(-1).item())
                    src = c0 + 1                      # payload column
                    k = next((kk for kk in range(n_parts)
                              if bounds[kk] + 1 <= src <= bounds[kk + 1]), 0)
                    plan = []
                    for si in range(n_parts):
                        seg = (k + si) % n_parts
                        # same formulas as perm_anchors (verified): segment
                        # anchor = bounds[seg]+2, length excludes the left
                        # separator; separator mini-run anchors at 1+ands[0].
                        plan.append((bounds[seg] + 2,
                                     bounds[seg + 1] - bounds[seg] - 1))
                        if si < n_parts - 1:
                            plan.append((1 + ands[0], 1))   # separator 'and'
                    plan_i = 0
                cur_c, L = plan[plan_i]
                plan_i += 1
                nxt = ids[cur_c + 1]
                cur = (cur_c, t_idx)
                run_left = L - 1
                seeded += 1
            gates.append(float(out["g"][0, -1]))
            if nxt in (eos, asid):
                break
            ids.append(nxt)
    return tok.decode(ids[len(prompt_ids):]).strip(), (
        sum(gates) / len(gates) if gates else 0.0), seeded


def pos_eos_done(t_idx, ans_len):
    return ans_len is not None and t_idx >= ans_len


def eval_reorders(model, tok, a_slots, b_slots, seed=0, mode="hard",
                  max_new=48, pos_eos=True, device=None):
    """v5 omega-seam: exact-match on the REORDER task (the M12 wall).

    gold text is the canonical decoding of the swapped prompt-token sequence
    (ByteLevel keeps the prompt's space-prefixed variants; .strip() normalizes
    only the leading boundary). Returns (accuracy, avg_gate)."""
    model.eval()
    rng = random.Random(seed)
    ok = tot = 0
    gates = []
    for _ in a_slots:
        a = rng.choice(a_slots)
        b = rng.choice(b_slots)
        ids, asi_pos, _ = make_reorder_ids(tok, a, b)
        prompt = ids[:asi_pos + 1]
        gold_text = tok.decode(ids[asi_pos + 1:-1]).strip()
        out, g, _ng = decode_v33(model, tok, prompt, max_new=max_new,
                                 mode=mode, seam=True, pos_eos=pos_eos,
                                 device=device)
        tot += 1
        ok += int(out.strip() == gold_text)
        gates.append(g)
    model.train()
    return ok / tot, (sum(gates) / len(gates) if gates else 0.0)


def copy_prob_sparse(attn, nxt, targets):
    """v4 sparse copy-marginal: p_copy(row, t) for target token ids.

    attn    : (B, T, T) position attention over prompt columns
    nxt     : (B, T) payload token id per column (0 = none)
    targets : (B, T) target token ids per row (Yc; -100 = ignore row)

    p = sum_j a[t, j] * [nxt[j] == target]  — i.e. attention over exactly the
    payload columns whose next token IS the target. Identical by construction
    to dense copy distribution lookup (masked columns carry zero attention),
    computed without ever materializing the (B, T, V) copy tensor. Ignore rows
    (target == -100) return 0 and are filtered by the caller.
    """
    eq = (nxt.unsqueeze(1) == targets.unsqueeze(-1))            # (B,T,T)
    return (attn * eq.float()).sum(-1)                          # (B,T)


def _logaddexp(a, b):
    m = max(a, b)
    return m + math.log(math.exp(min(a, b) - m) + 1.0)


def blend_argmax(gen_logp, g, pay_vals, pay_fracs):
    """v6 M1-C: EXACT blended-distribution argmax from candidate sets only.

    The snapped copy lane has support ONLY on the seed group's payloads P
    (fractions fr>0); everywhere else p(v) = (1-g)*p_gen(v). Bound: let
    w = argmax_v p_gen(v). For any u outside P,
        p(u) <= (1-g)*p_gen(u) <= (1-g)*p_gen(w) <= p(w),
    so the global argmax always lies in {w} ∪ P — no gen top-k needed.
    Ties go to the gen side (strict >), matching torch.argmax convention of
    preferring the earlier candidate deterministically.

    gen_logp : (V,) log-probs of the gen head at the row
    g        : float gate value in [0, 1]
    pay_vals : LongTensor payload ids in P
    pay_fracs: FloatTensor copy fractions aligned with pay_vals
    Returns next-token id (int).
    """
    lg1 = math.log(max(1.0 - g, 1e-12))
    lg2 = math.log(max(g, 1e-12))
    w = int(gen_logp.argmax())
    payset = set(pay_vals.tolist()) if hasattr(pay_vals, "tolist") else set(pay_vals)
    # the non-support champion is the gen argmax ONLY when it lies outside P;
    # inside P it is scored exactly by the loop below (adding its copy mass)
    best_v, best_lp = (None, float("-inf"))
    if w not in payset:
        best_v, best_lp = w, lg1 + float(gen_logp[w])
    for v, f in zip(pay_vals.tolist(), pay_fracs.tolist()):
        lp = _logaddexp(lg1 + float(gen_logp[v]), lg2 + math.log(max(float(f), 1e-12)))
        if lp > best_lp:
            best_v, best_lp = v, lp
    if best_v is None:
        best_v = w
    return best_v


def loss_v33(out, Y, Yc, G, lossf=None, w_copy=1.0, w_gate=0.0):
    """v3.3 loss = blend CE + w_copy*gen CE (masked) + w_copy*copy CE (manual).

    out must be HMN3.forward's dict. Two regimes:

    v6 M1-B stats path ("stats" in out — no (B,T,V) blend tensor exists):
      blend CE : per-target exact logaddexp
                 -log[(1-g)·p_gen(y) + g·p_copy(y)] on every answer row,
                 where p_copy comes from the IRStats payload histogram
                 (st.prob_at). Algebraically the SAME convex blend as the
                 dense path, evaluated only where it is consumed.
      gen CE   : identical to below.
      copy CE  : manual -log p_copy(target) from st.prob_at (Yc rows).

    Dense oracle regime ("logits"/"copy_dist" present, --exact-blend):

      blend CE : CE((1-g)*gen + g*copy_dist, Y) on every answer row.
      gen CE   : CE(gen_logits, Y) MASKED to gen rows only (G == 0.0: first
                 answer token 'pip', EOS, targets absent from prompt). Copy rows
                 (G == 1.0) are masked OUT — training gen to emit the copied
                 token makes it memorize the slot digits and fire the SAME token
                 again at the boundary row instead of EOS (pkg060->'OST',
                 pkg066->'6' loop, 2026-08-13). gen_logits is log_softmax, so
                 CE() on it == -log p_gen(target).
      copy CE  : manual -log p_target on copy_dist / position gather. THIS MUST
                 BE -log p, NOT CrossEntropyLoss(copy_dist, Y): copy_dist is
                 already a probability distribution (sums to 1), and CE would
                 log-softmax it a second time, pinning the loss at
                 ~ln(VOCAB)=7.07 with no gradient (the root cause of every
                 'frozen' curve since v3.1).
    Returns (loss, l_blend, l_gen, l_copy).
    """
    if lossf is None:
        lossf = nn.CrossEntropyLoss(ignore_index=-100)
    if "stats" in out:
        # ---------------- v6 M1-B: stats API, zero T^2/T*V tensors ----------
        st = out["stats"]
        genlp = out["gen_logits"]                      # (B,T,V) log-probs
        vocab = genlp.shape[-1]
        g = out["g"].squeeze(-1)                       # (B,T)
        ymask = Y != -100
        lg_y = genlp.gather(2, Y.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        pc = st.prob_at(Yc)                            # (B,T) copy prob of target
        # blend needs the TRUE zero: a missing payload must make the copy
        # branch -inf (never a clamped floor that could beat the gen branch
        # on low-gate rows); the copy CE below keeps the historical 1e-9 cap.
        lp_c = torch.full_like(pc, float("-inf"))
        nz = pc > 0
        lp_c[nz] = pc[nz].log()
        # NOTE: no artificial clamp on (1-g): the deterministic gate is born
        # in [0.0025, 0.9975] (sigmoid input pre-clamped to +-6), so log1p(-g)
        # is finite; the guard below only shields a degenerate learned gate.
        l_one = torch.logaddexp(torch.log1p(-g.clamp(max=1 - 1e-7)) + lg_y,
                                torch.log(g.clamp(min=1e-12)) + lp_c)
        l_blend = -l_one[ymask].mean() if ymask.any() else genlp.new_zeros(())
        gen_tgt = Y.reshape(-1).clone()
        gen_tgt[G.reshape(-1) != 0.0] = -100           # gen rows only
        l_gen = lossf(genlp.reshape(-1, vocab), gen_tgt)
        cmask = Yc != -100
        l_copy = -(pc.clamp(min=1e-9).log()[cmask]).mean() \
            if cmask.any() else genlp.new_zeros(())
        loss = l_blend + w_copy * l_gen + w_copy * l_copy
        if w_gate > 0.0:
            vm = G.reshape(-1) >= 0.0
            if vm.any():
                l_gate = nn.functional.binary_cross_entropy(
                    out["g"].reshape(-1)[vm], (G.reshape(-1)[vm] > 0.5).float())
                loss = loss + w_gate * l_gate
        return loss, l_blend, l_gen, l_copy
    vocab = out["logits"].shape[-1]
    l_blend = lossf(out["logits"].reshape(-1, vocab), Y.reshape(-1))
    gen_tgt = Y.reshape(-1).clone()
    gen_tgt[G.reshape(-1) != 0.0] = -100     # gen rows only
    l_gen = lossf(out["gen_logits"].reshape(-1, vocab), gen_tgt)
    if "attn" in out and "nxt" in out:
        # v4 sparse copy-marginal path: NO (B,T,V) copy tensor was built —
        # p_copy(target) = sum over payload columns j with ids[j+1]==target of a.
        # This is algebraically identical to dense copy_dist[., target] (the
        # position-gather carries zero attention for masked columns), so the
        # loss is bit-equivalent to the dense path without the O(T·V) memory.
        yv = Yc.reshape(-1)
        vmask = yv != -100
        p = copy_prob_sparse(out["attn"], out["nxt"], Yc).reshape(-1)[vmask]
        lc = -p.clamp(min=1e-9).log()
        l_copy = lc.mean() if lc.numel() else torch.zeros((), device=out["logits"].device)
    else:
        probs = out["copy_dist"].reshape(-1, vocab).clamp(min=1e-9)
        yv = Yc.reshape(-1)
        if (yv != -100).any():
            lc = -probs[yv != -100].gather(1, yv[yv != -100].unsqueeze(1)).squeeze(-1).log()
            l_copy = lc.mean()
        else:
            l_copy = torch.zeros((), device=out["logits"].device)
    loss = l_blend + w_copy * l_gen + w_copy * l_copy
    if w_gate > 0.0:
        # v4 M2: supervise the learned gate DIRECTLY against the copy mask G.
        # G==-1 rows (prompt/ASI) carry no copy target — mask them out.
        vm = G.reshape(-1) >= 0.0
        if vm.any():
            l_gate = nn.functional.binary_cross_entropy(
                out["g"].reshape(-1)[vm], (G.reshape(-1)[vm] > 0.5).float())
            loss = loss + w_gate * l_gate
    return loss, l_blend, l_gen, l_copy


def decode_v33(model, tok, prompt_ids, max_new=16, mode="blend", gate_thr=0.5,
               boundary_eos=False, device=None, cycle_break=False, pos_eos=False,
               seam=False):
    """Greedy decode. mode:
      blend  -> argmax of (1-g)*gen + g*copy
      hard   -> if g > gate_thr: argmax(copy_dist) else argmax(blend)
      copy   -> always argmax(copy_dist)  (channel-quality diagnostic)
    boundary_eos: structural rule (v3.3-final, 2026-08-13). In the slot-copy
      task every answer token after the first has an exact twin in the prompt,
      so its register gate is ~1. The ONLY row where the gate can collapse
      mid-answer is the final one (seed's twin points at ASI/EOS payload and
      was masked) -> the model is at the end and MUST emit EOS. Relying on the
      gen head to discover this per-slot is a lottery (pkg094 'rude' bug, ~1-2
      fails/seed); forcing EOS on a low gate makes the boundary deterministic.
      step 0 (seed=ASI, first answer token) is exempt: len(ids)==len(prompt).
    cycle_break (v4 M4b): if the same 2-token emission (prev, next) pair
      occurs twice while decoding, we are replaying an already-emitted
      segment (the register found a twin again and repeats its payload —
      e.g. the chain task's " and deploy lib039 and deploy ..." loop after the
      true end). Force EOS on the repeat. NOTE: a 1-token pair is NOT enough
      (the chain task legitimately re-emits tokens like '0' across segments);
      the (prev,next) window is unique per true answer. Deterministic
      (decoder-time stop rule, not a model change).
    pos_eos (v4 M6): answer-length-bounded stop. In the echo task the answer is
      a positional copy of the user content, so its token count is KNOWN at
      decode: len(answer) = len(user tokens) = len(prompt)-3 (<s>,<|user|>,
      <|assistant|>). When the emitted count reaches that length force EOS.
      This closes the repeated-subtoken loop (pkg333 -> '333333'): the anchor
      guarantees the CONTENT, and the length bound guarantees WHEN it ends —
      the gate stays ~0.93 (seed '3' still has a twin) so boundary_eos cannot
      fire, but the answer is structurally finished. Same decoder-time
      determinism family as boundary_eos/cycle_break. Safe only for echo tasks
      where user == gold (slot + chain); default OFF.
    Returns (text, gate_avg, n_gen).

    seam (v5 omega-seam): fragment-run decoding for reorder/transform tasks.
    State = (run anchor base column, run start row, tokens left in run).
    Within a run the anchor echoes +1 per row and the emitted token is the
    forced payload ids[c+1] (deterministic — no second forward needed). When
    the run is exhausted the SeedPointer heads on the CURRENT forward pick
    the next run: c_new = argmax ptr_logits, L_new = argmax len_logits + 1.
    Termination via pos_eos (a pure permutation keeps |answer| == |user|) or
    an ASI/EOS payload guard. Requires HMN3(seam_addr=True).
    """
    ids = list(prompt_ids)
    eos = tok.token_to_id(EOS)
    asid = tok.token_to_id(ASSIST)
    gates, n_gen = [], 0
    seen_pairs = set()
    # v4 M6: expected answer length = user content tokens, known a priori for
    # echo tasks (user == gold). Weaker than it looks: if a future non-echo
    # task needs it, the model must learn to emit EOS itself (pos_eos OFF).
    ans_len = max(0, len(prompt_ids) - 3) if pos_eos else None
    run_base = run_start = None
    run_left = 0
    with torch.no_grad():
        for _ in range(max_new):
            t_idx = len(ids) - len(prompt_ids)          # answer row (0-based)
            inp = torch.tensor([ids], device=device)
            if seam:
                if run_left > 0:
                    c = run_base + (t_idx - run_start)
                    anch = torch.full((1, inp.shape[1]), -100,
                                      dtype=torch.long, device=device)
                    anch[0, -1] = c
                    out = model(inp, seam_anchor=anch)
                    nxt = ids[c + 1]
                    run_left -= 1
                else:
                    out = model(inp)
                    c_new = int(out["ptr_logits"][0, -1].argmax(-1).item())
                    l_new = int(out["len_logits"][0, -1].argmax(-1).item()) + 1
                    nxt = ids[c_new + 1]
                    run_base, run_start, run_left = c_new, t_idx, l_new - 1
                gates.append(float(out["g"][0, -1]))
                if pos_eos and t_idx >= ans_len:
                    break                               # structurally complete
                if nxt in (eos, asid):
                    break
                ids.append(nxt)
                continue
            out = model(inp)
            g = out["g"][0, -1].item()
            gates.append(g)
            if "stats" in out:
                # v6 M1-C: copy candidate = per-group payload MODE (argmax of
                # the snapped copy distribution); the blend argmax is EXACT
                # over {gen argmax} ∪ {own-group payloads} — proven bound, no
                # gen top-k needed (see blend_argmax).
                st = out["stats"]
                gl = out["gen_logits"][0, -1]
                mv = int(st.copy_mode[0, -1])
                cand_copy = mv if mv >= 0 else int(gl.argmax())
                pay, fr = st.payloads_of(0, inp.shape[1] - 1)
                blend_arg = blend_argmax(gl, g, pay, fr)
            else:
                logits = out["logits"]
                copy = out["copy_dist"][0, -1]
                cand_copy = int(copy.argmax(-1).item())
                blend_arg = None
            pair = None
            if len(ids) > len(prompt_ids) + 1:
                prev = ids[-1]
                if prev != cand_copy:                 # ignore (x,x) self-pairs:
                    pair = (prev, cand_copy)          # repeated identical tokens
                                                      # (99999) are legitimately
                                                      # consecutive, not a replay
            if pos_eos and len(ids) - len(prompt_ids) >= ans_len:
                nxt = eos                          # answer structurally complete
            elif cycle_break and pair is not None and pair in seen_pairs:
                nxt = eos                          # segment replay -> stop
            elif boundary_eos and len(ids) > len(prompt_ids) and g < gate_thr:
                nxt = eos
            elif mode == "copy":
                nxt = cand_copy
            elif mode == "hard" and g > gate_thr:
                nxt = cand_copy
            else:
                n_gen += 1
                nxt = blend_arg if blend_arg is not None \
                    else int(logits[0, -1].argmax(-1).item())
            if pair is not None:
                seen_pairs.add(pair)
            if nxt == eos:
                break
            ids.append(nxt)
    return tok.decode(ids[len(prompt_ids):]), (sum(gates) / len(gates) if gates else 0.0), n_gen


def eval_slots(model, tok, slots, template=DEFAULT_TEMPLATE, mode="blend", seed=0,
               boundary_eos=False, max_new=16, pos_eos=False, device=None):
    """Exact-match accuracy on a slot list (unseen validation by convention).

    Returns (accuracy, avg_gate, avg_gen_tokens). eval_slots is the seed-42
    guardrail: pass the SAME slot list + template and the result is bit-stable
    given a fixed checkpoint.
    """
    model.eval()
    ok = tot = 0
    gates, ngen = [], 0
    for p in slots:
        gold = template.format(slot=p)
        prompt = make_chat_ids(tok, gold)
        out, g, ng = decode_v33(model, tok, prompt, mode=mode, max_new=max_new,
                                boundary_eos=boundary_eos, pos_eos=pos_eos,
                                device=device)
        tot += 1
        ok += int(out.strip() == gold)
        gates.append(g); ngen += ng
    model.train()
    return ok / tot, (sum(gates) / len(gates) if gates else 0.0), ngen / max(1, tot)


def seed_guardrail(seed=42):
    """Deterministic RNG reset for reproducible training/eval."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


# ---------------------------------------------------------------------------
# v7: convenience training API
# ---------------------------------------------------------------------------

def train(model, tok, slots=None, n_steps=200, bs=1, lr=3e-4,
          template=DEFAULT_TEMPLATE, templates=None, seed=42,
          eval_every=50, eval_slots_list=None, device=None,
          w_copy=1.0, seam=False, a_slots=None, b_slots=None,
          callback=None):
    """One-call training loop for slot-copy (and optionally reorder).

    Args:
        model: HMN3 or HMN3AttentionWR instance.
        tok: tokenizer (tokenizers.Tokenizer).
        slots: list of slot strings for training (default: pkg000-pkg039).
        n_steps: number of training steps.
        bs: batch size.
        lr: learning rate.
        template: format string with {slot} placeholder.
        templates: list of format strings (multi-template training).
        seed: random seed for reproducibility.
        eval_every: evaluate every N steps (0 to disable).
        eval_slots_list: slots for evaluation (default: unseen pkg060-pkg079).
        device: compute device (auto-detect if None).
        w_copy: copy loss weight.
        seam: use seam/reorder training (requires a_slots, b_slots).
        a_slots: slot A family for reorder training.
        b_slots: slot B family for reorder training.
        callback: optional fn(step, loss, metrics_dict) called each step.

    Returns:
        dict with training history: {losses: [...], eval_acc: [...], ...}.
    """
    dev = resolve_device(device)
    model = model.to(dev)
    seed_guardrail(seed)

    if slots is None:
        slots = [f"pkg{i:03d}" for i in range(40)]
    if eval_slots_list is None:
        eval_slots_list = [f"pkg{i:03d}" for i in range(60, 80)]
    if seam and a_slots is None:
        a_slots = [f"pkg{i:03d}" for i in range(40)]
    if seam and b_slots is None:
        b_slots = [f"lib{i:03d}" for i in range(40, 80)]

    eos = tok.token_to_id(EOS)
    asid = tok.token_to_id(ASSIST)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    history = {"losses": [], "eval_acc": [], "eval_gate": [], "steps": []}
    model.train()

    for step in range(1, n_steps + 1):
        if seam:
            Xb, Yb, YcB, Gb, Ab, Sb, Rb = make_reorder_batch(
                tok, a_slots, b_slots, bs=bs, seed=step, device=dev)
            out = model(Xb, seam_anchor=Ab)
            loss, lb, lg, lc = loss_v33(out, Yb, YcB, Gb, w_copy=w_copy)
        else:
            Xb, Yb, YcB, Gb = make_slot_batch(
                tok, slots, bs=bs, seed=step, template=template,
                templates=templates, device=dev)
            out = model(Xb)
            loss, lb, lg, lc = loss_v33(out, Yb, YcB, Gb, w_copy=w_copy)

        if hasattr(model, 'moe_aux_loss'):
            loss = loss + model.moe_aux_loss()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        history["losses"].append(loss.item())

        if callback:
            callback(step, loss.item(), {"blend": lb.item(),
                                          "gen": lg.item(),
                                          "copy": lc.item()})

        if eval_every > 0 and (step % eval_every == 0 or step == n_steps):
            acc, gate, ngen = eval_slots(
                model, tok, eval_slots_list, template=template,
                boundary_eos=True, device=dev)
            history["eval_acc"].append(acc)
            history["eval_gate"].append(gate)
            history["steps"].append(step)
            print(f"step {step:5d} loss={loss.item():.4f} "
                  f"eval={acc:.3f} gate={gate:.3f}")
            model.train()

    return history
