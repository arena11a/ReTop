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


def make_chat_targets(ids, asid, eos):
    """Per-sequence answer-region labels -> (y, yc, gt).

    y  : shifted targets (all answer tokens incl. EOS) for blend CE
    yc : copy-channel targets; EOS and the FIRST answer token (seed=ASI, absent
         from the prompt) are -100 — the register cannot copy them by design
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
        if targets[t] == eos or t == a:
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
                    templates=None):
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
        y, yc, gt = make_chat_targets(ids, asid, eos)
        X.append(ids); Y.append(y); YC.append(yc); G.append(gt)
    T = max(len(x) for x in X)
    Xb = torch.full((bs, T), eos, dtype=torch.long)
    Yb = torch.full((bs, T), -100, dtype=torch.long)
    YcB = torch.full((bs, T), -100, dtype=torch.long)
    Gb = torch.full((bs, T), -1.0, dtype=torch.float)
    for j in range(bs):
        Xb[j, :len(X[j])] = torch.tensor(X[j])
        Yb[j, :len(Y[j])] = torch.tensor(Y[j])
        YcB[j, :len(YC[j])] = torch.tensor(YC[j])
        Gb[j, :len(G[j])] = torch.tensor(G[j])
    return Xb, Yb, YcB, Gb


def make_slot_chain_batch(tok, bs, seed):
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
        y, yc, gt = make_chat_targets(ids, asid, eos)
        X.append(ids); Y.append(y); YC.append(yc); G.append(gt)
    T = max(len(x) for x in X)
    Xb = torch.full((bs, T), eos, dtype=torch.long)
    Yb = torch.full((bs, T), -100, dtype=torch.long)
    YcB = torch.full((bs, T), -100, dtype=torch.long)
    Gb = torch.full((bs, T), -1.0, dtype=torch.float)
    for j in range(bs):
        Xb[j, :len(X[j])] = torch.tensor(X[j])
        Yb[j, :len(Y[j])] = torch.tensor(Y[j])
        YcB[j, :len(YC[j])] = torch.tensor(YC[j])
        Gb[j, :len(G[j])] = torch.tensor(G[j])
    return Xb, Yb, YcB, Gb


def eval_slot_chains(model, tok, a_slots, b_slots, seed=0, mode="blend",
                     max_new=24, boundary_eos=False):
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
                                boundary_eos=boundary_eos)
        tot += 1
        ok += int(out.strip() == gold)
        gates.append(g); ngen += ng
    model.train()
    return ok / tot, (sum(gates) / len(gates) if gates else 0.0), ngen / max(1, tot)


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


def loss_v33(out, Y, Yc, G, lossf=None, w_copy=1.0, w_gate=0.0):
    """v3.3 loss = blend CE + w_copy*gen CE (masked) + w_copy*copy CE (manual).

    out must be HMN3.forward's dict: {"logits", "gen_logits", "copy_dist", ...}.

      blend CE : CE((1-g)*gen + g*copy_dist, Y) on every answer row.
      gen CE   : CE(gen_logits, Y) MASKED to gen rows only (G == 0.0: first
                 answer token 'pip', EOS, targets absent from prompt). Copy rows
                 (G == 1.0) are masked OUT — training gen to emit the copied
                 token makes it memorize the slot digits and fire the SAME token
                 again at the boundary row instead of EOS (pkg060->'OST',
                 pkg066->'6' loop, 2026-08-13). gen_logits is log_softmax, so
                 CE() on it == -log p_gen(target).
      copy CE  : manual -log p_target on copy_dist. THIS MUST BE -log p, NOT
                 CrossEntropyLoss(copy_dist, Y): copy_dist is already a
                 probability distribution (sums to 1), and CE would log-softmax
                 it a second time, pinning the loss at ~ln(VOCAB)=7.07 with no
                 gradient (the root cause of every 'frozen' curve since v3.1).
    Returns (loss, l_blend, l_gen, l_copy).
    """
    if lossf is None:
        lossf = nn.CrossEntropyLoss(ignore_index=-100)
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
               boundary_eos=False, device=None):
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
    Returns (text, gate_avg, n_gen).
    """
    ids = list(prompt_ids)
    eos = tok.token_to_id(EOS)
    gates, n_gen = [], 0
    with torch.no_grad():
        for _ in range(max_new):
            inp = torch.tensor([ids], device=device)
            out = model(inp)
            logits = out["logits"]
            g = out["g"][0, -1].item()
            copy = out["copy_dist"][0, -1]
            gates.append(g)
            if boundary_eos and len(ids) > len(prompt_ids) and g < gate_thr:
                nxt = eos
            elif mode == "copy":
                nxt = copy.argmax(-1).item()
            elif mode == "hard" and g > gate_thr:
                nxt = copy.argmax(-1).item()
            else:
                n_gen += 1
                nxt = logits[0, -1].argmax(-1).item()
            if nxt == eos:
                break
            ids.append(nxt)
    return tok.decode(ids[len(prompt_ids):]), (sum(gates) / len(gates) if gates else 0.0), n_gen


def eval_slots(model, tok, slots, template=DEFAULT_TEMPLATE, mode="blend", seed=0,
               boundary_eos=False, max_new=16):
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
                                boundary_eos=boundary_eos)
        tot += 1
        ok += int(out.strip() == gold)
        gates.append(g); ngen += ng
    model.train()
    return ok / tot, (sum(gates) / len(gates) if gates else 0.0), ngen / max(1, tot)


def seed_guardrail(seed=42):
    """Deterministic RNG reset for reproducible training/eval."""
    random.seed(seed)
    torch.manual_seed(seed)
    return seed
