"""HMN v3 — Helix Register-Network (hmn_v3_design.md).

The core new contribution vs v2 / vs vanilla single-pass LLMs:

  Dual-Head Decoder with Identity-Register copy channel.

  - Working Register (WR): existing reversible coupling backbone (+MoE) for
    contextual abstraction (reused from hmn_v2).
  - Identity Register (IR): a RAW-token lane. During a forward, it captures the
    token identity of every distinct token seen in the sequence (rows = token
    embeddings, content-addressable by row similarity).
  - Dual Head: generate-logits (softmax head) ⊕ copy-logits (similarity of a WR
    query against IR rows, masked to the token set actually present in context).
    Final logits = gen + gate*x.copy, so when the WR learns to emit a query that
    matches the identity of a token present in the context, the argmax can pick
    that EXACT token — unblocking hard slot-copy that v2's pure-softmax head could
    never do (val slot-copy 0/40).

Reuses verified v2 ingredients: Pre-LN coupling blocks (reconstruction-safe),
helper moe_aux_loss hook. Latent thinking buffer kept as a separate flag
(--thinking-buffer wiring added in the trainer/experiment).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from hmn.v2 import HelixCouplingBlock, ReversibleFunction, SparseConditionalCompute


class IdentityRegister(nn.Module):
    """Literal token lane (v3.3 — "next-token lookup"). 

    v3.1 used a contextual query vs context keys: the COPY channel then had to
    decode 'which token do I want next' through a single linear map — impossible
    per-position routing (measured 2026-08-12: ptr CE frozen at 2.08, copy CE
    frozen at 7.27 for 1500 steps; attention never sharpened past ~uniform;
    seen-only memorization, unseen 0/40). v3.2 (raw keys, same query) fixed
    nothing — the blocker is the QUERY, not the keys.

    v3.3 makes the register a pure identity lookup:
      query  = RAW embedding of the CURRENT token ids[t] (keys, unshifted)
      keys   = RAW embeddings of the prompt region (identity lane)
      a      = softmax(beta * cos(query, key_j))   <- self-match is EXACT (=1)
      payload= the token at position j+1 (what FOLLOWS the matched token)

    => 'the token right after the last thing I emitted/positioned'. For
    slot-copy ("pip install pkg061" -> answer "pip install pkg061") the chain
    is: seed=pip -> self-match prompt 'pip' -> copy 'install'; seed=install ->
    copy 'pkg061' UNSEEN token. The gen head emits the stable first answer
    token and EOS (both gutter rows: seed=ASI unreachable, payload=ASI != EOS);
    the gate learns when the lookup is reliable (tends ON in the mirror body,
    OFF at gutter rows). Corrected 2026-08-13: the seed MUST be ids[t], not
    ids[t-1] — the prev-shift made the register answer the already-seen token.

    This mirrors the verified v2 lever (task2 #10): addressing that is
    IDENTITY-based (raw embed self-match) hit 97-99% recall; contextual
    addressing stayed 51-62%. Identity lookup generalizes to unseen tokens
    because no knowledge about the token is needed — only its position.
    """

    def __init__(self, dim, beta_init=30.0, asi_id=None, eos_id=1,
                 user_id=None, stem_addr=False):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        self.asi_id = asi_id
        self.eos_id = eos_id
        self.user_id = user_id
        self.stem_addr = stem_addr

    def _attn(self, keys, ids, seam_anchor=None):
        """Shared identity-lane attention (v4 refactor). Returns everything the
        dense and sparse paths need WITHOUT building the (B, T, V) copy mass:
          a        (B, T, T) position attention over prompt columns
          nxt      (B, T) payload token id per column (0 for the last column)
          n_legal  (B, T) # of legal copy columns per row
          ctx      (B, T, D) attended raw-key blend (gate's memory read)
          behind   (B, T) bool rows at/before the ASI boundary (forced gen)
          mass_same (B, T) sum of attention over same-token-id columns (gate)
          mask     (B, T, T) bool legal-column mask (needed by position-gather)

        v5 omega-seam: `seam_anchor` (B, T long, -100 = no force) overrides
        row t's attention onto exactly column seam_anchor[b, t]. This is the
        run-echo generalization of stem-addr: within a fragment the anchor
        advances +1 per answer row (deterministic, supplied by recipe.py);
        at seams the SeedPointer-predicted start column re-seeds the chain.
        Forced rows open the gate (mass_same=1.0) and leave `behind`.
        Default None -> bit-identical to v3.3/v4 behavior.

        v6 M1-A (index-derived gate stats): `n_legal` is a prefix-sum over a
        per-column legality vector and `mass_same` is read off an inverted
        token->columns index (stable sort + searchsorted + composite keys),
        removing BOTH the second (B, T, T) bool matrix (`same`) AND the O(T^2)
        (~mask).sum reduction. The dense attention itself is untouched.

        mass_same identity: cos(q,k)=1 EXACTLY for same-id columns, so every
        legal twin of row t carries the identical softmax weight exp(beta)/Z_t
        ("twins-uniform"); therefore
            mass_same[t] = c_t * exp(beta - lse_t),
        where c_t = #legal twin columns with j <= t (from the index) and
        lse_t is the row normalizer of the already-materialized masked sim.
        Verified against the brute-force (a * same).sum formula in test_hmn.
        """
        B, T, D = keys.shape
        beta = self.beta.abs() + 1.0
        qk = F.normalize(keys, dim=-1)                              # (B,T,D) ids[t]
        ek = qk                                                     # identity addrs
        sim = (qk @ ek.transpose(-1, -2)) * beta                    # (B,T,T)
        pos = torch.arange(T, device=keys.device)
        # ASI boundary: first illegal column per sequence (T when absent)
        bound = torch.full((B,), T, device=keys.device, dtype=torch.long)
        if self.asi_id is not None and (ids == self.asi_id).any():
            idx = (ids == self.asi_id).float().argmax(-1)                # (B,)
            have = (ids == self.asi_id).any(-1)                          # (B,)
            bound = torch.where(have, idx, torch.full_like(idx, T)).long()
        legal_col = pos.unsqueeze(0) < bound.unsqueeze(-1)               # (B,T)
        # payload per column: ids[j+1]; last column has no successor
        nxt_col = torch.cat([ids[:, 1:], torch.full_like(ids[:, :1], -1)], dim=1)
        bad_payload = (nxt_col == self.asi_id) | (nxt_col == self.eos_id) | (nxt_col == -1)
        col_ok = legal_col & ~bad_payload                                # (B,T)
        mask = (torch.triu(torch.ones(T, T, device=keys.device,
                                       dtype=torch.bool), 1).unsqueeze(0)
                | (~legal_col).unsqueeze(1) | bad_payload.unsqueeze(1))
        sim = sim.masked_fill(mask, float("-inf"))
        lse = sim.logsumexp(-1)                                # row normalizer (B,T)
        a = sim.softmax(-1)                                    # position attention
        n_legal = col_ok.cumsum(-1)                             # (B,T) prefix count
        # inverted index: token -> sorted member columns (stable => positions
        # ascend within each token group; invalid members sort past position T)
        sids, perm = ids.sort(dim=-1, stable=True)
        grp = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=keys.device),
                         sids[:, 1:] != sids[:, :-1]], 1).cumsum(-1) - 1
        col_grp = torch.empty_like(grp).scatter_(1, perm, grp)   # group id / column
        ok_sorted = col_ok.gather(1, perm)
        pos_sorted = pos.unsqueeze(0).expand(B, T).gather(1, perm)
        key = grp * (T + 1) + torch.where(ok_sorted, pos_sorted, T)
        start = torch.searchsorted(sids, ids, side="left")
        c_same = torch.searchsorted(key, col_grp * (T + 1) + pos.unsqueeze(0)
                                    .expand(B, T), side="right") - start
        mass_same = c_same.float() * torch.exp(beta - lse)      # twins-uniform
        nxt = torch.cat([ids[:, 1:], torch.zeros_like(ids[:, :1])], dim=1)
        ctx = (a.unsqueeze(-1) * keys.unsqueeze(1)).sum(2)     # (B,T,D) raw blend
        behind = torch.zeros(B, T, dtype=torch.bool, device=keys.device)
        if self.asi_id is not None and (ids == self.asi_id).any():
            behind = pos.unsqueeze(0) <= bound.unsqueeze(-1)
            if self.stem_addr and self.user_id is not None and (ids == self.user_id).any():
                # v4 M2-dev: stem-addressing. The row-0 (ASI boundary row) query
                # of the identity register would self-match the ASI column, which
                # is a masked (unaddressable) boundary — so v3.3 forces gen there
                # and gen is lexically bound to seen template verbs (M3/M2
                # measured: 30/30 probes pick a seen-verb token). Stem-addr
                # anchors that row's attention onto the USER-token column whose
                # payload (ids[j+1]) is the FIRST token of the template prefix —
                # i.e. the answer's first segment becomes addressable by design,
                # not by what the gen head memorized.
                #
                # v4 M2-dev N-gram: the anchor extends BEYOND row 0, keyed by
                # stem position. Answer row a+i is anchored onto user-region
                # column u+i — a deterministic positional echo of the template
                # prefix (payload ids[j+1] = ids[u+i+1] = gold token i). This
                # is what resolves repeated
                # subtokens (e.g. "check" -> c,he,c,k): raw identity on the
                # seed 'c' ties col2 (c->he) vs col4 (c->k), but the positional
                # anchor picks col3=he unambiguously. Rows beyond the user
                # region (t-a >= user len) are left to identity fallback, which
                # at the EOS row correctly closes the gate.
                #
                # Deterministic: works the same at train and decode; default
                # OFF (v3.3 unchanged).
                u = (ids == self.user_id).float().argmax(-1)                # (B,) USER col
                r = bound                                              # (B,) ASI row
                if seam_anchor is None:
                    a = a.clone()
                    pos_stem = torch.arange(T, device=ids.device)
                    # valid rows: r <= t and u + (t - r) + 1 < r  =>  t < 2r - u - 1
                    upper = (2 * r - u - 1).clamp(max=T)
                    row_valid = ((pos_stem.unsqueeze(0) >= r.unsqueeze(1))
                                 & (pos_stem.unsqueeze(0) < upper.unsqueeze(1)))
                    # column index per valid row
                    col = u.unsqueeze(1) + (pos_stem.unsqueeze(0) - r.unsqueeze(1))
                    one_hot = torch.zeros(B, T, T, dtype=a.dtype, device=a.device)
                    one_hot.scatter_(2, col.unsqueeze(-1).clamp(min=0),
                                    row_valid.unsqueeze(-1).float())
                    a = torch.where(row_valid.unsqueeze(-1), one_hot, a)
                    behind.masked_fill_(row_valid, False)
                    mass_same.masked_fill_(row_valid, 1.0)
        if seam_anchor is not None:
            a = a.clone()
            forced_mask = seam_anchor >= 0                    # (B, T) bool
            one_hot = torch.zeros(B, T, T, dtype=a.dtype, device=a.device)
            safe_col = seam_anchor.clamp(min=0)
            one_hot.scatter_(2, safe_col.unsqueeze(-1),
                             forced_mask.unsqueeze(-1).float())
            a = torch.where(forced_mask.unsqueeze(-1), one_hot, a)
            behind.masked_fill_(forced_mask, False)
            mass_same.masked_fill_(forced_mask, 1.0)
        return a, nxt, n_legal, ctx, behind, mass_same, mask

    def _forced_columns(self, ids, seam_anchor=None):
        """Anchor overrides as a (B,T) column tensor (-100 = free row).

        Vectorized version — no Python loops. Same contract as before:
        seam anchors take precedence; stem echo runs from the ASI row.
        """
        B, T = ids.shape
        forced = torch.full((B, T), -100, dtype=torch.long, device=ids.device)
        if seam_anchor is not None:
            forced = torch.where(seam_anchor >= 0, seam_anchor, forced)
            return forced
        if (self.stem_addr and self.user_id is not None
                and self.asi_id is not None
                and (ids == self.asi_id).any()
                and (ids == self.user_id).any()):
            u = (ids == self.user_id).float().argmax(-1)          # (B,) USER col
            idx = (ids == self.asi_id).float().argmax(-1)         # (B,) ASI row
            have = (ids == self.asi_id).any(-1)
            r = torch.where(have, idx, torch.full_like(idx, T)).long()
            pos_fc = torch.arange(T, device=ids.device)
            # valid rows: r <= t and u + (t - r) + 1 < r  =>  t < 2r - u - 1
            upper = (2 * r - u - 1).clamp(max=T)
            row_valid = ((pos_fc.unsqueeze(0) >= r.unsqueeze(1))
                         & (pos_fc.unsqueeze(0) < upper.unsqueeze(1)))
            col = u.unsqueeze(1) + (pos_fc.unsqueeze(0) - r.unsqueeze(1))
            forced = torch.where(row_valid, col.clamp(min=0), forced)
        return forced

    def stats(self, keys, ids, seam_anchor=None):
        """v6 M1-B: build the IRStats index container (no T^2/T*V tensors)."""
        B = ids.shape[0]
        pos = torch.arange(ids.shape[1], device=ids.device)
        bound = torch.full((B,), ids.shape[1], device=ids.device,
                           dtype=torch.long)
        if self.asi_id is not None and (ids == self.asi_id).any():
            idx = (ids == self.asi_id).float().argmax(-1)
            have = (ids == self.asi_id).any(-1)
            bound = torch.where(have, idx,
                                torch.full_like(idx, ids.shape[1])).long()
        forced = self._forced_columns(ids, seam_anchor)
        return IRStats(ids=ids, keys=keys, beta=self.beta.abs() + 1.0,
                       bound=bound, forced=forced, vocab=self.vocab,
                       asi_id=self.asi_id, eos_id=self.eos_id)

    def forward(self, keys, ids, query, return_attn=False, seam_anchor=None):
        # keys:  (B, T, dim) RAW token embeddings (identity lane, addressing)
        # ids:   (B, T) raw token ids (payload)
        # query: (B, T, dim) — v3.3: IGNORED for addressing (contextual query
        #        was the v3.1 blocker); kept in signature for API compat.
        B, T, D = keys.shape
        a, nxt, n_legal, ctx, behind, mass_same, _ = self._attn(
            keys, ids, seam_anchor=seam_anchor)
        # PAYLOAD = token at position j+1 (masked-off columns carry 0 weight,
        # so their zeroed payload index never scatters).
        mass = torch.zeros(B, T, self.vocab, device=keys.device)
        mass = mass.scatter_add(-1, nxt.unsqueeze(1).expand(B, T, T), a)  # (B,T,V)

        if return_attn:
            return mass, ctx, a, n_legal, behind, mass_same
        return mass, ctx, n_legal, behind, mass_same

    def sparse_forward(self, keys, ids, query, seam_anchor=None):
        """v4 sparse copy-marginal: return position attention + payload ONLY —
        no (B, T, V) copy-mass tensor is materialized. Copy probabilities are
        recovered on demand with copy_prob_sparse() (position gather). For long
        contexts this removes the O(T·V) memory that grows linearly with T (V =
        3190 fixed) and is the base for template-general copy (M2+).
        """
        a, nxt, n_legal, ctx, behind, mass_same, mask = self._attn(
            keys, ids, seam_anchor=seam_anchor)
        return a, nxt, n_legal, ctx, behind, mass_same, mask

    def set_vocab(self, v):
        self.vocab = v


class IRStats:
    """v6 M1-B: register statistics read off the token inverted index.

    Replaces every consumer's access to the dense position attention `a`:

      gate inputs  mass_same / n_legal / behind — EXACT vs the dense path:
          attention similarity is constant within a token group (raw-embed
          self-match), so the (B, T, T) column space collapses LOSSLESSLY onto
          a (B, T, G) group space (G = distinct ids): the row normalizer, the
          softmax weights and the same-id mass are reproduced bit-close while
          never materializing T^2.
      ctx          raw-key segment means weighted by the same group weights —
          EXACT at answer rows t >= bound (where every consumer reads it).
          Prompt rows use the full-group segment mean instead of the <=t
          truncation (documented approximation; nothing consumes them).
      copy lane    OWN-GROUP payload histograms over a flattened
          (b, gid, payload) sort: p_copy(y | t) = hist_g[t][y] / C_g. This is
          the DECLARED semantic change of M1 (release-note item): cross-id
          epsilon-mass is dropped ("twins-uniform snap"). Exact at answer rows
          t >= bound; prompt rows carry the full-group histogram (unconsumed
          downstream: losses and decode read answer rows only).
      decode       per-group payload MODE table (= argmax of the snapped copy
          distribution, ties -> smallest id like torch.argmax) plus
          payloads_of() slices for the exact blend argmax.

    Forced rows (stem-addr / omega-seam anchors) open the gate (mass_same=1),
    leave `behind` False and pin p_copy onto the anchored payload — matching
    the dense path, which mutates `a` AFTER ctx was computed (ctx keeps the
    identity-fallback value there, deliberately).
    """

    def __init__(self, ids, keys, beta, bound, forced, vocab, asi_id, eos_id):
        B, T = ids.shape
        dev = ids.device
        D = keys.shape[-1]
        self.B, self.T, self.vocab, self.G = B, T, vocab, None
        pos = torch.arange(T, device=dev)
        nxt_col = torch.cat([ids[:, 1:], torch.full_like(ids[:, :1], -1)], dim=1)
        bad_payload = (nxt_col == asi_id) | (nxt_col == eos_id) | (nxt_col == -1)
        col_ok = (pos.unsqueeze(0) < bound.unsqueeze(-1)) & ~bad_payload
        self.n_legal = col_ok.cumsum(-1)                        # (B,T)
        behind = torch.zeros(B, T, dtype=torch.bool, device=dev)
        if asi_id is not None and (ids == asi_id).any():
            behind = pos.unsqueeze(0) <= bound.unsqueeze(-1)

        # token groups: stable sort keeps member positions ascending
        sids, perm = ids.sort(dim=-1, stable=True)
        grp_sorted = torch.cat(
            [torch.ones(B, 1, dtype=torch.bool, device=dev),
             sids[:, 1:] != sids[:, :-1]], 1).cumsum(-1) - 1
        G = int(grp_sorted.max()) + 1
        self.G = G
        col_grp = torch.empty_like(grp_sorted).scatter_(1, perm, grp_sorted)
        self.col_grp = col_grp                                  # (B,T) group/column
        rep_slot = torch.searchsorted(
            grp_sorted.contiguous(),
            torch.arange(G, device=dev).unsqueeze(0).expand(B, G).contiguous(),
            side="left").clamp(max=T - 1)   # rows with fewer groups: unused
        rep_col = perm.gather(1, rep_slot)                      # first member col
        # representatives must be normalized EXACTLY like _attn's ek lane
        # (within a group every member shares one vector, so this is exact)
        Krep = F.normalize(keys.gather(1, rep_col.unsqueeze(-1).expand(B, G, D)),
                           dim=-1)

        # ---- (B,T,G) group space: lossless collapse of the dense attention
        H = torch.zeros(B, T, G, device=dev)
        H.scatter_(2, col_grp.unsqueeze(-1), col_ok.float().unsqueeze(-1))
        C = H.cumsum(1)                                         # legal members <=t
        qk = F.normalize(keys, dim=-1)
        sim = (qk @ Krep.transpose(1, 2)) * beta                # (B,T,G)
        logC = C.clamp(min=1).log()
        tot = logC + sim.masked_fill(C == 0, float("-inf"))
        lse = tot.logsumexp(-1)                                 # (B,T) normalizer
        w = torch.exp(tot - lse.unsqueeze(-1))                  # rows sum to 1
        self.mass_same = w.gather(2, col_grp.unsqueeze(-1)).squeeze(-1)
        seg = torch.zeros(B, G, D, device=dev)
        seg.scatter_add_(1, col_grp.unsqueeze(-1).expand(B, T, D),
                         keys * col_ok.unsqueeze(-1).float())
        denom = torch.zeros(B, G, device=dev).scatter_add_(1, col_grp,
                                                          col_ok.float())
        self.denom = denom                                      # full ok count/group
        self.ctx = torch.matmul(w, seg / denom.clamp(min=1.0).unsqueeze(-1))

        self.forced = forced                                    # (B,T) long, -100 none
        self.behind = behind & ~(forced >= 0)
        self.mass_same = torch.where(forced >= 0,
                                     torch.ones_like(self.mass_same),
                                     self.mass_same)
        self.anchor_pay = torch.full((B, T), -100, dtype=torch.long, device=dev)
        fm = forced >= 0
        if fm.any():
            self.anchor_pay[fm] = torch.gather(
                nxt_col, 1, forced.clamp(min=0))[fm]

        # ---- payload histogram runs over OK columns, keyed (b*G+g)*V+y ----
        sel = col_ok.reshape(-1).nonzero(as_tuple=True)[0]
        if sel.numel():
            fb = sel // T
            key = ((fb * G + col_grp.reshape(-1)[sel]) * vocab
                   + nxt_col.reshape(-1)[sel])
            key, order = key.sort()
            starts = torch.ones(key.numel(), dtype=torch.bool, device=dev)
            starts[1:] = key[1:] != key[:-1]
            self.h_key = key                                    # sorted, unique
            self.h_bnd = starts.nonzero(as_tuple=True)[0]
            self.h_cnt = torch.diff(torch.cat(
                [self.h_bnd, torch.tensor([key.numel()], device=dev)]))
            rkey = self.h_key[self.h_bnd]
            self.h_b = rkey // (G * vocab)
            rem = rkey % (G * vocab)
            self.h_g = rem // vocab
            self.h_y = rem % vocab
            # MODE per group: cnt desc (stable) keeps payload asc on ties
            om = self.h_cnt.sort(descending=True, stable=True)[1]
            rb, rg, ry, rc = (self.h_b[om], self.h_g[om],
                              self.h_y[om], self.h_cnt[om])
            first = torch.ones(rb.numel(), dtype=torch.bool, device=dev)
            first[1:] = (rb[1:] != rb[:-1]) | (rg[1:] != rg[:-1])
            m_idx = rb[first] * G + rg[first]
            self.mode_val = torch.full((B * G,), -100, dtype=torch.long,
                                       device=dev)
            self.mode_val[m_idx] = ry[first]
            self.mode_cnt = torch.zeros(B * G, dtype=torch.long, device=dev)
            self.mode_cnt[m_idx] = rc[first]
        else:
            self.h_key = torch.zeros(0, dtype=torch.long, device=dev)
            self.h_bnd = self.h_key
            self.h_cnt = self.h_key.clone()
            self.h_b = self.h_g = self.h_y = self.h_key
            self.mode_val = torch.full((B * G,), -100, dtype=torch.long,
                                       device=dev)
            self.mode_cnt = torch.zeros(B * G, dtype=torch.long, device=dev)

        idx2d = (torch.arange(B, device=dev).unsqueeze(1) * G + col_grp)
        self.copy_mode = self.mode_val[idx2d.reshape(-1)].reshape(B, T)
        self.copy_mode_p = (self.mode_cnt[idx2d.reshape(-1)].float()
                            / denom.clamp(min=1.0).reshape(-1)[idx2d.reshape(-1)]
                            ).reshape(B, T)
        if fm.any():
            # a forced row's dense attention is ONE-HOT on the anchor column,
            # so its snapped copy distribution is a single payload at 1.0 —
            # expose that through the decode candidates too.
            self.copy_mode[fm] = self.anchor_pay[fm]
            self.copy_mode_p[fm] = 1.0

    def _run_slice(self, b, g):
        """[lo, hi) index range of group (b, g)'s runs in the flat histogram."""
        base = (b * self.G + g) * self.vocab
        lo = int(torch.searchsorted(self.h_key,
                                    torch.tensor(base, device=self.h_key.device),
                                    side="left"))
        hi = int(torch.searchsorted(
            self.h_key,
            torch.tensor(base + self.vocab - 1, device=self.h_key.device),
            side="right"))
        return lo, hi

    def payloads_of(self, b, t):
        """(values, fractions) of row (b, t)'s own-group payload histogram.

        Forced rows return exactly the anchored payload at 1.0 (one-hot
        attention semantics)."""
        if bool(self.forced[b, t] >= 0):
            dev = self.anchor_pay.device
            return (torch.tensor([int(self.anchor_pay[b, t])], device=dev),
                    torch.tensor([1.0], device=dev))
        lo, hi = self._run_slice(b, int(self.col_grp[b, t]))
        vals = self.h_y[lo:hi]
        fr = self.h_cnt[lo:hi].float() / float(self.denom[b, int(self.col_grp[b, t])].clamp(min=1))
        return vals, fr

    def prob_at(self, targets):
        """p_copy(target | row) for target token ids (B,T), -100 ignored.

        Snapped semantics: the mass of the row's own token group spreads
        uniformly over its legal members' PAYLOADS; hist fraction = count /
        group size. Rows with a forced anchor pin p onto the anchored payload
        (1 or 0) exactly like the overridden dense attention does.
        """
        B, T = self.B, self.T
        dev = targets.device
        yq = targets.clamp(min=0)
        qkey = ((torch.arange(B, device=dev).unsqueeze(1) * self.G
                 + self.col_grp.to(dev)) * self.vocab + yq).reshape(-1)
        if self.h_key.numel():
            lo = torch.searchsorted(self.h_key, qkey, side="left")
            hi = torch.searchsorted(self.h_key, qkey, side="right")
            hit = hi > lo
            cnt = torch.where(hit, self.h_cnt[lo.clamp(max=self.h_key.numel() - 1)],
                              torch.zeros_like(lo))
        else:
            cnt = torch.zeros_like(qkey)
        denom = self.denom.reshape(-1).clamp(min=1.0)[
            (torch.arange(B, device=dev).unsqueeze(1) * self.G
             + self.col_grp).reshape(-1).to(self.denom.device)]
        p = cnt.reshape(B, T).to(self.denom.device).float() \
            / denom.reshape(B, T)
        p = torch.where(targets >= 0, p, torch.zeros_like(p))
        fpay = (targets.to(self.anchor_pay.device) == self.anchor_pay).float()
        p = torch.where(self.forced.to(dev) >= 0, fpay.to(dev), p.to(dev))
        return p


class DualHeadDecoder(nn.Module):
    """gen ⊕ copy. Pointer-generator style: final distribution is a convex blend
      p = (1-g)*softmax(gen) + g*copy_dist
    returned as log p. copy_dist = register attention mass over vocab (an exact
    context token can get ~1.0 mass -> hard copy achievable).

    g is now DETERMINISTIC (v3.3-final, measured 2026-08-13): the copy lane is
    trustworthy iff the query's attention mass lands on a column holding the
    SAME token id as the seed (an exact string duplicate in the prompt). We sum
    attention over same-token-id columns:
        gate_mass = sum_j a_j * [ids[j] == ids[t]]
        g = sigmoid(tau * (gate_mass - thr))
    An exact in-prompt twin gives ~0.99 (on); when the seed's only twin is the
    ASI/EOS-boundary column (excluded by the payload mask) that column carries
    zero softmax mass, gate_mass collapses toward 0, and gen must emit
    (EOS/first token). Top-1 mass and entropy gates were both tried and failed:
    at a boundary row attention falls back 'medium-sharp' on a WRONG token and
    both signals stay high, so decode looped the slot value forever.
    A learned gate on [h, ctx] (v3.1-v3.3) collapsed to ~0.05 — same outcome
    via a different mechanism (see identity-register postmortem).
    """

    def __init__(self, dim, vocab, tie_embed=None, gate_bias=0.0, gate=None,
                 gate_threshold=0.5, gate_clamp=6.0, tau_init=12.0):
        super().__init__()
        # gen takes [h, gate_mass, behind] (dim+2): the two extra channels are a
        # DIRECT boundary signal. At the last answer row gate_mass collapses to
        # ~0 while behind=0, so gen can learn a reliable "emit EOS" rule there
        # instead of a per-slot lottery (pkg094 'rude' bug, 2026-08-13). At the
        # first-answer row behind=1 -> 'pip'. Without these gen is a bare
        # Linear(h) and ~1-2/40 unseen slots flip the EOS decision per seed.
        self.gen = nn.Linear(dim + 2, vocab, bias=False)
        self.gate_bias = gate_bias
        self.tau = nn.Parameter(torch.tensor(float(tau_init)))
        self.gate_threshold = gate_threshold
        self.gate_clamp = gate_clamp
        # v4 M2: pluggable learned gate (RelativeGate); None keeps the
        # deterministic same-token-id gate that is v3.3-final.
        self.gate = gate
        if tie_embed is not None:
            # tie only the h-cols (embed is dim-wide); the two extra boundary
            # channels stay free.
            self.gen.weight.data[:, :dim] = tie_embed.detach()

    def gen_probs(self, h, gate_mass, behind):
        """Conditioned gen distribution for external logging/aux CE. Mirrors
        forward() so HMN3.forward's gen_logits uses the SAME [h,gm,behind]
        input the blend does (otherwise the aux CE trains a different head)."""
        b = behind.unsqueeze(-1).float()
        gm = gate_mass.unsqueeze(-1)
        return torch.log_softmax(self.gen(torch.cat([h, gm, b], -1)), -1)

    def gate_and_gen(self, h, gate_mass, behind, n_legal=None):
        """v6 M1-B: stats-path entry — gen log-probs + gate value WITHOUT
        building any copy tensor or blended logits. Gate logic is identical
        to forward()'s; the blend itself moved to loss_v33/decode_v33 via
        exact logaddexp over the target/candidate support."""
        b = behind.unsqueeze(-1).float()
        gm = gate_mass.unsqueeze(-1)
        gen = torch.log_softmax(self.gen(torch.cat([h, gm, b], -1)), -1)
        if self.gate is not None:
            top2 = gen.topk(2, dim=-1)
            margin = (top2[0][..., 0] - top2[0][..., 1]).detach()
            g = self.gate(h, gate_mass, margin, behind, n_legal)
        else:
            g = torch.sigmoid((self.tau * (gate_mass.unsqueeze(-1) - self.gate_threshold))
                              .clamp(-self.gate_clamp, self.gate_clamp))
            g = g * (1.0 - b)                      # prompt/ASI rows: gen
        return gen, g

    def forward(self, h, copy_logits, ctx, attn=None, n_legal=None, behind=None,
                gate_mass=None, eps=1e-8, sparse=False, nxt=None):
        if gate_mass is not None:
            b = behind.unsqueeze(-1).float()
            gm = gate_mass.unsqueeze(-1)
            gen = torch.log_softmax(self.gen(torch.cat([h, gm, b], -1)), -1)
            if self.gate is not None:
                # v4 M2 relative gate: gen_margin = top1-top2 log prob =
                # the gen head's own confidence. Detached so the gate learns
                # to READ confidence without warping the gen head's gradient.
                top2 = gen.topk(2, dim=-1)
                margin = (top2[0][..., 0] - top2[0][..., 1]).detach()
                g = self.gate(h, gate_mass, margin, behind, n_legal)
            else:
                # deterministic same-token-id gate (see class docstring).
                # threshold and clamp now configurable (v9 M22).
                g = torch.sigmoid((self.tau * (gate_mass.unsqueeze(-1) - self.gate_threshold))
                                  .clamp(-self.gate_clamp, self.gate_clamp))
                g = g * (1.0 - b)                      # prompt/ASI rows: gen
        else:
            # legacy path (no attn provided): fall back to a static open gate
            gen = torch.log_softmax(self.gen(
                torch.cat([h, torch.zeros_like(h[..., :1]), torch.zeros_like(h[..., :1])], -1)), -1)
            g = torch.sigmoid(torch.full_like(gen[..., :1], self.gate_bias).clamp(-3, 3))
        if sparse:
            # v4 sparse copy-marginal: forward does NOT materialize (B,T,V).
            # We still need a (B,T,V) copy dist for the BLEND, but it is built
            # from position attention via a gather that skips the masked columns
            # (they carry zero attention). This keeps blend/decode semantics
            # identical to the dense path while the LOSS can read p_copy purely
            # from (attn, nxt) without ever expanding to vocab width.
            B, T = nxt.shape
            copy = torch.zeros(B, T, copy_logits.shape[-1] if copy_logits is not None
                               else self.gen.out_features, device=nxt.device)
            copy = copy.scatter_add(-1, nxt.unsqueeze(1).expand(B, T, T), attn)
            copy_dist = F.normalize(copy.clamp(min=0.0), p=1, dim=-1)
            p = (1 - g) * gen.exp() + g * copy_dist
            return torch.log(p.clamp(min=eps)), g, copy_dist
        # copy mass already sums ~1 over vocab; renormalize to be safe
        copy_dist = F.normalize(copy_logits, p=1, dim=-1)
        p = (1 - g) * gen.exp() + g * copy_dist
        return torch.log(p.clamp(min=eps)), g


class RelativeGate(nn.Module):
    """v4 M2: LEARNED copy gate on the register statistics, supervised directly
    against the copy-mask G (BCE). The v3.1 learned gate (Linear[h,ctx] ->
    sigmoid) collapsed to ~0.05 because it was trained only through the blend CE
    — a free-running scalar had no pressure to discriminate copy vs gen rows.
    Here g is trained with an explicit target (G==1 rows should open copy,
    G==0 rows should close it), and it sees the same-token-id mass PLUS the
    counter-signal v3.1 lacked:

        gen_margin = gen top-1 minus top-2 log prob. High gen margin means the
        gen head is already confident, so copy is not needed (it would only
        overwrite a correct prediction with a memorized twin); low margin means
        gen is unsure and an exact in-prompt twin is the reliable lane.

    Input channels: [h, gate_mass, gen_margin, behind, n_legal_norm].
    Deterministic gate stays the DEFAULT; this module only swaps it in when
    HMN3(gate_mode="relative")."""

    def __init__(self, dim, gate_bias=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim + 4, dim + 4),
            nn.SiLU(),
            nn.Linear(dim + 4, 1),
        )
        self.bias = nn.Parameter(torch.tensor(float(gate_bias)), requires_grad=False)

    def forward(self, h, gate_mass, gen_margin, behind, n_legal):
        x = torch.cat([h.float(),
                       gate_mass.unsqueeze(-1),
                       gen_margin.unsqueeze(-1),
                       behind.unsqueeze(-1).float(),
                       (n_legal.float().clamp(min=0) / 2.0).unsqueeze(-1)], -1)
        # v4 M2b: NO forced (1-behind) mask — unlike the deterministic gate
        # (v3.3), which needs it because it has no confidence signal, the
        # relative gate watches gen_margin itself and can decide row 0 (the
        # template verb) independently. That is precisely the unseen-template
        # failure M3 measured (gen emits a *seen* verb like 'pip' for probes).
        g = torch.sigmoid(self.mlp(x) + self.bias)
        return g


class SeedPointer(nn.Module):
    """v5 omega-seam: fragment-run seeder for the Identity Register.

    The reorder wall (M12a-d) decomposes into (a) fragment-seam gate collapse
    and (b) no mechanism to START a new copy run mid-answer. This module owns
    exactly that start: at a SEAM row it predicts
      - which prompt column the run's payload chain starts from (pointer CE
        against gold run anchors), and
      - how many tokens the run lasts (length CE against gold run lengths).
    Between seams the register runs the proven deterministic positional echo
    (stem-addr generalized: anchor col c advances +1 per answer row), so the
    learned module is only consulted at run boundaries — the smallest possible
    learned surface, everything else stays content-addressable identity.

    ptr_logits = beta * cos(Wq h_t, raw keys), masked to columns before ASI.
    len_logits = Linear(h_t) over classes 1..max_run (class i <-> length i+1).
    Default OFF; HMN3(seam_addr=True) instantiates it.
    """

    def __init__(self, dim, max_run=16, beta=20.0):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.len_head = nn.Linear(dim, max_run)
        self.max_run = max_run
        self.beta = beta

    def forward(self, h, keys, bound):
        # h, keys: (B,T,D); bound: (B,) long — first illegal column (ASI col or T)
        q = F.normalize(self.q(h.float()), dim=-1)
        k = F.normalize(keys.float(), dim=-1)
        logits = (q @ k.transpose(-1, -2)) * self.beta          # (B,T,T)
        cols = torch.arange(logits.shape[-1], device=h.device)
        legal = cols.unsqueeze(0) < bound.unsqueeze(-1)          # (B,T)
        logits = logits.masked_fill(~legal.unsqueeze(1), float("-inf"))
        len_logits = self.len_head(h.float())                    # (B,T,max_run)
        return logits, len_logits


class AttentionSeedPointer(nn.Module):
    """v8 M17: multi-head cross-attention seed pointer.

    Replaces v5's cosine-similarity pointer with full multi-head attention:
      1. h (last hidden) as query, x (raw embeddings) as key/value
      2. Output is contextualized per-column representation
      3. Attention weights become the pointer logits (softmax-normalized
         distribution over prompt columns, scaled by learnable temperature)

    Advantage: the pointer can attend to positional structure and context
    jointly, not just raw embedding similarity.  The length head is unchanged.

    Default OFF; HMN3(seam_addr=True, attn_ptr=True) instantiates it.
    """

    def __init__(self, dim, max_run=16, n_heads=4, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.len_head = nn.Linear(dim, max_run)
        self.max_run = max_run
        # learnable temperature for pointer logits (initialized ~beta=20)
        self.ptr_temp = nn.Parameter(torch.tensor(3.0))

    def forward(self, h, keys, bound):
        B, T, D = h.shape
        nh, hd = self.n_heads, self.head_dim
        q = self.q_proj(h.float()).view(B, T, nh, hd).transpose(1, 2)
        k = self.k_proj(keys.float()).view(B, T, nh, hd).transpose(1, 2)
        v = self.v_proj(keys.float()).view(B, T, nh, hd).transpose(1, 2)
        attn = (q @ k.transpose(-1, -2)) * self.scale          # (B,nh,T,T)
        cols = torch.arange(T, device=h.device)
        legal = cols.unsqueeze(0) < bound.unsqueeze(-1)         # (B,T)
        attn = attn.masked_fill(~legal.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn_w = F.softmax(attn, dim=-1)
        attn_w = self.dropout(attn_w)
        out = (attn_w @ v).transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)
        # pointer logits: average attention across heads, scaled by temperature
        ptr_logits = attn_w.mean(dim=1) * self.ptr_temp.exp()  # (B,T,T)
        len_logits = self.len_head(h.float())
        return ptr_logits, len_logits


class LatentThinkingBuffer(nn.Module):
    """Adaptive deliberation: re-run the WR block sequence over the last hidden
    state, refining it in latent space without decoding to text tokens. Stops
    when the top-1 output confidence stops increasing (convergence) or K_max
    reached. Zero new vocab cost — pure compute scaling with input difficulty."""

    def __init__(self, dim, k_max, thr=0.02):
        super().__init__()
        self.k_max = k_max
        self.thr = thr
        self.adapt = nn.Linear(dim, dim)
        # v4 M4: init adapt as ~identity so at init think is roughly a no-op
        # (the +adapt residual barely shifts h). Keeps the think on/off
        # ablation honest: whatever M4 measures comes from training, not from
        # an arbitrary random hop.
        nn.init.zeros_(self.adapt.weight)
        nn.init.zeros_(self.adapt.bias)

    def forward(self, h, block_fn):
        prev_conf = None
        hk = h
        for k in range(self.k_max):
            hk = block_fn(hk)
            hk = hk + self.adapt(hk)
            conf = torch.softmax(hk[..., 0, :], -1).max(-1).values
            if prev_conf is not None and ((conf - prev_conf).abs().mean() < self.thr):
                break
            prev_conf = conf
        return hk


class HMN3(nn.Module):
    """Full v3: embed -> WR (L coupling blocks) -> IR (raw-id lane) ->
    dual head (gated copy⊕gen). Optionally LatentThinkingBuffer before the head
    (compute-scaling deliberation, enabled via use_think).
    If aux_copy>0, forward also returns copy_dist (register attention mass);
    the trainer adds CE(copy_dist, Y) so the register is shaped DIRECTLY to
    point at the correct context token — otherwise a gen head that memorizes
    seen slots starves the copy path of gradient (seen=learn, unseen=0
    deadlock observed in the PoC).
    """

    def __init__(self, vocab_size, dim=96, state_dim=8, n_layers=3, n_experts=16,
                 top_k=2, use_moe=False, use_think=False, k_max=4, tie_weights=True,
                 gate_bias=0.0, aux_copy=True, asi_id=None,
                 sparse_marginal=False, gate_mode="deterministic", user_id=None,
                 stem_addr=False, seam_addr=False, max_run=16,
                 exact_blend=False, attn_ptr=False,
                 ir_beta_init=30.0, ir_gate_threshold=0.5, ir_gate_clamp=6.0, ir_tau_init=12.0,
                 ssm_chunk_size=8, ssm_clamp=-9.0,
                 rope_base=10000.0, ffn_mult=4.0,
                 moe_key_dim=16, moe_capacity_factor=1.25, moe_z_loss_coef=0.001):
        super().__init__()
        # v6 M1-B: False (default) = index/stats path (no (B,T,T)/(B,T,V));
        # True = legacy dense oracle (bit-faithful pre-M1 behavior) for parity
        # tests and --exact-blend debugging.
        self.exact_blend = exact_blend
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList(
            [HelixCouplingBlock(dim, state_dim, chunk_size=ssm_chunk_size,
                                clamp=ssm_clamp) for _ in range(n_layers)]
        )
        self.moe_off = nn.Identity()
        self.use_moe = use_moe
        self.moe_list = nn.ModuleList([
            SparseConditionalCompute(dim, n_experts, top_k) if use_moe else None
            for _ in range(n_layers)])
        self.ir = IdentityRegister(dim, asi_id=asi_id,
                                   user_id=user_id, stem_addr=stem_addr)
        self.ir.set_vocab(vocab_size)
        # v4 M2: pluggable gate — deterministic stays default (v3.3-final),
        # RelativeGate is the learned alternative supervised by the copy mask.
        gate = RelativeGate(dim, gate_bias) if gate_mode == "relative" else None
        self.dual = DualHeadDecoder(dim, vocab_size,
                                    tie_embed=self.embed.weight if tie_weights else None,
                                    gate_bias=gate_bias, gate=gate,
                                    gate_threshold=ir_gate_threshold,
                                    gate_clamp=ir_gate_clamp,
                                    tau_init=ir_tau_init)
        self.use_think = use_think
        self.think = LatentThinkingBuffer(dim, k_max) if use_think else None
        self.aux_copy = aux_copy
        self.sparse_marginal = sparse_marginal
        # v5 omega-seam: fragment-run seeding (reorder/transform tasks).
        # Default OFF -> parameter set identical to v3.3/v4 checkpoints.
        self.seam_addr = seam_addr
        # v8 M17: attention-based seed pointer (attn_ptr=True) or cosine (default)
        self.seed_ptr = (AttentionSeedPointer(dim, max_run=max_run)
                         if seam_addr and attn_ptr
                         else SeedPointer(dim, max_run=max_run) if seam_addr
                         else None)

    def moe_aux_loss(self):
        if not self.use_moe:
            return torch.tensor(0.0)
        return sum(m.last_aux_loss * m.aux_coef for m in self.moe_list if m is not None)

    def blocks_apply(self, x=None):
        def _block_fn(h):
            for i, blk in enumerate(self.blocks):
                h = ReversibleFunction.apply(h, [blk])
                if self.moe_list[i] is not None:
                    h = h + self.moe_list[i](h)
            return h
        return _block_fn

    def forward(self, input_ids, return_gate=False, return_attn=False,
                seam_anchor=None, exact_blend=None):
        ex = self.exact_blend if exact_blend is None else exact_blend
        ids = input_ids
        x = self.embed(ids)                       # raw token lane (IR keys)
        block_fn = self.blocks_apply()
        h = block_fn(x)
        if self.use_think:
            h = self.think(h, block_fn)
        if not ex:
            # ---- v6 M1-B stats path: consumers read the IRStats API; no
            # dense `a`, no (B,T,V) copy mass, no blended logits tensor.
            st = self.ir.stats(x, ids, seam_anchor=seam_anchor)
            gen_logits, g = self.dual.gate_and_gen(h, st.mass_same, st.behind,
                                                   n_legal=st.n_legal)
            d = {"gen_logits": gen_logits, "g": g, "stats": st,
                 "n_legal": st.n_legal}
        elif self.sparse_marginal:
            a, nxt, n_legal, ctx, behind, gate_mass, _ = \
                self.ir.sparse_forward(x, ids, h, seam_anchor=seam_anchor)
            logits, g, copy_dist = self.dual(h, None, ctx, attn=a, n_legal=n_legal,
                                             behind=behind, gate_mass=gate_mass,
                                             sparse=True, nxt=nxt)
            attn = a
            gen_logits = self.dual.gen_probs(h, gate_mass, behind)
            d = {"logits": logits, "g": g, "copy_dist": copy_dist, "attn": attn,
                 "n_legal": n_legal, "gen_logits": gen_logits, "nxt": nxt}
        else:
            copy_logits, ctx, attn, n_legal, behind, gate_mass = \
                self.ir(x, ids, h, return_attn=True, seam_anchor=seam_anchor)
            logits, g = self.dual(h, copy_logits, ctx, attn=attn, n_legal=n_legal,
                                  behind=behind, gate_mass=gate_mass)
            copy_dist = F.normalize(copy_logits.clamp(min=0.0), p=1, dim=-1)
            gen_logits = self.dual.gen_probs(h, gate_mass, behind)
            d = {"logits": logits, "g": g, "copy_dist": copy_dist, "attn": attn,
                 "n_legal": n_legal, "gen_logits": gen_logits}
        if self.seed_ptr is not None:
            # v5 omega-seam: run-start pointer + run-length heads. Legal seed
            # columns are the prompt region (before the ASI boundary column).
            B, T = ids.shape
            bound = torch.full((B,), T, dtype=torch.long, device=ids.device)
            if self.ir.asi_id is not None and (ids == self.ir.asi_id).any():
                idx = (ids == self.ir.asi_id).float().argmax(-1)
                have = (ids == self.ir.asi_id).any(-1)
                bound = torch.where(have, idx, torch.full_like(idx, T)).long()
            ptr_logits, len_logits = self.seed_ptr(h, x, torch.clamp(bound - 1, min=0))
            d["ptr_logits"] = ptr_logits
            d["len_logits"] = len_logits
        if return_gate and ex:
            return logits, g
        return d


class HMN3_NoReg(nn.Module):
    """Ablation: same WR + dual-head but copy channel disabled (pure softmax).
    Control for measuring what the register itself contributes."""

    def __init__(self, vocab_size, dim=96, state_dim=8, n_layers=3, tie_weights=True):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList(
            [HelixCouplingBlock(dim, state_dim) for _ in range(n_layers)])
        self.head = nn.Linear(dim, vocab_size, bias=False)
        if tie_weights:
            self.head.weight = self.embed.weight

    def forward(self, input_ids):
        h = self.embed(input_ids)
        for blk in self.blocks:
            h = ReversibleFunction.apply(h, [blk])
        return self.head(h)

    def moe_aux_loss(self):
        return torch.tensor(0.0)