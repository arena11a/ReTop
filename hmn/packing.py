"""v6 M5 — sequence packing utilities (docs/v6_scaling_roadmap.md, M5).

Scale training needs many short documents per step. Two doc-faithful forms
live here ("packing is per-document with loss masks carried through, not
naive concat"):

  pack_sequences()      greedy concatenation of short docs into bins of at
                        most max_len tokens. Doc boundaries are tracked as
                        (start, end) spans + a per-position doc index, so
                        unpack_outputs() restores the originals EXACTLY and
                        position ids can restart at every doc start (IR
                        compatibility: the register addresses raw token ids,
                        never absolute offsets — a future RoPE-WR variant
                        must see document-local positions too).
  doc_masked_padding()  ONE doc per row, right-padded with EOS + attention
                        mask. Right padding is the parity-safe form for this
                        architecture: every consumer is causal (chunked SSM
                        scan in hmn/v2.py, triu-masked register in
                        hmn/v3.py, teacher-forced CE), so a pad token
                        strictly RIGHT of a document can never enter any
                        consumed activation. Packed-batch rows are therefore
                        numerically identical to the individual forwards —
                        the property the M5 loss/gradient-parity gate pins.
  unpack_outputs()      split model outputs back to per-doc lengths: batched
                        (B, T, ...) tensors are cut at each row's length;
                        flat 1-D packed streams are cut at cumulative lens.

No HMN3 changes: pure data-shape utilities; the model keeps its
(input_ids) contract. The FSDP2/DeepSpeed wrap consumes them at the trainer
layer; experiments/v6/m5_packed_sequence.py verifies the roadmap's explicit
precondition FIRST — ReversibleFunction autograd compat with packed batches.
"""

from collections import namedtuple

import torch

PackedBin = namedtuple("PackedBin",
                       ["ids", "lens", "spans", "doc_index", "position_ids"])
PaddedDocs = namedtuple("PaddedDocs",
                        ["ids", "attn_mask", "position_ids", "lens"])

DOC_NONE = -100  # doc_index value for separator slots (repo sentinel)


def _as_long_tensor(doc):
    if torch.is_tensor(doc):
        return doc.detach().to(torch.long).cpu()
    return torch.tensor(list(doc), dtype=torch.long)


def pack_sequences(docs, max_len=None, sep_id=None):
    """Greedily concatenate short docs into bins of <= max_len tokens.

    docs    : iterable of int sequences / 1-D LongTensors. Non-empty (chat
              docs already end with their own </s>, so no separator is
              needed by default).
    max_len : bin capacity in tokens; None -> everything in one bin. A doc
              longer than max_len gets its OWN oversized bin — never split,
              because splitting would corrupt the answer-region masks.
    sep_id  : optional separator id inserted BETWEEN consecutive docs inside
              a bin (counted toward max_len). Separators belong to no doc:
              doc_index marks them DOC_NONE and they trail the preceding
              doc's position counter.

    Returns [PackedBin(ids, lens, spans, doc_index, position_ids), ...] in
    input order. Round trip is exact:
        unpack_outputs(b.ids, b.lens) == original docs   for every bin b.
    """
    bins = []
    pieces, spans, lens, size = [], [], [], 0

    def _close():
        nonlocal pieces, spans, lens, size
        if not pieces:
            return
        ids = torch.cat(pieces)
        L = int(ids.numel())
        pos = torch.zeros(L, dtype=torch.long)
        idx = torch.full((L,), DOC_NONE, dtype=torch.long)
        prev_end = 0
        for k, (s, e) in enumerate(spans):
            pos[prev_end:s] = torch.arange(prev_end, s)  # separators trail
            pos[s:e] = torch.arange(e - s)               # per-doc reset
            idx[s:e] = k
            prev_end = e
        bins.append(PackedBin(ids=ids,
                              lens=torch.tensor(lens, dtype=torch.long),
                              spans=list(spans), doc_index=idx,
                              position_ids=pos))
        pieces, spans, lens, size = [], [], [], 0

    for doc in docs:
        d = _as_long_tensor(doc)
        n = int(d.numel())
        if n == 0:
            raise ValueError("empty documents break span bookkeeping")
        sep_here = 1 if (sep_id is not None and pieces) else 0
        if max_len is not None and pieces and size + sep_here + n > max_len:
            _close()
            sep_here = 0                      # new bin starts without a sep
        if sep_here:
            pieces.append(torch.tensor([sep_id], dtype=torch.long))
            size += 1
        spans.append((size, size + n))
        lens.append(n)
        pieces.append(d)
        size += n
    _close()
    return bins


def doc_position_ids(lens):
    """Per-document position ids for a flat concatenation of doc `lens`:
    counters restart at 0 at every doc boundary (packed-concat form)."""
    lens = [int(l) for l in lens]
    return (torch.cat([torch.arange(l, dtype=torch.long) for l in lens])
            if lens else torch.zeros(0, dtype=torch.long))


def doc_masked_padding(docs, max_len, pad_id):
    """Right-pad one doc per row to max_len with pad_id (EOS in practice).

    Returns PaddedDocs(ids (B,L), attn_mask (B,L long, 1=real / 0=pad),
    position_ids (B,L), lens (B,)). One doc per row means the position ids
    ARE the per-document counters already (each row restarts at 0); the
    cross-boundary reset matters only in the concatenated form.

    The Y/Yc/G loss triple travels through the SAME padding: pad Y/Yc with
    -100 and G with -1.0 so every loss_v33 mask ignores pad rows by
    construction (packing carries the masks through, never drops them).
    """
    tens = [_as_long_tensor(d) for d in docs]
    B = len(tens)
    longest = max(int(d.numel()) for d in tens)
    if max_len < longest:
        raise ValueError(f"max_len={max_len} < longest doc={longest}")
    ids = torch.full((B, max_len), int(pad_id), dtype=torch.long)
    attn = torch.zeros(B, max_len, dtype=torch.long)
    for b, d in enumerate(tens):
        n = int(d.numel())
        ids[b, :n] = d
        attn[b, :n] = 1
    pos = torch.arange(max_len, dtype=torch.long).unsqueeze(0) \
        .expand(B, -1).contiguous()
    return PaddedDocs(ids=ids, attn_mask=attn, position_ids=pos,
                      lens=torch.tensor([int(d.numel()) for d in tens],
                                        dtype=torch.long))


def unpack_outputs(tensor, lens):
    """Split model outputs back to per-doc lengths.

    Batched (B, T, ...) input: [tensor[b, :lens[b]]] — the inverse of
    doc_masked_padding (pad suffixes dropped).
    Flat 1-D input: one concatenated packed stream cut at cumulative lens —
    the inverse of pack_sequences (spans/separators outside doc content are
    ignored, so the round trip is exact).
    """
    lens = [int(l) for l in lens]
    if tensor.dim() == 1:
        out, off = [], 0
        for l in lens:
            out.append(tensor[off:off + l])
            off += l
        if off != int(tensor.numel()):
            raise ValueError(f"lens sum {off} != stream length "
                             f"{tensor.numel()}")
        return out
    if tensor.shape[0] != len(lens):
        raise ValueError(f"batch {tensor.shape[0]} != len(lens) {len(lens)}")
    return [tensor[b, :l] for b, l in enumerate(lens)]
