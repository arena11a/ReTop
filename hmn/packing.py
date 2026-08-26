"""v6 M5 — sequence packing utilities (doc-masked, per-document positions).

Roadmap item (docs/v6_scaling_roadmap.md M5): "sequence packing (doc-masked,
position_ids)" — the data-side half of the distributed milestone. Section 2.4
fixes the contract these helpers implement: packing is PER-DOCUMENT with the
loss-mask triple (Y/Yc/G) carried through — never a naive concat.

Guarantees (all verified in experiments/v6/m5_packed_sequence.py):

  * pack_sequences -> doc_masked_padding -> unpack_outputs round-trips
    EXACTLY: unpack(pack(docs)) == docs, token for token.
  * position_ids restart at every document boundary (0-based inside each
    doc, 0 on padding). The Identity Register is content-addressed (raw-id
    self-match) and imposes no positional scheme of its own, but every
    position-consuming consumer (RoPE on the roadmap-2.2 attention-WR
    variant, HF `position_ids` inputs) must not see doc k+1 continuing
    doc k's position clock.
  * Padding is EOS with an explicit attention mask (1 real / 0 pad) and a
    doc_id map, so padded positions drop out of every loss/decode consumer
    through the standard -100/-1 target masks.

Honest boundary: these utilities move DOCUMENTS and MASKS; they do not
change model math. The production WR is a causal SSM recurrence and the IR
binds the FIRST <|assistant|> column of the row it sees, so forwarding a
packed row lets doc k>1 read doc<k SSM state and share token-twin groups
across documents. Exact per-document isolation therefore needs doc-aware
masking / state resets at the MODEL level (the FSDP2/DeepSpeed wrap half of
M5). The M5 experiment proves the EXACT case (architecturally isolated
documents: packed loss == sum of per-document losses, packed grads == their
summed grads), measures — rather than hides — the residual cross-doc effect
of the recurrent WR on later documents, and shows earlier documents are
causally untouched.
"""
import torch


def pack_sequences(sequences):
    """Concatenate short documents into ONE sequence, tracking boundaries.

    sequences : list of token-id lists (or 1-D LongTensors). Documents are
    expected to carry their own terminator (the chat recipe ends every gold
    with EOS); no separator is injected — boundaries are tracked by LENGTH,
    which is what keeps the round-trip exact.

    Returns (packed_ids list[int], doc_lens list[int]) with
    sum(doc_lens) == len(packed_ids).
    """
    if not sequences:
        raise ValueError("pack_sequences: need at least one document")
    packed, lens = [], []
    for d in sequences:
        if isinstance(d, torch.Tensor):
            d = d.tolist()
        d = [int(t) for t in d]
        if not d:
            raise ValueError("pack_sequences: empty document")
        packed.extend(d)
        lens.append(len(d))
    return packed, lens


def doc_masked_padding(packed_ids, doc_lens, max_len, pad_id, device=None):
    """Pad a packed row to max_len with EOS + per-document attention mask.

    Returns a dict:
      input_ids      (1, L) long   — packed tokens then pad_id fill
      attention_mask (1, L) long   — 1 on real document tokens, 0 on padding
                                     ("doc-masked": mask edges ARE doc edges)
      doc_id         (1, L) long   — index of the owning document per column,
                                     -1 on padding
    """
    L = len(packed_ids)
    if max_len < L:
        raise ValueError(f"doc_masked_padding: max_len {max_len} < packed "
                         f"length {L}")
    ids = torch.full((1, max_len), int(pad_id), dtype=torch.long, device=device)
    am = torch.zeros((1, max_len), dtype=torch.long, device=device)
    did = torch.full((1, max_len), -1, dtype=torch.long, device=device)
    ids[0, :L] = torch.tensor(packed_ids, dtype=torch.long, device=device)
    am[0, :L] = 1
    col = 0
    for k, ln in enumerate(doc_lens):
        did[0, col:col + ln] = k
        col += ln
    return {"input_ids": ids, "attention_mask": am, "doc_id": did}


def doc_position_ids(doc_lens, total_len, device=None):
    """Per-document position_ids: restart at 0 at EVERY doc boundary.

    Returns (1, total_len) long. Doc k occupies columns off_k..off_k+len_k-1
    and receives positions 0..len_k-1; padding columns get 0 (they are
    attention-masked out everywhere the ids are consumed).
    """
    pos = torch.zeros((1, total_len), dtype=torch.long, device=device)
    col = 0
    for ln in doc_lens:
        if col + ln > total_len:
            raise ValueError("doc_position_ids: doc_lens exceed total_len")
        pos[0, col:col + ln] = torch.arange(ln, device=device)
        col += ln
    return pos


def unpack_outputs(outputs, doc_lens):
    """Split model outputs back to per-doc lengths.

    outputs : a Tensor with batch dim 1 (sliced along dim 1), or a dict as
              returned by HMN3.forward — every TENSOR value is sliced along
              dim 1; non-tensor entries (e.g. the packed IRStats object,
              which indexes the whole row) are carried through unchanged —
              evaluate per-document copy probabilities against it by
              row-masking prob_at's targets, as the M5 experiment does.

    Returns a list with one entry per document: the tensor slice, or a dict
    of that document's slices.
    """
    offs = [0]
    for ln in doc_lens:
        offs.append(offs[-1] + ln)

    def _slice(x, lo, hi):
        if isinstance(x, torch.Tensor):
            if x.dim() < 2 or x.shape[0] != 1:
                raise ValueError("unpack_outputs: expected leading shape "
                                 f"(1, L, ...) got {tuple(x.shape)}")
            return x[:, lo:hi]
        return x

    if isinstance(outputs, torch.Tensor):
        return [outputs[:, offs[k]:offs[k + 1]] for k in range(len(doc_lens))]
    if isinstance(outputs, dict):
        out = []
        for k in range(len(doc_lens)):
            lo, hi = offs[k], offs[k + 1]
            out.append({key: _slice(v, lo, hi) for key, v in outputs.items()})
        return out
    raise TypeError("unpack_outputs: expected Tensor or dict")


def pack_batch(sequences, max_len, pad_id, device=None):
    """One-call trainer entry: pack -> pad -> positions.

    Returns the doc_masked_padding dict plus "position_ids" (per-doc reset)
    and "doc_lens". Feed input_ids/position_ids to the model; carry
    attention_mask/doc_id so targets can be placed per document and padding
    masked out of every loss term.
    """
    packed, lens = pack_sequences(sequences)
    batch = doc_masked_padding(packed, lens, max_len, pad_id, device=device)
    batch["position_ids"] = doc_position_ids(lens, max_len, device=device)
    batch["doc_lens"] = lens
    return batch
