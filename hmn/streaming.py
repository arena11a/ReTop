"""v6 M6 — streaming data pipeline (docs/v6_scaling_roadmap.md M6).

Community-scale training on corpora that never fit in RAM: the reader walks
jsonl line-by-line, the shuffle holds only a bounded reservoir window, and
chat-id conversion happens on the fly (`hmn.recipe.make_chat_ids` — identity-
register labels stay computed per document at use time, no precomputed label
storage, roadmap section 2.5).

Pass criterion (roadmap): stream w/o RAM growth. The seeded bounded shuffle
also gives "val split stable across restarts" for free: a fixed seed fixes
the eviction sequence, so restarting reproduces the exact stream order.

Composition (plain iterators end to end):

  reader = StreamJsonlReader("/data/shard-*.jsonl")   # or HF Hub stream later
  shuf   = BoundedBufferShuffle(reader, buffer_size=8192, seed=0)
  ids    = ChatIdConverter(tok).stream(shuf)          # {"text": ...} echo task
  ds     = InfiniteStreamDataset(lambda: ids, epoch_size=100_000)
  DataLoader(ds, batch_size=B,
             collate_fn=partial(pad_collate, pad_id=eos_id))
"""
import json
import random

import torch
from torch.utils.data import Dataset

from hmn.recipe import make_chat_ids


class StreamJsonlReader:
    """Line-by-line jsonl reader — NEVER loads the file into memory.

    Yields parsed dict records (or `rec[key]` when `key=` is given, for
    single-field corpora). Blank lines are skipped. Accepts a single path or
    an ordered list of shard paths (shards are read in the given order, so a
    fixed shard order + fixed seed reproduces the exact stream).

    The instance IS an iterator (`__next__`), and exhausting it resets it, so
    a reader can be re-iterated like it was fresh. Error handling via
    `on_error`: "strict" lets a malformed line raise its json.JSONDecodeError
    (default — silent corruption is worse than a crash); "skip" drops the
    line and counts it in `.skipped`.
    """

    def __init__(self, paths, key=None, on_error="strict"):
        if on_error not in ("strict", "skip"):
            raise ValueError(f"on_error must be 'strict' or 'skip', got {on_error!r}")
        self.paths = [paths] if isinstance(paths, str) else list(paths)
        self.key = key
        self.on_error = on_error
        self.skipped = 0
        self._gen = None

    def _records(self):
        for path in self.paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        if self.on_error == "skip":
                            self.skipped += 1
                            continue
                        raise
                    yield rec[self.key] if self.key is not None else rec

    def __iter__(self):
        return self

    def __next__(self):
        if self._gen is None:
            self._gen = self._records()
        try:
            return next(self._gen)
        except StopIteration:
            self._gen = None                  # exhausted -> reusable
            raise

    def count(self):
        """Record count via full rescan (O(1) memory)."""
        return sum(1 for _ in self)


class BoundedBufferShuffle:
    """Reservoir-style shuffle with a FIXED memory bound.

    Fill the buffer to `buffer_size`; each later sample evicts a uniformly
    drawn slot and takes its place (the evicted sample is emitted). At end of
    stream the buffer drains in shuffled order. Every input sample is emitted
    EXACTLY once; peak memory is min(seen, buffer_size) samples.
    buffer_size=1 degenerates to file order (a one-slot window cannot
    reorder anything).

    Honest boundary: with B << N this is decorrelation for SGD, NOT a uniform
    permutation — samples move at most ~O(N/B) positions.

    Fixed seed => deterministic eviction sequence => identical order after a
    restart (the roadmap's stable-val-split requirement).
    """

    def __init__(self, source, buffer_size=10000, seed=0):
        if buffer_size < 1:
            raise ValueError("buffer_size must be >= 1")
        self.source = source
        self.buffer_size = int(buffer_size)
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed)
        buf = []
        for x in self.source:
            if len(buf) < self.buffer_size:
                buf.append(x)
                continue
            j = rng.randrange(self.buffer_size)
            yield buf[j]
            buf[j] = x
        while buf:
            j = rng.randrange(len(buf))
            buf[j], buf[-1] = buf[-1], buf[j]
            yield buf.pop()


class ChatIdConverter:
    """Stream record -> chat-id token list via recipe.make_chat_ids.

    Defaults target text-field corpora (fineweb/slimpajama-class): the user
    text comes from rec["text"] and is trained as an ECHO task (gold == user),
    matching the guardrail corpus shape where every answer token has an exact
    prompt twin. Records with explicit chat fields win by constructor:
    ChatIdConverter(tok, user_key="user", gold_key="gold"); a missing gold
    field degrades to echo. Per-part encoding keeps every special as its
    exact single id (see the make_chat_ids docstring).
    """

    def __init__(self, tok, user_key="text", gold_key="text"):
        self.tok = tok
        self.user_key = user_key
        self.gold_key = gold_key

    def __call__(self, rec):
        user = rec[self.user_key]
        gold = rec.get(self.gold_key, user) if self.gold_key else None
        return make_chat_ids(self.tok, user, gold)

    def stream(self, records):
        """Lazily map the converter over any record iterable."""
        return (self(r) for r in records)


class InfiniteStreamDataset(Dataset):
    """Map-style torch Dataset over an UNBOUNDED stream.

    The stream is cycled: when it exhausts, `make_iterator()` rebuilds it
    transparently (with a seeded shuffle the rebuilt pass is identical, so
    epochs are reproducible). `__len__` is the configured epoch_size — samples
    per epoch, NOT corpus size (a stream has no corpus size).

    Access contract: sequential index order (DataLoader's SequentialSampler,
    num_workers=0). Out-of-order access raises instead of silently returning
    wrong data — a stream cannot serve random access, and multi-worker
    DataLoaders interleave indices, so they are rejected by the same check.
    Call `.reset()` between epochs (DataLoader re-reads indices 0..len-1).
    """

    def __init__(self, make_iterator, epoch_size=100000):
        self.make_iterator = make_iterator
        self.epoch_size = int(epoch_size)
        self._it = None
        self._pos = 0

    def __len__(self):
        return self.epoch_size

    def reset(self):
        """Restart the stream at position 0 (call between epochs)."""
        self._it = None
        self._pos = 0

    def _next(self):
        if self._it is None:
            self._it = iter(self.make_iterator())
        try:
            item = next(self._it)
        except StopIteration:
            self._it = iter(self.make_iterator())
            item = next(self._it)
        return item

    def __getitem__(self, idx):
        if idx != self._pos:
            raise RuntimeError(
                f"InfiniteStreamDataset is sequential-access: got index {idx}, "
                f"expected {self._pos} (use num_workers=0)")
        item = self._next()
        self._pos += 1
        return item

    def __iter__(self):
        self.reset()
        for idx in range(self.epoch_size):
            yield self[idx]


def pad_collate(batch, pad_id):
    """Right-pad variable-length id lists -> (X LongTensor (B, T_max), lengths).

    Padding mirrors make_slot_batch (pad with EOS so padded rows look like
    finished sequences); loss targets must be masked downstream.
    """
    t_max = max(len(x) for x in batch)
    X = torch.full((len(batch), t_max), pad_id, dtype=torch.long)
    for i, x in enumerate(batch):
        X[i, :len(x)] = torch.as_tensor(x, dtype=torch.long)
    lengths = torch.tensor([len(x) for x in batch], dtype=torch.long)
    return X, lengths
