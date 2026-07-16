"""Loads the goal's PINNED corpus and tokenizer (read-only files; paths come
from run_config.toml [assets]) with a fixed train/val split.

Editable like everything in the lab — but the underlying files are not.

The corpus asset directory is expected to contain a PRE-TOKENIZED flat token
stream produced once during box setup (see scripts/download_dataset.py):
  <corpus>/tokens.bin   flat little-endian uint16/uint32 token ids, no header
  <corpus>/meta.json    {"dtype", "n_tokens", "vocab_size", "tokenizer", ...}
The tokenizer asset is tokenizer.json (either the file itself or a directory
containing it) — used here only for vocab size and for decoding samples.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

_DTYPES = {"uint16": np.uint16, "uint32": np.uint32}

# Held-out validation set size, in TOKENS (a fixed count, not a fraction). The
# pinned corpus is a few hundred M tokens; 1M held-out tokens is ~0.3-0.5% of it
# — a rounding error against training data, yet ~1000x a single eval batch, so
# val loss is measured on a large, stable, run-comparable slice with negligible
# sampling noise. A fixed count (vs a fraction) keeps the val set identical if
# the corpus is ever re-tokenized to a slightly different length. The small-
# corpus guard below never lets val exceed half the stream (matters for the CPU
# smoke corpus, which is far smaller than VAL_SIZE).
VAL_SIZE = 1_000_000


def resolve_tokenizer_path(tokenizer_asset: str) -> Path:
    p = Path(tokenizer_asset)
    if p.is_dir():
        p = p / "tokenizer.json"
    if not p.is_file():
        raise FileNotFoundError(f"tokenizer.json not found at {tokenizer_asset}")
    return p


def load_tokenizer(tokenizer_asset: str) -> Tokenizer:
    return Tokenizer.from_file(str(resolve_tokenizer_path(tokenizer_asset)))


class TokenData:
    """Memmapped flat token stream with a fixed, deterministic train/val split.

    The split is by position: the last `VAL_SIZE` tokens of the stream are the
    val set (capped at half the corpus for tiny streams), the rest is train.
    Deterministic across runs (no shuffling of the split boundary), so runs stay
    comparable.
    """

    def __init__(self, corpus_asset: str):
        corpus = Path(corpus_asset)
        meta_path = corpus / "meta.json"
        tokens_path = corpus / "tokens.bin"
        if not meta_path.is_file() or not tokens_path.is_file():
            raise FileNotFoundError(
                f"expected pre-tokenized corpus at {corpus} (tokens.bin + meta.json); "
                "run scripts/download_dataset.py during setup"
            )
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        dtype = _DTYPES[self.meta["dtype"]]
        self.tokens = np.memmap(tokens_path, dtype=dtype, mode="r")
        self.vocab_size = int(self.meta["vocab_size"])

        n = len(self.tokens)
        n_val = min(VAL_SIZE, n // 2)       # never let val swallow the corpus
        self.n_val = n_val
        self._train = self.tokens[: n - n_val]
        self._val = self.tokens[n - n_val :]

    def split(self, name: str) -> np.memmap:
        return self._train if name == "train" else self._val

    def get_batch(self, name: str, batch_size: int, block_size: int, device, generator=None):
        """Contiguous-chunk sampler: random start offsets, then (x, y) shifted by one."""
        data = self.split(name)
        high = len(data) - block_size - 1
        if high <= 0:
            raise ValueError(f"{name} split too small ({len(data)} tokens) for block_size {block_size}")
        ix = torch.randint(high, (batch_size,), generator=generator)
        x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
        if device.type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)
