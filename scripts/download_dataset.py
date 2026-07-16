"""One-time corpus download for the EXAMPLE goal:
erhwenkuo/wikipedia-zhtw (~1 GB) -> data/wikipedia-zhtw/ (the goal's `corpus` asset).

Run once during box setup:  python scripts/download_dataset.py
Other goals bring their own asset-prep script. Subagents never run these and
never touch the outputs (DESIGN.md — a goal's assets are pinned).

This does two things, both under data/wikipedia-zhtw/ (the mounted `corpus` asset):
  1. save_to_disk of the raw HF dataset (kept for provenance / re-tokenization).
  2. pre-tokenize the whole corpus with ../tokenizer.json into a flat token
     stream the lab reads directly:
       tokens.bin  flat little-endian uint16 token ids, no header
       meta.json   format descriptor (dtype, n_tokens, vocab_size, tokenizer, ...)
Pre-tokenizing here (once) keeps every training run cheap and deterministic and
means the lab never needs the `datasets` dependency.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path("data/wikipedia-zhtw")
TOKENIZER = Path("tokenizer.json")   # repo-root pinned tokenizer asset
TEXT_COLUMN = "text"
DTYPE = np.uint16                    # vocab is 8000 < 2**16, so uint16 is enough
SHARD_TOKENS = 8_000_000            # flush to disk in chunks to bound memory


def pretokenize(dataset, tokenizer, out_dir: Path) -> dict:
    """Tokenize every document, append <eos> between docs, write a flat memmap."""
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        eos_id = tokenizer.token_to_id("<|endoftext|>")
    tokens_path = out_dir / "tokens.bin"

    n_tokens = 0
    n_docs = 0
    buf: list[int] = []
    with tokens_path.open("wb") as f:
        for split in dataset:                       # DatasetDict -> iterate splits
            for row in dataset[split]:
                text = row.get(TEXT_COLUMN)
                if not text:
                    continue
                ids = tokenizer.encode(text).ids
                buf.extend(ids)
                if eos_id is not None:
                    buf.append(eos_id)
                n_docs += 1
                if len(buf) >= SHARD_TOKENS:
                    np.asarray(buf, dtype=DTYPE).tofile(f)
                    n_tokens += len(buf)
                    buf.clear()
        if buf:
            np.asarray(buf, dtype=DTYPE).tofile(f)
            n_tokens += len(buf)

    meta = {
        "format": "flat little-endian token-id stream, no header; documents joined by <eos>",
        "dtype": DTYPE.__name__,
        "n_tokens": n_tokens,
        "docs": n_docs,
        "vocab_size": tokenizer.get_vocab_size(),
        "eos_id": eos_id,
        "tokenizer": TOKENIZER.name,
        "source": "erhwenkuo/wikipedia-zhtw",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    if (OUT / "tokens.bin").exists():
        print(f"{OUT}/tokens.bin already exists — nothing to do (delete {OUT} to re-run)")
        return
    from datasets import load_dataset, load_from_disk  # requires the [experiment] extra
    from tokenizers import Tokenizer

    OUT.mkdir(parents=True, exist_ok=True)
    hf_dir = OUT / "hf"
    if hf_dir.exists():
        ds = load_from_disk(str(hf_dir))
    else:
        ds = load_dataset("erhwenkuo/wikipedia-zhtw")
        ds.save_to_disk(str(hf_dir))
        print(f"saved raw dataset to {hf_dir}")

    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    meta = pretokenize(ds, tokenizer, OUT)
    print(f"pre-tokenized -> {OUT}/tokens.bin ({meta['n_tokens']:,} tokens, "
          f"{meta['docs']:,} docs) + meta.json")


if __name__ == "__main__":
    main()
