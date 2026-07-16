"""One-time corpus download for the EXAMPLE goal:
erhwenkuo/wikipedia-zhtw (~1 GB) -> data/wikipedia-zhtw/ (the goal's `corpus` asset).

Run once during box setup:  python scripts/download_dataset.py
Other goals bring their own asset-prep script. Subagents never run these and
never touch the outputs (DESIGN.md — a goal's assets are pinned).
"""
from __future__ import annotations

from pathlib import Path

OUT = Path("data/wikipedia-zhtw")


def main() -> None:
    if OUT.exists():
        print(f"{OUT} already exists — nothing to do (delete it to re-download)")
        return
    from datasets import load_dataset  # requires the [experiment] extra

    ds = load_dataset("erhwenkuo/wikipedia-zhtw")
    OUT.mkdir(parents=True)
    ds.save_to_disk(str(OUT))
    print(f"saved to {OUT}")
    # TODO: pre-tokenize with ../tokenizer.json into a flat memmap token file
    # (harness/data.py reads that, with a fixed train/val split).


if __name__ == "__main__":
    main()
