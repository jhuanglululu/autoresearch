"""Loads the goal's PINNED corpus and tokenizer from their read-only mounts
(/assets/corpus, /assets/tokenizer) with a fixed train/val split.

Editable like everything in the lab — but the underlying files are not.
TODO(implement): pre-tokenized memmap reader + batch sampler.
"""
