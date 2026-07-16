"""KV-cache correctness proof (CPU, tiny random-weights model).

Asserts that greedy generation WITH the KV cache produces byte-for-byte identical
token ids to the naive full-recompute path, over >=64 decode steps, for batch 1
and batch 4. This is the guarantee that speed experiments measure architecture,
not a broken cache.

Run it (from inside the lab):

    uv run --project . python lab/test_kvcache.py

Exits non-zero on any mismatch. This is a standalone check, not a pytest module —
the lab is a template, so its self-checks live next to the code they guard rather
than in the repo's tests/ tree.
"""
from __future__ import annotations

import sys

import torch

from lab.model import GPT, ModelConfig

DECODE_STEPS = 80          # comfortably above the required 64
PROMPT_LEN = 16
BLOCK_SIZE = 256           # PROMPT_LEN + DECODE_STEPS < BLOCK_SIZE -> no window effects


def _tiny_model() -> GPT:
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=257, d_model=64, n_layers=3, n_heads=4, d_ff=128,
        block_size=BLOCK_SIZE, dropout=0.0,
    )
    return GPT(cfg).eval()


def _check(model: GPT, batch: int) -> None:
    torch.manual_seed(123 + batch)
    prompt = torch.randint(0, model.cfg.vocab_size, (batch, PROMPT_LEN))
    naive = model.generate(prompt, max_new_tokens=DECODE_STEPS, greedy=True, use_cache=False)
    cached = model.generate(prompt, max_new_tokens=DECODE_STEPS, greedy=True, use_cache=True)

    assert naive.shape == cached.shape, f"shape mismatch {naive.shape} vs {cached.shape}"
    if not torch.equal(naive, cached):
        diff = (naive != cached)
        first = int(diff.float().argmax())
        row, col = divmod(first, naive.shape[1])
        raise AssertionError(
            f"batch {batch}: token ids diverge at [row={row}, col={col}] "
            f"(naive={naive[row, col].item()} cached={cached[row, col].item()}); "
            f"{int(diff.sum())}/{diff.numel()} positions differ"
        )
    print(f"OK  batch {batch}: {DECODE_STEPS} decode steps, {naive.shape[1]} total ids identical")


def main() -> int:
    model = _tiny_model()
    for batch in (1, 4):
        _check(model, batch)
    print("PASS: KV-cached greedy decoding == naive greedy decoding")
    return 0


if __name__ == "__main__":
    sys.exit(main())
