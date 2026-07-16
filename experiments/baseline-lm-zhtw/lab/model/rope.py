"""Rotary position embeddings (RoPE): cos/sin cache + application.

Kept in its own file (not folded into attention) because positional encoding is
an independent axis an experiment swaps on its own — e.g. try ALiBi, NoPE, or a
different theta schedule — by editing ONLY this file. `build_rope_cache`
(the cache) and `apply_rope` (the per-tensor rotation) belong together: they are
the two halves of one scheme and always change in lockstep. `attention.py`
imports `apply_rope`; `gpt.py` owns the cache via `build_rope_cache`.
"""
from __future__ import annotations

import torch


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype):
    """Precompute cos/sin tables of shape (seq_len, head_dim) for RoPE."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)            # (seq_len, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)       # (seq_len, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, n_heads, T, head_dim). cos/sin: (T, head_dim)."""
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin
