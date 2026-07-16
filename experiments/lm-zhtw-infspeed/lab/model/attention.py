"""Causal multi-head self-attention with RoPE applied to q/k, plus an optional
per-layer KV cache for incremental (autoregressive) decoding.

Swap the attention mechanism (GQA, sliding window, a different mask) by editing
this file alone. Positional encoding lives in `rope.py`; this module only
*applies* it via `apply_rope`.

KV-cache contract
-----------------
A `LayerKVCache` holds this layer's post-RoPE keys and values for every token
seen so far, each shaped (B, n_heads, T_cached, head_dim). `forward(..., cache=...)`
computes q/k/v for the CURRENT chunk of tokens, RoPE-rotates q and k for their
absolute positions, appends the new k/v to the cache, and attends the current
queries against the FULL cached k/v. Keys are cached AFTER RoPE (RoPE is
absolute, so a token's rotated key is fixed forever — see rope.apply_rope).

The training / plain-forward path passes `cache=None` and is byte-for-byte the
original implementation: one `is_causal=True` SDPA call, no cache, same numerics.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from lab.model.rope import apply_rope

if TYPE_CHECKING:
    from lab.model.gpt import ModelConfig


class LayerKVCache:
    """Per-layer key/value cache for one attention module.

    Stores post-RoPE k and v of shape (B, n_heads, T_cached, head_dim). `update`
    appends the current step's (already RoPE-applied) k/v and returns the full
    cached tensors. Growth is by concatenation — simple and readable; the cost is
    dwarfed by the matmuls it lets us skip (preallocation is a later optimization).
    """

    def __init__(self):
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    def update(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = torch.cat((self.k, k), dim=2)
            self.v = torch.cat((self.v, v), dim=2)
        return self.k, self.v

    def __len__(self) -> int:
        return 0 if self.k is None else self.k.shape[2]


def _sliced_causal_mask(offset: int, t_new: int, t_total: int, device) -> torch.Tensor:
    """Boolean attend-mask (True = keep) for `t_new` queries at absolute positions
    [offset, offset+t_new) against `t_total` keys at positions [0, t_total).
    Query i may attend to key j iff j <= offset + i (causality with an arbitrary
    prefix already in the cache). Shape (t_new, t_total); broadcasts over B/heads.
    """
    q_pos = torch.arange(offset, offset + t_new, device=device).unsqueeze(1)  # (t_new, 1)
    k_pos = torch.arange(t_total, device=device).unsqueeze(0)                 # (1, t_total)
    return k_pos <= q_pos


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, cache: LayerKVCache | None = None):
        """x: (B, T, C). cos/sin: (T, head_dim) for the ABSOLUTE positions of x's
        T tokens. `cache` None -> plain training/forward path (unchanged). When a
        cache is given, the T new tokens' k/v are appended and the queries attend
        over the whole cached sequence.
        """
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is None:
            # Training / naive forward: square lower-triangular causal attention.
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            )
        else:
            k, v = cache.update(k, v)                 # (B, nh, T_total, hd)
            t_total = k.shape[2]
            offset = t_total - T                      # tokens already cached before this chunk
            if offset == 0:
                # Prefill over the whole prompt: identical to the training path.
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
            else:
                # Decode a suffix (usually T==1) against an existing prefix.
                mask = _sliced_causal_mask(offset, T, t_total, x.device)
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)
