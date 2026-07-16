"""Pre-norm transformer block: attention + FFN, each with a residual.

Uses torch's builtin RMSNorm (torch>=2.4). Swap the block wiring (norm
placement, parallel attn+ffn, extra sublayers) by editing this file alone.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn

from lab.model.attention import CausalSelfAttention
from lab.model.ffn import SwiGLU

if TYPE_CHECKING:
    from lab.model.gpt import ModelConfig


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.d_model, eps=1e-5)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = nn.RMSNorm(cfg.d_model, eps=1e-5)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache=None):
        # `cache` is None on the training/forward path (unchanged); during cached
        # decoding it is this block's LayerKVCache, threaded straight to attention.
        x = x + self.attn(self.attn_norm(x), cos, sin, cache=cache)   # pre-norm
        x = x + self.ffn(self.ffn_norm(x))
        return x
