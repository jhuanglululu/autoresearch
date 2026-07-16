"""SwiGLU feed-forward network.

Swap the FFN (plain MLP, GeGLU, MoE) by editing this file alone.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from lab.model.gpt import ModelConfig


class SwiGLU(nn.Module):
    """SwiGLU FFN: down(silu(gate(x)) * up(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))
