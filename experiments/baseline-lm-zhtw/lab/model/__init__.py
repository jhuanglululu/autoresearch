"""Model package: the assembled GPT plus its swappable pieces.

Public API — `from lab.model import GPT, ModelConfig`. The internals live
one-concern-per-file so an experiment can swap a piece by editing one file:
  lab/model/rope.py       cos/sin cache + RoPE application
  lab/model/attention.py  causal multi-head self-attention
  lab/model/ffn.py        SwiGLU feed-forward
  lab/model/block.py      pre-norm transformer block (uses torch.nn.RMSNorm)
  lab/model/gpt.py        ModelConfig + assembled GPT
"""
from __future__ import annotations

from lab.model.gpt import GPT, ModelConfig

__all__ = ["GPT", "ModelConfig"]
