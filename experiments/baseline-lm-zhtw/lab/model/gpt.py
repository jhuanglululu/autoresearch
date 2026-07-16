"""Baseline ~50M decoder-only LM (nanoGPT-style): causal MHA + SwiGLU + RMSNorm + RoPE.

This file owns the assembled model — `ModelConfig`, the embeddings, the block
stack, the tied head, `generate`, and `num_params`. The pieces live in their own
files so an experiment can swap one by editing one file:
  lab/model/rope.py       cos/sin cache + RoPE application
  lab/model/attention.py  causal multi-head self-attention
  lab/model/ffn.py        SwiGLU feed-forward
  lab/model/block.py      pre-norm transformer block (uses torch.nn.RMSNorm)

Normalization is torch's builtin `nn.RMSNorm` (torch>=2.4), not a hand-rolled one.

Param arithmetic (defaults below, tokenizer vocab = 8000):
  Let d = d_model = 640, L = n_layers = 9, F = d_ff = 1728, V = vocab = 8000.
  Per layer:
    attention (q,k,v,o, no bias)     = 4 * d*d           = 4 * 409_600 = 1_638_400
    SwiGLU FFN (gate, up, down)      = 3 * d*F           = 3 * 640*1728 = 3_317_760
    2x RMSNorm gains                 = 2 * d             =        1_280
    per-layer total                                      =    4_957_440
  All layers: 9 * 4_957_440                              =   44_616_960
  Token embedding (weight-tied w/ lm_head, counted once): V*d = 8000*640 = 5_120_000
  Final RMSNorm gain: d                                  =          640
  RoPE: no learned params.
  TOTAL = 44_616_960 + 5_120_000 + 640                   =   49_737_600  (~49.7M)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from lab.model.block import Block
from lab.model.rope import build_rope_cache


@dataclass
class ModelConfig:
    vocab_size: int = 8000    # set from the tokenizer at load time
    d_model: int = 640
    n_layers: int = 9
    n_heads: int = 10         # head_dim = 640 / 10 = 64
    d_ff: int = 1728          # ~ 8/3 * d_model, rounded to a multiple of 64
    block_size: int = 1024    # max context length
    dropout: float = 0.0
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        return self.d_model // self.n_heads


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = nn.RMSNorm(cfg.d_model, eps=1e-5)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight   # weight tying

        self._rope_cos = None
        self._rope_sin = None
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _rope(self, T, device, dtype):
        if self._rope_cos is None or self._rope_cos.shape[0] < T or self._rope_cos.device != device:
            cos, sin = build_rope_cache(
                max(T, self.cfg.block_size), self.cfg.head_dim, self.cfg.rope_theta, device, dtype
            )
            self._rope_cos, self._rope_sin = cos, sin
        return self._rope_cos[:T], self._rope_sin[:T]

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx))
        cos, sin = self._rope(T, idx.device, x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, nxt), dim=1)
        return idx

    def num_params(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        embedding = self.tok_emb.weight.numel()   # tied with lm_head, counted once
        return {"params": total, "embedding_params": embedding}
