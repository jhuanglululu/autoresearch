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

from lab.model.attention import LayerKVCache
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

    def _rope_at(self, pos_offset, T, device, dtype):
        """cos/sin rows for the T absolute positions [pos_offset, pos_offset+T).
        Grows the (lazily built) cache if a decode step reaches past its length."""
        end = pos_offset + T
        if self._rope_cos is None or self._rope_cos.shape[0] < end or self._rope_cos.device != device:
            cos, sin = build_rope_cache(
                max(end, self.cfg.block_size), self.cfg.head_dim, self.cfg.rope_theta, device, dtype
            )
            self._rope_cos, self._rope_sin = cos, sin
        return self._rope_cos[pos_offset:end], self._rope_sin[pos_offset:end]

    def _rope(self, T, device, dtype):
        return self._rope_at(0, T, device, dtype)

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

    # ------------------------------------------------------------------ decoding

    def _forward_cached(self, idx, pos_offset: int, caches: list[LayerKVCache]):
        """Forward a chunk of tokens whose FIRST token sits at absolute position
        `pos_offset`, threading a per-layer KV cache through the blocks. RoPE uses
        the absolute positions [pos_offset, pos_offset+T); each block appends its
        new k/v to its cache and attends over the whole cached history. Returns
        logits (B, T, V). No autograd/dropout — decoding only."""
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx))
        cos, sin = self._rope_at(pos_offset, T, idx.device, x.dtype)
        for block, cache in zip(self.blocks, caches):
            x = block(x, cos, sin, cache=cache)
        x = self.norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def prefill(self, idx):
        """Run a prompt through a FRESH KV cache. Returns (next_token_logits,
        caches): the logits (B, V) that predict the first new token, and the
        populated per-layer caches to keep decoding against. Exposed so
        time-to-first-token can be measured as exactly the prefill cost."""
        caches = [LayerKVCache() for _ in self.blocks]
        logits = self._forward_cached(idx, 0, caches)
        return logits[:, -1, :], caches

    def _sample_next(self, logits, temperature: float, top_k: int | None, greedy: bool):
        """Pick the next token from last-position logits (B, V) -> (B, 1)."""
        if greedy:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / max(temperature, 1e-6)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < v[:, [-1]], -float("inf"))
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, *, greedy: bool = False, use_cache: bool = True):
        """Autoregressive generation.

        use_cache=True (default): prefill the prompt once, then decode one token
        at a time against a KV cache — the fast path and the reference for speed
        experiments. use_cache=False: the naive full-recompute-per-token path,
        kept only for the equivalence test.

        Both paths are numerically identical (identical token ids under greedy
        decoding) as long as generation stays within block_size. Past block_size
        they DIVERGE by design: the naive path crops to a sliding window and keeps
        going, while the cached path STOPS (the KV cache / RoPE table only span
        the trained causal window; cropping the cache would corrupt absolute RoPE
        positions). Benchmarks and samples stay well within block_size.
        """
        if not use_cache:
            return self._generate_naive(idx, max_new_tokens, temperature, top_k, greedy)

        next_logits, caches = self.prefill(idx)
        out = idx
        for i in range(max_new_tokens):
            if out.shape[1] >= self.cfg.block_size:
                break  # causal window full — stop (see docstring)
            nxt = self._sample_next(next_logits, temperature, top_k, greedy)
            out = torch.cat((out, nxt), dim=1)
            if i == max_new_tokens - 1 or out.shape[1] >= self.cfg.block_size:
                break  # produced the last token; skip the unused decode forward
            pos_offset = out.shape[1] - 1              # absolute position of `nxt`
            next_logits = self._forward_cached(nxt, pos_offset, caches)[:, -1, :]
        return out

    def _generate_naive(self, idx, max_new_tokens, temperature, top_k, greedy):
        """Original generate: full forward over the (cropped) context every step."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            nxt = self._sample_next(logits[:, -1, :], temperature, top_k, greedy)
            idx = torch.cat((idx, nxt), dim=1)
        return idx

    def num_params(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        embedding = self.tok_emb.weight.numel()   # tied with lm_head, counted once
        return {"params": total, "embedding_params": embedding}
