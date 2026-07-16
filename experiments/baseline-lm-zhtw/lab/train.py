"""Baseline training + validation loop. Fully editable — add metrics/logging
whenever a run needs explaining.

AdamW + cosine schedule with linear warmup, grad clipping, bf16 autocast when
CUDA is available (plain fp32 on CPU). Validation runs periodically; metrics.json
is rewritten atomically after every eval so a killed run still leaves partial
metrics behind. The best (lowest val loss) weights are checkpointed to
model.safetensors (+ model.json sidecar). Every train point, eval, and
checkpoint is also appended to records.jsonl via the RecordLog.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from lab.checkpoint import save_checkpoint


@dataclass
class TrainConfig:
    steps: int = 20000
    batch_size: int = 32
    grad_accum: int = 1
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 500
    eval_iters: int = 100
    log_interval: int = 20
    seed: int = 1337


def make_optimizer(model, cfg: TrainConfig):
    # Weight-decay only 2D+ params (matmuls/embeddings); leave norms/biases alone.
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    if step >= cfg.steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


@torch.no_grad()
def estimate_val_loss(model, data, cfg: TrainConfig, device, autocast_ctx) -> float:
    model.eval()
    losses = torch.zeros(cfg.eval_iters)
    for i in range(cfg.eval_iters):
        x, y = data.get_batch("val", cfg.batch_size, model.cfg.block_size, device)
        with autocast_ctx():
            _, loss = model(x, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


def train(model, data, cfg: TrainConfig, device, metrics: dict, save_metrics, records) -> dict:
    """Train in place, updating `metrics` and calling save_metrics(metrics) after
    every eval. Appends train_log/eval/checkpoint events to `records` (a
    RecordLog). Returns the metrics dict. Writes best weights to
    model.safetensors (+ model.json sidecar)."""
    torch.manual_seed(cfg.seed)
    use_cuda = device.type == "cuda"
    if use_cuda:
        autocast_ctx = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        # No autocast on CPU (bf16 matmuls are slow/unsupported); run fp32.
        import contextlib
        autocast_ctx = contextlib.nullcontext

    optimizer = make_optimizer(model, cfg)
    tokens_per_step = cfg.batch_size * model.cfg.block_size * cfg.grad_accum

    metrics.setdefault("val_history", [])
    metrics["steps"] = cfg.steps
    best_val = float("inf")
    train_seconds = 0.0
    tokens_seen = 0
    model.train()

    for step in range(cfg.steps):
        lr = lr_at(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(cfg.grad_accum):
            x, y = data.get_batch("train", cfg.batch_size, model.cfg.block_size, device)
            with autocast_ctx():
                _, loss = model(x, y)
            (loss / cfg.grad_accum).backward()
            loss_accum += loss.item() / cfg.grad_accum
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        if use_cuda:
            torch.cuda.synchronize()
        train_seconds += time.perf_counter() - t0
        tokens_seen += tokens_per_step

        metrics["train_loss"] = round(loss_accum, 4)

        if step % cfg.log_interval == 0 or step == cfg.steps - 1:
            tok_s = tokens_seen / train_seconds if train_seconds > 0 else 0.0
            print(
                f"step {step:>6}/{cfg.steps} | loss {loss_accum:7.4f} | lr {lr:.2e} "
                f"| grad {grad_norm:5.2f} | {tok_s:8.0f} tok/s",
                flush=True,
            )
            records.log(
                "train_log", step=step, loss=round(loss_accum, 4), lr=lr,
                grad_norm=round(float(grad_norm), 4), tokens_per_sec=round(tok_s, 1),
            )

        if (step % cfg.eval_interval == 0 and step > 0) or step == cfg.steps - 1:
            val_loss = estimate_val_loss(model, data, cfg, device, autocast_ctx)
            metrics["val_loss"] = round(val_loss, 4)
            metrics["val_perplexity"] = round(math.exp(min(val_loss, 20)), 3)
            metrics["val_history"].append({"step": step, "val_loss": round(val_loss, 4)})
            metrics["tokens_per_sec"] = round(tokens_seen / train_seconds, 1)
            metrics["tokens_trained"] = tokens_seen
            print(f"  eval @ {step}: val_loss {val_loss:.4f} "
                  f"ppl {metrics['val_perplexity']:.2f}", flush=True)
            records.log(
                "eval", step=step, val_loss=round(val_loss, 4),
                val_perplexity=metrics["val_perplexity"], tokens_trained=tokens_seen,
            )
            is_best = val_loss < best_val
            if is_best:
                best_val = val_loss
                save_checkpoint(model, step=step, val_loss=val_loss)
                records.log(
                    "checkpoint", step=step, val_loss=round(val_loss, 4),
                    best=True, path="model.safetensors",
                )
            save_metrics(metrics)

    metrics["tokens_per_sec"] = round(tokens_seen / train_seconds, 1) if train_seconds else 0.0
    metrics["tokens_trained"] = tokens_seen
    metrics["train_seconds"] = round(train_seconds, 2)
    metrics["best_val_loss"] = round(best_val, 4)
    save_metrics(metrics)
    return metrics
