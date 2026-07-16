"""Standard evals: val loss/perplexity, tokens/sec, peak VRAM, param counts.
Keep these for comparability; extend freely. Deviations show up in the run
record for the orchestrator to notice.

Most numbers are produced during training (tokens/sec, val loss); this module
finalizes the suite: a clean full-val-set perplexity, peak VRAM, param counts,
and a short generated text sample for the run record.
"""
from __future__ import annotations

import math

import torch


@torch.no_grad()
def final_val_loss(model, data, device, batch_size: int, iters: int = 200) -> float:
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = data.get_batch("val", batch_size, model.cfg.block_size, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    return sum(losses) / len(losses)


@torch.no_grad()
def sample_text(model, tokenizer, device, max_new_tokens: int = 120, prompt: str = "") -> str:
    model.eval()
    if prompt:
        ids = tokenizer.encode(prompt).ids
    else:
        bos = tokenizer.token_to_id("<bos>")
        ids = [bos] if bos is not None else [0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.8, top_k=200)
    return tokenizer.decode(out[0].tolist())


def evaluate(model, data, tokenizer, device, cfg, metrics: dict, save_metrics, records) -> dict:
    """Finalize the standard eval suite, merge into `metrics`, and append a
    `final_evals` event to `records` (a RecordLog)."""
    counts = model.num_params()
    metrics.update(counts)

    val = final_val_loss(model, data, device, cfg.batch_size, iters=cfg.eval_iters)
    metrics["val_loss"] = round(val, 4)
    metrics["val_perplexity"] = round(math.exp(min(val, 20)), 3)

    if device.type == "cuda":
        metrics["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    else:
        metrics["peak_vram_mb"] = 0

    try:
        metrics["sample"] = sample_text(model, tokenizer, device)
    except Exception as e:  # a bad sample must never sink a finished run
        metrics["sample"] = f"<sample failed: {e}>"

    save_metrics(metrics)
    records.log(
        "final_evals",
        val_loss=metrics["val_loss"],
        val_perplexity=metrics["val_perplexity"],
        params=counts["params"],
        embedding_params=counts["embedding_params"],
        peak_vram_mb=metrics["peak_vram_mb"],
        sample=str(metrics.get("sample", ""))[:600],
    )
    print(
        f"evals: params {counts['params']:,} (emb {counts['embedding_params']:,}) | "
        f"val_loss {metrics['val_loss']} ppl {metrics['val_perplexity']} | "
        f"peak_vram {metrics['peak_vram_mb']}MB",
        flush=True,
    )
    return metrics
