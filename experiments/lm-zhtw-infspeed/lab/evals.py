"""Standard evals: val loss/perplexity, tokens/sec, peak VRAM, param counts, plus
the INFERENCE-SPEED suite this goal is built around (KV-cached generation
throughput, time-to-first-token) and a handful of eyeballable generation samples.
Keep these for comparability; extend freely. Deviations show up in the run record
for the orchestrator to notice.

Most numbers are produced during training (tokens/sec, val loss); this module
finalizes the suite: a clean full-val-set perplexity, peak VRAM, param counts,
the inference benchmarks below, and generation samples for the run record.

Inference-speed metric (the honest measuring stick every architecture idea is
compared against):
  gen_tok_per_sec_b1/b8  aggregate generated tokens/sec (total new tokens / wall)
                         with the KV cache, GREEDY, 128-token val prompts, 256
                         generated tokens, batch 1 and 8, timed AFTER a warmup
                         generation (CUDA: synchronize around the timer).
  ttft_ms                time-to-first-token: the prefill cost for a 128-token
                         prompt at batch 1 (ms), the moment the first token's
                         logits are ready.
"""
from __future__ import annotations

import math
from time import perf_counter

import numpy as np
import torch

# Fixed metric definition — do not vary per run, or speed numbers stop comparing.
BENCH_PROMPT_LEN = 128
BENCH_GEN_TOKENS = 256
BENCH_SEED = 1234
SAMPLES_SEED = 20240


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


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _val_prompts(data, batch_size: int, prompt_len: int, device, seed: int) -> torch.Tensor:
    """Deterministically sample `batch_size` contiguous prompts of length
    `prompt_len` from the val split (fixed seed -> identical prompts every run,
    so speed and samples are comparable). Returns (batch_size, prompt_len) longs."""
    val = data.split("val")
    high = len(val) - prompt_len - 1
    if high <= 0:
        raise ValueError(f"val split too small ({len(val)} tokens) for prompt_len {prompt_len}")
    g = torch.Generator().manual_seed(seed)
    ix = torch.randint(high, (batch_size,), generator=g)
    x = torch.stack([torch.from_numpy(val[i : i + prompt_len].astype(np.int64)) for i in ix])
    return x.to(device)


@torch.no_grad()
def inference_benchmarks(model, data, device,
                         prompt_len: int = BENCH_PROMPT_LEN,
                         gen_tokens: int = BENCH_GEN_TOKENS) -> dict:
    """Measure KV-cached greedy generation throughput (batch 1 and 8) and
    time-to-first-token (batch 1), each after a warmup. See module docstring for
    the exact metric definition. Reports aggregate throughput = (batch * gen_tokens)
    generated tokens / wall-clock seconds."""
    model.eval()

    def throughput(batch_size: int) -> float:
        idx = _val_prompts(data, batch_size, prompt_len, device, seed=BENCH_SEED)
        model.generate(idx, max_new_tokens=gen_tokens, greedy=True, use_cache=True)  # warmup
        _sync(device)
        t0 = perf_counter()
        model.generate(idx, max_new_tokens=gen_tokens, greedy=True, use_cache=True)
        _sync(device)
        dt = perf_counter() - t0
        return (batch_size * gen_tokens) / dt if dt > 0 else 0.0

    idx1 = _val_prompts(data, 1, prompt_len, device, seed=BENCH_SEED)
    model.prefill(idx1)  # warmup the prefill path
    _sync(device)
    t0 = perf_counter()
    model.prefill(idx1)
    _sync(device)
    ttft_ms = (perf_counter() - t0) * 1000.0

    return {
        "gen_tok_per_sec_b1": round(throughput(1), 2),
        "gen_tok_per_sec_b8": round(throughput(8), 2),
        "ttft_ms": round(ttft_ms, 3),
        "bench_prompt_len": prompt_len,
        "bench_gen_tokens": gen_tokens,
    }


@torch.no_grad()
def inference_samples(model, tokenizer, data, device, n: int = 8,
                      prompt_len: int = 32, gen_tokens: int = 80) -> list[dict]:
    """A few eyeballable generations from varied, deterministically chosen val
    prompts (fixed seed). Each entry: {prompt: <decoded ~50 chars>,
    generated: <decoded <=200 chars>}. Sampled (temp 0.8, top_k 200) under a
    fixed seed so the text is reproducible run-to-run."""
    model.eval()
    torch.manual_seed(SAMPLES_SEED)
    prompts = _val_prompts(data, n, prompt_len, device, seed=SAMPLES_SEED)
    samples = []
    for i in range(n):
        p = prompts[i : i + 1]
        out = model.generate(p, max_new_tokens=gen_tokens, temperature=0.8, top_k=200, use_cache=True)
        prompt_txt = tokenizer.decode(p[0].tolist())
        gen_txt = tokenizer.decode(out[0, prompt_len:].tolist())
        samples.append({"prompt": prompt_txt[:50], "generated": gen_txt[:200]})
    return samples


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

    # Inference-speed suite + eyeballable samples. Wrapped so a benchmark hiccup
    # (e.g. a val split too small on a smoke corpus) never sinks a finished run.
    try:
        metrics.update(inference_benchmarks(model, data, device))
    except Exception as e:
        metrics["inference_benchmarks_error"] = str(e)
    try:
        metrics["samples"] = inference_samples(model, tokenizer, data, device)
    except Exception as e:
        metrics["samples"] = []
        metrics["samples_error"] = str(e)

    save_metrics(metrics)
    records.log(
        "final_evals",
        val_loss=metrics["val_loss"],
        val_perplexity=metrics["val_perplexity"],
        params=counts["params"],
        embedding_params=counts["embedding_params"],
        peak_vram_mb=metrics["peak_vram_mb"],
        gen_tok_per_sec_b1=metrics.get("gen_tok_per_sec_b1"),
        gen_tok_per_sec_b8=metrics.get("gen_tok_per_sec_b8"),
        ttft_ms=metrics.get("ttft_ms"),
        device=device.type,
        n_samples=len(metrics.get("samples", [])),
        samples=metrics.get("samples", []),
        sample=str(metrics.get("sample", ""))[:600],
    )
    print(
        f"evals: params {counts['params']:,} (emb {counts['embedding_params']:,}) | "
        f"val_loss {metrics['val_loss']} ppl {metrics['val_perplexity']} | "
        f"peak_vram {metrics['peak_vram_mb']}MB | "
        f"gen tok/s b1 {metrics.get('gen_tok_per_sec_b1')} b8 {metrics.get('gen_tok_per_sec_b8')} | "
        f"ttft {metrics.get('ttft_ms')}ms",
        flush=True,
    )
    return metrics
