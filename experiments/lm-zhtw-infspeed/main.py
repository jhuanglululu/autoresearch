"""Run entry point, launched by the GPU worker as a sandboxed subprocess
(own uv env, own process group, cwd = the run dir, host-side wall-clock kill).

Reads run_config.toml from the run directory (the cwd), trains, and writes
metrics.json + model.safetensors (+ model.json sidecar) + records.jsonl +
record.md there. No CLI args, no env vars.

The lab owns run_config.toml at its root — that file IS the run configuration.
`run_experiment` takes no arguments: it snapshots the lab (this file included)
into the run dir and runs it, so editing run_config.toml is how you change the
next run. The worker resolves the goal's pinned read-only asset paths into the
snapshot's [assets] section at launch (so the lab copy omits [assets]).

run_config.toml schema ([assets] is REQUIRED at run time — the worker injects the
goal's pinned read-only asset paths into it; all other fields optional, shown with
baseline defaults):

    [assets]
    corpus    = "…/data/wikipedia-zhtw"   # dir with tokens.bin + meta.json
    tokenizer = "…/tokenizer.json"        # tokenizer.json (file or containing dir)

    [model]                           # omit any field -> ~50M baseline default
    d_model    = 640
    n_layers   = 9
    n_heads    = 10
    d_ff       = 1728
    block_size = 1024
    dropout    = 0.0

    [train]
    steps         = 20000
    batch_size    = 32
    grad_accum    = 1
    lr            = 3e-4
    min_lr        = 3e-5
    warmup_steps  = 200
    weight_decay  = 0.1
    grad_clip     = 1.0
    eval_interval = 500
    eval_iters    = 100
    seed          = 1337

Experiments vary a run purely by editing run_config.toml (and/or the lab code);
overriding [model] gives a smaller/larger net, [train].steps/batch_size scale
the run. Vocab size is always taken from the tokenizer, never overridden.
"""
from __future__ import annotations

import dataclasses
import json
import os
import time
import tomllib
import traceback
from pathlib import Path

import torch

from lab.data import TokenData, load_tokenizer
from lab.evals import evaluate
from lab.model import GPT, ModelConfig
from lab.records import RecordLog
from lab.train import TrainConfig, train

RUN_DIR = Path.cwd()
METRICS_PATH = RUN_DIR / "metrics.json"
RECORDS_PATH = RUN_DIR / "records.jsonl"


def load_run_config() -> dict:
    path = RUN_DIR / "run_config.toml"
    if not path.is_file():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def _apply(dc_cls, overrides: dict):
    """Build a dataclass from its defaults, applying only known keys."""
    fields = {f.name for f in dataclasses.fields(dc_cls)}
    known = {k: v for k, v in overrides.items() if k in fields}
    return dc_cls(**known)


def save_metrics(metrics: dict) -> None:
    """Atomic write so a killed run still leaves valid partial metrics behind."""
    tmp = METRICS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, METRICS_PATH)


def write_record(metrics: dict, status: str, wall: float) -> None:
    """Factual, machine-written run record (research-bot format)."""
    m = metrics
    lines = [
        f"# Experiment run: {RUN_DIR.name}",
        f"- status: {status} · wall: {wall}s",
        f"- tokenizer: {m.get('tokenizer')} · vocab: {m.get('vocab_size')} · "
        f"tokens: {m.get('tokens')} · steps: {m.get('steps')}",
        f"- params: {m.get('params')} (embedding {m.get('embedding_params')})",
        f"- train_loss: {m.get('train_loss')} · val_loss: {m.get('val_loss')} · "
        f"val_perplexity: {m.get('val_perplexity')}",
        f"- tokens/sec: {m.get('tokens_per_sec')} · peak_vram_mb: {m.get('peak_vram_mb')}",
        f"- gen tok/s (KV-cached, greedy): b1 {m.get('gen_tok_per_sec_b1')} · "
        f"b8 {m.get('gen_tok_per_sec_b8')} · ttft: {m.get('ttft_ms')}ms "
        f"(prompt {m.get('bench_prompt_len')} + {m.get('bench_gen_tokens')} gen, "
        f"device {m.get('device')})",
        "",
        "Full metrics in `metrics.json`; weights in `model.safetensors` "
        "(+ `model.json` sidecar); event log in `records.jsonl`.",
    ]
    if m.get("sample"):
        lines += ["", "## sample", "```", str(m["sample"])[:600], "```"]
    # Preview the first few inference samples; the full set lives in metrics.json.
    samples = m.get("samples") or []
    if samples:
        preview = samples[:3]
        lines += ["", f"## inference samples ({len(samples)} in `metrics.json`, "
                      f"showing {len(preview)})"]
        for i, s in enumerate(preview):
            lines += [
                f"{i + 1}. prompt: `{str(s.get('prompt', ''))[:50]}`",
                f"   generated: `{str(s.get('generated', ''))[:200]}`",
            ]
    (RUN_DIR / "record.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    start = time.monotonic()
    cfg = load_run_config()
    assets = cfg.get("assets", {})
    # No defaults: the worker always writes resolved asset paths; a missing key
    # means a broken launch and should fail loudly here.
    corpus_asset = assets["corpus"]
    tokenizer_asset = assets["tokenizer"]

    train_cfg = _apply(TrainConfig, cfg.get("train", {}))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(int(os.environ.get("EXP_THREADS", os.cpu_count() or 1)))
    print(f"device: {device} | threads: {torch.get_num_threads()}", flush=True)

    records = RecordLog(RECORDS_PATH, t0=start)

    # Data + tokenizer from the pinned read-only assets.
    tokenizer = load_tokenizer(tokenizer_asset)
    data = TokenData(corpus_asset)
    vocab_size = data.vocab_size

    model_cfg = _apply(ModelConfig, {**cfg.get("model", {}), "vocab_size": vocab_size})
    model = GPT(model_cfg).to(device)

    metrics = {
        "tokenizer": Path(data.meta.get("tokenizer", "tokenizer.json")).name,
        "vocab_size": vocab_size,
        "tokens": data.n_tokens,
        "block_size": model_cfg.block_size,
        "batch_size": train_cfg.batch_size,
        "device": device.type,
        "model_config": vars(model_cfg),
        **model.num_params(),
    }
    save_metrics(metrics)
    print(f"model: {metrics['params']:,} params "
          f"(emb {metrics['embedding_params']:,}) | tokens: {data.n_tokens:,}", flush=True)

    # Full resolved config, first line of the machine log.
    records.log(
        "run_start",
        step=0,
        run_dir=RUN_DIR.name,
        device=device.type,
        threads=torch.get_num_threads(),
        assets={"corpus": corpus_asset, "tokenizer": tokenizer_asset},
        tokenizer=metrics["tokenizer"],
        vocab_size=vocab_size,
        tokens=data.n_tokens,
        val_tokens=data.n_val,
        model_config=vars(model_cfg),
        train_config=vars(train_cfg),
        params=metrics["params"],
        embedding_params=metrics["embedding_params"],
    )

    status = "ok"
    try:
        train(model, data, train_cfg, device, metrics, save_metrics, records)
        evaluate(model, data, tokenizer, device, train_cfg, metrics, save_metrics, records)
    except Exception:
        status = "failed"
        save_metrics(metrics)
        records.log("failure", status=status, traceback=traceback.format_exc())
        write_record(metrics, status, round(time.monotonic() - start, 2))
        raise

    wall = round(time.monotonic() - start, 2)
    metrics["wall_seconds"] = wall
    save_metrics(metrics)
    write_record(metrics, status, wall)
    print(f"done in {wall}s — metrics.json, model.safetensors, records.jsonl, "
          f"record.md written", flush=True)


if __name__ == "__main__":
    main()
