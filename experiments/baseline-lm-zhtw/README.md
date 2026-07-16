# Baseline lab — 50M zhtw LM

A ~50M-param decoder-only LM trained from scratch on the pinned wikipedia-zhtw
corpus. This whole project is yours to edit — model, training loop, validation
loop, logging, dependencies. Add metrics when you need them to understand a run.

Hard rules:
- The goal's pinned assets are read-only files outside the lab; their resolved
  paths arrive in `run_config.toml` `[assets]` — here `corpus` (dataset dir) and
  `tokenizer` (tokenizer.json). Never copy, regenerate, or substitute them.
- Runs start via the `run_experiment` tool only, and it takes no arguments: it
  snapshots this lab as-is and runs it. `run_config.toml` at the lab root IS the run
  configuration — edit it (and/or the code) to change the next run. `main.py` reads
  `run_config.toml` from the run directory — no CLI args, no env vars.
- Write `metrics.json` and human-readable progress to stdout; keep the standard
  metrics (val loss/ppl, tokens/sec, peak VRAM, param count) unless you have a
  reason to drop one — the run record notes any deviation.

Layout: `main.py` (entry); `run_config.toml` (the lab-owned run configuration —
edit it to define the next run; `[assets]` is injected by the worker at launch, so
it is absent here); the model is the `lab/model/` package, split
one-concern-per-file so an experiment can swap a piece by editing one file —
`lab/model/rope.py` (RoPE cache + apply), `lab/model/attention.py` (causal MHA),
`lab/model/ffn.py` (SwiGLU), `lab/model/block.py` (pre-norm block),
`lab/model/gpt.py` (ModelConfig + assembled GPT, re-exported from
`lab/model/__init__.py`); plus `lab/data.py`, `lab/train.py`, `lab/evals.py`,
`lab/checkpoint.py` (safetensors + JSON sidecar), `lab/records.py`
(records.jsonl event log), `NOTES.md` (your journal).

Each run writes `metrics.json` (at-a-glance summary), `model.safetensors` +
`model.json` sidecar (weights + provenance metadata), `records.jsonl`
(append-only machine log), and `record.md`.
