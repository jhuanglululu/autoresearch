# Baseline lab — 50M zhtw LM

A ~50M-param decoder-only LM trained from scratch on the pinned wikipedia-zhtw
corpus. This whole project is yours to edit — model, training loop, validation
loop, logging, dependencies. Add metrics when you need them to understand a run.

Hard rules (enforced by mounts, not honor):
- The goal's pinned assets are mounted read-only at `/assets/<name>` — here
  `/assets/corpus` (dataset) and `/assets/tokenizer` (tokenizer.json). Never
  copy, regenerate, or substitute them.
- Runs start via the `run_experiment` tool only. `main.py` reads `run_config.toml`
  from the run directory — no CLI args, no env vars.
- Write `metrics.json` and human-readable progress to stdout; keep the standard
  metrics (val loss/ppl, tokens/sec, peak VRAM, param count) unless you have a
  reason to drop one — the run record notes any deviation.

Layout: `main.py` (entry), `lab/model.py`, `lab/data.py`, `lab/train.py`,
`lab/evals.py`, `NOTES.md` (your journal).
