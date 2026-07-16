# autoresearch

Always-on autonomous LM-architecture research on a single-GPU box, steered from
Discord. See **DESIGN.md** for the approved spec (rev 3).

## Layout

```
models.toml            models + endpoints (env-var NAMES only — safe to commit)
goals/<id>.toml        one research goal: id + seed template + [experiment] block
                       (baseline lab template + pinned read-only assets); known id = resume
templates/*.md         goal seed prompts
prompt/                all agent behavior lives in these text files, not code
src/autoresearch/
  config.py            models.toml + goal loading (implemented)
  llm/                 thin async clients over raw Anthropic/OpenAI HTTP APIs
  orchestrator/        the planning loop, checkpoints, subagent sessions
  subagent/            one-shot agent runner (executor / researcher toolsets)
  wiki/                two-tier wiki store (ported from ../research-bot) + idea graph
  queue/               filesystem GPU job queue + worker process
  harness/             fixed nanoGPT-style trainer with pluggable module slots
  bot/                 thin discord.py front-end
scripts/download_dataset.py   one-time corpus download + pre-tokenize for the example goal
tokenizer.json         pinned tokenizer asset of the example goal
tests/                 pytest suite (wiki store/tools, analyze tool)
```

## Running (once implemented)

```
cp .env.example .env             # fill in tokens
python scripts/download_dataset.py
python -m autoresearch goals/example.toml    # bot + orchestrator
python -m autoresearch.queue.worker          # GPU worker (separate process)
```

## Status

Done:
- `config.py` — models.toml + goal loading (with `[experiment]` domain block)
- `wiki/` — two-tier store (immutable sources / must-cite summaries), typed idea
  graph, dual-path search, rebuildable index, 17 per-action tools (`wiki_*`)
- `experiments/baseline-lm-zhtw/` — full lab template: `lab/model/` package
  (rope/attention/ffn/block/gpt), fixed VAL_SIZE split, safetensors weights +
  `model.json` provenance sidecar, `records.jsonl` event log, `record.md`;
  CPU smoke-tested end-to-end (49,737,600-param default config)
- `subagent/analyze.py` + `analyze_records` tool — inline read-only Python over
  run bookkeeping
- Sandbox decision: **subprocess, no Docker** (the GPU box is an unprivileged
  container; see DESIGN.md — Experiments)
- `llm/` — Anthropic + OpenAI clients over raw httpx: message/tool translation,
  prompt caching (system prompt), retries with Retry-After, cumulative usage
- `subagent/` — the one-shot `Subagent` runner: per-type toolsets (executor 24
  tools / researcher 22), path-guarded lab `write`, crash-proof tool dispatch,
  kept sessions with `follow_up()`; `run_experiment` takes NO arguments — the
  lab's own `run_config.toml` + code define the run, the tool snapshots and runs

In progress: `queue/` (filesystem job queue + worker: snapshot, asset injection,
sandboxed subprocess launch, record capture).

Remaining: `orchestrator/loop.py` + spawn/checkpoint wiring, `bot/discord_bot.py`.
