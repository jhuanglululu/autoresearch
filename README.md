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

## Running

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

- `queue/` — filesystem job queue (atomic-rename lanes, fail-not-requeue stale
  sweep), lab lifecycle, worker: snapshot + sha256, [assets] injection,
  per-snapshot uv env, setsid subprocess with SIGTERM→SIGKILL group timeout,
  automated record.md capture as wiki source `exp-<lab>-r<n>`;
  `make_run_experiment()` plugs straight into the Subagent runner

- `orchestrator/` — the always-on loop: races user messages / subagent tasks /
  digest timer; fire-and-forget spawns with one prompt-driven retry; kept
  sessions + `follow_up_subagent`; kill/steer mid-run; programmatic digests;
  checkpoint per turn and per finished subagent (restart resumes); per-model
  USD spend caps from models.toml (orchestrator stops at its cap; a capped
  subagent fails its session)

- `bot/` + `__main__.py` — thin discord.py front-end: one steering channel where
  only admin messages that @mention the bot reach the orchestrator (mention
  stripped); /status /sessions /stop /kill-subagent answered from state without
  LLM turns; /btw asks a side question over the orchestrator's context (no
  tools, nothing appended); optional AUTORESEARCH_FORUM live feed — one thread
  per subagent, batched tool-call briefs every ~7s; 2000-char fence-aware
  splitting everywhere; entry point with .env loading and linked shutdown

**The build order is complete.** Next: first live deployment (sync to the GPU
box, download the corpus once, fill .env, run bot+orchestrator and the worker).
