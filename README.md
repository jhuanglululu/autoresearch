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
scripts/download_dataset.py   one-time corpus download for the example goal
tokenizer.json         pinned tokenizer asset of the example goal
```

## Running (once implemented)

```
cp .env.example .env             # fill in tokens
python scripts/download_dataset.py
python -m autoresearch goals/example.toml    # bot + orchestrator
python -m autoresearch.queue.worker          # GPU worker (separate process)
```

## Status

Scaffold: structure, configs, prompts, and contracts are in place; modules marked
TODO(implement) are stubs. Suggested build order:
1. `wiki/store.py` (port from research-bot) — everything else writes into it
2. `experiments/baseline-lm-zhtw/` + `queue/` — a baseline run end-to-end, hand-submitted
3. `llm/` clients + `subagent/runner.py`
4. `orchestrator/loop.py` + checkpoints
5. `bot/discord_bot.py`
