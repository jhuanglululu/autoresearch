# autoresearch — Design (rev 4, 2026-07-16)

An always-on autonomous research system on a remote box (single GPU: RTX 5090 or
RTX 6000-class), steered from Discord. It researches LM architectures, generates novel
ideas (combining papers, transferring mechanisms to new uses), tests them as ~50M-param
from-scratch training runs, and grows a persistent knowledge base. It keeps working
until told to stop.

Review page: https://claude.ai/code/artifact/fa9d38d2-9ba2-4058-833c-ec5460247720

## Topology

```
You (Discord) ⇄ Orchestrator (per models.toml) ⇄ subagents ⇄ wiki / archive / GPU queue
```

Subagents cannot spawn subagents. Details land in the wiki/archive; only short
summaries flow up to the orchestrator. Finished subagent sessions are **kept** so the
orchestrator can ask follow-up questions without respawning an agent to rediscover
the same context.

## Orchestrator

- The only agent you talk to. One Discord channel: natural language + slash commands
  (`/status`, `/stop`, `/kill-subagent`).
- Small, idle context: plans, composes a prompt + detailed rules per spawn, receives a
  short summary, evaluates, decides next step. **Never reads or writes code** unless you
  explicitly ask an implementation question.
- Fire-and-forget + timeout; one retry with an amended prompt, then reports to you.
  You can kill/redirect a running subagent by messaging mid-run.
- Picks the subagent model per task using the `description` fields in `models.toml`;
  can request a specific model to re-review an unexpected result.
- Assigns architecture-level ideas only — never trivia like "test a wider d_model".
  The engineer may vary training dynamics for a given architecture on its own.
- **Watches for anomalies and investigates.** Nothing about experiments is strictly
  predefined; when a result looks wrong (loss spikes, suspicious speedups, metrics
  that don't add up) the orchestrator's job is to notice and dig in — e.g. spawn a
  reviewer, or have the engineer add the metrics/logging needed to explain it.
- Simple checkpoint restore: state snapshotted after each subagent finishes. A crash
  resumes from the last completed subagent, not the exact instruction it died on.
- Periodic Discord digests.

## Subagents — two types only

Roles (brainstormer, literature scout, idea synthesizer, experiment engineer, utility
lookups) are **prompt templates the orchestrator composes at spawn time**, not types.

| Type | Capabilities |
|---|---|
| **executor** | Writes experiment code, executes it, submits GPU jobs via tool call, writes wiki. |
| **researcher** | No code editing or execution (it couldn't run what it writes). Searches web/arXiv, reads everything, **writes wiki notes directly**. |

**Tools are per-action, and write tools are strictly separated:**

- Wiki tools are one-per-action (`wiki_search`, `wiki_read`, `wiki_graph_*`, … read-only;
  `wiki_capture_source`, `wiki_write_summary`, `wiki_retract_source` writes). The write
  tools can only touch the wiki, through the store, so wiki structure and invariants
  (citations, types, immutability of sources) are always maintained.
- `write` — can only write inside the current editable **lab**. It cannot touch the
  archive (past run dirs) and cannot touch the wiki. Executor-only.
- `analyze_records` — runs inline Python with **read-only** filesystem access, for
  programmatic analysis of run bookkeeping (`records.jsonl`, metrics across runs).

Brainstormer is skipped when you propose an idea yourself or a previous brainstorm
already yielded multiple ideas.

## Knowledge base — research-bot's wiki, extended

Base design ported from `../research-bot/` (`src/wiki/store.py`); markdown folders are
the ground truth:

- `sources/<id>.md` — **immutable evidence**, write-once: fetched papers/pages, captures,
  auto-generated experiment run records (`origin: url | capture | experiment`).
  Never edited or deleted — only *retracted* with a reason.
- `summary/<slug>.md` — **editable synthesis that must cite** via inline `(source: id)`;
  a write citing an unknown id is rejected.
- **New here: typed relations.** Notes carry a type (paper, mechanism, idea, experiment,
  result) and typed links (extends, combines, refutes) so agents can query the idea
  graph, not just full-text search.
- **Rebuildable index** beside the files: citations/backlinks, lexical FTS, semantic
  embeddings; dual-path search (semantic + lexical) with a divergence warning.
  Markdown is canonical; the index can always be regenerated.
- Experiment run records auto-captured as immutable sources (`exp-<lab>-r<n>`) by the
  pipeline — artifact documentation is never delegated to a subagent.
- **Shared across goals**: knowledge accumulates; per-goal state stays separate.

Carried over from research-bot: two-tier store with enforced citations,
retraction-not-deletion + citation audit, dual-path search, auto-archiving run records,
prompt text in files (`prompt/tools/*.md`), sandboxed runs with per-run code
snapshot + hash and host-side kill timeout, `lab/<id>/runs/<n>/{code, log, metrics,
weights, record.md}` layout.
Dropped: tamper-evidence layer (research-bot's devs removed it themselves), the
~1,500-line Discord session machinery, per-guild shared chat contexts.

## Experiments

Lab design taken from research-bot: a **lab is a proper uv project** (its own
`pyproject.toml`), created by copying the **goal's baseline template** (the example
goal uses `experiments/baseline-lm-zhtw/`).

- **The domain comes from the goal, not the code.** Each goal pins its own baseline
  template and read-only assets, so another goal can target a TTS model, a different
  language, or any other trainable domain without touching the system.
- **The engineer owns the whole lab.** Model, training loop, validation loop, logging,
  evals — all freely editable. The baseline template is a starting point, not a fixed
  harness. If an idea needs extra metrics to be understood, the engineer edits the
  training loop to collect them.
- **The only untouchables are the goal's pinned assets** (for the example goal: the
  wikipedia-zhtw corpus + `tokenizer.json`) — kept outside every lab, read-only by
  file permissions, never edited by any subagent. The worker resolves their paths
  into each run's `run_config.toml` `[assets]` section.
- Baseline templates ship with standard evals (for the LM template: val loss /
  perplexity, tokens/sec, peak VRAM, param counts) so runs stay comparable by default;
  deviations are visible in the run record and it's the orchestrator's job to question
  comparisons that no longer hold.
- Every run leaves two records: `metrics.json` (at-a-glance summary, read directly) and
  `records.jsonl` (append-only machine log of everything — config, train/eval events,
  failures — for bookkeeping and Python analysis via `analyze_records`). Weights are
  saved as safetensors.
- Single-GPU sequential job queue. Launching an experiment is a **zero-argument tool
  call**: `run_experiment` snapshots the entire current lab (including its
  `run_config.toml`) into `lab/<id>/runs/<n>/code/` (sha256-hashed) and runs that
  snapshot — the configuration can never depend on tool-call args, only on what is in
  the files. To run something different, the engineer edits the lab files and calls
  again. No CLI args, no env vars anywhere. Runs execute as **sandboxed subprocesses**: a per-run uv env built
  from the lab's pyproject, launched in its own process group (`setsid`) with
  cwd = run dir and a host-side wall-clock kill of the whole group; assets stay
  read-only via file permissions. Docker is deliberately not used: the rented GPU
  box (gputw) is itself an unprivileged container where no container runtime can
  work, and the single-operator threat model (the sandbox prevents mistakes, not
  adversaries) makes subprocess isolation sufficient. Consequence to stay honest
  about: no network cutoff inside runs.

## Goals & configuration

The orchestrator script is generic. A **goal** is a TOML with an `id`, an initial
`template` (seed prompt: goal statement, constraints, success criteria), and an
`[experiment]` block pinning the domain: the baseline lab template plus the goal's
read-only assets (`[experiment.assets]`, resolved into each run's config). One goal might
optimize inference speed of a 50M zhtw LM; another could train a TTS model on a
different corpus. A known id **resumes** from its last checkpoint; a new id starts
fresh from the template.

Models are pinned **only in `models.toml`** — never CLI args, never env vars — so
upgrading models is a one-line edit (the current three are placeholders). The config
stores **env-var names, never tokens**, so it is safe to commit. Each
`[[subagent_model]]` carries a prose `description` written *for the orchestrator* —
that is how it decides which model to spawn per task.

## Discord & stack

- One steering channel: only admin messages that **@mention the bot** reach the
  orchestrator (it doesn't know it's on Discord; ambient chat must not confuse it) —
  the mention is stripped before delivery. Slash commands + digests live there too.
- One optional **forum channel** (`AUTORESEARCH_FORUM`): a read-only live feed of
  subagent activity — a new thread per spawned subagent, batched updates every ~7s
  with brief tool-call descriptions and truncated results (never full contents).
  No steering there.
- Python monorepo; custom thin async loop over raw Anthropic + OpenAI APIs (async HTTP,
  no agent frameworks); `discord.py`.
- Plain folders everywhere: wiki + archive are directory trees subagents explore with
  ordinary file reads; queue + orchestrator checkpoints are small JSON files; the only
  database is the rebuildable search index.
- Two processes: (1) bot + orchestrator, (2) GPU worker popping the queue folder —
  connected only through the filesystem, restartable independently.
- KB + archive under git, auto-committed after each subagent finishes (this is both the
  automated-documentation step and the restore checkpoint).

## Open items

- Final GPU choice: RTX 5090 (32 GB) vs RTX 6000-class — affects max batch size only.
- Discord bot token + server setup.
- ~~No API spend cap~~ Per-model spend caps: each models.toml entry carries `cap`
  (USD) + `price_in`/`price_out` (USD per 1M tokens); a client whose accumulated
  cost reaches its cap refuses further calls. The orchestrator reports and stops
  on its own cap; a capped subagent surfaces as a failed session (not retryable
  with the same model). Digests show $spent/$cap per model.
