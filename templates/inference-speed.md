# Goal: optimize inference speed of the ~50M zhtw LM

Find architecture changes that make **autoregressive inference faster** at a
similar parameter budget and similar final quality. The trained models will NOT
be used for anything — the deliverable is **knowledge**: well-cited wiki
summaries and informative, comparable results. A slightly worse model with a
clearly understood speed/quality trade-off beats a slightly better model with
muddy measurements.

## Hard constraints
- Parameter budget: 45–55M total (baseline is 49.7M). Report exact counts.
- Quality bar: final val loss within **2% relative** of the baseline reference
  run at the same training-token budget. Outside that, the result is still worth
  recording — as a rejected idea with its trade-off curve.
- Training time: MAY exceed the baseline's (quality bar matters more), but treat
  training-time reduction as a secondary win worth measuring and reporting.
- The pinned tokenizer/corpus and the fixed val split are untouchable, as always.

## Step zero — the measuring stick (already built into this baseline)
The lab template already ships with the honest reference implementation:
- **KV-cached generation** — and `lab/test_kvcache.py`, which proves cached greedy
  decoding produces identical tokens to the naive path. Rerun it inside the lab
  after ANY change to attention or caching, and report the result.
- **Inference benchmarks in every run's metrics**: `gen_tok_per_sec_b1`,
  `gen_tok_per_sec_b8` (greedy, KV-cached, 128-token val prompts → 256 generated,
  measured after warmup, CUDA-synchronized) and `ttft_ms` (time-to-first-token).
  These constants are fixed so numbers stay comparable across all runs.
- **8 seeded inference samples** in every run's metrics.json (`samples` list) —
  quality must be eyeballable from the record.

Your actual step zero is therefore just: **run the unmodified baseline once**
(default config). That reference run's val loss, params, and speed numbers are
what every idea is compared against. Do not start any idea before the reference
run exists in the wiki.

## What to explore
Architecture-level ideas that plausibly change the inference-speed/quality
trade-off (examples, not an exhaustive list: attention variants that shrink KV
cache or per-token FLOPs, cheaper mixing for some layers, layer-wise
heterogeneity). Brainstorm from literature; prefer ideas with a mechanism-level
reason to be faster, not just smaller. Training-dynamics tweaks are the
engineer's discretion per architecture, but must be reported when they change.

## Trade-offs: sacrifice one metric, recover it later
A change is allowed to sacrifice one metric to win another — e.g. an architecture
that gives 2× faster inference at 2× the training time is a KEEPABLE intermediate
state, not an automatic failure. When an executor reports such a trade-off:
- The ORCHESTRATOR decides: keep the change and spawn follow-up work to recover
  the sacrificed metric, or revert the lab to the last good snapshot
  (`revert_lab` — an orchestrator-only tool; the run archive always survives).
- Executors must never manually undo a change because its results look bad —
  report the trade-off honestly and let the orchestrator own the keep/revert call.
- The leaderboard marks such entries as "kept, recovering <metric>" until the
  follow-up lands or the change is reverted.

## Deliverables (what "done" looks like at any point)
- One wiki summary per tested idea (type `experiment` or `result`): mechanism,
  cited motivation (papers as sources), exact config, speed vs baseline
  (batch 1 / batch 8 / TTFT), val-loss delta, training-time delta, verdict
  (adopted / rejected / inconclusive) — every claim citing run sources.
- A living comparison note (type `result`, slug `inference-speed-leaderboard`)
  ranking all tested ideas: speedup × quality delta × training time, updated
  after every run.
- Ideas that fail are documented as carefully as ideas that work.

## Cadence
- One experiment at a time (single GPU). Between runs, use researchers to read
  and to keep the wiki current.
- This runs over a weekend: keep going without operator input, but respect the
  spend caps and report honestly in digests.
