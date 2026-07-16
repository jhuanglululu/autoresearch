# Goal: system debug — exercise every feature once, then wait

This is NOT a research goal. You are verifying that the system works, feature by
feature, spending as few tokens and as little GPU time as possible. Work through
the checklist in order, tell the operator the result of each step as you go, and
when the checklist is done, STOP working — report a final pass/fail summary and
wait for operator instructions. Do not start any research.

## Checklist

1. **Hello.** Send the operator a short hello message confirming you are alive and
   name the goal you are running.
2. **Researcher spawn.** Spawn a `researcher` with a trivial task: capture one
   source into the wiki (a short made-up note is fine, e.g. a paragraph about
   RMSNorm titled `debug-test-source`), then write a summary note
   (slug `debug-test-note`, type `idea`) citing it, and return a one-line summary.
3. **Follow-up.** Ask that same researcher one follow-up question (e.g. what slug
   it used) via `follow_up_subagent` — verify the answer is consistent.
4. **Executor spawn + tiny run.** Spawn an `executor` in lab id `debug-lab` with
   this exact task: edit `run_config.toml` to a TINY configuration —
   `[model]` d_model = 64, n_layers = 2, n_heads = 4, d_ff = 128, block_size = 128;
   `[train]` steps = 30, batch_size = 4, eval_interval = 10, eval_iters = 5,
   log_interval = 5 — also create a scratch file `NOTES.md` entry describing what
   it changed, then call `run_experiment` (no arguments) and report the result
   summary including the wiki source id. This tests file editing, the queue, the
   GPU, and automated record capture, in about a minute of compute.
5. **Verify the record.** Yourself (read-only wiki tools): `wiki_read_source` the
   `exp-debug-lab-r1` source and confirm it contains a val_loss.
6. **Status.** Call `get_status` and relay the queue depth and per-model spend to
   the operator.

## Rules
- One step at a time; never two subagents at once.
- If a step fails, report the failure clearly and continue with the remaining
  steps that still make sense.
- Total budget for this whole checklist: a few dollars. Keep prompts short.
