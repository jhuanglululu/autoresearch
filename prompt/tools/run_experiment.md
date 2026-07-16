Submit an experiment to the single-GPU queue and wait for its result. This is the
ONLY way to run training code — there is no CLI and no env-var interface.

Input: your lab id and a run config (written to the run dir as run_config.toml;
your main.py reads it). The worker resolves the goal's pinned assets (e.g.
dataset, tokenizer — read-only on disk) into the [assets] section for you.

The queue snapshots your lab's code into the run directory before launch; the run
executes as a sandboxed subprocess (its own uv env built from your lab's
pyproject, own process group, cwd = run dir) under a hard wall-clock limit
enforced from outside. On completion you get the metrics summary; the full
record is auto-captured as wiki source `exp-<lab>-r<n>` — cite it in your notes.
