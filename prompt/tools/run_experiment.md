Submit an experiment to the single-GPU queue and wait for its result. This is the
ONLY way to run training code — there is no CLI and no env-var interface.

Input: your lab id and a run config (written to the run dir as run_config.toml;
your main.py reads it). The goal's pinned assets (e.g. dataset, tokenizer) are
mounted read-only at /assets/<name>.

The queue snapshots your lab's code into the run directory before launch; the run
executes offline in a sandboxed container (env built from your lab's uv project)
with a hard wall-clock limit. On completion you get the metrics summary; the full
record is auto-captured as wiki source `exp-<lab>-r<n>` — cite it in your notes.
