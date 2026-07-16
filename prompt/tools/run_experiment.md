Submit an experiment to the single-GPU queue and wait for its result. This is the
ONLY way to run training code — there is no CLI and no env-var interface.

Takes NO arguments. The run is defined entirely by your lab's current files: it
snapshots your ENTIRE current lab — including `run_config.toml` — into the
immutable archive at `lab/<id>/runs/<n>/code/`, the worker injects the goal's
resolved read-only asset paths into that snapshot's `run_config.toml` `[assets]`
section, then executes `main.py` in the run dir as a sandboxed subprocess (its own
uv env built from your lab's pyproject, own process group, cwd = run dir) under a
hard wall-clock limit enforced from outside.

To vary anything about a run, edit your lab files first — `run_config.toml` for
hyperparameters/model shape, the code for anything deeper — with `write`, then call
`run_experiment` again. The tool never takes configuration; the snapshot does.

On completion you get the metrics summary back; the full record is auto-captured as
wiki source `exp-<lab>-r<n>` — cite it in your notes.
