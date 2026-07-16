Restore a lab's working directory to the code snapshot of a past run. ORCHESTRATOR-ONLY:
subagents have no revert tool and must never undo a lab change by hand — a manual revert
does not work well, so the keep-or-revert decision is yours alone.

- `lab_id`: the lab to restore.
- `run_number`: the run whose snapshot (`runs/<n>/code`) the working directory is reset to.

This rewrites ONLY the live working tree — the run archive (`runs/`, including the very
snapshot you restore from and every other run's records/metrics/logs) is NEVER touched.
The working tree ends up byte-identical to that snapshot; live edits made after the run
are discarded.

Use after you decide an experimental direction should be abandoned. Typical flow: an
executor reports a change with a bad trade-off (e.g. it won one metric but wrecked
another); you weigh keep-and-fix (spawn follow-up work to recover the lost metric) against
reverting, and if you choose to abandon it you `revert_lab` to the last good run's snapshot
before spawning the next experiment.
