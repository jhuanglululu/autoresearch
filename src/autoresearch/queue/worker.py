"""GPU worker process:  python -m autoresearch.queue.worker

Separate process from the bot+orchestrator (DESIGN.md — two processes, connected
only through the filesystem, restartable independently).

Loop: claim next pending job -> build/reuse the per-lab uv env (`uv sync` against
the lab's pyproject) -> snapshot the lab code into the run dir (sha256-hashed) ->
copy the lab's run_config.toml into the run dir and inject the resolved [assets]
section (refuse to launch if a declared asset is missing) -> launch main.py as a SANDBOXED
SUBPROCESS: own process group (setsid), cwd = run dir, HOST-SIDE wall-clock
timeout that kills the whole group -> on completion the run dir holds
metrics.json, records.jsonl, model.safetensors (+ model.json), log, record.md;
auto-capture record.md as an immutable wiki source (exp-<lab>-r<n>). That capture
is the "automated documentation" step — never a subagent's job.

No Docker by design: the GPU box (gputw) is an unprivileged container where no
container runtime can run; assets are kept read-only via file permissions and
the threat model is operator mistakes, not adversaries. No network cutoff inside
runs — documented, not pretended.

TODO(implement): the claim loop + subprocess launch (adapt src/lab/runner.py from
research-bot, replacing docker with setsid + process-group kill).
"""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
