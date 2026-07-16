"""GPU worker process:  python -m autoresearch.queue.worker

Separate process from the bot+orchestrator (DESIGN.md — two processes, connected
only through the filesystem, restartable independently).

Loop: claim next pending job -> build/reuse the container env from the lab's uv
project -> launch the run container (research-bot's sandbox pattern: offline,
the goal's pinned assets mounted read-only at /assets/<name>, lab code snapshot
in the run dir, RW only on the run dir, GPU passthrough, HOST-SIDE wall-clock
kill via `docker kill`; refuses to launch if a declared asset is missing) -> on
completion write metrics/log/weights into the run dir, write record.md, and
auto-capture it as an immutable wiki source (exp-<lab>-r<n>). That capture is the
"automated documentation" step — never a subagent's job.

TODO(implement): the claim loop + container launch (port src/lab/runner.py from
research-bot, add --gpus).
"""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
