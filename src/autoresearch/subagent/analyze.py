"""`analyze_records` implementation: run operator-supplied inline Python against
run bookkeeping (records.jsonl / metrics.json across runs) with read-only intent.

The snippet runs in a fresh, isolated child interpreter
(`sys.executable -I -c <code>`): `-I` ignores PYTHONPATH / PYTHON* env vars and
the user site-packages, so the child sees only the stdlib + whatever is
importable from `cwd`. stdout is the result; the caller reads it back.

READ-ONLY IS BEST-EFFORT, NOT ENFORCED. This function does NOT sandbox the
filesystem or the network — a determined snippet could still write files or open
sockets. That is accepted by design: the GPU box is an unprivileged container
where no container runtime (and no privilege-dropping) is possible, and the
threat model is operator mistakes, not adversaries (DESIGN.md — Experiments).
What this DOES do:
  * isolate the interpreter (`-I`) so host env/config can't leak in;
  * point TMPDIR at a throwaway dir (so well-behaved temp writes land there and
    vanish), while leaving the real cwd readable;
  * cap wall-clock time (kills infinite loops) and output size.
The prompt (prompt/tools/analyze_records.md) tells the model plainly: never
write, move, or delete — analysis only.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TIMEOUT = 30.0        # wall-clock seconds
OUTPUT_CAP = 64_000           # bytes kept per stream (head), rest truncated


@dataclass
class AnalyzeResult:
    stdout: str
    stderr: str
    returncode: int | None     # None if the snippet was killed on timeout
    timed_out: bool


def _truncate(s: str, cap: int = OUTPUT_CAP) -> str:
    if len(s) <= cap:
        return s
    return s[:cap] + f"\n...[truncated, {len(s) - cap} more chars]"


def analyze(
    code: str,
    cwd: str | Path = ".",
    timeout: float = DEFAULT_TIMEOUT,
    output_cap: int = OUTPUT_CAP,
) -> AnalyzeResult:
    """Run `code` as a one-shot Python snippet with `cwd` as its working dir.

    stdout is the intended result channel. Returns captured stdout/stderr
    (each truncated to `output_cap`), the child return code, and whether it was
    killed for exceeding `timeout`.
    """
    cwd = Path(cwd)
    if not cwd.is_dir():
        raise NotADirectoryError(f"analyze cwd does not exist: {cwd}")

    with tempfile.TemporaryDirectory(prefix="analyze-tmp-") as tmp:
        # Minimal env: a throwaway writable TMPDIR, keep PATH so the interpreter
        # can find shared libs. `-I` already drops PYTHON* vars inside the child.
        env = {"TMPDIR": tmp, "PATH": os.environ.get("PATH", "")}
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", code],
                cwd=str(cwd),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            out = e.stdout or ""
            err = e.stderr or ""
            if isinstance(out, bytes):
                out = out.decode("utf-8", "replace")
            if isinstance(err, bytes):
                err = err.decode("utf-8", "replace")
            return AnalyzeResult(
                stdout=_truncate(out, output_cap),
                stderr=_truncate(err, output_cap)
                + f"\n[analyze: killed after {timeout}s timeout]",
                returncode=None,
                timed_out=True,
            )

    return AnalyzeResult(
        stdout=_truncate(proc.stdout, output_cap),
        stderr=_truncate(proc.stderr, output_cap),
        returncode=proc.returncode,
        timed_out=False,
    )
