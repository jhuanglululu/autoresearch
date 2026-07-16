"""GPU worker process:  python -m autoresearch.queue.worker

Separate process from the bot+orchestrator (DESIGN.md — two processes, connected
only through the filesystem, restartable independently).

Per job the worker:
  1. allocates lab/<id>/runs/<n>/ (next_run_dir),
  2. snapshots the whole lab into runs/<n>/code/ (sha256-hashed, written to
     runs/<n>/code.sha256) — excluding runs/, .venv, __pycache__, .git,
  3. copies the lab's run_config.toml into runs/<n>/run_config.toml and INJECTS the
     goal's resolved [assets] paths — refusing to launch (job fails cleanly) if a
     declared asset is missing on disk,
  4. launches the snapshot's main.py as a SANDBOXED SUBPROCESS: its own uv env
     (`uv run --project runs/<n>/code`), its own process group (start_new_session /
     setsid), cwd = the RUN DIR so ./run_config.toml resolves, stdout+stderr streamed
     to runs/<n>/log.txt, under a HOST-SIDE wall-clock timeout that SIGTERM→SIGKILLs
     the whole group,
  5. auto-captures the run's record.md into the wiki as an immutable source
     `exp-<lab>-r<n>` (origin=experiment) — synthesizing one from the log tail if the
     run crashed before writing its own. That capture is the "automated
     documentation" step; it is never a subagent's job.

uv env strategy: the launch is `uv run --project runs/<n>/code python <code>/main.py`.
`uv run --project DIR` resolves/creates DIR's environment from its pyproject (and
uv.lock if present), installs the project itself, then runs inside it. uv keeps a
shared, content-addressed package cache, so torch et al. are downloaded/built once
and hard-linked into each new run's env — the FIRST run of a lab pays the sync cost,
later runs with an unchanged dependency set reuse the cache and start fast. Each run
snapshot gets its OWN resolved env keyed to code/pyproject.toml, so a run is
reproducible from exactly the deps it captured and two runs never share a mutable env.

No Docker by design: the GPU box (gputw) is an unprivileged container where no
container runtime can run; assets are kept read-only via file permissions and the
threat model is operator mistakes, not adversaries. No network cutoff inside runs —
documented, not pretended.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
import uuid
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path

from autoresearch.config import GoalConfig
from autoresearch.queue.jobs import Job, JobQueue
from autoresearch.queue.labs import LABS_ROOT, SNAPSHOT_EXCLUDE, next_run_dir
from autoresearch.wiki.store import WikiStore

# ~6h. A ~50M-param from-scratch run (20k steps) on a single RTX 5090/6000 is the
# baseline; 6h leaves generous headroom for larger sweeps while still bounding a hung
# or pathological run so the single GPU is never monopolised indefinitely. Overridable
# per goal via [experiment].run_timeout_s.
DEFAULT_TIMEOUT_S = 6 * 60 * 60

# Grace between SIGTERM and SIGKILL when killing a timed-out run's process group.
_TERM_GRACE_S = 10.0
# How much of log.txt to fold into a synthesized failure record.
_LOG_TAIL = 4000
# Worker idle sleep between empty claim attempts.
_IDLE_SLEEP_S = 2.0

# A launcher builds the argv PREFIX (everything before the script path) for a run's
# code dir. The default uses uv; tests inject a plain interpreter to skip env builds.
Launcher = Callable[[Path], Sequence[str]]


def default_launcher(code_dir: Path) -> list[str]:
    """`uv run --project <code_dir> python` — a fresh, cached env per snapshot."""
    return ["uv", "run", "--project", str(code_dir), "python"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AssetError(RuntimeError):
    """A declared read-only asset is missing on disk — the run must not launch."""


# ----- snapshot + config -----

def _snapshot_code(lab_dir: Path, run_dir: Path) -> str:
    """Copy the lab into run_dir/code (minus SNAPSHOT_EXCLUDE) and return a content
    hash over the sorted (relative-path, bytes) stream — a stable fingerprint of the
    exact code that produced the run."""
    dest = run_dir / "code"
    shutil.copytree(lab_dir, dest, ignore=shutil.ignore_patterns(*SNAPSHOT_EXCLUDE))
    h = hashlib.sha256()
    for path in sorted(p for p in dest.rglob("*") if p.is_file()):
        h.update(path.relative_to(dest).as_posix().encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
    return h.hexdigest()


def _toml_str(value: str) -> str:
    """Encode a string as a TOML basic string (paths never contain newlines)."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _write_run_config(lab_dir: Path, run_dir: Path, assets: dict[str, Path]) -> None:
    """Copy the lab's run_config.toml into the run dir with an injected [assets] block.

    Refuses (AssetError) if any declared asset path is missing on disk — the worker
    must never launch a run against a phantom corpus/tokenizer. The block is appended
    as a fresh table so it can never collide with the engineer's other sections."""
    missing = {name: p for name, p in assets.items() if not p.exists()}
    if missing:
        detail = ", ".join(f"{n}={p}" for n, p in sorted(missing.items()))
        raise AssetError(f"declared asset path(s) missing on disk: {detail}")
    src = lab_dir / "run_config.toml"
    base = src.read_text(encoding="utf-8") if src.is_file() else ""
    lines = [base.rstrip("\n"), "", "# [assets] injected by the GPU worker at launch.", "[assets]"]
    for name in sorted(assets):
        lines.append(f"{name} = {_toml_str(str(assets[name].resolve()))}")
    (run_dir / "run_config.toml").write_text("\n".join(lines).lstrip("\n") + "\n", encoding="utf-8")


# ----- record + summary -----

def _write_synthetic_record(
    run_dir: Path, lab_id: str, n: int, status: str, wall: float, code_hash: str
) -> str:
    """Build a minimal failure record when the run left no record.md of its own."""
    log = run_dir / "log.txt"
    tail = log.read_text(encoding="utf-8", errors="replace")[-_LOG_TAIL:] if log.exists() else ""
    lines = [
        f"# Experiment run: {lab_id} / run {n}",
        f"- status: {status} · wall: {wall}s · code sha256: {code_hash}",
        "",
        "No record.md was written — the run did not finish cleanly. This record was "
        "synthesized by the worker from the run's log tail.",
        "",
        "## log tail",
        "```",
        tail or "(log is empty)",
        "```",
    ]
    text = "\n".join(lines) + "\n"
    (run_dir / "record.md").write_text(text, encoding="utf-8")
    return text


def _capture_record(
    wiki_store: WikiStore, lab_id: str, n: int, status: str, record_text: str
) -> str:
    """Capture the run record as an immutable wiki source. Returns the source id."""
    source_id = f"exp-{lab_id}-r{n}"
    wiki_store.capture_source(
        source_id=source_id,
        title=f"Experiment run {lab_id}/r{n} ({status})",
        content=record_text,
        origin="experiment",
    )
    return source_id


def _read_metrics(run_dir: Path) -> dict | None:
    path = run_dir / "metrics.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _summary(status: str, n: int, wall: float, metrics: dict | None, source_id: str) -> str:
    """One-line result for the executor: status, run number, key metrics, wiki id."""
    parts = [f"run {n} {status}"]
    metric_bits = []
    if metrics:
        for key, label in (
            ("val_loss", "val_loss"),
            ("val_perplexity", "val_perplexity"),
            ("tokens_per_sec", "tokens_per_sec"),
        ):
            if metrics.get(key) is not None:
                metric_bits.append(f"{label}={metrics[key]}")
    metric_bits.append(f"wall={wall}s")
    return f"{parts[0]} — " + ", ".join(metric_bits) + f" — source: {source_id}"


# ----- subprocess launch + host-side timeout -----

def _kill_group(proc: subprocess.Popen) -> None:
    """SIGTERM the run's process group, grace, then SIGKILL whatever survives."""
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=_TERM_GRACE_S)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=_TERM_GRACE_S)
    except subprocess.TimeoutExpired:
        pass


def _launch(cmd: Sequence[str], run_dir: Path, timeout_s: float) -> tuple[int | None, float, bool]:
    """Run ``cmd`` with cwd=run_dir in its own process group, streaming merged
    stdout/stderr to run_dir/log.txt, under a host-side wall-clock kill.

    Returns (returncode, wall_seconds, timed_out)."""
    log_path = run_dir / "log.txt"
    start = time.monotonic()
    with log_path.open("wb") as log:
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(run_dir),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # setsid: own process group, killable as a whole
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        deadline = start + timeout_s
        timed_out = False
        while True:
            try:
                proc.wait(timeout=1.0)
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() >= deadline:
                    timed_out = True
                    _kill_group(proc)
                    break
    wall = round(time.monotonic() - start, 2)
    return proc.returncode, wall, timed_out


# ----- the run machinery -----

def execute_job(
    job: Job,
    goal: GoalConfig,
    wiki_store: WikiStore,
    *,
    labs_root: Path | str = LABS_ROOT,
    launcher: Launcher = default_launcher,
    queue: JobQueue | None = None,
) -> dict:
    """Execute one claimed job end-to-end and return the result fields (status,
    run_number, run_dir, summary) for JobQueue.complete().

    Snapshots the lab, injects the assets, runs the snapshot under a host timeout, and
    auto-captures the run record into the wiki. A missing asset fails the job cleanly
    (no subprocess is launched). ``launcher`` is a seam: production uses uv; tests
    inject a plain interpreter so no uv env is built (the subprocess sandbox — process
    group, cwd, host timeout, log streaming — is exercised unchanged).

    ``queue`` (the owning JobQueue) is optional: when given, the run dir is written back
    into running/<id>.json as soon as it is allocated, so a live observer (the forum
    feed) can tail this run's log while it is still in flight."""
    lab_dir = Path(labs_root) / job.lab_id
    if not lab_dir.is_dir():
        raise FileNotFoundError(f"lab does not exist: {lab_dir}")

    run_dir = next_run_dir(lab_dir)
    n = int(run_dir.name)
    if queue is not None:  # publish the run dir onto the running record for live tailing
        queue.annotate_running(job, run_dir=str(run_dir))
    code_hash = _snapshot_code(lab_dir, run_dir)
    (run_dir / "code.sha256").write_text(code_hash + "\n", encoding="utf-8")

    # Inject assets / refuse on a missing one — job fails cleanly, still documented.
    try:
        _write_run_config(lab_dir, run_dir, goal.experiment.assets)
    except AssetError as e:
        status = "failed"
        (run_dir / "log.txt").write_text(f"launch refused: {e}\n", encoding="utf-8")
        record = _write_synthetic_record(run_dir, job.lab_id, n, status, 0.0, code_hash)
        source_id = _capture_record(wiki_store, job.lab_id, n, status, record)
        summary = f"run {n} failed — {e} — source: {source_id}"
        return {"status": status, "run_number": n, "run_dir": str(run_dir), "summary": summary}

    code_dir = run_dir / "code"
    cmd = [*launcher(code_dir), str(code_dir / "main.py")]
    returncode, wall, timed_out = _launch(cmd, run_dir, float(job.timeout_s))

    if timed_out:
        status = "timeout"
    elif returncode == 0:
        status = "ok"
    else:
        status = "failed"

    record_path = run_dir / "record.md"
    if record_path.exists():
        record = record_path.read_text(encoding="utf-8")
    else:
        record = _write_synthetic_record(run_dir, job.lab_id, n, status, wall, code_hash)
    source_id = _capture_record(wiki_store, job.lab_id, n, status, record)

    metrics = _read_metrics(run_dir)
    summary = _summary(status, n, wall, metrics, source_id)
    return {"status": status, "run_number": n, "run_dir": str(run_dir), "summary": summary}


def _goal_timeout(goal: GoalConfig) -> int:
    return goal.experiment.run_timeout_s or DEFAULT_TIMEOUT_S


def make_run_experiment(
    lab_id: str,
    goal: GoalConfig,
    wiki_store: WikiStore,
    *,
    queue: JobQueue | None = None,
    timeout_s: int | None = None,
    poll_interval: float = 2.0,
):
    """Build the ZERO-ARG async callable wired into Subagent(run_experiment_callable=…).

    Calling it submits a job for ``lab_id`` and waits — polling the queue's done/ dir
    with asyncio.sleep — for the worker (a separate process) to finish, then returns
    the one-line summary. The run is defined entirely by the lab snapshot, so the
    callable needs no arguments (DESIGN.md — run_experiment takes no args)."""
    q = queue or JobQueue()
    limit = timeout_s if timeout_s is not None else _goal_timeout(goal)

    async def run_experiment() -> str:
        job_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        q.submit(Job(id=job_id, lab_id=lab_id, submitted_at=_now(), timeout_s=limit))
        while True:
            done = q.get_done(job_id)
            if done is not None:
                return done.summary or f"run {done.run_number} {done.status}"
            await asyncio.sleep(poll_interval)

    return run_experiment


# ----- standalone worker process -----

def run_forever(
    goal: GoalConfig,
    wiki_store: WikiStore,
    *,
    queue: JobQueue | None = None,
    labs_root: Path | str = LABS_ROOT,
    launcher: Launcher = default_launcher,
    idle_sleep_s: float = _IDLE_SLEEP_S,
) -> None:
    """Sweep stale runs once, then claim→execute→complete forever with an idle sleep.
    A single GPU, so one job at a time. Ctrl-C exits cleanly between jobs."""
    q = queue or JobQueue()
    for swept in q.sweep_stale():
        print(f"[worker] swept stale job {swept.id}: {swept.status}", flush=True)
    print("[worker] ready — polling for jobs (Ctrl-C to stop)", flush=True)
    while True:
        job = q.claim_next()
        if job is None:
            time.sleep(idle_sleep_s)
            continue
        print(f"[worker] claimed {job.id} (lab {job.lab_id})", flush=True)
        try:
            result = execute_job(
                job, goal, wiki_store, labs_root=labs_root, launcher=launcher, queue=q
            )
        except Exception as e:  # a broken job must not take the worker down
            result = {"status": "failed", "summary": f"worker error: {e!r}"}
        q.complete(job, **result)
        print(f"[worker] done {job.id}: {result.get('summary')}", flush=True)


def main() -> None:
    """`python -m autoresearch.queue.worker <goal.toml> [wiki_dir]`."""
    import sys

    from autoresearch.config import load_goal

    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m autoresearch.queue.worker <goal.toml> [wiki_dir]")
    goal = load_goal(sys.argv[1])
    wiki_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("wiki-library")
    wiki_store = WikiStore(wiki_dir)
    try:
        run_forever(goal, wiki_store)
    except KeyboardInterrupt:
        print("\n[worker] stopped.", flush=True)


if __name__ == "__main__":
    main()
