"""Filesystem GPU job queue — small JSON files, no database.

Layout:
    queue/jobs/pending/<job-id>.json    submitted, waiting for the GPU
    queue/jobs/running/<job-id>.json    claimed by the worker (atomic rename)
    queue/jobs/done/<job-id>.json       finished; result fields appended

A job names a lab (lab/<lab-id>/) whose current files — including run_config.toml —
the worker snapshots into a fresh runs/<n>/ at execution time. Submitting a job IS
the run_experiment tool call: there is deliberately no CLI and no env-var interface.
Single GPU: the worker claims one job at a time, oldest first.

State transitions are single moves on one filesystem, so they are atomic:

    submit       -> writes pending/<id>.json (tmp + os.replace)
    claim_next   -> os.rename pending/<id>.json -> running/<id>.json (the claim IS
                    the rename; whoever wins the rename owns the job)
    complete     -> writes done/<id>.json then unlinks running/<id>.json
    sweep_stale  -> on worker start, anything left in running/ is from a crash

Timestamps are supplied by the caller (``submitted_at``) so the queue itself stays
clock-free and deterministic under test.
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from pathlib import Path

QUEUE_DIR = Path("queue/jobs")

# The three lanes of the queue, in order of a job's life.
_LANES = ("pending", "running", "done")


@dataclass
class Job:
    """One GPU job. The first four fields are set at submit time; the rest are the
    result, appended by the worker on completion (and absent while pending)."""

    id: str
    lab_id: str
    submitted_at: str
    timeout_s: int
    # ----- result fields (None until completed) -----
    status: str | None = None  # ok | failed | timeout
    run_number: int | None = None
    run_dir: str | None = None
    summary: str | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Job":
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in fields})


class JobQueue:
    """Filesystem-backed single-GPU job queue rooted at ``root`` (default
    ``queue/jobs``). All state is small JSON files under pending/, running/, done/;
    there is no database and no lock — the atomic ``os.rename`` claim is the only
    coordination needed for the single worker."""

    def __init__(self, root: Path | str = QUEUE_DIR) -> None:
        self.root = Path(root)
        self.pending = self.root / "pending"
        self.running = self.root / "running"
        self.done = self.root / "done"
        for d in (self.pending, self.running, self.done):
            d.mkdir(parents=True, exist_ok=True)

    # ----- io helpers -----

    @staticmethod
    def _write(path: Path, job: Job) -> None:
        """Atomic write: dump to a temp file in the same dir, then os.replace."""
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(job.to_dict(), indent=2), encoding="utf-8")
        os.replace(tmp, path)

    @staticmethod
    def _read(path: Path) -> Job:
        return Job.from_dict(json.loads(path.read_text(encoding="utf-8")))

    # ----- operations -----

    def submit(self, job: Job) -> Job:
        """Enqueue a job (write pending/<id>.json). Refuses a duplicate id."""
        for lane in (self.pending, self.running, self.done):
            if (lane / f"{job.id}.json").exists():
                raise FileExistsError(f"job id already in the queue: {job.id!r}")
        self._write(self.pending / f"{job.id}.json", job)
        return job

    def claim_next(self) -> Job | None:
        """Claim the oldest pending job by atomically renaming it into running/.

        Oldest is by (submitted_at, id). The rename IS the claim: if it loses a race
        (the file vanished), the next candidate is tried. Returns the claimed Job, or
        None when nothing is pending."""
        candidates = sorted(
            self.pending.glob("*.json"),
            key=lambda p: (self._read(p).submitted_at, p.stem),
        )
        for src in candidates:
            dst = self.running / src.name
            try:
                os.rename(src, dst)
            except OSError:
                continue  # someone/something took it first — try the next
            return self._read(dst)
        return None

    def complete(self, job: Job, **result) -> Job:
        """Record a finished job in done/ and drop it from running/.

        ``result`` overrides fields on the job (status, run_number, run_dir, summary).
        done/ is written first so a crash between the two steps leaves the result on
        disk (the running/ leftover is then harmless — sweep_stale sees the done copy
        and just clears it)."""
        for key, value in result.items():
            if not hasattr(job, key):
                raise AttributeError(f"Job has no result field {key!r}")
            setattr(job, key, value)
        self._write(self.done / f"{job.id}.json", job)
        running = self.running / f"{job.id}.json"
        if running.exists():
            running.unlink()
        return job

    def get_done(self, job_id: str) -> Job | None:
        """Return the completed job if it is in done/, else None (used by waiters)."""
        path = self.done / f"{job_id}.json"
        if not path.exists():
            return None
        return self._read(path)

    def sweep_stale(self) -> list[Job]:
        """Reconcile running/ on worker start. Anything here is orphaned by a crash.

        Policy — FAIL, do not re-queue. This is a single-GPU, single-worker system:
        a job in running/ means the previous worker died mid-run, and its training
        subprocess (its own setsid process group) may still be alive holding the GPU.
        Re-queueing would risk launching a second run onto an occupied GPU and would
        silently repeat a possibly-expensive run behind the executor's back. Marking
        it failed (with a note that GPU state is unknown) instead surfaces the anomaly
        to the orchestrator, unblocks any waiter polling done/, and lets a human decide
        whether to resubmit. If the job already has a done/ copy (crash mid-complete),
        we just clear the running/ leftover. Returns the swept jobs."""
        swept: list[Job] = []
        for path in sorted(self.running.glob("*.json")):
            job = self._read(path)
            done_path = self.done / path.name
            if done_path.exists():
                path.unlink()  # completion already recorded; drop the stale copy
                continue
            job.status = "failed"
            job.summary = (
                f"job {job.id} was running when the worker restarted — failed without "
                "a result; the GPU state at crash time is unknown. Resubmit if needed."
            )
            self._write(done_path, job)
            path.unlink()
            swept.append(job)
        return swept
