"""Filesystem GPU job queue — small JSON files, no database.

Layout:
    queue/jobs/pending/<job-id>.json    submitted, waiting for the GPU
    queue/jobs/running/<job-id>.json    claimed by the worker (atomic rename)
    queue/jobs/done/<job-id>.json       finished; result fields appended

A job references a lab run dir (lab/<lab-id>/runs/<n>/) that already contains the
config + snapshotted code. Submitting a job IS the run_experiment tool call —
there is deliberately no CLI and no env-var interface.
Single GPU: the worker claims one job at a time, in submission order.

TODO(implement): submit(), claim_next() via atomic rename, complete(), and a
stale-running sweep for worker crashes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

QUEUE_DIR = Path("queue/jobs")


@dataclass
class Job:
    id: str
    lab_id: str
    run_number: int
    wall_clock_limit_s: int
