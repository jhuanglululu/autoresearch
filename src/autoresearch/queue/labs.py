"""Lab lifecycle — create an editable lab from a goal's baseline, allocate run dirs.

A lab (lab/<lab-id>/) is a full uv project seeded by copying the goal's baseline
template (DESIGN.md — Experiments). The executor subagent then owns and edits it
freely; the worker snapshots it per run into lab/<lab-id>/runs/<n>/code/.

Nothing here touches the wiki or the pinned assets — those live outside every lab.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from autoresearch.config import GoalConfig

LABS_ROOT = Path("lab")

# Never copied into a lab or a per-run snapshot: the run archive (would recurse /
# balloon), the built uv env, and byte-caches. ".git" is only relevant to snapshots.
COPY_EXCLUDE = ("runs", ".venv", "__pycache__")
SNAPSHOT_EXCLUDE = ("runs", ".venv", "__pycache__", ".git")


def create_lab(goal: GoalConfig, lab_id: str, *, labs_root: Path | str = LABS_ROOT) -> Path:
    """Create lab/<lab_id>/ by copying the goal's baseline template.

    Excludes runs/, .venv, and __pycache__ so a fresh lab is just the source project.
    Errors if the lab already exists (a known id resumes an existing lab; it is not
    re-seeded from the template)."""
    dest = Path(labs_root) / lab_id
    if dest.exists():
        raise FileExistsError(f"lab already exists: {dest} (a known id resumes, not re-seeds)")
    baseline = goal.experiment.baseline
    if not (baseline / "pyproject.toml").is_file():
        raise FileNotFoundError(f"baseline is not a uv project: {baseline}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(baseline, dest, ignore=shutil.ignore_patterns(*COPY_EXCLUDE))
    return dest


def revert_lab(lab_dir: Path | str, run_number: int) -> str:
    """Restore the LIVE lab working directory to a run's code snapshot in place.

    Semantics: restore the working tree, KEEP the archives. runs/<n>/code/ is the
    captured code that produced run n; this makes the lab's working tree byte-identical
    to it again. The whole live tree is wiped first EXCEPT runs/ — that directory is the
    permanent archive and is never touched, including the very snapshot used here and
    every other run's artifacts. .venv/__pycache__ leftovers are wiped too: they are
    rebuildable (uv resolves a fresh env per run) and never belong to a snapshot.

    NOTES.md and any other working file: the SNAPSHOT's copy wins. It was part of the
    captured code, so after a revert the file reads exactly as it did at snapshot time —
    a live edit made after the run is discarded, by design.

    Orchestrator-only (wired as the `revert_lab` tool in orchestrator/loop.py). No
    subagent may revert a lab manually. Raises FileNotFoundError with a clear message if
    the lab or the requested snapshot is missing. Returns a one-line summary."""
    lab_dir = Path(lab_dir)
    if not lab_dir.is_dir():
        raise FileNotFoundError(f"lab does not exist: {lab_dir}")
    snapshot = lab_dir / "runs" / str(run_number) / "code"
    if not snapshot.is_dir():
        raise FileNotFoundError(
            f"no code snapshot for run {run_number}: {snapshot} does not exist "
            "(is the run number right? the archive is under runs/)"
        )
    # Wipe the live working tree, preserving ONLY runs/ (the untouchable archive).
    for child in lab_dir.iterdir():
        if child.name == "runs":
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
    # Copy the snapshot's contents back in (its copy of every file wins). The snapshot
    # never contains runs/ (SNAPSHOT_EXCLUDE), so the archive can never be re-created here.
    for child in snapshot.iterdir():
        dest = lab_dir / child.name
        if child.is_dir() and not child.is_symlink():
            shutil.copytree(child, dest)
        else:
            shutil.copy2(child, dest)
    return f"lab {lab_dir.name} restored to runs/{run_number} snapshot; archive untouched"


def next_run_dir(lab_dir: Path | str) -> Path:
    """Allocate and create lab/<id>/runs/<n>/ with n = max existing numbered run + 1.

    Numbering starts at 1 and ignores any non-numeric entries under runs/."""
    runs = Path(lab_dir) / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    highest = 0
    for child in runs.iterdir():
        if child.is_dir() and child.name.isdigit():
            highest = max(highest, int(child.name))
    run_dir = runs / str(highest + 1)
    run_dir.mkdir()
    return run_dir
