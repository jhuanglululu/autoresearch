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
