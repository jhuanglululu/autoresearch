"""Tests for the filesystem GPU job queue + worker (autoresearch.queue.*).

No torch, no network, no real uv env builds. A FAKE minimal lab (stub pyproject +
a stdlib-only main.py) stands in for the real training lab, and the worker's
`launcher` seam is used to run that main.py under the plain test interpreter instead
of `uv run` — so the subprocess sandbox (own process group, cwd = run dir, host-side
wall-clock kill, log streaming) is exercised for real while the (slow, network-bound)
uv env build is skipped. Everything below the launcher is the production code path.
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
import threading
import time
from pathlib import Path

import pytest

from autoresearch.config import ExperimentSpec, GoalConfig
from autoresearch.queue import worker as W
from autoresearch.queue.jobs import Job, JobQueue
from autoresearch.queue.labs import COPY_EXCLUDE, create_lab, next_run_dir
from autoresearch.wiki.store import WikiStore

# A stdlib-only main.py: reads run_config.toml from cwd (proving [assets] injection),
# writes the same artifacts the real lab does, quickly.
FAST_MAIN = '''\
import json, tomllib
from pathlib import Path
run = Path.cwd()
cfg = tomllib.loads((run / "run_config.toml").read_text())
assets = cfg["assets"]
metrics = {"val_loss": 2.5, "val_perplexity": 12.18, "tokens_per_sec": 9999,
           "wall_seconds": 0.01, "params": 123}
(run / "metrics.json").write_text(json.dumps(metrics))
(run / "records.jsonl").write_text(json.dumps({"event": "run_start"}) + "\\n")
(run / "record.md").write_text(
    "# fake run\\ncorpus=" + assets["corpus"] + "\\nval_loss=2.5\\n"
)
print("fake run ok")
'''

# A crashing main.py: exits non-zero before writing record.md (hard crash path).
CRASH_MAIN = '''\
import sys
print("about to crash")
sys.exit(3)
'''

# A slow main.py: spawns a child (same process group) and sleeps, so a host timeout
# must kill the WHOLE group. Records the child pid so the test can assert it died.
SLOW_MAIN = '''\
import subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
Path.cwd().joinpath("child.pid").write_text(str(child.pid))
time.sleep(60)
'''

TEST_LAUNCHER = lambda code_dir: [sys.executable]  # noqa: E731 — the injected seam


# ----- fixtures / builders -----

def _make_baseline(tmp_path: Path, main_src: str = FAST_MAIN) -> Path:
    baseline = tmp_path / "baseline"
    (baseline).mkdir()
    (baseline / "pyproject.toml").write_text(
        '[project]\nname = "lab"\nversion = "0.1.0"\nrequires-python = ">=3.11"\n'
    )
    (baseline / "run_config.toml").write_text("[train]\nsteps = 1\n")
    (baseline / "main.py").write_text(main_src)
    return baseline


def _make_assets(tmp_path: Path) -> dict[str, Path]:
    corpus = tmp_path / "assets" / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / "tokens.bin").write_bytes(b"\0\0")
    tok = tmp_path / "assets" / "tokenizer.json"
    tok.write_text("{}")
    return {"corpus": corpus, "tokenizer": tok}


def _goal(baseline: Path, assets: dict[str, Path], run_timeout_s=None) -> GoalConfig:
    template = baseline.parent / "template.md"
    template.write_text("goal\n")
    return GoalConfig(
        id="g",
        template_path=template,
        experiment=ExperimentSpec(
            baseline=baseline, assets=assets, run_timeout_s=run_timeout_s
        ),
    )


@pytest.fixture
def wiki(tmp_path):
    return WikiStore(tmp_path / "wiki-library")


def _job(jid: str, submitted_at: str, lab_id: str = "lab-a", timeout_s: int = 30) -> Job:
    return Job(id=jid, lab_id=lab_id, submitted_at=submitted_at, timeout_s=timeout_s)


# ----- (1) queue order + atomic claim + complete -----

def test_claim_orders_oldest_first_then_id(tmp_path):
    q = JobQueue(tmp_path / "queue")
    q.submit(_job("b", "2026-01-01T00:00:02"))
    q.submit(_job("a", "2026-01-01T00:00:01"))
    q.submit(_job("c", "2026-01-01T00:00:01"))  # tie with 'a' -> break by id
    assert q.claim_next().id == "a"
    assert q.claim_next().id == "c"
    assert q.claim_next().id == "b"
    assert q.claim_next() is None


def test_claim_is_atomic_move_and_complete_moves_to_done(tmp_path):
    q = JobQueue(tmp_path / "queue")
    q.submit(_job("j1", "2026-01-01T00:00:00"))
    claimed = q.claim_next()
    # The claim IS the rename: gone from pending, present in running.
    assert not (q.pending / "j1.json").exists()
    assert (q.running / "j1.json").exists()
    q.complete(claimed, status="ok", run_number=1, run_dir="lab/lab-a/runs/1", summary="run 1 ok")
    assert not (q.running / "j1.json").exists()
    done = q.get_done("j1")
    assert done.status == "ok" and done.run_number == 1 and done.summary == "run 1 ok"


def test_submit_rejects_duplicate_id(tmp_path):
    q = JobQueue(tmp_path / "queue")
    q.submit(_job("dup", "2026-01-01T00:00:00"))
    with pytest.raises(FileExistsError):
        q.submit(_job("dup", "2026-01-01T00:00:01"))


# ----- (1b) live-run plumbing: run_dir onto the running record + find_running -----

def test_execute_job_annotates_running_with_run_dir(tmp_path, wiki):
    baseline = _make_baseline(tmp_path)
    goal = _goal(baseline, _make_assets(tmp_path))
    create_lab(goal, "lab-a", labs_root=tmp_path / "lab")
    q = JobQueue(tmp_path / "queue")
    q.submit(_job("r", "2026-01-01T00:00:00"))
    job = q.claim_next()

    # execute_job does not complete the job (the worker loop does) — so afterwards it is
    # still in running/, now carrying the run_dir the worker allocated at launch.
    result = W.execute_job(
        job, goal, wiki, labs_root=tmp_path / "lab", launcher=TEST_LAUNCHER, queue=q
    )
    running = q.find_running("lab-a")
    assert running is not None
    assert running.run_dir == result["run_dir"]


def test_annotate_running_is_noop_when_not_in_running(tmp_path):
    q = JobQueue(tmp_path / "queue")
    job = _job("gone", "2026-01-01T00:00:00")  # never submitted/claimed
    q.annotate_running(job, run_dir="whatever")  # must not raise, writes nothing
    assert not (q.running / "gone.json").exists()


def test_find_running_matches_lab_id(tmp_path):
    q = JobQueue(tmp_path / "queue")
    q.submit(_job("j1", "2026-01-01T00:00:00", lab_id="lab-a"))
    q.submit(_job("j2", "2026-01-01T00:00:01", lab_id="lab-b"))
    q.claim_next()  # lab-a -> running
    q.claim_next()  # lab-b -> running
    assert q.find_running("lab-a").id == "j1"
    assert q.find_running("lab-b").id == "j2"
    assert q.find_running("lab-z") is None


# ----- (2) stale-running sweep -----

def test_sweep_stale_fails_orphaned_running_jobs(tmp_path):
    q = JobQueue(tmp_path / "queue")
    q.submit(_job("orphan", "2026-01-01T00:00:00"))
    q.claim_next()  # now in running/, then pretend the worker died
    swept = q.sweep_stale()
    assert [j.id for j in swept] == ["orphan"]
    assert not (q.running / "orphan.json").exists()
    done = q.get_done("orphan")
    assert done.status == "failed"
    assert "worker restarted" in done.summary


def test_sweep_clears_running_leftover_when_done_exists(tmp_path):
    # Crash between writing done/ and unlinking running/: sweep just clears running/.
    q = JobQueue(tmp_path / "queue")
    q.submit(_job("j", "2026-01-01T00:00:00"))
    job = q.claim_next()
    q._write(q.done / "j.json", job)  # done already recorded
    swept = q.sweep_stale()
    assert swept == []  # not re-failed
    assert not (q.running / "j.json").exists()


# ----- (3) create_lab copy/exclusions + next_run_dir -----

def test_create_lab_copies_and_excludes(tmp_path):
    baseline = _make_baseline(tmp_path)
    for junk in COPY_EXCLUDE:
        (baseline / junk).mkdir()
        (baseline / junk / "x").write_text("junk")
    goal = _goal(baseline, _make_assets(tmp_path))
    lab = create_lab(goal, "lab-a", labs_root=tmp_path / "lab")
    assert (lab / "main.py").is_file() and (lab / "pyproject.toml").is_file()
    for junk in COPY_EXCLUDE:
        assert not (lab / junk).exists(), junk
    # A known id resumes, it does not re-seed: a second create errors.
    with pytest.raises(FileExistsError):
        create_lab(goal, "lab-a", labs_root=tmp_path / "lab")


def test_next_run_dir_increments_max(tmp_path):
    lab = tmp_path / "lab-a"
    (lab / "runs").mkdir(parents=True)
    assert next_run_dir(lab).name == "1"
    (lab / "runs" / "7").mkdir()
    (lab / "runs" / "junk").mkdir()  # non-numeric ignored
    assert next_run_dir(lab).name == "8"


# ----- (4) snapshot excludes runs/ and .venv -----

def test_snapshot_excludes_runs_and_venv(tmp_path, wiki):
    baseline = _make_baseline(tmp_path)
    goal = _goal(baseline, _make_assets(tmp_path))
    lab = create_lab(goal, "lab-a", labs_root=tmp_path / "lab")
    (lab / ".venv").mkdir()
    (lab / ".venv" / "big").write_text("env")
    (lab / "runs").mkdir(exist_ok=True)
    (lab / "runs" / "old").write_text("prior")
    job = _job("s", "2026-01-01T00:00:00")
    result = W.execute_job(job, goal, wiki, labs_root=tmp_path / "lab", launcher=TEST_LAUNCHER)
    code = Path(result["run_dir"]) / "code"
    assert (code / "main.py").is_file()
    assert not (code / ".venv").exists()
    assert not (code / "runs").exists()  # no recursion into the archive
    assert (Path(result["run_dir"]) / "code.sha256").is_file()


# ----- (5) [assets] injection + missing-asset refusal -----

def test_assets_injected_into_run_config(tmp_path, wiki):
    baseline = _make_baseline(tmp_path)
    assets = _make_assets(tmp_path)
    goal = _goal(baseline, assets)
    create_lab(goal, "lab-a", labs_root=tmp_path / "lab")
    job = _job("a", "2026-01-01T00:00:00")
    result = W.execute_job(job, goal, wiki, labs_root=tmp_path / "lab", launcher=TEST_LAUNCHER)
    cfg = (Path(result["run_dir"]) / "run_config.toml").read_text()
    assert "[assets]" in cfg
    assert str(assets["corpus"].resolve()) in cfg
    assert str(assets["tokenizer"].resolve()) in cfg
    assert "[train]" in cfg  # the engineer's original section is preserved


def test_missing_asset_refuses_launch_and_documents(tmp_path, wiki):
    baseline = _make_baseline(tmp_path)
    assets = {"corpus": tmp_path / "nope" / "corpus", "tokenizer": tmp_path / "nope.json"}
    goal = _goal(baseline, assets)
    create_lab(goal, "lab-a", labs_root=tmp_path / "lab")
    job = _job("m", "2026-01-01T00:00:00")
    result = W.execute_job(job, goal, wiki, labs_root=tmp_path / "lab", launcher=TEST_LAUNCHER)
    assert result["status"] == "failed"
    # No subprocess ran (no metrics), but the failure is captured for the record.
    assert not (Path(result["run_dir"]) / "metrics.json").exists()
    captured = wiki.read_source(f"exp-lab-a-r{result['run_number']}")
    assert "missing" in captured.lower()


# ----- (6) end-to-end execute_job on the fake lab -----

def test_execute_job_end_to_end(tmp_path, wiki):
    baseline = _make_baseline(tmp_path)
    assets = _make_assets(tmp_path)
    goal = _goal(baseline, assets)
    create_lab(goal, "lab-a", labs_root=tmp_path / "lab")
    job = _job("e", "2026-01-01T00:00:00")
    result = W.execute_job(job, goal, wiki, labs_root=tmp_path / "lab", launcher=TEST_LAUNCHER)

    run_dir = Path(result["run_dir"])
    assert result["status"] == "ok" and result["run_number"] == 1
    # Artifacts landed in the run dir.
    for artifact in ("metrics.json", "records.jsonl", "record.md", "log.txt"):
        assert (run_dir / artifact).is_file(), artifact
    # Wiki source captured with origin=experiment and the record's content.
    src_id = "exp-lab-a-r1"
    captured = wiki.read_source(src_id)
    assert "origin: experiment" in captured
    assert str(assets["corpus"].resolve()) in captured  # record.md echoed the asset
    # Summary string is sane: status, run number, key metrics, and the wiki id.
    # (wall is timing-dependent, so its value is matched by shape, not exact number.)
    summary = result["summary"]
    assert summary.startswith("run 1 ok — ")
    assert "val_loss=2.5" in summary
    assert "val_perplexity=12.18" in summary
    assert "tokens_per_sec=9999" in summary
    assert re.search(r"wall=\d+(\.\d+)?s", summary)
    assert summary.endswith(f"source: {src_id}")


def test_hard_crash_synthesizes_record(tmp_path, wiki):
    baseline = _make_baseline(tmp_path, main_src=CRASH_MAIN)
    goal = _goal(baseline, _make_assets(tmp_path))
    create_lab(goal, "lab-a", labs_root=tmp_path / "lab")
    job = _job("c", "2026-01-01T00:00:00")
    result = W.execute_job(job, goal, wiki, labs_root=tmp_path / "lab", launcher=TEST_LAUNCHER)
    assert result["status"] == "failed"
    captured = wiki.read_source("exp-lab-a-r1")
    assert "synthesized" in captured.lower()
    assert "about to crash" in captured  # log tail folded into the record


# ----- (7) timeout path: process group killed, status timeout, record synthesized -----

def test_timeout_kills_process_group(tmp_path, wiki):
    baseline = _make_baseline(tmp_path, main_src=SLOW_MAIN)
    goal = _goal(baseline, _make_assets(tmp_path))
    create_lab(goal, "lab-a", labs_root=tmp_path / "lab")
    job = _job("t", "2026-01-01T00:00:00", timeout_s=1)
    result = W.execute_job(job, goal, wiki, labs_root=tmp_path / "lab", launcher=TEST_LAUNCHER)
    assert result["status"] == "timeout"

    run_dir = Path(result["run_dir"])
    child_pid = int((run_dir / "child.pid").read_text())
    # The whole process group was killed — the grandchild must be gone too.
    for _ in range(40):
        try:
            os.kill(child_pid, 0)
        except (ProcessLookupError, PermissionError):
            break
        time.sleep(0.05)
    else:
        pytest.fail("child process survived the group kill")
    # A record was synthesized (the slow run wrote none) and captured.
    captured = wiki.read_source("exp-lab-a-r1")
    assert "timeout" in captured.lower()


# ----- (8) make_run_experiment round-trip -----

def test_make_run_experiment_round_trip(tmp_path, wiki):
    baseline = _make_baseline(tmp_path)
    goal = _goal(baseline, _make_assets(tmp_path))
    create_lab(goal, "lab-a", labs_root=tmp_path / "lab")
    q = JobQueue(tmp_path / "queue")

    def worker_once() -> None:
        job = None
        while job is None:
            job = q.claim_next()
            if job is None:
                time.sleep(0.01)
        result = W.execute_job(
            job, goal, wiki, labs_root=tmp_path / "lab", launcher=TEST_LAUNCHER
        )
        q.complete(job, **result)

    async def scenario() -> str:
        runner = W.make_run_experiment(
            "lab-a", goal, wiki, queue=q, poll_interval=0.02
        )
        t = threading.Thread(target=worker_once)
        t.start()
        try:
            return await runner()  # zero-arg callable: submit + wait for done/
        finally:
            t.join()

    summary = asyncio.run(scenario())
    assert summary.startswith("run 1 ok")
    assert "source: exp-lab-a-r1" in summary


def test_run_timeout_from_goal_config(tmp_path, wiki):
    baseline = _make_baseline(tmp_path)
    goal = _goal(baseline, _make_assets(tmp_path), run_timeout_s=42)
    assert W._goal_timeout(goal) == 42
    goal_default = _goal(baseline, _make_assets(tmp_path))
    assert W._goal_timeout(goal_default) == W.DEFAULT_TIMEOUT_S
