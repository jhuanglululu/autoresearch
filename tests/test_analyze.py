"""Tests for the analyze_records tool implementation (subagent/analyze.py).

These exercise the two behaviours that matter: a snippet can read a run's
records.jsonl from its cwd and compute an answer to stdout, and a runaway
snippet is killed by the wall-clock timeout instead of hanging.
"""
from __future__ import annotations

import json

from autoresearch.subagent.analyze import analyze


def _write_records(dir_path):
    """A tiny records.jsonl fixture: three eval events with known val_losses."""
    lines = [
        {"event": "run_start", "step": 0, "t_wall": 0.0},
        {"event": "eval", "step": 100, "t_wall": 1.0, "val_loss": 4.0},
        {"event": "eval", "step": 200, "t_wall": 2.0, "val_loss": 3.0},
        {"event": "eval", "step": 300, "t_wall": 3.0, "val_loss": 2.0},
    ]
    path = dir_path / "records.jsonl"
    path.write_text("\n".join(json.dumps(o) for o in lines) + "\n", encoding="utf-8")


def test_snippet_reads_records_and_computes(tmp_path):
    _write_records(tmp_path)
    # Best (minimum) val_loss across eval events. Hand-computed expectation: 2.0
    # (min of 4.0, 3.0, 2.0), independent of the implementation under test.
    code = (
        "import json\n"
        "vals = [json.loads(l)['val_loss'] for l in open('records.jsonl')\n"
        "        if json.loads(l)['event'] == 'eval']\n"
        "print(min(vals))\n"
    )
    result = analyze(code, cwd=tmp_path)

    assert not result.timed_out
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "2.0"


def test_infinite_loop_times_out(tmp_path):
    result = analyze("while True:\n    pass\n", cwd=tmp_path, timeout=1.0)

    assert result.timed_out
    assert result.returncode is None
    assert "timeout" in result.stderr.lower()
