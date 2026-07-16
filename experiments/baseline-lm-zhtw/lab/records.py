"""Append-only machine log for a run: records.jsonl in the run dir.

Distinct from metrics.json (the at-a-glance summary, overwritten in place):
records.jsonl is a monotonic, append-only event stream for bookkeeping and
later programmatic analysis (compare loss curves across runs, etc. — see the
`analyze_records` agent tool). One JSON object per line, each with an "event"
field and a monotonic "t_wall" (seconds since run start); step-tied events also
carry "step".

Event vocabulary written by this lab:
  run_start   full resolved config (assets, model, train, device, data stats)
  train_log   periodic training point: loss, lr, grad_norm, tok/s
  eval        a validation pass: val_loss, val_perplexity
  checkpoint  weights written: step, val_loss, best (bool), path
  final_evals the finalized eval suite (val loss/ppl, params, vram, sample)
  failure     an unhandled exception, with traceback
"""
from __future__ import annotations

import json
import time
from pathlib import Path


class RecordLog:
    """Append-only JSONL writer. Reopens per line so a killed run keeps a valid
    (line-complete) log; volume is low (one line per log/eval interval)."""

    def __init__(self, path, t0: float | None = None):
        self.path = Path(path)
        self.t0 = t0 if t0 is not None else time.monotonic()

    def log(self, event: str, step: int | None = None, **fields) -> None:
        rec = {"event": event, "t_wall": round(time.monotonic() - self.t0, 3)}
        if step is not None:
            rec["step"] = step
        rec.update(fields)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
