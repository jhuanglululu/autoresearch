"""Per-goal checkpoints: state/<goal-id>/checkpoint.json (+ git auto-commit of KB).

Deliberately simple (DESIGN.md): a snapshot is taken after each subagent finishes,
NOT at the exact instruction the process died on. Restarting a goal id loads the
latest snapshot and continues.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

STATE_DIR = Path("state")


@dataclass
class GoalState:
    goal_id: str
    orchestrator_messages: list[dict] = field(default_factory=list)
    completed_subagents: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def path(self) -> Path:
        return STATE_DIR / self.goal_id / "checkpoint.json"

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=1))

    @classmethod
    def load_or_create(cls, goal_id: str) -> "GoalState":
        path = STATE_DIR / goal_id / "checkpoint.json"
        if path.is_file():
            return cls(**json.loads(path.read_text()))
        return cls(goal_id=goal_id)
