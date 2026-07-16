"""Per-goal checkpoints: state/<goal-id>/checkpoint.json (+ git auto-commit of KB).

Deliberately simple (DESIGN.md): a snapshot is taken after each subagent finishes
(and after each orchestrator turn), NOT at the exact instruction the process died
on. Restarting a goal id loads the latest snapshot and continues.

What a checkpoint holds:
- ``orchestrator_messages`` — the orchestrator's own conversation, serialized
  ``llm.base.Message`` records (role, content, tool_calls, tool_call_id). Restoring
  rebuilds this history verbatim so the model resumes with its full context.
- ``completed_sessions`` — one small record per finished subagent (id, type, model,
  prompt, summary, status). The subagents themselves are NOT revived on restart:
  their live context is gone, but their one-line summaries already live inside the
  orchestrator messages, which is all the orchestrator ever saw of them. A follow-up
  after a restart is therefore only possible on sessions spawned in the current
  process; a resumed orchestrator treats prior sessions as read-only history.
- ``last_digest_index`` — how many completed sessions have already been folded into
  a digest, so the next digest reports only what is new.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..llm.base import Message, ToolCall

STATE_DIR = Path("state")


# ----- Message (de)serialization -----

def message_to_dict(m: Message) -> dict:
    """Serialize a base.Message (including tool calls) to a JSON-safe dict."""
    return {
        "role": m.role,
        "content": m.content,
        "tool_calls": [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
            for tc in m.tool_calls
        ],
        "tool_call_id": m.tool_call_id,
    }


def message_from_dict(d: dict) -> Message:
    return Message(
        role=d["role"],
        content=d.get("content", "") or "",
        tool_calls=[
            ToolCall(id=tc["id"], name=tc["name"], arguments=tc.get("arguments", {}))
            for tc in d.get("tool_calls", [])
        ],
        tool_call_id=d.get("tool_call_id"),
    )


@dataclass
class GoalState:
    goal_id: str
    orchestrator_messages: list[dict] = field(default_factory=list)
    completed_sessions: list[dict] = field(default_factory=list)
    last_digest_index: int = 0
    notes: str = ""

    def __post_init__(self) -> None:
        # Where this checkpoint lives. Not a dataclass field (so it never lands in
        # the JSON); set by load_or_create / save so tests can point at a tmp dir.
        self._root: Path = STATE_DIR

    # ----- helpers the orchestrator uses to rebuild its own history -----

    def messages(self) -> list[Message]:
        return [message_from_dict(d) for d in self.orchestrator_messages]

    def set_messages(self, messages: list[Message]) -> None:
        self.orchestrator_messages = [message_to_dict(m) for m in messages]

    def record_session(
        self, *, id: str, type: str, model: str, prompt: str, summary: str, status: str
    ) -> None:
        self.completed_sessions.append(
            {
                "id": id,
                "type": type,
                "model": model,
                "prompt": prompt,
                "summary": summary,
                "status": status,
            }
        )

    def path_in(self, root: Path | str) -> Path:
        return Path(root) / self.goal_id / "checkpoint.json"

    def save(self, root: Path | str | None = None) -> None:
        root = Path(root) if root is not None else self._root
        self._root = root
        path = self.path_in(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=1))

    @classmethod
    def load_or_create(cls, goal_id: str, root: Path | str = STATE_DIR) -> "GoalState":
        path = Path(root) / goal_id / "checkpoint.json"
        if path.is_file():
            state = cls(**json.loads(path.read_text()))
        else:
            state = cls(goal_id=goal_id)
        state._root = Path(root)
        return state
