"""The orchestrator: the only agent the user talks to.

Contract (DESIGN.md — Orchestrator):
- Small idle context. It plans, composes a prompt + detailed rules per spawn,
  receives a short summary back, evaluates, decides the next step. It NEVER reads
  or writes code unless the user explicitly asks an implementation question.
- Spawns subagents fire-and-forget with a timeout; one retry with an amended
  prompt, then reports failure to the user.
- Keeps finished subagent sessions (see spawn.SubagentSession) and may send them
  follow-up questions instead of respawning.
- Picks the subagent model per task from ModelsConfig.subagent_models using their
  `description` fields; may request a specific model to re-review an unexpected
  result.
- Assigns architecture-level ideas only — never trivia like "test a wider d_model".
- Runs until the user says stop. Checkpoints after each subagent finishes
  (checkpoint.py); restart resumes from the last completed subagent.

TODO(implement): the async loop — one turn per (user message | subagent completion
| digest timer), driven through llm.base.LLMClient with tools from prompt/tools/.
"""
from __future__ import annotations

from ..config import GoalConfig, ModelsConfig


class Orchestrator:
    def __init__(self, models: ModelsConfig, goal: GoalConfig):
        self.models = models
        self.goal = goal

    async def run(self) -> None:
        raise NotImplementedError
