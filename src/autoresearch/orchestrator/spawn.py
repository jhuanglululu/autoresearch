"""Subagent sessions, spawned and owned by the orchestrator.

Exactly two subagent types (DESIGN.md):
- "executor":   writes + executes experiment code, submits GPU jobs, writes wiki.
- "researcher": no code editing or execution; searches web/arXiv, reads everything,
                writes wiki notes directly.

Roles (brainstormer, scout, synthesizer, engineer, utility) are prompt templates
the orchestrator composes into the spawn prompt — not separate types.

A finished session is KEPT (self.messages retained) so the orchestrator can call
follow_up() instead of respawning an agent to rediscover the same context.
Subagents cannot spawn subagents.

TODO(implement): the subagent tool-call loop lives in subagent/runner.py; this
module manages lifecycle (timeout, single retry with amended prompt, kill on user
request) and the session registry.
"""
from __future__ import annotations

from typing import Literal

SubagentType = Literal["executor", "researcher"]


class SubagentSession:
    def __init__(self, type_: SubagentType, model_name: str, prompt: str):
        self.type = type_
        self.model_name = model_name
        self.prompt = prompt
        self.messages: list = []
        self.summary: str | None = None

    async def run(self, timeout_s: float) -> str:
        """Run to completion; returns the short summary for the orchestrator."""
        raise NotImplementedError

    async def follow_up(self, question: str) -> str:
        """Ask a finished session a question with its full context intact."""
        raise NotImplementedError
