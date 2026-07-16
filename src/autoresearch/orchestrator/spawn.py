"""Subagent sessions, spawned and owned by the orchestrator.

Exactly two subagent types (DESIGN.md):
- "executor":   writes + executes experiment code, submits GPU jobs, writes wiki.
- "researcher": no code editing or execution; searches web/arXiv, reads everything,
                writes wiki notes directly.

Roles (brainstormer, scout, synthesizer, engineer, utility) are prompt templates
the orchestrator composes into the spawn prompt — not separate types.

A finished session is KEPT (its runner and messages are retained) so the orchestrator
can call follow_up() instead of respawning an agent to rediscover the same context.
Subagents cannot spawn subagents.

This module owns lifecycle only — a session wraps a ``subagent.runner`` runner and
runs it under a timeout as an asyncio task the orchestrator can cancel (kill). The
actual tool-call loop lives in ``subagent/runner.py``; the orchestrator's own
scheduling (interruptible waits, retry-via-prompt, checkpoints) lives in ``loop.py``.
"""
from __future__ import annotations

import asyncio
import itertools
import time
from typing import Literal, Protocol

SubagentType = Literal["executor", "researcher"]
SessionStatus = Literal["running", "done", "failed", "killed", "timeout"]


class SubagentRunner(Protocol):
    """What a session drives — satisfied by ``subagent.runner.Subagent``."""

    async def run(self, initial_prompt: str) -> str: ...

    async def follow_up(self, question: str) -> str: ...


class SubagentSession:
    """One spawned subagent: a runner plus lifecycle bookkeeping.

    ``run(timeout_s)`` drives the runner to a summary under a wall-clock timeout;
    the resulting ``status`` / ``summary`` are what the orchestrator reports and
    checkpoints. The session is kept after it finishes so ``follow_up`` can reuse the
    runner's full context instead of respawning.
    """

    def __init__(
        self,
        session_id: str,
        type_: SubagentType,
        model_name: str,
        prompt: str,
        runner: SubagentRunner,
    ) -> None:
        self.id = session_id
        self.type = type_
        self.model_name = model_name
        self.prompt = prompt
        self._runner = runner
        self.status: SessionStatus = "running"
        self.summary: str | None = None
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.task: asyncio.Task | None = None

    async def run(self, timeout_s: float) -> str:
        """Run to a final summary under ``timeout_s``; set status/summary; return it.

        A timeout or any runner exception is caught and turned into a failure summary
        so the orchestrator can feed it back and (per its prompt) retry once or report.
        A cancellation (operator/orchestrator kill) sets ``killed`` and re-raises so the
        task ends cancelled — the killer already knows the outcome.
        """
        self.started_at = time.monotonic()
        self.status = "running"
        try:
            self.summary = await asyncio.wait_for(self._runner.run(self.prompt), timeout_s)
            self.status = "done"
        except asyncio.TimeoutError:
            self.status = "timeout"
            self.summary = (
                f"timed out after {timeout_s:.0f}s without returning a summary"
            )
        except asyncio.CancelledError:
            self.status = "killed"
            self.summary = "killed before it returned a summary"
            self.finished_at = time.monotonic()
            raise
        except Exception as e:  # a broken subagent must not crash the orchestrator
            self.status = "failed"
            self.summary = f"failed with {type(e).__name__}: {e}"
        self.finished_at = time.monotonic()
        return self.summary or ""

    async def follow_up(self, question: str) -> str:
        """Ask a finished session a question with its full context intact."""
        return await self._runner.follow_up(question)

    @property
    def elapsed_s(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at if self.finished_at is not None else time.monotonic()
        return end - self.started_at


class SessionManager:
    """Registry of every session the orchestrator has spawned. Finished sessions are
    kept for follow-ups; ids are ``exec-N`` / ``res-N`` per type."""

    def __init__(self) -> None:
        self._sessions: dict[str, SubagentSession] = {}
        self._counters: dict[SubagentType, itertools.count] = {
            "executor": itertools.count(1),
            "researcher": itertools.count(1),
        }

    def _next_id(self, type_: SubagentType) -> str:
        prefix = "exec" if type_ == "executor" else "res"
        return f"{prefix}-{next(self._counters[type_])}"

    def create(
        self,
        type_: SubagentType,
        model_name: str,
        prompt: str,
        runner: SubagentRunner,
    ) -> SubagentSession:
        session = SubagentSession(self._next_id(type_), type_, model_name, prompt, runner)
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> SubagentSession | None:
        return self._sessions.get(session_id)

    def all(self) -> list[SubagentSession]:
        return list(self._sessions.values())

    def active(self) -> list[SubagentSession]:
        return [s for s in self._sessions.values() if s.status == "running"]
