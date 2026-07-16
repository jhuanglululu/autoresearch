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
- Runs until the user says stop. Checkpoints after each subagent finishes and after
  each orchestrator turn (checkpoint.py); restart resumes from the last checkpoint.

Loop semantics
--------------
One persistent driver races three things with ``asyncio.wait`` (FIRST_COMPLETED):
the operator's next message, the tasks of any running subagents, and a digest timer.

- A subagent is spawned FIRE-AND-FORGET: ``spawn_subagent`` starts an asyncio task and
  immediately returns "running" as the tool result, so the spawning turn ends with a
  note to the operator. The subagent's summary arrives later as its OWN turn (opened
  with the outcome), which keeps every turn self-contained — no dangling tool calls —
  and every turn checkpoint-safe. On a timeout/exception the failure opens that turn
  instead, and the PROMPT (not this code) decides to retry once with an amended prompt
  or report.
- A user message mid-run does not wait for the subagent: the driver wakes, runs a quick
  turn (which may ``kill_subagent`` or just answer), then resumes racing the still-running
  subagent — exactly ``asyncio.wait([subagent_task, user_message])``.
- Stop (`request_stop()` or the literal message "stop") lets the in-flight subagent
  finish, drains it, then exits.

Everything below the LLM boundary is injectable for tests: the orchestrator's own
``llm_client``, a ``subagent_factory`` (default builds a real subagent + creates the lab
+ wires the zero-arg run_experiment callable), the labs/state roots and the job queue.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..config import GoalConfig, ModelsConfig
from ..llm.base import LLMClient, Message, SpendCapExceeded, ToolCall, ToolSpec
from ..llm.openai import make_client as _default_make_client
from ..queue.jobs import JobQueue
from ..queue.labs import LABS_ROOT, create_lab
from ..queue.worker import make_run_experiment
from ..subagent.runner import Subagent
from ..wiki import WIKI_READ_TOOLS, WikiStore, execute_wiki_tool
from .channel import Channel
from .checkpoint import STATE_DIR, GoalState
from .spawn import SessionManager, SubagentRunner, SubagentSession, SubagentType

log = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).resolve().parents[3] / "prompt"

# Sane default per-spawn wall-clock timeout (overridable per spawn). Subagents plan +
# read + (executor) launch runs; a run itself is bounded separately by the GPU worker,
# so 2h bounds a wedged subagent loop without cutting off legitimate work.
DEFAULT_SPAWN_TIMEOUT_S = 2 * 60 * 60

# Cap on LLM rounds within a single orchestrator turn (defensive; a turn normally ends
# in a few rounds with plain text).
DEFAULT_MAX_ROUNDS = 40

# A fire-and-forget observability callback for one subagent's activity (see
# subagent.runner.EventCallback), plus the factory that mints one per session — given
# (session_id, type, model, prompt, lab_id) it returns a callback or None (feature
# disabled). lab_id lets the feed tail an executor's in-flight run log.
EventCallback = Callable[[dict], Awaitable[None]]
SessionObserverFactory = Callable[
    [str, SubagentType, str, str, "str | None"], "EventCallback | None"
]

# A factory that builds the runner a session drives, given (type, model, prompt, lab_id,
# on_event). The trailing on_event is the observability hook (None = no feed).
SubagentFactory = Callable[
    [SubagentType, str, str, "str | None", "EventCallback | None"], SubagentRunner
]
MakeClient = Callable[[Any], LLMClient]


@lru_cache(maxsize=None)
def _tool_description(name: str) -> str:
    """Load an orchestrator tool's description verbatim from prompt/tools/<name>.md."""
    return (_PROMPT_DIR / "tools" / f"{name}.md").read_text(encoding="utf-8").strip()


def _spec(name: str, properties: dict[str, Any], required: list[str]) -> ToolSpec:
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return ToolSpec(name=name, description=_tool_description(name), input_schema=schema)


# ----- the orchestrator's own toolset -----

SPAWN_SUBAGENT_TOOL = _spec(
    "spawn_subagent",
    {
        "type": {
            "type": "string",
            "enum": ["executor", "researcher"],
            "description": "executor = writes+runs experiment code, submits GPU jobs, "
            "writes wiki. researcher = searches/reads/writes wiki notes, no code.",
        },
        "model": {
            "type": "string",
            "description": "Name of a configured subagent model (see the model list in "
            "your system prompt). Pick per the task.",
        },
        "prompt": {
            "type": "string",
            "description": "The full composed instruction: role, task, detailed rules, "
            "and what to put in the wiki/lab vs. return as a summary.",
        },
        "lab_id": {
            "type": "string",
            "description": "REQUIRED for executor spawns: the lab to work in. Created "
            "from the goal baseline if new; reused if it exists. Ignored for researcher.",
        },
        "timeout_s": {
            "type": "number",
            "description": "Optional per-spawn wall-clock timeout in seconds.",
        },
    },
    ["type", "model", "prompt"],
)

FOLLOW_UP_SUBAGENT_TOOL = _spec(
    "follow_up_subagent",
    {
        "session_id": {
            "type": "string",
            "description": "A finished session id (e.g. exec-1, res-2).",
        },
        "question": {"type": "string", "description": "The follow-up question."},
    },
    ["session_id", "question"],
)

KILL_SUBAGENT_TOOL = _spec(
    "kill_subagent",
    {"session_id": {"type": "string", "description": "The running session to kill."}},
    ["session_id"],
)

LIST_SESSIONS_TOOL = _spec("list_sessions", {}, [])
GET_STATUS_TOOL = _spec("get_status", {}, [])


class Orchestrator:
    def __init__(
        self,
        models: ModelsConfig,
        goal: GoalConfig,
        *,
        channel: Channel,
        wiki_store: WikiStore,
        labs_root: Path | str = LABS_ROOT,
        state_root: Path | str = STATE_DIR,
        queue: JobQueue | None = None,
        llm_client: LLMClient | None = None,
        subagent_factory: SubagentFactory | None = None,
        make_client: MakeClient | None = None,
        session_observer_factory: SessionObserverFactory | None = None,
        digest_interval_s: float = 3600,
        default_timeout_s: float = DEFAULT_SPAWN_TIMEOUT_S,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
    ) -> None:
        self.models = models
        self.goal = goal
        self.channel = channel
        self.wiki_store = wiki_store
        self._labs_root = Path(labs_root)
        self._state_root = Path(state_root)
        self._queue = queue if queue is not None else JobQueue()
        self._make_client = make_client or _default_make_client
        self._own_client = llm_client or self._make_client(self.models.orchestrator)
        self._subagent_factory = subagent_factory or self._build_subagent
        self._session_observer_factory = session_observer_factory
        self._digest_interval_s = digest_interval_s
        self._default_timeout_s = default_timeout_s
        self._max_rounds = max_rounds

        self._subagent_clients: dict[str, LLMClient] = {}
        # on_event callbacks for the read-only forum feed, kept per live session so the
        # loop can emit a closing "end" event when the session finalizes.
        self._session_observers: dict[str, EventCallback] = {}
        self._sessions = SessionManager()
        # Tasks of subagents whose outcome still needs an evaluation turn.
        self._watching: dict[asyncio.Task, SubagentSession] = {}
        self._stop = False
        self._stop_event = asyncio.Event()

        self._state = GoalState.load_or_create(goal.id, self._state_root)
        self._messages: list[Message] = self._state.messages()
        self._last_digest_monotonic = time.monotonic()

        self._system_prompt = self._build_system_prompt()
        self._wiki_read_names = {t.name for t in WIKI_READ_TOOLS}
        self._tool_specs: list[ToolSpec] = [
            SPAWN_SUBAGENT_TOOL,
            FOLLOW_UP_SUBAGENT_TOOL,
            KILL_SUBAGENT_TOOL,
            LIST_SESSIONS_TOOL,
            GET_STATUS_TOOL,
            *WIKI_READ_TOOLS,
        ]

    # ----- public control -----

    def request_stop(self) -> None:
        """Ask the loop to exit after the in-flight subagent finishes."""
        self._stop = True
        self._stop_event.set()

    async def aside(self, question: str) -> str:
        """Answer a side-channel question (Discord ``/btw``) from the CURRENT context
        without touching it.

        Builds a one-off message list — a snapshot copy of ``self._messages`` plus a
        single framing user message — and calls the orchestrator's own client with
        ``tools=[]`` (no tools available for an aside). Nothing is mutated: the history
        is neither appended to nor checkpointed, and no session/digest state is touched,
        so a ``/btw`` may run concurrently with an in-flight turn — asides never mutate
        the list, and the snapshot copy taken here is a consistent view at call time.

        A spent budget is reported as text (the same wording as a turn), never raised —
        an aside must never stop the loop.
        """
        framed = Message(
            role="user",
            content=(
                "Side question from the operator — answer directly from what you "
                "already know; you have no tools for this: " + question
            ),
        )
        one_off = [*self._messages, framed]  # snapshot copy; self._messages untouched
        try:
            response = await self._own_client.complete(self._system_prompt, one_off, [])
        except SpendCapExceeded as e:
            return (
                f"orchestrator spend cap reached: ${e.spent:.2f} of ${e.cap:.2f} "
                "— raise cap in models.toml and restart"
            )
        return response.message.content

    async def run(self) -> None:
        """Drive the orchestrator until stopped."""
        recv_task: asyncio.Task = asyncio.ensure_future(self.channel.recv())
        stop_task: asyncio.Task = asyncio.ensure_future(self._stop_event.wait())
        try:
            if not self._messages:  # fresh goal: kick off planning
                await self._run_turn(Message(role="user", content=self._kickoff_text()))

            while not self._stop:
                wait_set: set[asyncio.Task] = {recv_task, stop_task, *self._watching}
                done, _pending = await asyncio.wait(
                    wait_set,
                    timeout=self._digest_timeout(),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:  # digest timer elapsed
                    await self._maybe_digest()
                    self._last_digest_monotonic = time.monotonic()
                    continue

                # Subagent completions (killed ones were already removed from _watching).
                finished = [self._watching.pop(t) for t in done if t in self._watching]

                user_msg: str | None = None
                if recv_task in done:
                    user_msg = recv_task.result()
                    recv_task = asyncio.ensure_future(self.channel.recv())
                if stop_task in done:
                    self._stop = True

                for session in finished:
                    if self._stop:  # shutting down: record it, no new evaluation turn
                        self._finalize_session(session)
                    else:
                        await self._handle_completion(session)
                if user_msg is not None and not self._stop:
                    await self._handle_user_message(user_msg)

            await self._drain()
        finally:
            for t in (recv_task, stop_task):
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await t

    # ----- event handlers -----

    async def _handle_user_message(self, msg: str) -> None:
        if msg.strip().lower() == "stop":
            self.request_stop()
            await self.channel.send("Stopping after the in-flight subagent finishes.")
            return
        await self._run_turn(Message(role="user", content=msg))

    async def _handle_completion(self, session: SubagentSession) -> None:
        self._finalize_session(session)
        await self._run_turn(Message(role="user", content=self._completion_notice(session)))
        await self._maybe_digest()

    async def _drain(self) -> None:
        """Let in-flight subagents finish, record them, send a final digest — no more
        evaluation turns, we are shutting down."""
        for task, session in list(self._watching.items()):
            self._watching.pop(task, None)
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            self._finalize_session(session)
        await self._maybe_digest()
        self._checkpoint()

    # ----- the turn driver -----

    async def _run_turn(self, opening: Message) -> None:
        """Append ``opening`` and run LLM rounds until the model replies with plain
        text (sent to the operator) or the per-turn round cap is hit."""
        self._messages.append(opening)
        for _ in range(self._max_rounds):
            try:
                response = await self._own_client.complete(
                    self._system_prompt, self._messages, self._tool_specs
                )
            except SpendCapExceeded as e:
                # The orchestrator's own budget is spent — never crash: tell the
                # operator, checkpoint, and stop cleanly. Restart after raising the cap.
                await self.channel.send(
                    f"orchestrator spend cap reached: ${e.spent:.2f} of ${e.cap:.2f} "
                    "— stopping; raise cap in models.toml and restart"
                )
                self._checkpoint()
                self.request_stop()
                return
            self._messages.append(response.message)
            self._checkpoint()
            if not response.message.tool_calls:
                if response.message.content.strip():
                    await self.channel.send(response.message.content)
                return
            for tool_call in response.message.tool_calls:
                result = await self._execute_tool(tool_call)
                self._messages.append(
                    Message(role="tool", content=result, tool_call_id=tool_call.id)
                )
            self._checkpoint()
        await self.channel.send(
            "[orchestrator reached its per-turn round cap without a reply]"
        )

    # ----- tool dispatch (never crashes a turn) -----

    async def _execute_tool(self, tool_call: ToolCall) -> str:
        name = tool_call.name
        args = tool_call.arguments or {}
        try:
            if name == "spawn_subagent":
                return await self._tool_spawn(args)
            if name == "follow_up_subagent":
                return await self._tool_follow_up(args)
            if name == "kill_subagent":
                return await self._tool_kill(args)
            if name == "list_sessions":
                return self._tool_list_sessions()
            if name == "get_status":
                return self._tool_get_status()
            if name in self._wiki_read_names:
                if self.wiki_store is None:
                    return "Wiki is unavailable in this session."
                return execute_wiki_tool(tool_call, self.wiki_store)
            return f"Unknown tool: {name!r}."
        except Exception as e:  # any tool failure becomes a tool-result string
            return f"Tool {name} raised {type(e).__name__}: {e}"

    async def _tool_spawn(self, args: dict) -> str:
        type_ = args.get("type")
        model = args.get("model")
        prompt = args.get("prompt")
        lab_id = args.get("lab_id")
        if type_ not in ("executor", "researcher"):
            return "type must be 'executor' or 'researcher'."
        if not isinstance(prompt, str) or not prompt.strip():
            return "prompt is required and must be a non-empty string."
        if not isinstance(model, str) or not model.strip():
            return "model is required."
        try:
            self.models.subagent(model)  # validate against models.toml
        except KeyError:
            names = ", ".join(m.name for m in self.models.subagent_models)
            return f"unknown model {model!r}. Choose one of: {names}."
        if type_ == "executor" and (not isinstance(lab_id, str) or not lab_id.strip()):
            return "executor spawns require a lab_id."

        # Mint the (optional) observability hook BEFORE building the runner so the runner
        # streams into the feed from its first event. session_id must exist first.
        session_id = self._sessions.peek_next_id(type_)
        on_event = self._make_observer(session_id, type_, model, prompt, lab_id)
        try:
            runner = self._subagent_factory(type_, model, prompt, lab_id, on_event)
        except Exception as e:
            return f"could not spawn: {type(e).__name__}: {e}"

        session = self._sessions.create(type_, model, prompt, runner)
        if on_event is not None:
            self._session_observers[session.id] = on_event
        timeout_s = args.get("timeout_s")
        timeout_s = float(timeout_s) if timeout_s else self._default_timeout_s
        task = asyncio.ensure_future(session.run(timeout_s))
        session.task = task
        self._watching[task] = session
        return (
            f"spawned {session.id} ({type_} on model {model}); it is now running. "
            "Reply to the operator now and end your turn — I will open a new turn with "
            "its summary when it finishes."
        )

    async def _tool_follow_up(self, args: dict) -> str:
        session_id = args.get("session_id")
        question = args.get("question")
        if not isinstance(question, str) or not question.strip():
            return "question is required."
        session = self._sessions.get(session_id) if isinstance(session_id, str) else None
        if session is None:
            return f"no such session: {session_id!r}."
        if session.status == "running":
            return f"{session_id} is still running; wait for it or kill it first."
        if session.status == "killed":
            return f"{session_id} was killed; its context is gone — respawn instead."
        try:
            answer = await session.follow_up(question)
        except Exception as e:
            return f"follow-up on {session_id} failed: {type(e).__name__}: {e}"
        return f"{session_id} (follow-up): {answer}"

    async def _tool_kill(self, args: dict) -> str:
        return await self.kill_session(args.get("session_id"))

    # ----- public accessors (shared by the tool executors AND the Discord bot) -----

    async def kill_session(self, session_id: Any) -> str:
        """Kill a running session directly (Discord ``/kill-subagent`` and the
        ``kill_subagent`` tool both route here). Returns a short ack string."""
        session = self._sessions.get(session_id) if isinstance(session_id, str) else None
        if session is None:
            return f"no such session: {session_id!r}."
        if session.status != "running" or session.task is None:
            return f"{session_id} is not running (status={session.status})."
        task = session.task
        self._watching.pop(task, None)  # we own the outcome here — no evaluation turn
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
        self._finalize_session(session)
        return f"killed {session_id}."

    def sessions_text(self) -> str:
        """One line per session (id, type/model, status, elapsed, first summary line).
        Backs the ``list_sessions`` tool and the Discord ``/sessions`` command."""
        sessions = self._sessions.all()
        if not sessions:
            return "no sessions yet."
        lines = []
        for s in sessions:
            summary = (s.summary or "").splitlines()[0] if s.summary else ""
            lines.append(
                f"- {s.id} [{s.type}/{s.model_name}] {s.status} "
                f"({s.elapsed_s:.0f}s): {summary}"
            )
        return "\n".join(lines)

    def _tool_list_sessions(self) -> str:
        return self.sessions_text()

    def status_text(self) -> str:
        """Programmatic status snapshot (no LLM turn): running subagents, GPU queue
        depth, cumulative token/spend. Backs the ``get_status`` tool and ``/status``."""
        active = self._sessions.active()
        if active:
            running = ", ".join(
                f"{s.id} ({s.type}/{s.model_name}, {s.elapsed_s:.0f}s)" for s in active
            )
        else:
            running = "none"
        pending, running_jobs, done = self._queue_counts()
        lines = [
            f"running subagents: {running}",
            f"GPU queue: pending={pending}, running={running_jobs}, done={done}",
            "token usage (cumulative):",
            *self._usage_lines(),
        ]
        return "\n".join(lines)

    def _tool_get_status(self) -> str:
        return self.status_text()

    # ----- digests (programmatic; no LLM call) -----

    async def _maybe_digest(self) -> None:
        new = self._state.completed_sessions[self._state.last_digest_index :]
        if not new:
            return
        lines = [f"Digest — {len(new)} subagent(s) finished since the last update:"]
        for rec in new:
            summary = (rec.get("summary") or "").splitlines()[0] if rec.get("summary") else ""
            lines.append(
                f"- {rec['id']} [{rec['type']}/{rec['model']}] {rec['status']}: {summary}"
            )
        pending, running_jobs, done = self._queue_counts()
        lines.append(f"GPU queue: pending={pending}, running={running_jobs}, done={done}")
        lines.append("token usage (cumulative):")
        lines.extend(self._usage_lines())
        await self.channel.send("\n".join(lines))
        self._state.last_digest_index = len(self._state.completed_sessions)
        self._last_digest_monotonic = time.monotonic()
        self._checkpoint()

    def _digest_timeout(self) -> float | None:
        if not self._digest_interval_s or self._digest_interval_s <= 0:
            return None
        elapsed = time.monotonic() - self._last_digest_monotonic
        return max(0.0, self._digest_interval_s - elapsed)

    # ----- status helpers -----

    def _queue_counts(self) -> tuple[int, int, int]:
        def n(d: Path) -> int:
            return len(list(d.glob("*.json")))

        return n(self._queue.pending), n(self._queue.running), n(self._queue.done)

    def _usage_lines(self) -> list[str]:
        entries: list[tuple[Any, Any]] = [
            (self.models.orchestrator, getattr(self._own_client, "usage", None))
        ]
        for name, client in self._subagent_clients.items():
            entries.append((self.models.subagent(name), getattr(client, "usage", None)))
        lines = []
        for endpoint, usage in entries:
            name = endpoint.name
            if usage is None:
                lines.append(f"  {name}: (usage unavailable)")
            elif endpoint.price_in is not None and endpoint.price_out is not None:
                # Prices configured: report dollars as "$spent/$cap" (or just spend
                # when no cap is set) instead of raw token counts.
                spent = usage.cost_usd(endpoint.price_in, endpoint.price_out)
                cap = f"/${endpoint.cap:.0f}" if endpoint.cap is not None else ""
                lines.append(f"  {name}: {usage.calls} calls, ${spent:.2f}{cap}")
            else:
                lines.append(
                    f"  {name}: {usage.calls} calls, "
                    f"{usage.input_tokens} in / {usage.output_tokens} out tokens"
                )
        return lines

    # ----- default subagent construction (the injectable seam's default) -----

    def _client_for(self, model_name: str) -> LLMClient:
        client = self._subagent_clients.get(model_name)
        if client is None:
            client = self._make_client(self.models.subagent(model_name))
            self._subagent_clients[model_name] = client
        return client

    def _make_observer(
        self,
        session_id: str,
        type_: SubagentType,
        model: str,
        prompt: str,
        lab_id: str | None,
    ) -> "EventCallback | None":
        """Mint the (optional) observability hook for a new session. A missing factory or
        one that raises leaves the session unobserved — the feed is best-effort only."""
        if self._session_observer_factory is None:
            return None
        try:
            return self._session_observer_factory(session_id, type_, model, prompt, lab_id)
        except Exception:  # the feed must never affect spawning
            log.warning("session_observer_factory failed for %s", session_id, exc_info=True)
            return None

    def _build_subagent(
        self,
        type_: SubagentType,
        model: str,
        prompt: str,
        lab_id: str | None,
        on_event: "EventCallback | None" = None,
    ) -> SubagentRunner:
        client = self._client_for(model)
        if type_ == "executor":
            lab_dir = self._labs_root / lab_id  # type: ignore[operator]
            if not lab_dir.exists():
                create_lab(self.goal, lab_id, labs_root=self._labs_root)
            run_callable = make_run_experiment(
                lab_id, self.goal, self.wiki_store, queue=self._queue
            )
            return Subagent(
                client,
                "executor",
                self._subagent_system_prompt("executor"),
                lab_dir=lab_dir,
                archive_dir=self._labs_root,
                wiki_store=self.wiki_store,
                pinned_assets=list(self.goal.experiment.assets.values()),
                run_experiment_callable=run_callable,
                on_event=on_event,
            )
        return Subagent(
            client,
            "researcher",
            self._subagent_system_prompt("researcher"),
            archive_dir=self._labs_root,
            wiki_store=self.wiki_store,
            on_event=on_event,
        )

    @staticmethod
    def _subagent_system_prompt(type_: SubagentType) -> str:
        role = (
            "You write and run experiment code, submit GPU jobs, and write the wiki."
            if type_ == "executor"
            else "You search, read, and write wiki notes. You cannot edit or run code."
        )
        return (
            f"You are a one-shot {type_} subagent in an autonomous research system. "
            f"{role} Put all detail in the wiki (and the lab, if you have one); the final "
            "message you return with no tool call is a SHORT summary for the orchestrator. "
            "Follow the assignment you were given exactly."
        )

    # ----- checkpoint + prompt assembly -----

    def _finalize_session(self, session: SubagentSession) -> None:
        self._state.record_session(
            id=session.id,
            type=session.type,
            model=session.model_name,
            prompt=session.prompt,
            summary=session.summary or "",
            status=session.status,
        )
        self._close_observer(session)
        self._checkpoint()

    def _close_observer(self, session: SubagentSession) -> None:
        """Emit a final ``end`` event to the session's forum feed (if any) so it flushes
        and closes its thread. Fire-and-forget: scheduled, never awaited, never raises —
        it covers every termination (done/failed/timeout/killed), including those where
        the runner emitted no summary (it was cancelled)."""
        cb = self._session_observers.pop(session.id, None)
        if cb is None:
            return
        event = {
            "kind": "end",
            "status": session.status,
            "summary": (session.summary or "")[:200],
        }
        try:
            asyncio.ensure_future(self._safe_emit(cb, event))
        except RuntimeError:  # no running loop (e.g. a synchronous test) — skip
            pass

    @staticmethod
    async def _safe_emit(cb: "EventCallback", event: dict) -> None:
        try:
            await cb(event)
        except Exception:
            log.warning("session observer end hook failed", exc_info=True)

    def _checkpoint(self) -> None:
        self._state.set_messages(self._messages)
        self._state.save(self._state_root)

    def _build_system_prompt(self) -> str:
        text = (_PROMPT_DIR / "orchestrator.md").read_text(encoding="utf-8")
        text = text.replace("{{GOAL_TEMPLATE}}", self.goal.template_text())
        text = text.replace("{{SUBAGENT_MODELS}}", self._render_models())
        return text

    def _render_models(self) -> str:
        return "\n".join(
            f"- **{m.name}** (`{m.model}`): {m.description}"
            for m in self.models.subagent_models
        )

    def _kickoff_text(self) -> str:
        return (
            "[system] You are online. Review the wiki for prior work, then plan and "
            "spawn your first subagent toward the goal. Tell the operator your plan."
        )

    def _completion_notice(self, session: SubagentSession) -> str:
        return (
            f"[system] Subagent {session.id} ({session.type}, model "
            f"{session.model_name}) finished with status={session.status} after "
            f"{session.elapsed_s:.0f}s.\nSummary:\n{session.summary}\n\n"
            "Evaluate it and decide the next step. If it failed or timed out, per your "
            "rules retry ONCE with an amended prompt, otherwise report to the operator."
        )
