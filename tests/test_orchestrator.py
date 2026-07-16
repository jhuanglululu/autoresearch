"""Tests for the orchestrator loop (orchestrator/loop.py + spawn.py + checkpoint.py).

Everything below the LLM boundary is real (checkpoint files, session manager, the
programmatic digest); the orchestrator's own model is a scripted FakeLLM and subagents
are faked at the factory seam (a FakeRunner with async run/follow_up) so no network, no
real lab, and no GPU are touched. Two scenarios that need the wiring of the DEFAULT
factory (executor lab creation + run_experiment callable) monkeypatch the module-level
create_lab / make_run_experiment / Subagent instead.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from autoresearch.config import ExperimentSpec, GoalConfig, ModelEndpoint, ModelsConfig
from autoresearch.llm.base import LLMResponse, Message, SpendCapExceeded, ToolCall, Usage
from autoresearch.orchestrator import loop as loop_mod
from autoresearch.orchestrator.checkpoint import GoalState
from autoresearch.orchestrator.loop import Orchestrator
from autoresearch.subagent.runner import Subagent
from autoresearch.wiki.store import WikiStore


# ----- scripted fake LLM (same shape as test_runner's) -----

class FakeLLM:
    def __init__(self, script=None):
        self._script = list(script or [])
        self.sent: list[tuple[str, list[Message], list[str]]] = []
        self.usage = Usage()

    def push(self, *items):
        self._script.extend(items)

    async def complete(self, system, messages, tools):
        self.sent.append((system, list(messages), [t.name for t in tools]))
        if not self._script:
            raise AssertionError("FakeLLM ran out of scripted responses")
        item = self._script.pop(0)
        self.usage.record(1, 1)
        return item(messages) if callable(item) else item


def _text(content: str) -> LLMResponse:
    return LLMResponse(
        message=Message(role="assistant", content=content),
        input_tokens=1,
        output_tokens=1,
        stop_reason="end_turn",
    )


def _tool(name: str, call_id: str = "c1", **args) -> LLMResponse:
    return LLMResponse(
        message=Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id=call_id, name=name, arguments=args)],
        ),
        input_tokens=1,
        output_tokens=1,
        stop_reason="tool_use",
    )


# ----- fake channel + subagent -----

class FakeChannel:
    def __init__(self):
        self.inbound: asyncio.Queue = asyncio.Queue()
        self.sent: list[str] = []

    async def recv(self) -> str:
        return await self.inbound.get()

    async def send(self, text: str) -> None:
        self.sent.append(text)

    def feed(self, text: str) -> None:
        self.inbound.put_nowait(text)


class FakeRunner:
    """Stands in for subagent.runner.Subagent at the factory seam."""

    def __init__(self, summary="did the work", sleep=0.0, exc=None):
        self.summary = summary
        self.sleep = sleep
        self.exc = exc
        self.ran_with: list[str] = []
        self.followups: list[str] = []

    async def run(self, prompt):
        self.ran_with.append(prompt)
        if self.sleep:
            await asyncio.sleep(self.sleep)
        if self.exc:
            raise self.exc
        return self.summary

    async def follow_up(self, question):
        self.followups.append(question)
        return f"followup:{question}"


# ----- config helpers -----

def _models() -> ModelsConfig:
    orch = ModelEndpoint("orchestrator", "https://api.anthropic.com", "claude-x", "K")
    subs = (
        ModelEndpoint("opus", "https://api.anthropic.com", "claude-opus", "K", "coder"),
        ModelEndpoint("gpt", "https://api.openai.com/v1", "gpt-x", "K2", "second opinion"),
    )
    return ModelsConfig(orchestrator=orch, subagent_models=subs)


def _goal(tmp_path: Path) -> GoalConfig:
    template = tmp_path / "template.md"
    template.write_text("Optimize the zhtw LM.\n")
    return GoalConfig(
        id="t",
        template_path=template,
        experiment=ExperimentSpec(baseline=tmp_path / "baseline", assets={}),
    )


def _orch(tmp_path, script, *, factory=None, make_client=None, state_root=None,
          channel=None, digest_interval_s=1000.0, default_timeout_s=100.0):
    from autoresearch.queue.jobs import JobQueue

    return Orchestrator(
        _models(),
        _goal(tmp_path),
        channel=channel or FakeChannel(),
        wiki_store=WikiStore(tmp_path / "wiki"),
        labs_root=tmp_path / "lab",
        state_root=state_root or (tmp_path / "state"),
        queue=JobQueue(tmp_path / "queue"),
        llm_client=FakeLLM(script),
        subagent_factory=factory,
        make_client=make_client,
        digest_interval_s=digest_interval_s,
        default_timeout_s=default_timeout_s,
    )


async def _wait_until(cond, tries=300, delay=0.01):
    for _ in range(tries):
        if cond():
            return
        await asyncio.sleep(delay)
    raise AssertionError("condition was never met")


def _llm_sent(orch) -> list[str]:
    """Every message-content string the orchestrator's own model was shown."""
    out: list[str] = []
    for _system, messages, _tools in orch._own_client.sent:
        out.extend(m.content for m in messages)
    return out


# ===== (a) spawn -> summary -> checkpoint -> digest =====

def test_spawn_summary_checkpoint_digest(tmp_path):
    runner = FakeRunner(summary="engine ran: val_loss=2.1")
    script = [
        _tool("spawn_subagent", type="executor", model="opus", prompt="build X", lab_id="lab-a"),
        _text("Started the engineer on lab-a."),
        _text("Good — val_loss 2.1 recorded."),
    ]

    async def scenario():
        orch = _orch(tmp_path, script, factory=lambda *a, **k: runner)
        task = asyncio.ensure_future(orch.run())
        await _wait_until(lambda: any(m.lstrip("`text\n").startswith("Digest") for m in orch.channel.sent))
        orch.request_stop()
        await asyncio.wait_for(task, 2)
        return orch

    orch = asyncio.run(scenario())

    # The subagent ran with the composed prompt.
    assert runner.ran_with == ["build X"]

    # A digest was posted naming the finished session + queue depth + usage.
    digest = next(m for m in orch.channel.sent if m.lstrip("`text\n").startswith("Digest"))
    assert "exec-1" in digest and "engine ran: val_loss=2.1" in digest
    assert "GPU queue:" in digest and "token usage" in digest

    # Checkpoint on disk holds the orchestrator messages + the completed session record.
    ckpt = tmp_path / "state" / "t" / "checkpoint.json"
    data = json.loads(ckpt.read_text())
    assert data["orchestrator_messages"], "messages were not persisted"
    recs = data["completed_sessions"]
    assert len(recs) == 1
    assert recs[0]["id"] == "exec-1" and recs[0]["status"] == "done"
    assert recs[0]["model"] == "opus" and recs[0]["prompt"] == "build X"


# ===== (b) executor spawn creates the lab + wires the run_experiment callable =====

def test_executor_spawn_creates_lab_and_wires_callable(tmp_path, monkeypatch):
    created, wired, built = {}, {}, {}

    def fake_create_lab(goal, lab_id, *, labs_root):
        created["lab_id"] = lab_id
        (Path(labs_root) / lab_id).mkdir(parents=True, exist_ok=True)

    def fake_make_run_experiment(lab_id, goal, wiki_store, *, queue=None):
        wired["lab_id"] = lab_id
        wired["queue"] = queue
        return lambda: "ran"

    class FakeSub:
        def __init__(self, client, type_, system_prompt, **kw):
            built.update(kw)
            built["type"] = type_

        async def run(self, prompt):  # pragma: no cover - not driven here
            return "s"

        async def follow_up(self, q):  # pragma: no cover
            return "f"

    monkeypatch.setattr(loop_mod, "create_lab", fake_create_lab)
    monkeypatch.setattr(loop_mod, "make_run_experiment", fake_make_run_experiment)
    monkeypatch.setattr(loop_mod, "Subagent", FakeSub)

    # Default factory (subagent_factory=None) + a stub make_client so no real client.
    orch = _orch(tmp_path, [], make_client=lambda ep: FakeLLM([]))
    runner = orch._build_subagent("executor", "opus", "do it", "lab-x")

    assert isinstance(runner, FakeSub)
    assert created["lab_id"] == "lab-x"          # lab created from the baseline
    assert wired["lab_id"] == "lab-x"            # callable bound to the same lab
    assert wired["queue"] is orch._queue         # wired to the orchestrator's queue
    assert built["type"] == "executor"
    assert built["lab_dir"] == orch._labs_root / "lab-x"
    assert built["run_experiment_callable"] is not None  # zero-arg callable wired in


def test_executor_spawn_requires_lab_id(tmp_path):
    orch = _orch(tmp_path, [])
    result = asyncio.run(
        orch._tool_spawn({"type": "executor", "model": "opus", "prompt": "go"})
    )
    assert "lab_id" in result and orch._sessions.all() == []


# ===== (c) follow_up hits the same finished session =====

def test_follow_up_hits_same_session(tmp_path):
    runner = FakeRunner(summary="s1")

    async def scenario():
        orch = _orch(tmp_path, [])
        session = orch._sessions.create("researcher", "gpt", "prompt", runner)
        session.status = "done"
        session.summary = "s1"
        session.started_at, session.finished_at = 0.0, 1.0
        out = await orch._tool_follow_up({"session_id": session.id, "question": "why?"})
        return out

    out = asyncio.run(scenario())
    assert out == "res-1 (follow-up): followup:why?"
    assert runner.followups == ["why?"]  # the same runner's context was reused


# ===== (d) timeout -> failure fed back -> scripted retry spawn =====

def test_timeout_feeds_failure_and_prompt_retries(tmp_path):
    retry_runner = FakeRunner(summary="retry ok")
    script = [
        # eval turn opened with the timeout failure -> the (scripted) model retries once.
        _tool("spawn_subagent", type="executor", model="opus",
              prompt="amended: smaller scope", lab_id="lab-a"),
        _text("Retried the engineer with a tighter scope."),
    ]

    async def scenario():
        orch = _orch(tmp_path, script, factory=lambda *a, **k: retry_runner)
        # A session that already timed out (as spawn.SubagentSession.run would leave it).
        timed = orch._sessions.create("executor", "opus", "original", FakeRunner())
        timed.status = "timeout"
        timed.summary = "timed out after 100s without returning a summary"
        timed.started_at, timed.finished_at = 0.0, 100.0

        await orch._handle_completion(timed)

        # The failure was fed back verbatim as the turn's opening message.
        opening = next(m for m in _llm_sent(orch) if "status=timeout" in m)
        assert "timed out after 100s" in opening
        for t in list(orch._watching):  # tidy the retry task
            t.cancel()
        return orch

    orch = asyncio.run(scenario())
    # Exactly one retry was spawned (the original + one retry = 2 sessions).
    assert len(orch._sessions.all()) == 2
    assert orch._sessions.all()[1].prompt == "amended: smaller scope"
    # The timeout was surfaced in a digest.
    assert any("timeout" in m for m in orch.channel.sent)


# ===== (e) kill via a user message mid-run =====

def test_kill_subagent_via_user_message_midrun(tmp_path):
    slow = FakeRunner(summary="should never return", sleep=100.0)
    script = [
        _tool("spawn_subagent", type="executor", model="opus", prompt="long job", lab_id="lab-a"),
        _text("Started the long job."),
        _tool("kill_subagent", session_id="exec-1"),
        _text("Killed exec-1 as requested."),
    ]

    async def scenario():
        orch = _orch(tmp_path, script, factory=lambda *a, **k: slow)
        orch._messages = [Message(role="user", content="prior")]  # skip kickoff
        task = asyncio.ensure_future(orch.run())

        orch.channel.feed("start the long job")
        await _wait_until(
            lambda: orch._sessions.get("exec-1") is not None
            and orch._sessions.get("exec-1").status == "running"
        )
        orch.channel.feed("kill it")
        await _wait_until(lambda: orch._sessions.get("exec-1").status == "killed")

        orch.request_stop()
        await asyncio.wait_for(task, 2)
        return orch

    orch = asyncio.run(scenario())
    session = orch._sessions.get("exec-1")
    assert session.status == "killed"
    assert session.summary and "should never return" not in session.summary
    assert any("Killed exec-1" in m for m in orch.channel.sent)
    # The killed session is not re-evaluated: no completion turn was opened for it.
    assert not any("status=killed" in m for m in _llm_sent(orch))


# ===== (f) resume: a new orchestrator over the same state dir sees prior messages =====

def test_resume_reads_prior_messages(tmp_path):
    state_root = tmp_path / "state"
    runner = FakeRunner(summary="engine ran")
    script = [
        _tool("spawn_subagent", type="executor", model="opus", prompt="build X", lab_id="lab-a"),
        _text("Started."),
        _text("Result recorded."),
    ]

    async def first_run():
        orch = _orch(tmp_path, script, factory=lambda *a, **k: runner, state_root=state_root)
        task = asyncio.ensure_future(orch.run())
        await _wait_until(lambda: any(m.lstrip("`text\n").startswith("Digest") for m in orch.channel.sent))
        orch.request_stop()
        await asyncio.wait_for(task, 2)

    asyncio.run(first_run())

    # A fresh orchestrator over the same state dir rebuilds the prior conversation.
    resumed = _orch(tmp_path, [], state_root=state_root)
    assert resumed._messages, "resume did not rebuild prior messages"
    assert any("Result recorded." in m.content for m in resumed._messages)
    # Session summaries survive in history; the live sessions are NOT revived.
    assert resumed._sessions.all() == []
    assert resumed._state.completed_sessions[0]["id"] == "exec-1"


# ===== (g) stop request exits the loop =====

def test_request_stop_exits_loop(tmp_path):
    async def scenario():
        orch = _orch(tmp_path, [])
        orch._messages = [Message(role="user", content="prior")]  # skip kickoff
        task = asyncio.ensure_future(orch.run())
        await asyncio.sleep(0.05)  # let it reach the wait
        assert not task.done()
        orch.request_stop()
        await asyncio.wait_for(task, 2)
        return task

    task = asyncio.run(scenario())
    assert task.done() and task.exception() is None


def test_stop_message_exits_loop(tmp_path):
    async def scenario():
        orch = _orch(tmp_path, [])
        orch._messages = [Message(role="user", content="prior")]  # skip kickoff
        task = asyncio.ensure_future(orch.run())
        orch.channel.feed("stop")
        await asyncio.wait_for(task, 2)
        return orch

    orch = asyncio.run(scenario())
    assert any("Stopping" in m for m in orch.channel.sent)


# ===== (h) get_status + unknown model guard (toolset smoke) =====

def test_get_status_and_unknown_model(tmp_path):
    orch = _orch(tmp_path, [])
    status = orch._tool_get_status()
    assert "running subagents: none" in status
    assert "GPU queue:" in status and "token usage" in status

    bad = asyncio.run(
        orch._tool_spawn({"type": "researcher", "model": "nope", "prompt": "go"})
    )
    assert "unknown model" in bad and "opus" in bad and "gpt" in bad


# ===== (h2) revert_lab: orchestrator-only tool, absent from subagent toolsets =====

def test_revert_lab_in_orchestrator_toolset_only(tmp_path):
    orch = _orch(tmp_path, [])
    assert "revert_lab" in {t.name for t in orch._tool_specs}

    # Build the real subagent runners and assert neither carries a revert tool — the
    # keep/revert decision is the orchestrator's alone.
    executor = Subagent(FakeLLM([]), "executor", "sys", lab_dir=tmp_path / "lab")
    researcher = Subagent(FakeLLM([]), "researcher", "sys")
    assert "revert_lab" not in {t.name for t in executor.tools}
    assert "revert_lab" not in {t.name for t in researcher.tools}


def test_revert_lab_tool_calls_labs_revert_and_returns_summary(tmp_path, monkeypatch):
    calls = {}

    def fake_revert(lab_dir, run_number):
        calls["lab_dir"] = lab_dir
        calls["run_number"] = run_number
        return "lab lab-a restored to runs/2 snapshot; archive untouched"

    monkeypatch.setattr(loop_mod, "revert_lab", fake_revert)

    script = [
        _tool("revert_lab", lab_id="lab-a", run_number=2),
        _text("Reverted lab-a to run 2's snapshot."),
    ]

    async def scenario():
        orch = _orch(tmp_path, script)
        orch._messages = [Message(role="user", content="prior")]  # skip kickoff
        await orch._run_turn(Message(role="user", content="revert lab-a to run 2"))
        return orch

    orch = asyncio.run(scenario())

    # The tool routed to labs.revert_lab with the resolved lab dir + run number.
    assert calls["lab_dir"] == orch._labs_root / "lab-a"
    assert calls["run_number"] == 2
    # revert_lab's one-line summary was fed back to the model as the tool result.
    tool_results = [m.content for m in orch._messages if m.role == "tool"]
    assert any("restored to runs/2 snapshot; archive untouched" in c for c in tool_results)


# ===== (i) own-client spend cap -> channel message + stop, no crash =====

class CapLLM:
    """A fake orchestrator model that refuses with a spend-cap error."""

    def __init__(self):
        self.usage = Usage()
        self.sent: list = []

    async def complete(self, system, messages, tools):
        self.sent.append((system, list(messages), [t.name for t in tools]))
        raise SpendCapExceeded("orchestrator", 55.0, 50.0)


def test_own_client_spend_cap_stops_and_messages(tmp_path):
    async def scenario():
        orch = _orch(tmp_path, [])
        orch._own_client = CapLLM()
        orch._messages = [Message(role="user", content="prior")]  # skip kickoff
        await orch._run_turn(Message(role="user", content="go"))
        return orch

    orch = asyncio.run(scenario())
    # The operator was told, with the exact spend/cap figures — and no exception escaped.
    msg = next(m for m in orch.channel.sent if "spend cap reached" in m)
    assert "$55.00 of $50.00" in msg
    assert "raise cap in models.toml" in msg
    # The loop was asked to stop and the turn was checkpointed.
    assert orch._stop is True
    ckpt = tmp_path / "state" / "t" / "checkpoint.json"
    assert ckpt.exists()


def test_usage_lines_report_dollars_when_priced(tmp_path):
    # Prices on the orchestrator endpoint -> status reports "$spent/$cap", not tokens.
    orch = _orch(tmp_path, [])
    priced = ModelEndpoint(
        "orchestrator", "https://api.anthropic.com", "claude-x", "K",
        cap=50.0, price_in=10.0, price_out=50.0,
    )
    orch.models = ModelsConfig(orchestrator=priced, subagent_models=orch.models.subagent_models)
    orch._own_client.usage.record(1_000_000, 200_000)  # $10 + $10 = $20.00
    status = orch._tool_get_status()
    assert "$20.00/$50" in status


# ===== (j) session observer factory wiring (forum feed seam) =====

def test_session_observer_factory_is_wired_and_closed(tmp_path):
    runner = FakeRunner(summary="engine ran: val_loss=2.1")
    calls, events = [], []

    def obs_factory(session_id, type_, model, prompt, lab_id):
        calls.append((session_id, type_, model, prompt, lab_id))

        async def sink(event):
            events.append(event)

        return sink

    script = [
        _tool("spawn_subagent", type="executor", model="opus", prompt="build X", lab_id="lab-a"),
        _text("Started."),
        _text("Recorded."),
    ]

    async def scenario():
        orch = _orch(tmp_path, script, factory=lambda *a, **k: runner)
        orch._session_observer_factory = obs_factory
        task = asyncio.ensure_future(orch.run())
        await _wait_until(lambda: any(e.get("kind") == "end" for e in events))
        orch.request_stop()
        await asyncio.wait_for(task, 2)

    asyncio.run(scenario())

    # The factory was called once with (session_id, type, model, prompt, lab_id) — id
    # previewed BEFORE the session was created, so it matches the assigned id.
    assert calls == [("exec-1", "executor", "opus", "build X", "lab-a")]
    # A closing "end" event carried the final status + summary to the feed.
    end = next(e for e in events if e["kind"] == "end")
    assert end["status"] == "done" and "val_loss=2.1" in end["summary"]


def test_no_observer_factory_stores_nothing(tmp_path):
    # Default: no factory -> no per-session observer bookkeeping (pure no-op seam).
    orch = _orch(tmp_path, [])
    assert orch._session_observer_factory is None
    assert orch._session_observers == {}


# ===== (k) aside: /btw side-channel question uses context, mutates nothing =====

def test_aside_uses_context_with_no_tools(tmp_path):
    orch = _orch(tmp_path, [_text("the val_loss was 2.1")])
    orch._messages = [
        Message(role="user", content="earlier: run the trainer"),
        Message(role="assistant", content="ran it, val_loss=2.1"),
    ]

    answer = asyncio.run(orch.aside("what was the val_loss again?"))
    assert answer == "the val_loss was 2.1"

    # The model saw the prior history + the framed question, and NO tools.
    system, messages_sent, tool_names = orch._own_client.sent[-1]
    assert tool_names == []  # an aside has no tools
    contents = [m.content for m in messages_sent]
    assert "ran it, val_loss=2.1" in contents  # prior context arrived
    assert any("Side question from the operator" in c and "what was the val_loss again?" in c
               for c in contents)
    # The framed question is last and carries the "no tools" framing.
    assert "what was the val_loss again?" in messages_sent[-1].content


def test_aside_does_not_mutate_context_or_checkpoint(tmp_path):
    state_root = tmp_path / "state"
    orch = _orch(tmp_path, [_text("answer")], state_root=state_root)
    orch._messages = [Message(role="user", content="prior")]
    orch._checkpoint()  # establish a baseline checkpoint on disk

    ckpt = state_root / "t" / "checkpoint.json"
    before_len = len(orch._messages)
    before_bytes = ckpt.read_bytes()
    before_mtime = ckpt.stat().st_mtime_ns

    answer = asyncio.run(orch.aside("a side question"))
    assert answer == "answer"

    # History is untouched; nothing was checkpointed by the aside.
    assert len(orch._messages) == before_len
    assert orch._messages[-1].content == "prior"  # no framed message appended
    assert ckpt.read_bytes() == before_bytes
    assert ckpt.stat().st_mtime_ns == before_mtime


def test_aside_spend_cap_returns_message_not_raise(tmp_path):
    orch = _orch(tmp_path, [])
    orch._own_client = CapLLM()  # complete() raises SpendCapExceeded(55, 50)
    orch._messages = [Message(role="user", content="prior")]

    answer = asyncio.run(orch.aside("anything?"))
    assert "spend cap reached" in answer and "$55.00 of $50.00" in answer
    # The loop was NOT asked to stop (an aside must never stop it).
    assert orch._stop is False


# ===== checkpoint round-trip of tool-call messages =====

def test_checkpoint_message_roundtrip(tmp_path):
    state = GoalState.load_or_create("g", tmp_path / "state")
    state.set_messages([
        Message(role="user", content="hi"),
        Message(role="assistant", content="", tool_calls=[
            ToolCall(id="c1", name="spawn_subagent", arguments={"type": "researcher"})
        ]),
        Message(role="tool", content="spawned res-1", tool_call_id="c1"),
    ])
    state.save()

    reloaded = GoalState.load_or_create("g", tmp_path / "state")
    msgs = reloaded.messages()
    assert msgs[1].tool_calls[0].name == "spawn_subagent"
    assert msgs[1].tool_calls[0].arguments == {"type": "researcher"}
    assert msgs[2].tool_call_id == "c1"
