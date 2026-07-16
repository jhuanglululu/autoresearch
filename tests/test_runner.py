"""Tests for the one-shot subagent runner (subagent/runner.py + subagent/tools.py).

Everything below the LLM boundary is real (path guards, wiki store in a tmp dir);
the model itself is a scripted FakeLLMClient that returns queued LLMResponses and
records exactly what it was sent. Network tools (web_search / fetch_page) are only
exercised behind mocks — no test hits the real network.
"""
from __future__ import annotations

import asyncio

import pytest

from autoresearch.llm.base import LLMResponse, Message, ToolCall, ToolSpec
from autoresearch.subagent import tools as T
from autoresearch.subagent.runner import Subagent
from autoresearch.wiki.store import WikiStore


# ----- scripted fake LLM -----

class FakeLLMClient:
    """Returns queued responses in order; records every complete() call.

    Each queued item is either an ``LLMResponse`` or a callable taking the current
    message list and returning one (so a test can assert on the context at that
    turn). ``sent`` captures (system, messages copy, tool names) per call.
    """

    def __init__(self, script):
        self._script = list(script)
        self.sent: list[tuple[str, list[Message], list[str]]] = []

    async def complete(self, system, messages, tools):
        self.sent.append((system, list(messages), [t.name for t in tools]))
        if not self._script:
            raise AssertionError("FakeLLMClient ran out of scripted responses")
        item = self._script.pop(0)
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


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def wiki(tmp_path):
    return WikiStore(tmp_path / "wiki-library")


def _executor(tmp_path, wiki, **kw):
    lab = tmp_path / "lab"
    lab.mkdir(exist_ok=True)
    archive = tmp_path / "archive"
    archive.mkdir(exist_ok=True)
    return Subagent(
        kw.pop("llm"),
        "executor",
        "SYS",
        lab_dir=lab,
        archive_dir=archive,
        wiki_store=wiki,
        **kw,
    )


# ----- (a) write path guards -----

def test_write_happy_path(tmp_path, wiki):
    llm = FakeLLMClient([_tool("write", path="main.py", content="print(1)\n"), _text("done")])
    agent = _executor(tmp_path, wiki, llm=llm)
    summary = _run(agent.run("go"))
    assert summary == "done"
    assert (tmp_path / "lab" / "main.py").read_text() == "print(1)\n"


@pytest.mark.parametrize(
    "path",
    ["/etc/passwd", "../escape.py", "runs/3/record.md", "runs"],
)
def test_write_guards_reject_and_loop_survives(tmp_path, wiki, path):
    # One write attempt (rejected), then the model gives up with a summary.
    llm = FakeLLMClient([_tool("write", path=path, content="x"), _text("blocked, summarizing")])
    agent = _executor(tmp_path, wiki, llm=llm)
    summary = _run(agent.run("go"))
    assert summary == "blocked, summarizing"  # loop survived the rejection
    # The rejection was fed back as a tool result.
    tool_results = [m for m in agent.messages if m.role == "tool"]
    assert len(tool_results) == 1
    # Rejected with a clear reason, whatever the specific guard that fired.
    assert any(
        w in tool_results[0].content for w in ("refused", "absolute", "'..'")
    ), tool_results[0].content
    # Nothing escaped: no file was created outside/inside forbidden areas.
    assert not (tmp_path / "escape.py").exists()
    assert not (tmp_path / "lab" / "runs" / "3" / "record.md").exists()


def test_write_symlink_escape_rejected(tmp_path, wiki):
    lab = tmp_path / "lab"
    lab.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (lab / "link").symlink_to(outside)  # lab/link -> tmp_path/outside
    llm = FakeLLMClient([_tool("write", path="link/pwned.py", content="x"), _text("done")])
    agent = Subagent(llm, "executor", "SYS", lab_dir=lab, wiki_store=wiki)
    _run(agent.run("go"))
    assert not (outside / "pwned.py").exists()
    tool_results = [m for m in agent.messages if m.role == "tool"]
    assert "refused" in tool_results[0].content


def test_write_onto_pinned_asset_rejected(tmp_path, wiki):
    lab = tmp_path / "lab"
    lab.mkdir()
    asset = lab / "data" / "corpus.bin"
    asset.parent.mkdir()
    asset.write_text("original")
    llm = FakeLLMClient([_tool("write", path="data/corpus.bin", content="tampered"), _text("d")])
    agent = Subagent(llm, "executor", "SYS", lab_dir=lab, wiki_store=wiki, pinned_assets=[asset])
    _run(agent.run("go"))
    assert asset.read_text() == "original"  # untouched


# ----- (b) toolsets differ by type -----

def test_toolsets_by_type(tmp_path, wiki):
    ex = _executor(tmp_path, wiki, llm=FakeLLMClient([]))
    rs = Subagent(FakeLLMClient([]), "researcher", "SYS", archive_dir=tmp_path, wiki_store=wiki)
    ex_names = {t.name for t in ex.tools}
    rs_names = {t.name for t in rs.tools}

    # Executor has the lab-write + run tools; researcher has neither.
    assert "write" in ex_names and "run_experiment" in ex_names
    assert "write" not in rs_names and "run_experiment" not in rs_names

    # Both share wiki read+write, archive read, analysis, and the research tools.
    for shared in ("wiki_read", "wiki_write_summary", "read_file", "list_dir",
                   "analyze_records", "web_search", "fetch_page"):
        assert shared in ex_names, shared
        assert shared in rs_names, shared

    # ToolSpec integrity: every exposed tool carries a non-empty description.
    for spec in [*ex.tools, *rs.tools]:
        assert isinstance(spec, ToolSpec) and spec.description.strip()


def test_researcher_cannot_dispatch_write(tmp_path, wiki):
    # Even if the model somehow emits a write call, a researcher refuses it.
    llm = FakeLLMClient([_tool("write", path="x.py", content="x"), _text("done")])
    agent = Subagent(llm, "researcher", "SYS", archive_dir=tmp_path, wiki_store=wiki)
    _run(agent.run("go"))
    tool_results = [m for m in agent.messages if m.role == "tool"]
    assert "not available to a researcher" in tool_results[0].content


# ----- (c) a scripted 3-round session -----

def test_three_round_session(tmp_path, wiki):
    # round 1: capture a source; round 2: read it back; round 3: final text.
    llm = FakeLLMClient([
        _tool("wiki_capture_source", call_id="a", id="src-1", title="T", content="body text"),
        _tool("wiki_read_source", call_id="b", id="src-1"),
        _text("summary: captured and verified src-1"),
    ])
    agent = _executor(tmp_path, wiki, llm=llm)
    summary = _run(agent.run("capture and verify"))
    assert summary == "summary: captured and verified src-1"
    assert len(llm.sent) == 3
    # The second tool result fed back into round 3 contains the source body.
    tool_results = [m for m in agent.messages if m.role == "tool"]
    assert len(tool_results) == 2
    assert "body text" in tool_results[1].content


# ----- (d) follow_up reuses context -----

def test_follow_up_reuses_context(tmp_path, wiki):
    llm = FakeLLMClient([_text("first summary"), _text("second summary")])
    agent = _executor(tmp_path, wiki, llm=llm)
    first = _run(agent.run("initial"))
    assert first == "first summary"
    count_after_first = len(agent.messages)

    second = _run(agent.follow_up("and now?"))
    assert second == "second summary"
    # Context grew and the prior messages are intact (not reset).
    assert len(agent.messages) > count_after_first
    # The second complete() call was sent the full prior history + the new question.
    _, messages_sent_second, _ = llm.sent[1]
    assert messages_sent_second[0].content == "initial"
    assert any(m.content == "first summary" for m in messages_sent_second)
    assert messages_sent_second[-1].content == "and now?"


# ----- (e) run_experiment without an injected callable -----

def test_run_experiment_schema_takes_no_arguments():
    # The tool contract is argument-free: the run is defined by the lab snapshot,
    # never by tool-call args.
    schema = T.RUN_EXPERIMENT_TOOL.input_schema
    assert schema.get("properties") == {}
    assert "required" not in schema


def test_run_experiment_without_callable(tmp_path, wiki):
    llm = FakeLLMClient([_tool("run_experiment"), _text("blocked")])
    agent = _executor(tmp_path, wiki, llm=llm)  # no run_experiment_callable
    _run(agent.run("run it"))
    tool_results = [m for m in agent.messages if m.role == "tool"]
    assert "unavailable" in tool_results[0].content


def test_run_experiment_with_injected_callable(tmp_path, wiki):
    calls = []

    def fake_runner():
        calls.append(True)
        return "run finished: val_loss=2.1"

    llm = FakeLLMClient([_tool("run_experiment"), _text("ok")])
    agent = _executor(tmp_path, wiki, llm=llm, run_experiment_callable=fake_runner)
    _run(agent.run("run it"))
    assert calls == [True]  # the callable was invoked exactly once, with no args
    tool_results = [m for m in agent.messages if m.role == "tool"]
    assert tool_results[0].content == "run finished: val_loss=2.1"


# ----- (f) max_rounds forced-summary path -----

def test_max_rounds_forces_summary(tmp_path, wiki):
    # The model never stops calling tools; after max_rounds a forced no-tools call
    # is made and the summary carries the truncation notice.
    script = [_tool("wiki_tags", call_id=f"t{i}") for i in range(3)]
    script.append(_text("forced final summary"))  # answer to the forced no-tools call
    llm = FakeLLMClient(script)
    agent = _executor(tmp_path, wiki, llm=llm, max_rounds=3)
    summary = _run(agent.run("loop forever"))
    assert "reached max_rounds=3" in summary
    assert "forced final summary" in summary
    # The forced call was made with NO tools available.
    assert llm.sent[-1][2] == []  # tool names empty on the last complete() call


# ----- tool execution never crashes the loop -----

def test_tool_exception_becomes_error_string(tmp_path, wiki, monkeypatch):
    def boom(tool_call, store):
        raise RuntimeError("kaboom")

    monkeypatch.setitem(
        __import__("autoresearch.wiki", fromlist=["WIKI_EXECUTORS"]).WIKI_EXECUTORS,
        "wiki_tags",
        boom,
    )
    llm = FakeLLMClient([_tool("wiki_tags"), _text("survived")])
    agent = _executor(tmp_path, wiki, llm=llm)
    summary = _run(agent.run("go"))
    assert summary == "survived"
    tool_results = [m for m in agent.messages if m.role == "tool"]
    assert "RuntimeError" in tool_results[0].content and "kaboom" in tool_results[0].content


# ----- web_search / fetch_page behind mocks (never real network) -----

def test_web_search_mocked(tmp_path, wiki, monkeypatch):
    class FakeDDGS:
        def text(self, query, max_results):
            return [{"title": "A", "href": "http://a", "body": "snippet"}]

    monkeypatch.setattr(T, "DDGS", FakeDDGS)
    llm = FakeLLMClient([_tool("web_search", query="rope scaling"), _text("done")])
    agent = _executor(tmp_path, wiki, llm=llm)
    _run(agent.run("search"))
    tool_results = [m for m in agent.messages if m.role == "tool"]
    assert "http://a" in tool_results[0].content and "snippet" in tool_results[0].content


def test_fetch_page_mocked(tmp_path, wiki, monkeypatch):
    class FakeResponse:
        headers = {"content-type": "text/html"}
        text = "<html><head><title>x</title></head><body><p>Hello world</p>"

    async def fake_get(url):
        return FakeResponse()

    monkeypatch.setattr(T, "_http_get", fake_get)
    llm = FakeLLMClient([_tool("fetch_page", url="http://x"), _text("done")])
    agent = _executor(tmp_path, wiki, llm=llm)
    _run(agent.run("fetch"))
    tool_results = [m for m in agent.messages if m.role == "tool"]
    assert "Hello world" in tool_results[0].content


# ----- read_file / list_dir / analyze_records over real dirs -----

def test_read_file_and_list_dir(tmp_path, wiki):
    lab = tmp_path / "lab"
    lab.mkdir()
    (lab / "main.py").write_text("print('hi')\n")
    (lab / "runs").mkdir()
    llm = FakeLLMClient([
        _tool("list_dir", call_id="l", path="."),
        _tool("read_file", call_id="r", path="main.py"),
        _tool("read_file", call_id="x", path="../secret"),
        _text("done"),
    ])
    agent = Subagent(llm, "executor", "SYS", lab_dir=lab, wiki_store=wiki)
    _run(agent.run("explore"))
    results = [m.content for m in agent.messages if m.role == "tool"]
    assert "main.py" in results[0] and "runs/" in results[0]
    assert results[1] == "print('hi')\n"
    assert "'..'" in results[2]  # traversal blocked
