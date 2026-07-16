"""Tests for the read-only forum feed (bot/discord_bot.py: SessionFeed / SessionThread).

No real Discord and no real clocks: FakeForum/FakeThread/FakeMessage record what would
be posted/edited, and a callable clock is injected so elapsed times are deterministic.
The live run window is driven against a real (tmp-dir) JobQueue plus a hand-built run dir.
Everything is driven by explicit flush()/close() — the ~7s flusher is never relied on.
"""
from __future__ import annotations

import asyncio
import itertools

from autoresearch.bot.discord_bot import (
    DISCORD_LIMIT,
    RUN_TAIL_LINES,
    SessionFeed,
    SessionThread,
    _fit_window,
    _fmt_elapsed,
    _format_event,
    _tail_log,
)
from autoresearch.queue.jobs import Job, JobQueue


def _run(coro):
    return asyncio.run(coro)


class _Clock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


class _NotFound(Exception):
    """A stand-in for discord.NotFound (its type name is what _is_gone checks)."""


class FakeMessage:
    _ids = itertools.count(1)

    def __init__(self, content: str):
        self.id = next(FakeMessage._ids)
        self.content = content
        self.edits: list[str] = []
        self.edit_fail: Exception | None = None

    async def edit(self, *, content: str) -> None:
        if self.edit_fail is not None:
            raise self.edit_fail
        self.content = content
        self.edits.append(content)


class FakeThread:
    def __init__(self):
        self.messages: list[FakeMessage] = []

    async def send(self, content: str) -> FakeMessage:
        m = FakeMessage(content)
        self.messages.append(m)
        return m


class FakeForum:
    def __init__(self, fail: bool = False):
        self.created: list[tuple[str, str]] = []
        self.threads: list[FakeThread] = []
        self.fail = fail

    async def create_thread(self, *, name: str, content: str):
        if self.fail:
            raise RuntimeError("discord down")
        self.created.append((name, content))
        thread = FakeThread()
        self.threads.append(thread)

        class _Wrapper:
            pass

        w = _Wrapper()
        w.thread = thread
        return w


def _thread(forum, clock=None, **kw):
    kw.setdefault("interval_s", 999)  # never fire on its own; we drive flush() by hand
    return SessionThread(
        forum, "exec-1", "executor", "opus", "build the trainer",
        clock=clock or _Clock(), **kw,
    )


# ===== pure helpers =====

def test_fmt_elapsed_compact():
    assert _fmt_elapsed(0) == "0s"
    assert _fmt_elapsed(45) == "45s"
    assert _fmt_elapsed(4 * 60 + 32) == "4m32s"
    assert _fmt_elapsed(3600 + 4 * 60 + 32) == "1h04m32s"


def test_format_event_lines():
    assert _format_event({"kind": "tool_call", "tool": "wiki_search", "brief": "{query: 'x'}"}) == (
        "🔧 wiki_search {query: 'x'}"
    )
    assert _format_event({"kind": "tool_result", "text": "3 results"}) == "→ 3 results"
    assert _format_event({"kind": "summary", "text": "done well"}) == "✅ done: done well"
    assert _format_event({"kind": "follow_up", "question": "why?"}) == "❓ follow-up: why?"
    assert _format_event({"kind": "start", "type": "executor", "prompt": "p"}) == ""


def test_fit_window_truncates_monster_line_and_trims_oldest():
    # A single monster line is hard-truncated to fit.
    content = _fit_window(["x" * 5000], "⏱ 1s · running")
    assert len(content) <= DISCORD_LIMIT and "…" in content

    # Many long lines: oldest dropped until it fits; newest always kept.
    lines = [f"line{i} " + "y" * 100 for i in range(100)]
    content = _fit_window(lines, "⏱ 1s · running")
    assert len(content) <= DISCORD_LIMIT
    assert "line99 " in content and "line0 " not in content


def test_tail_log_reads_last_lines(tmp_path):
    (tmp_path / "log.txt").write_text("\n".join(f"L{i}" for i in range(100)))
    tail = _tail_log(tmp_path)
    lines = tail.splitlines()
    assert len(lines) == RUN_TAIL_LINES
    assert lines[-1] == "L99"
    assert _tail_log(tmp_path / "missing") is None


# ===== tool-call feed: a new posted message per flush =====

def test_feed_posts_a_new_message_per_flush():
    forum = FakeForum()
    st = _thread(forum)

    async def scenario():
        await st.on_event({"kind": "tool_call", "tool": "web_search", "brief": "{query: 'x'}"})
        await st.flush()  # posts batch #1
        await st.on_event({"kind": "tool_result", "text": "3 results"})
        await st.flush()  # posts batch #2 as its OWN new message

    _run(scenario())
    assert len(forum.created) == 1  # one thread
    msgs = forum.threads[0].messages
    assert len(msgs) == 2  # one message per flush
    assert not msgs[0].edits and not msgs[1].edits  # posted, never edited
    assert "🔧 web_search {query: 'x'}" in msgs[0].content
    assert "→ 3 results" in msgs[1].content


def test_feed_batches_lines_since_last_flush():
    forum = FakeForum()
    st = _thread(forum)

    async def scenario():
        await st.on_event({"kind": "tool_call", "tool": "a", "brief": "{q: 1}"})
        await st.on_event({"kind": "tool_result", "text": "r1"})
        await st.on_event({"kind": "summary", "text": "s"})
        await st.flush()  # all three accumulated lines go out as ONE batched message

    _run(scenario())
    msgs = forum.threads[0].messages
    assert len(msgs) == 1
    content = msgs[0].content
    assert content == "🔧 a {q: 1}\n→ r1\n✅ done: s"  # plain lines, newline-joined
    # No fence, no elapsed footer on the feed itself.
    assert "```" not in content and "⏱" not in content


def test_feed_skips_empty_flushes():
    forum = FakeForum()
    st = _thread(forum)

    async def scenario():
        await st.flush()  # nothing buffered -> no post, no thread

    _run(scenario())
    assert forum.created == [] and forum.threads == []


def test_feed_splits_a_long_batch_to_the_limit():
    forum = FakeForum()
    st = _thread(forum)

    async def scenario():
        await st.on_event({"kind": "tool_result", "text": "z" * 5000})  # one huge line
        await st.flush()

    _run(scenario())
    msgs = forum.threads[0].messages
    assert len(msgs) >= 2
    assert all(len(m.content) <= DISCORD_LIMIT for m in msgs)


def test_feed_final_status_message_on_close():
    clock = _Clock(0.0)
    forum = FakeForum()
    st = _thread(forum, clock=clock)

    async def scenario():
        await st.on_event({"kind": "summary", "text": "all good"})
        clock.t = 12 * 60 + 8  # 12m08s
        await st.close({"status": "done", "summary": "final summary text"})

    _run(scenario())
    msgs = forum.threads[0].messages
    # The buffered summary line goes out as a normal batch...
    assert any("✅ done: all good" in m.content for m in msgs)
    # ...and the closing status line is its OWN final message.
    status = msgs[-1]
    assert status.content.startswith("✅ done · 12m08s")
    assert "final summary text" in status.content
    assert not status.edits  # posted, not edited


def test_close_is_idempotent():
    forum = FakeForum()
    st = _thread(forum)

    async def scenario():
        await st.close({"status": "failed"})  # posts one final status message
        n = len(forum.threads[0].messages)
        await st.close({"status": "failed"})  # a second close is a no-op
        return n

    n = _run(scenario())
    assert n == 1
    assert len(forum.threads[0].messages) == n


def test_thread_create_failure_is_swallowed():
    forum = FakeForum(fail=True)
    st = _thread(forum)

    async def scenario():
        await st.on_event({"kind": "tool_call", "tool": "a", "brief": "{}"})
        await st.flush()  # create_thread raises -> logged, dropped

    _run(scenario())  # must not raise
    assert forum.threads == []


# ===== live run window =====

def _seed_running(q: JobQueue, run_dir, lab_id="lab-a", job_id="j1") -> Job:
    q.submit(Job(id=job_id, lab_id=lab_id, submitted_at="2026-01-01T00:00:00", timeout_s=30))
    job = q.claim_next()
    q.annotate_running(job, run_dir=str(run_dir))
    return job


def test_run_window_waiting_state(tmp_path):
    forum = FakeForum()
    clock = _Clock(0.0)
    q = JobQueue(tmp_path / "queue")  # empty: no running job yet
    st = _thread(forum, clock=clock, lab_id="lab-a", queue=q)

    async def scenario():
        await st.on_event({"kind": "tool_call", "tool": "run_experiment", "brief": "{}"})
        clock.t = 3
        await st.flush()

    _run(scenario())
    # messages[0] = feed, messages[1] = the live run window.
    run_msg = forum.threads[0].messages[1]
    assert "(waiting for run to start …)" in run_msg.content
    assert run_msg.content.endswith("⏱ 3s · running")


def test_run_window_inert_without_lab_or_queue():
    forum = FakeForum()
    st = _thread(forum)  # no lab_id, no queue

    async def scenario():
        await st.on_event({"kind": "tool_call", "tool": "run_experiment", "brief": "{}"})
        await st.flush()

    _run(scenario())
    # Only the feed message — no run window without a lab + queue to tail.
    assert len(forum.threads[0].messages) == 1


def test_run_window_live_tail_then_final_status(tmp_path):
    forum = FakeForum()
    clock = _Clock(0.0)
    q = JobQueue(tmp_path / "queue")
    run_dir = tmp_path / "lab" / "lab-a" / "runs" / "1"
    run_dir.mkdir(parents=True)
    (run_dir / "log.txt").write_text("\n".join(f"step {i} loss {2.0 - i * 0.01}" for i in range(50)))
    job = _seed_running(q, run_dir)

    st = _thread(forum, clock=clock, lab_id="lab-a", queue=q)

    async def scenario():
        await st.on_event({"kind": "tool_call", "tool": "run_experiment", "brief": "{}"})
        clock.t = 10
        await st.flush()  # live: tail of log.txt + running footer
        # The worker finishes: job leaves running/ for done/ with a status.
        q.complete(job, status="ok", run_number=1, run_dir=str(run_dir), summary="run 1 ok")
        clock.t = 25
        await st.flush()  # gone from running/ -> final edit with the done status

    _run(scenario())
    run_msg = forum.threads[0].messages[1]
    # It is ONE edited-in-place message (window), showing the last ~20 lines only.
    assert "step 49" in run_msg.content and "step 30" in run_msg.content
    assert "step 29" not in run_msg.content  # older lines outside the window
    # Final footer carries the run's total elapsed + terminal status.
    assert run_msg.content.endswith("⏱ 25s · ok")
    assert run_msg.edits  # the same message was edited, not re-posted
    assert len(forum.threads[0].messages) == 2  # feed + the single run window


def test_run_window_vanished_message_is_reposted(tmp_path):
    # The run window edits one message; if that message is deleted, it re-posts a fresh
    # one (editing is correct for a live status display — unlike the posted feed).
    forum = FakeForum()
    clock = _Clock(0.0)
    q = JobQueue(tmp_path / "queue")
    run_dir = tmp_path / "runs" / "1"
    run_dir.mkdir(parents=True)
    (run_dir / "log.txt").write_text("step 0\n")
    _seed_running(q, run_dir)
    st = _thread(forum, clock=clock, lab_id="lab-a", queue=q)

    async def scenario():
        await st.on_event({"kind": "tool_call", "tool": "run_experiment", "brief": "{}"})
        await st.flush()  # feed batch (messages[0]) + run window (messages[1])
        forum.threads[0].messages[1].edit_fail = _NotFound("gone")
        clock.t = 5
        await st.flush()  # run-window edit fails as 'gone' -> re-post a fresh message

    _run(scenario())
    assert len(forum.threads[0].messages) == 3  # feed, ghost window, re-posted window


def test_run_window_final_status_on_close_if_still_running(tmp_path):
    # Session ends while the run is still in running/ (no done record): the window is
    # stamped with the session's terminal status rather than left live.
    forum = FakeForum()
    clock = _Clock(0.0)
    q = JobQueue(tmp_path / "queue")
    run_dir = tmp_path / "runs" / "1"
    run_dir.mkdir(parents=True)
    (run_dir / "log.txt").write_text("step 0\nstep 1\n")
    _seed_running(q, run_dir)
    st = _thread(forum, clock=clock, lab_id="lab-a", queue=q)

    async def scenario():
        await st.on_event({"kind": "tool_call", "tool": "run_experiment", "brief": "{}"})
        clock.t = 8
        await st.close({"status": "killed"})

    _run(scenario())
    run_msg = forum.threads[0].messages[1]
    assert run_msg.content.endswith("⏱ 8s · killed")


# ===== SessionFeed: factory, lab_id/queue wiring, disabled-when-unset =====

def test_disabled_when_forum_unset():
    feed = SessionFeed()
    assert feed.observer_factory("exec-1", "executor", "opus", "p", "lab-a") is None
    feed.bind(None)
    assert feed.observer_factory("exec-1", "executor", "opus", "p", None) is None


def test_factory_wires_lab_id_and_queue(tmp_path):
    forum = FakeForum()
    q = JobQueue(tmp_path / "queue")
    feed = SessionFeed(forum, queue=q, interval_s=999)

    async def scenario():
        cb = feed.observer_factory("exec-1", "executor", "opus", "train", lab_id="lab-a")
        assert cb is not None
        st = feed.sessions["exec-1"]
        assert st.lab_id == "lab-a" and st._queue is q
        await st.close({"status": "done"})  # cancel the (interval=999) flusher task

    _run(scenario())


def test_factory_returns_working_on_event(tmp_path):
    forum = FakeForum()
    feed = SessionFeed(forum, interval_s=999, clock=_Clock(0.0))

    async def scenario():
        cb = feed.observer_factory("exec-1", "executor", "opus", "p", lab_id=None)
        await cb({"kind": "tool_result", "text": "ok"})
        await feed.sessions["exec-1"].close({"status": "done", "summary": "s"})

    _run(scenario())
    msgs = forum.threads[0].messages
    assert any("→ ok" in m.content for m in msgs)  # buffered line posted as a batch
    assert msgs[-1].content.startswith("✅ done · 0s")  # closing status line
