"""Tests for the Discord front-end (bot/discord_bot.py) and process wiring
(__main__.py) — the pure parts only, no real Discord connection or token.

``discord`` is imported lazily inside ``build_client``; nothing here touches it, so
the module and its helpers import token-free. Orchestrator accessors are checked
against the tool executors using the fakes from test_orchestrator.py.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from autoresearch.__main__ import _load_env
from autoresearch.bot.discord_bot import (
    DISCORD_LIMIT,
    DiscordChannel,
    answer_aside,
    extract_command_text,
    is_authorized,
    split_message,
)

# Reuse the orchestrator test fakes/builders (prepend import mode puts tests/ on path).
from test_orchestrator import FakeRunner, _orch


# ===== message splitting =====

def test_short_text_is_one_chunk():
    assert split_message("hello") == ["hello"]
    assert split_message("a" * DISCORD_LIMIT) == ["a" * DISCORD_LIMIT]


def test_long_text_splits_on_newlines_under_limit():
    text = "\n".join(f"line {i} " + "x" * 80 for i in range(200))
    chunks = split_message(text)
    assert len(chunks) > 1
    assert all(len(c) <= DISCORD_LIMIT for c in chunks)
    # Newline-joining the chunks recovers the original (breaks happen at newlines).
    assert "\n".join(chunks) == text


def test_hard_cap_on_a_single_overlong_line():
    text = "x" * 5000  # no newline to break on
    chunks = split_message(text)
    assert all(len(c) <= DISCORD_LIMIT for c in chunks)
    assert len(chunks) == 3  # 2000 + 2000 + 1000
    assert "".join(chunks) == text


def test_code_fences_preserved_across_split():
    code = "\n".join(f"print({i})  # padding padding padding" for i in range(120))
    text = f"intro paragraph\n```python\n{code}\n```\nouttro paragraph"
    chunks = split_message(text)

    assert len(chunks) > 1
    assert all(len(c) <= DISCORD_LIMIT for c in chunks)
    # Every emitted chunk has balanced fences (valid markdown on its own).
    assert all(c.count("```") % 2 == 0 for c in chunks)
    # No code line is lost across the boundary.
    for i in range(120):
        assert f"print({i})" in "".join(chunks)


# ===== DiscordChannel queue plumbing =====

def test_send_splits_and_posts_each_chunk():
    channel = AsyncMock()
    dc = DiscordChannel(channel=channel)
    asyncio.run(dc.send("x" * 5000))
    assert channel.send.await_count == 3
    posted = [c.args[0] for c in channel.send.await_args_list]
    assert all(len(p) <= DISCORD_LIMIT for p in posted)
    assert "".join(posted) == "x" * 5000


def test_send_without_bound_channel_drops_quietly():
    dc = DiscordChannel(channel=None)
    asyncio.run(dc.send("nowhere to go"))  # must not raise


def test_send_swallows_network_errors():
    channel = AsyncMock()
    channel.send.side_effect = RuntimeError("discord down")
    dc = DiscordChannel(channel=channel)
    asyncio.run(dc.send("hello"))  # logged + dropped, never propagates
    assert channel.send.await_count == 1


def test_send_ignores_empty_text():
    channel = AsyncMock()
    dc = DiscordChannel(channel=channel)
    asyncio.run(dc.send("   "))
    channel.send.assert_not_awaited()


def test_recv_returns_posted_inbound():
    dc = DiscordChannel()

    async def scenario():
        dc.post_inbound("first")
        dc.post_inbound("second")
        return await dc.recv(), await dc.recv()

    assert asyncio.run(scenario()) == ("first", "second")


# ===== admin / channel gating =====

def test_gating_allows_only_admin_in_bound_channel():
    kw = dict(admin_id=42, bound_channel_id=99)
    assert is_authorized(42, 99, **kw) is True
    assert is_authorized(7, 99, **kw) is False  # not the admin
    assert is_authorized(42, 100, **kw) is False  # wrong channel
    assert is_authorized(42, 99, is_bot=True, **kw) is False  # a bot (incl. itself)


def test_gating_denies_when_admin_unset():
    assert is_authorized(42, 99, admin_id=None, bound_channel_id=99) is False


# ===== mention gating (extract_command_text) =====

def test_extract_strips_plain_mention_at_start():
    assert extract_command_text("<@123> run the plan", 123) == "run the plan"


def test_extract_strips_nick_mention():
    # The <@!id> "nickname" form is recognised too.
    assert extract_command_text("<@!123> status please", 123) == "status please"


def test_extract_mention_not_at_start():
    # The mention may sit anywhere in the message, not only the first token.
    assert extract_command_text("hey <@123> spawn a researcher", 123) == "hey spawn a researcher"
    assert extract_command_text("do the thing <@!123>", 123) == "do the thing"


def test_extract_multiple_mentions_all_stripped():
    assert extract_command_text("<@123> ping <@!123> pong", 123) == "ping pong"


def test_extract_no_mention_returns_none():
    assert extract_command_text("just chatting in the channel", 123) is None
    # A DIFFERENT bot/user id mentioned is not our mention.
    assert extract_command_text("<@999> not for us", 123) is None
    assert extract_command_text("", 123) is None


def test_extract_mention_only_yields_empty_string():
    # Mentioned but nothing else said: "" (distinct from None) — caller posts nothing.
    assert extract_command_text("<@123>", 123) == ""


# ===== .env loader =====

def test_load_env_parses_and_never_overrides(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n"
        "\n"
        "DISCORD_BOT_TOKEN=abc123\n"
        'AUTORESEARCH_ADMIN="4242"\n'
        "AUTORESEARCH_CHANNEL = 99  \n"
        "WEIRD=a=b=c\n"
        "PRESET=fromfile\n"
    )
    for key in ("DISCORD_BOT_TOKEN", "AUTORESEARCH_ADMIN", "AUTORESEARCH_CHANNEL", "WEIRD"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PRESET", "already-set")

    _load_env(env)

    import os
    assert os.environ["DISCORD_BOT_TOKEN"] == "abc123"
    assert os.environ["AUTORESEARCH_ADMIN"] == "4242"  # quotes stripped
    assert os.environ["AUTORESEARCH_CHANNEL"] == "99"  # whitespace trimmed
    assert os.environ["WEIRD"] == "a=b=c"  # only the first '=' splits
    assert os.environ["PRESET"] == "already-set"  # existing env not overridden


def test_load_env_missing_file_is_noop(tmp_path):
    _load_env(tmp_path / "does-not-exist.env")  # must not raise


# ===== /btw side-channel handler (answer_aside) =====

class _AsideOrch:
    def __init__(self, result=None, exc=None):
        self._result = result
        self._exc = exc
        self.asked: list[str] = []

    async def aside(self, question: str) -> str:
        self.asked.append(question)
        if self._exc:
            raise self._exc
        return self._result


def test_answer_aside_returns_answer_not_error():
    orch = _AsideOrch(result="val_loss was 2.1")
    text, is_error = asyncio.run(answer_aside(orch, "what was the val_loss?"))
    assert (text, is_error) == ("val_loss was 2.1", False)
    assert orch.asked == ["what was the val_loss?"]


def test_answer_aside_swallows_exception_into_ephemeral_error():
    orch = _AsideOrch(exc=RuntimeError("boom"))
    text, is_error = asyncio.run(answer_aside(orch, "anything"))
    assert is_error is True
    assert "couldn't" in text and "RuntimeError" in text  # friendly, names the failure


def test_answer_aside_blank_becomes_placeholder():
    orch = _AsideOrch(result="   ")
    text, is_error = asyncio.run(answer_aside(orch, "q"))
    assert (text, is_error) == ("(no answer)", False)


# ===== Orchestrator public accessors match the tool executors =====

def test_status_and_sessions_accessors_match_tools(tmp_path):
    orch = _orch(tmp_path, [])
    orch._sessions.create("executor", "opus", "build X", FakeRunner())

    assert orch.status_text() == orch._tool_get_status()
    assert orch.sessions_text() == orch._tool_list_sessions()
    assert "GPU queue:" in orch.status_text()
    assert "exec-1" in orch.sessions_text()


def test_kill_session_accessor_matches_tool(tmp_path):
    orch = _orch(tmp_path, [])

    async def scenario():
        via_method = await orch.kill_session("nope")
        via_tool = await orch._tool_kill({"session_id": "nope"})
        return via_method, via_tool

    via_method, via_tool = asyncio.run(scenario())
    assert via_method == via_tool == "no such session: 'nope'."
