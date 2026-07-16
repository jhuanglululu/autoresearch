"""Tests for the two httpx LLM clients (llm/anthropic.py, llm/openai.py).

No network: httpx.MockTransport fakes both APIs. Tests are plain sync functions
driving the async clients with asyncio.run(), so the suite needs no
pytest-asyncio. Coverage: request translation (system + tool spec + tool_result
round-trip in the outgoing JSON, for BOTH providers), response parsing (text and
tool-call), retry-then-succeed on 429 with Retry-After, usage accumulation, and
make_client dispatch.
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from autoresearch.config import ModelEndpoint
from autoresearch.llm.anthropic import AnthropicClient
from autoresearch.llm.openai import OpenAIClient, make_client
from autoresearch.llm.base import Message, SpendCapExceeded, ToolCall, ToolSpec, Usage


ANTHROPIC_URL = "https://api.anthropic.com"
OPENAI_URL = "https://api.openai.com/v1"

TOOLS = [
    ToolSpec(
        name="search",
        description="Search the web.",
        input_schema={
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
    )
]

# user -> assistant(tool_call) -> two consecutive tool results (same turn on the wire)
ROUNDTRIP_MESSAGES = [
    Message(role="user", content="find X"),
    Message(
        role="assistant",
        content="I'll search",
        tool_calls=[ToolCall(id="tu_1", name="search", arguments={"q": "X"})],
    ),
    Message(role="tool", content="result A", tool_call_id="tu_1"),
    Message(role="tool", content="result B", tool_call_id="tu_2"),
]


@pytest.fixture(autouse=True)
def _api_keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai-test")


def _endpoint(provider: str, reasoning_effort: str | None = None) -> ModelEndpoint:
    if provider == "anthropic":
        return ModelEndpoint(
            name="orch",
            base_url=ANTHROPIC_URL,
            model="claude-opus-4-8",
            api_key_env="ANTHROPIC_API_KEY",
        )
    return ModelEndpoint(
        name="gpt",
        base_url=OPENAI_URL,
        model="gpt-5.6-sol",
        api_key_env="OPENAI_API_KEY",
        reasoning_effort=reasoning_effort,
    )


def _mock(provider, responses, *, reasoning_effort: str | None = None):
    """Build a client whose transport replays ``responses`` (list of dicts or
    httpx.Response) in order, and a ``captured`` list receiving each request's
    parsed JSON body. Returns (client, captured, calls) where calls is a
    single-item mutable counter of how many requests were made."""
    captured: list[dict] = []
    calls = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        item = responses[min(calls[0], len(responses) - 1)]
        calls[0] += 1
        if isinstance(item, httpx.Response):
            return item
        return httpx.Response(200, json=item)

    transport = httpx.MockTransport(handler)
    if provider == "anthropic":
        client = AnthropicClient(_endpoint(provider), transport=transport)
    else:
        client = OpenAIClient(
            _endpoint(provider, reasoning_effort), transport=transport
        )
    return client, captured, calls


# --------------------------------------------------------------------------- #
# Request translation
# --------------------------------------------------------------------------- #

def test_anthropic_request_translation():
    resp = {"content": [{"type": "text", "text": "ok"}], "usage": {}, "stop_reason": "end_turn"}
    client, captured, _ = _mock("anthropic", [resp])
    asyncio.run(client.complete("You are a researcher.", ROUNDTRIP_MESSAGES, TOOLS))

    body = captured[0]
    assert body["model"] == "claude-opus-4-8"
    assert body["max_tokens"] == 16000

    # System prompt carries the ephemeral cache breakpoint.
    assert body["system"] == [
        {
            "type": "text",
            "text": "You are a researcher.",
            "cache_control": {"type": "ephemeral"},
        }
    ]

    # Tool spec -> Anthropic tool shape.
    assert body["tools"] == [
        {
            "name": "search",
            "description": "Search the web.",
            "input_schema": TOOLS[0].input_schema,
        }
    ]

    msgs = body["messages"]
    # user, assistant(text+tool_use), and ONE merged tool_result user turn.
    assert len(msgs) == 3
    assert msgs[0] == {"role": "user", "content": "find X"}
    assert msgs[1] == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "I'll search"},
            {"type": "tool_use", "id": "tu_1", "name": "search", "input": {"q": "X"}},
        ],
    }
    assert msgs[2] == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "tu_1", "content": "result A"},
            {"type": "tool_result", "tool_use_id": "tu_2", "content": "result B"},
        ],
    }


def _openai_ok(text="ok"):
    """A minimal completed Responses reply carrying one text item."""
    return {
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": text}]}
        ],
        "usage": {},
        "status": "completed",
    }


def test_openai_request_translation():
    client, captured, _ = _mock("openai", [_openai_ok()])
    asyncio.run(client.complete("You are a researcher.", ROUNDTRIP_MESSAGES, TOOLS))

    body = captured[0]
    assert body["model"] == "gpt-5.6-sol"
    assert body["max_output_tokens"] == 16000
    # System prompt goes to the top-level instructions field.
    assert body["instructions"] == "You are a researcher."

    # Tool spec -> flat Responses tool shape (no nested "function" object).
    assert body["tools"] == [
        {
            "type": "function",
            "name": "search",
            "description": "Search the web.",
            "parameters": TOOLS[0].input_schema,
        }
    ]

    inp = body["input"]
    # user, assistant text, one function_call, and two function_call_output items.
    assert inp[0] == {"role": "user", "content": "find X"}
    assert inp[1] == {"role": "assistant", "content": "I'll search"}

    call = inp[2]
    assert call["type"] == "function_call"
    assert call["call_id"] == "tu_1"
    assert call["name"] == "search"
    # arguments is a JSON *string*.
    assert isinstance(call["arguments"], str)
    assert json.loads(call["arguments"]) == {"q": "X"}

    # tool results become function_call_output items keyed by matching call_id.
    assert inp[3] == {
        "type": "function_call_output",
        "call_id": "tu_1",
        "output": "result A",
    }
    assert inp[4] == {
        "type": "function_call_output",
        "call_id": "tu_2",
        "output": "result B",
    }
    assert len(inp) == 5


def test_openai_omits_instructions_tools_and_reasoning_when_unset():
    client, captured, _ = _mock("openai", [_openai_ok()])
    asyncio.run(client.complete("", [Message(role="user", content="hi")], []))
    body = captured[0]
    assert "instructions" not in body
    assert "tools" not in body
    assert "reasoning" not in body  # reasoning_effort not configured on the endpoint


def test_openai_reasoning_effort_passed_through_when_configured():
    client, captured, _ = _mock("openai", [_openai_ok()], reasoning_effort="medium")
    asyncio.run(client.complete("", [Message(role="user", content="hi")], TOOLS))
    assert captured[0]["reasoning"] == {"effort": "medium"}


def test_anthropic_omits_system_and_tools_when_empty():
    resp = {"content": [{"type": "text", "text": "ok"}], "usage": {}, "stop_reason": "end_turn"}
    client, captured, _ = _mock("anthropic", [resp])
    asyncio.run(client.complete("", [Message(role="user", content="hi")], []))
    body = captured[0]
    assert "system" not in body
    assert "tools" not in body


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #

def test_anthropic_response_text_only():
    resp = {
        "content": [{"type": "text", "text": "hello world"}],
        "usage": {"input_tokens": 12, "output_tokens": 4},
        "stop_reason": "end_turn",
    }
    client, _, _ = _mock("anthropic", [resp])
    out = asyncio.run(client.complete("", [Message(role="user", content="hi")], []))
    assert out.message.role == "assistant"
    assert out.message.content == "hello world"
    assert out.message.tool_calls == []
    assert out.input_tokens == 12
    assert out.output_tokens == 4
    assert out.stop_reason == "end_turn"


def test_anthropic_response_tool_call():
    resp = {
        "content": [
            {"type": "text", "text": "searching"},
            {"type": "tool_use", "id": "tu_9", "name": "search", "input": {"q": "cats"}},
        ],
        "usage": {"input_tokens": 3, "output_tokens": 8},
        "stop_reason": "tool_use",
    }
    client, _, _ = _mock("anthropic", [resp])
    out = asyncio.run(client.complete("", [Message(role="user", content="hi")], TOOLS))
    assert out.message.content == "searching"
    assert out.stop_reason == "tool_use"
    assert out.message.tool_calls == [
        ToolCall(id="tu_9", name="search", arguments={"q": "cats"})
    ]


def test_openai_response_text_only():
    resp = {
        "output": [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "thinking..."}],
            },
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "hello"}],
            },
        ],
        "usage": {"input_tokens": 20, "output_tokens": 6},
        "status": "completed",
    }
    client, _, _ = _mock("openai", [resp])
    out = asyncio.run(client.complete("", [Message(role="user", content="hi")], []))
    assert out.message.content == "hello"  # reasoning item skipped, output_text kept
    assert out.message.tool_calls == []
    assert out.input_tokens == 20
    assert out.output_tokens == 6
    assert out.stop_reason == "end_turn"  # completed, no function_call items


def test_openai_response_tool_call():
    resp = {
        "output": [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "picking a tool"}],
            },
            {
                "type": "function_call",
                "id": "fc_abc",  # the output item's own id...
                "call_id": "call_7",  # ...distinct from the call_id we must echo back
                "name": "search",
                "arguments": '{"q": "dogs"}',
            },
        ],
        "usage": {"input_tokens": 5, "output_tokens": 9},
        "status": "completed",
    }
    client, _, _ = _mock("openai", [resp])
    out = asyncio.run(client.complete("", [Message(role="user", content="hi")], TOOLS))
    assert out.message.content == ""
    assert out.stop_reason == "tool_use"  # completed + a function_call item present
    # ToolCall.id is the call_id (NOT the item id fc_abc) so it round-trips.
    assert out.message.tool_calls == [
        ToolCall(id="call_7", name="search", arguments={"q": "dogs"})
    ]


def test_openai_response_incomplete_maps_to_max_tokens():
    resp = {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "partial"}],
            }
        ],
        "usage": {"input_tokens": 4, "output_tokens": 16000},
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
    }
    client, _, _ = _mock("openai", [resp])
    out = asyncio.run(client.complete("", [Message(role="user", content="hi")], []))
    assert out.message.content == "partial"
    assert out.stop_reason == "max_tokens"


# --------------------------------------------------------------------------- #
# Retry policy
# --------------------------------------------------------------------------- #

def test_retry_then_succeed_on_429_honours_retry_after(monkeypatch):
    delays: list[float] = []

    async def fake_sleep(seconds):
        delays.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    success = {"content": [{"type": "text", "text": "ok"}], "usage": {"input_tokens": 1, "output_tokens": 1}, "stop_reason": "end_turn"}
    responses = [
        httpx.Response(429, headers={"retry-after": "2"}, text="slow down"),
        success,
    ]
    client, _, calls = _mock("anthropic", responses)
    out = asyncio.run(client.complete("", [Message(role="user", content="hi")], []))

    assert calls[0] == 2  # one 429, then one success
    assert out.message.content == "ok"
    assert delays == [2.0]  # Retry-After honoured verbatim


def test_retry_exhaustion_raises(monkeypatch):
    async def fake_sleep(seconds):
        pass

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    from autoresearch.llm._http import LLMError, MAX_TRIES

    responses = [httpx.Response(503, text="overloaded")]
    client, _, calls = _mock("openai", responses)
    with pytest.raises(LLMError):
        asyncio.run(client.complete("", [Message(role="user", content="hi")], []))
    assert calls[0] == MAX_TRIES


def test_non_retryable_4xx_raises_immediately():
    from autoresearch.llm._http import LLMError

    responses = [httpx.Response(400, text="bad request")]
    client, _, calls = _mock("anthropic", responses)
    with pytest.raises(LLMError):
        asyncio.run(client.complete("", [Message(role="user", content="hi")], []))
    assert calls[0] == 1  # no retries on a 4xx


# --------------------------------------------------------------------------- #
# Usage accumulation
# --------------------------------------------------------------------------- #

def test_usage_accumulates_across_calls():
    responses = [
        {"content": [{"type": "text", "text": "a"}], "usage": {"input_tokens": 10, "output_tokens": 3}, "stop_reason": "end_turn"},
        {"content": [{"type": "text", "text": "b"}], "usage": {"input_tokens": 5, "output_tokens": 7}, "stop_reason": "end_turn"},
    ]
    client, _, _ = _mock("anthropic", responses)
    assert client.usage.calls == 0
    asyncio.run(client.complete("", [Message(role="user", content="hi")], []))
    asyncio.run(client.complete("", [Message(role="user", content="hi")], []))
    assert client.usage.calls == 2
    assert client.usage.input_tokens == 15
    assert client.usage.output_tokens == 10


# --------------------------------------------------------------------------- #
# make_client dispatch
# --------------------------------------------------------------------------- #

def test_make_client_dispatch():
    assert isinstance(make_client(_endpoint("anthropic")), AnthropicClient)
    assert isinstance(make_client(_endpoint("openai")), OpenAIClient)


# --------------------------------------------------------------------------- #
# Spend caps
# --------------------------------------------------------------------------- #

def test_cost_usd_math():
    u = Usage()
    u.record(1_000_000, 500_000)
    # price_in=$10/1M, price_out=$50/1M -> $10 + $25 = $35.
    assert u.cost_usd(10.0, 50.0) == 35.0


def test_cap_refuses_before_any_request():
    """At/over cap, complete() raises SpendCapExceeded WITHOUT making an HTTP call."""
    calls = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        calls[0] += 1
        return httpx.Response(200, json={"content": [], "usage": {}, "stop_reason": "end_turn"})

    endpoint = ModelEndpoint(
        name="orch",
        base_url=ANTHROPIC_URL,
        model="claude-opus-4-8",
        api_key_env="ANTHROPIC_API_KEY",
        cap=0.5,
        price_in=10.0,
        price_out=30.0,
    )
    client = AnthropicClient(endpoint, transport=httpx.MockTransport(handler))
    # 100k output tokens at $30/1M = $3.00, already over the $0.50 cap.
    client.usage.record(0, 100_000)

    with pytest.raises(SpendCapExceeded) as exc:
        asyncio.run(client.complete("", [Message(role="user", content="hi")], []))

    assert calls[0] == 0  # refused before the request
    assert exc.value.endpoint == "orch"
    assert exc.value.cap == 0.5
    assert exc.value.spent == 3.0
    assert "spend cap" in str(exc.value)


def test_cap_allows_the_call_that_crosses_then_refuses_next():
    """The call that crosses the cap runs; the following one is refused."""
    resp = {
        "content": [{"type": "text", "text": "ok"}],
        "usage": {"input_tokens": 0, "output_tokens": 100_000},
        "stop_reason": "end_turn",
    }
    calls = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        calls[0] += 1
        return httpx.Response(200, json=resp)

    endpoint = ModelEndpoint(
        name="orch",
        base_url=ANTHROPIC_URL,
        model="claude-opus-4-8",
        api_key_env="ANTHROPIC_API_KEY",
        cap=1.0,
        price_in=0.0,
        price_out=30.0,
    )
    client = AnthropicClient(endpoint, transport=httpx.MockTransport(handler))

    # First call: under cap (spend is 0), runs and pushes spend to $3.00.
    asyncio.run(client.complete("", [Message(role="user", content="hi")], []))
    assert calls[0] == 1
    # Second call: already over cap -> refused, no new request.
    with pytest.raises(SpendCapExceeded):
        asyncio.run(client.complete("", [Message(role="user", content="hi")], []))
    assert calls[0] == 1


def test_no_cap_never_refuses():
    resp = {"content": [{"type": "text", "text": "ok"}], "usage": {}, "stop_reason": "end_turn"}
    client, _, calls = _mock("anthropic", [resp])
    client.usage.record(10_000_000, 10_000_000)  # huge spend, but no cap set
    asyncio.run(client.complete("", [Message(role="user", content="hi")], []))
    assert calls[0] == 1


def test_load_models_cap_without_price_raises(tmp_path):
    from autoresearch.config import load_models

    p = tmp_path / "models.toml"
    p.write_text(
        """
[orchestrator]
base_url    = "https://api.anthropic.com"
model       = "claude-fable-5"
api_key_env = "ANTHROPIC_API_KEY"
cap         = 50

[[subagent_model]]
name        = "opus"
base_url    = "https://api.anthropic.com"
model       = "claude-opus-4-8"
api_key_env = "ANTHROPIC_API_KEY"
price_in    = 5
price_out   = 25
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cap needs price_in/price_out"):
        load_models(p)
