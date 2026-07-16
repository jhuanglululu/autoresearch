"""OpenAI Chat Completions client over httpx (no SDK).

Translates the provider-agnostic base.Message/ToolCall types to and from the
Chat Completions message/tool_call shape, retries transient failures, and
accumulates token usage for the orchestrator's digests.
"""
from __future__ import annotations

import json
from typing import Any

import httpx

from ..config import ModelEndpoint
from ._http import post_json
from .base import (
    LLMClient,
    LLMResponse,
    Message,
    ToolCall,
    ToolSpec,
    Usage,
    raise_if_capped,
)

_COMPLETIONS_PATH = "chat/completions"

# Chat Completions finish_reason -> our provider-agnostic stop_reason vocabulary
# (aligned with Anthropic's, so callers see one set of stop reasons).
_FINISH_TO_STOP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "content_filter",
}


class OpenAIClient(LLMClient):
    def __init__(
        self, endpoint: ModelEndpoint, *, transport: httpx.BaseTransport | None = None
    ):
        self.endpoint = endpoint
        self.usage = Usage()
        self._http = httpx.AsyncClient(
            base_url=endpoint.base_url.rstrip("/") + "/",
            headers={"authorization": f"Bearer {endpoint.resolve_api_key()}"},
            timeout=600,
            transport=transport,
        )

    async def complete(
        self, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> LLMResponse:
        # Refuse BEFORE spending: if cumulative cost already reached the cap, the
        # previous call was the one allowed to cross the line — this one is refused.
        raise_if_capped(self.endpoint, self.usage)
        api_messages: list[dict] = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages.extend(_to_api_messages(messages))

        body: dict[str, Any] = {
            "model": self.endpoint.model,
            "messages": api_messages,
        }
        if tools:
            body["tools"] = [_to_api_tool(t) for t in tools]

        data = await post_json(self._http, _COMPLETIONS_PATH, body, provider="openai")
        return self._parse_response(data)

    def _parse_response(self, data: dict) -> LLMResponse:
        choice = data["choices"][0]
        msg = choice.get("message") or {}
        text = msg.get("content") or ""

        tool_calls: list[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            tool_calls.append(
                ToolCall(
                    id=tc["id"],
                    name=fn["name"],
                    arguments=json.loads(fn.get("arguments") or "{}"),
                )
            )

        usage = data.get("usage") or {}
        input_tokens = usage.get("prompt_tokens", 0) or 0
        output_tokens = usage.get("completion_tokens", 0) or 0
        self.usage.record(input_tokens, output_tokens)

        finish = choice.get("finish_reason") or ""
        return LLMResponse(
            message=Message(role="assistant", content=text, tool_calls=tool_calls),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=_FINISH_TO_STOP.get(finish, finish),
        )


def _to_api_tool(tool: ToolSpec) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


def _tool_call_payload(tc: ToolCall) -> dict:
    return {
        "id": tc.id,
        "type": "function",
        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
    }


def _to_api_messages(messages: list[Message]) -> list[dict]:
    """Build the OpenAI ``messages`` array. Assistant tool calls carry
    JSON-string arguments; role="tool" results carry their tool_call_id."""
    result: list[dict] = []
    for m in messages:
        if m.role == "tool":
            result.append(
                {
                    "role": "tool",
                    "tool_call_id": m.tool_call_id,
                    "content": m.content,
                }
            )
        elif m.role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": m.content or None}
            if m.tool_calls:
                entry["tool_calls"] = [_tool_call_payload(tc) for tc in m.tool_calls]
            result.append(entry)
        else:  # "user" (and any stray "system")
            result.append({"role": m.role, "content": m.content})
    return result


def make_client(endpoint: ModelEndpoint) -> LLMClient:
    """Pick a client implementation from the endpoint's base_url."""
    from .anthropic import AnthropicClient

    if "anthropic" in endpoint.base_url:
        return AnthropicClient(endpoint)
    return OpenAIClient(endpoint)
