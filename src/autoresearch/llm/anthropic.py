"""Anthropic Messages API client over httpx (no SDK).

Translates the provider-agnostic base.Message/ToolCall types to and from the
Messages API content-block shape, prompt-caches the system prompt, retries
transient failures, and accumulates token usage for the orchestrator's digests.
"""
from __future__ import annotations

from typing import Any

import httpx

from ..config import ModelEndpoint
from ._http import post_json
from .base import LLMClient, LLMResponse, Message, ToolCall, ToolSpec, Usage

MAX_TOKENS = 16000
_MESSAGES_PATH = "v1/messages"


class AnthropicClient(LLMClient):
    def __init__(
        self, endpoint: ModelEndpoint, *, transport: httpx.BaseTransport | None = None
    ):
        self.endpoint = endpoint
        self.usage = Usage()
        self._http = httpx.AsyncClient(
            base_url=endpoint.base_url.rstrip("/") + "/",
            headers={
                "x-api-key": endpoint.resolve_api_key(),
                "anthropic-version": "2023-06-01",
            },
            timeout=600,
            transport=transport,
        )

    async def complete(
        self, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self.endpoint.model,
            "max_tokens": MAX_TOKENS,
            "messages": _to_api_messages(messages),
        }
        if system:
            # Prompt caching: the system prompt is the stable prefix, so mark it
            # ephemeral. (research-bot caches only the system prompt, not tools.)
            body["system"] = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        if tools:
            body["tools"] = [_to_api_tool(t) for t in tools]

        data = await post_json(self._http, _MESSAGES_PATH, body, provider="anthropic")
        return self._parse_response(data)

    def _parse_response(self, data: dict) -> LLMResponse:
        text = ""
        tool_calls: list[ToolCall] = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block["id"],
                        name=block["name"],
                        arguments=block.get("input", {}),
                    )
                )

        usage = data.get("usage") or {}
        input_tokens = usage.get("input_tokens", 0) or 0
        output_tokens = usage.get("output_tokens", 0) or 0
        self.usage.record(input_tokens, output_tokens)

        return LLMResponse(
            message=Message(role="assistant", content=text, tool_calls=tool_calls),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=data.get("stop_reason") or "",
        )


def _to_api_tool(tool: ToolSpec) -> dict:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    }


def _tool_use_block(tc: ToolCall) -> dict:
    return {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}


def _is_tool_result_turn(entry: dict) -> bool:
    content = entry.get("content")
    return (
        entry["role"] == "user"
        and isinstance(content, list)
        and bool(content)
        and all(b.get("type") == "tool_result" for b in content)
    )


def _to_api_messages(messages: list[Message]) -> list[dict]:
    """Build the Anthropic ``messages`` array. role="tool" results become
    user-role tool_result blocks, and *consecutive* tool results merge into a
    single user turn (the API requires one user turn to carry them together)."""
    result: list[dict] = []
    for m in messages:
        if m.role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": m.tool_call_id,
                "content": m.content,
            }
            if result and _is_tool_result_turn(result[-1]):
                result[-1]["content"].append(block)
            else:
                result.append({"role": "user", "content": [block]})
        elif m.role == "assistant":
            content: list[dict] = []
            if m.content:
                content.append({"type": "text", "text": m.content})
            content.extend(_tool_use_block(tc) for tc in m.tool_calls)
            result.append({"role": "assistant", "content": content})
        else:  # "user" (and any stray "system" — fold into a plain user turn)
            result.append({"role": "user", "content": m.content})
    return result
