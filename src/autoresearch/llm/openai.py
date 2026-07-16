"""OpenAI Responses API client over httpx (no SDK).

Translates the provider-agnostic base.Message/ToolCall types to and from the
Responses API (``POST /v1/responses``) shape, retries transient failures, and
accumulates token usage for the orchestrator's digests.

Why the Responses API (not Chat Completions): the target model rejects function
tools combined with a reasoning effort on ``/v1/chat/completions`` — reasoning +
tools is only supported on ``/v1/responses``. This client is stateless: like the
Anthropic client, it resends the full conversation every turn and never sets
``store``/``previous_response_id``.
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

# Cap on generated tokens; mirrors the Anthropic client's MAX_TOKENS so both
# providers share one output budget.
MAX_OUTPUT_TOKENS = 16000
# base_url already ends in /v1, so this resolves to POST /v1/responses.
_RESPONSES_PATH = "responses"


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

        body: dict[str, Any] = {
            "model": self.endpoint.model,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "input": _to_api_input(messages),
        }
        if system:
            # The Responses API carries the system prompt as top-level instructions.
            body["instructions"] = system
        if tools:
            body["tools"] = [_to_api_tool(t) for t in tools]
        if self.endpoint.reasoning_effort:
            # Reasoning + function tools is exactly why we use /v1/responses.
            body["reasoning"] = {"effort": self.endpoint.reasoning_effort}

        data = await post_json(self._http, _RESPONSES_PATH, body, provider="openai")
        return self._parse_response(data)

    def _parse_response(self, data: dict) -> LLMResponse:
        text = ""
        tool_calls: list[ToolCall] = []
        for item in data.get("output") or []:
            item_type = item.get("type")
            if item_type == "reasoning":
                # Reasoning items are opaque summaries; nothing for us to surface.
                continue
            if item_type == "message":
                for part in item.get("content") or []:
                    if part.get("type") == "output_text":
                        text += part.get("text", "")
            elif item_type == "function_call":
                # A function_call item carries BOTH an "id" (the output item's own
                # id, e.g. "fc_...") and a "call_id" (e.g. "call_..."). The call_id
                # is what a function_call_output must echo back, so we store it as our
                # ToolCall.id to round-trip correctly on the next turn.
                tool_calls.append(
                    ToolCall(
                        id=item["call_id"],
                        name=item["name"],
                        arguments=json.loads(item.get("arguments") or "{}"),
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
            stop_reason=_stop_reason(data, bool(tool_calls)),
        )


def _stop_reason(data: dict, has_tool_calls: bool) -> str:
    """Map a Responses ``status`` (+ incomplete_details) onto our Anthropic-style
    stop_reason vocabulary."""
    status = data.get("status") or ""
    if status == "completed":
        return "tool_use" if has_tool_calls else "end_turn"
    if status == "incomplete":
        reason = (data.get("incomplete_details") or {}).get("reason") or ""
        if reason == "max_output_tokens":
            return "max_tokens"
        return reason or status
    return status


def _to_api_tool(tool: ToolSpec) -> dict:
    # Responses tools are flat (name/description/parameters at top level), unlike
    # Chat Completions which nests them under a "function" key.
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.input_schema,
    }


def _function_call_item(tc: ToolCall) -> dict:
    return {
        "type": "function_call",
        "call_id": tc.id,
        "name": tc.name,
        "arguments": json.dumps(tc.arguments),
    }


def _to_api_input(messages: list[Message]) -> list[dict]:
    """Build the Responses ``input`` array. Plain user/assistant turns are
    {role, content} items; an assistant tool call becomes its text item (if any)
    plus one function_call item per call; role="tool" results become
    function_call_output items keyed by call_id."""
    result: list[dict] = []
    for m in messages:
        if m.role == "tool":
            result.append(
                {
                    "type": "function_call_output",
                    "call_id": m.tool_call_id,
                    "output": m.content,
                }
            )
        elif m.role == "assistant":
            if m.content:
                result.append({"role": "assistant", "content": m.content})
            result.extend(_function_call_item(tc) for tc in m.tool_calls)
        else:  # "user" (and any stray "system"; system normally goes to instructions)
            result.append({"role": m.role, "content": m.content})
    return result


def make_client(endpoint: ModelEndpoint) -> LLMClient:
    """Pick a client implementation from the endpoint's base_url."""
    from .anthropic import AnthropicClient

    if "anthropic" in endpoint.base_url:
        return AnthropicClient(endpoint)
    return OpenAIClient(endpoint)
