"""Provider-agnostic message/tool types for the thin async LLM loop.

Both clients (anthropic.py, openai.py) translate to/from these; everything above
this layer is provider-blind. Pattern follows research-bot's src/llm/base.py, kept
minimal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # set on role="tool" results


@dataclass
class ToolSpec:
    name: str
    description: str  # loaded verbatim from prompt/tools/<name>.md
    input_schema: dict[str, Any]


@dataclass
class LLMResponse:
    message: Message
    input_tokens: int
    output_tokens: int
    stop_reason: str


class LLMClient:
    """Interface both provider clients implement."""

    async def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> LLMResponse:
        raise NotImplementedError
