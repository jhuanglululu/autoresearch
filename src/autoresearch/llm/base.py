"""Provider-agnostic message/tool types for the thin async LLM loop.

Both clients (anthropic.py, openai.py) translate to/from these; everything above
this layer is provider-blind. Pattern follows research-bot's src/llm/base.py, kept
minimal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..config import ModelEndpoint


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


@dataclass
class Usage:
    """Cumulative token/call totals for one client, summed across complete()
    calls. Mutable on purpose — the orchestrator's digests read these running
    totals off ``client.usage``."""

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def record(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.calls += 1

    def cost_usd(self, price_in: float, price_out: float) -> float:
        """Dollar cost of the accumulated tokens. ``price_in``/``price_out`` are
        USD per MILLION tokens (as configured in models.toml)."""
        return (
            self.input_tokens * price_in + self.output_tokens * price_out
        ) / 1_000_000


class SpendCapExceeded(Exception):
    """Raised when a client's cumulative spend has reached its endpoint's cap.

    Carries ``endpoint`` (name), ``spent`` (USD so far), and ``cap`` (USD) so the
    caller can report exactly how much of the budget was used."""

    def __init__(self, endpoint: str, spent: float, cap: float) -> None:
        self.endpoint = endpoint
        self.spent = spent
        self.cap = cap
        super().__init__(
            f"spend cap reached for {endpoint}: ${spent:.2f} of ${cap:.2f}"
        )


def raise_if_capped(endpoint: "ModelEndpoint", usage: Usage) -> None:
    """Refuse a request when ``usage`` has already reached the endpoint's cap.

    Called at the TOP of complete(), BEFORE the HTTP request. Semantics: the call
    that crosses the cap is allowed to finish (its cost lands in ``usage`` after it
    returns); the NEXT call — for which cumulative spend is already at/over ``cap`` —
    is the one refused. A cap with prices missing is unenforceable and skipped (this
    is rejected up front in load_models, so it should not happen in practice)."""
    cap = endpoint.cap
    if cap is None or endpoint.price_in is None or endpoint.price_out is None:
        return
    spent = usage.cost_usd(endpoint.price_in, endpoint.price_out)
    if spent >= cap:
        raise SpendCapExceeded(endpoint.name, spent, cap)


class LLMClient:
    """Interface both provider clients implement."""

    async def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> LLMResponse:
        raise NotImplementedError
