"""OpenAI Chat Completions client over httpx.

TODO(implement): translate base.Message <-> OpenAI messages/tool_calls, track usage.
"""
from __future__ import annotations

import httpx

from ..config import ModelEndpoint
from .base import LLMClient, LLMResponse, Message, ToolSpec


class OpenAIClient(LLMClient):
    def __init__(self, endpoint: ModelEndpoint):
        self.endpoint = endpoint
        self._http = httpx.AsyncClient(
            base_url=endpoint.base_url,
            headers={"authorization": f"Bearer {endpoint.resolve_api_key()}"},
            timeout=600,
        )

    async def complete(
        self, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> LLMResponse:
        raise NotImplementedError("port request/response translation from research-bot")


def make_client(endpoint: ModelEndpoint) -> LLMClient:
    """Pick a client implementation from the endpoint's base_url."""
    from .anthropic import AnthropicClient

    if "anthropic" in endpoint.base_url:
        return AnthropicClient(endpoint)
    return OpenAIClient(endpoint)
