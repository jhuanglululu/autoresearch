"""Anthropic Messages API client over httpx.

TODO(implement): translate base.Message <-> Anthropic content blocks, prompt-cache
the system prompt (cache_control: ephemeral), track usage totals for digests.
"""
from __future__ import annotations

import httpx

from ..config import ModelEndpoint
from .base import LLMClient, LLMResponse, Message, ToolSpec


class AnthropicClient(LLMClient):
    def __init__(self, endpoint: ModelEndpoint):
        self.endpoint = endpoint
        self._http = httpx.AsyncClient(
            base_url=endpoint.base_url,
            headers={
                "x-api-key": endpoint.resolve_api_key(),
                "anthropic-version": "2023-06-01",
            },
            timeout=600,
        )

    async def complete(
        self, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> LLMResponse:
        raise NotImplementedError("port request/response translation from research-bot")
