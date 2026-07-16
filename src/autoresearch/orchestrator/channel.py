"""The orchestrator's link to the operator — one duplex text channel.

The Discord bot implements this later (a channel-bound queue of inbound messages
plus ``channel.send``); the orchestrator only ever sees this interface, so tests
drive it with a trivial in-memory fake.

Contract:
- ``recv()`` blocks until the operator sends a message and returns its text. It MUST
  be cancellation-safe: the orchestrator races it against subagent completions and a
  digest timer with ``asyncio.wait`` and cancels the pending ``recv`` between waits,
  so a cancelled ``recv`` must not drop an already-delivered message. (The reference
  loop keeps a single ``recv`` task alive across waits and only ever cancels it on
  shutdown, so a queue-backed implementation is safe.)
- ``send(text)`` posts one message to the operator.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Channel(Protocol):
    async def recv(self) -> str: ...

    async def send(self, text: str) -> None: ...
